"""Addendum 230 (Amendment 230a): Option-1 Zero-LoRA Multimodal Living Buddy built on official `google/gemma-4-E4B-it`.

Combines all 3-seed proven winners from Addenda 226-229 directly on top of full multimodal Gemma 4 E4B (L=42, D=2560,
keeping native vision/video + streaming audio encoders 100% frozen):
  1. Awake online learning: Zero-initialized `O3` (Titans-Momentum) parallel SwiGLU fast-weight branches (12.1M params,
     permanently fp32 so small AdamW & gate updates never underflow!) grafted onto layers (12, 21, 30, 39) — bit-for-bit
     identical to base Gemma 4 E4B at step 0, strictly causal chunk C=64 in fp32, carrying live `fast_state` across
     prefill -> KV-cached decode (`logits_to_keep=0` for full KV-shared prefill!) across text, video & audio.
  2. Option-1 Zero-LoRA selective full-rank slow weights (compact ~656 MB bf16 diff slice saved, never duplicating 16 GB base!):
     - Stage A (`stage_a`): Upcast ONLY full-rank Layer 11-12 MLPs (157.3M params) to fp32 master during training under
       bf16 autocast + training-only `plato_head` (2560->256, dropped after Stage A -> 0 extra inference params),
       annealed `plato_w` [0.65, 0.90] -> 0.0 against 5-family Platonic Consensus `C5` + boosted `U_jung_c5` (3.5x),
       KL-anchored to exact base Gemma 4 E4B via chunked `lm_head` + `use_reentrant=False` gradient checkpointing (~18 GB peak!).
     - Stage B (`stage_b`): Freeze layers 0..39; unfreeze ONLY the 4 `O3` Titans-Momentum branches (12.1M fp32) + upper
       full-rank MLPs 40-41 (157.3M fp32 master during training, cast back to bf16 after), KL-anchored to Stage-A base,
       trained on `buddy_train230.jsonl` (natural spoken voice, zero 'As an AI...' boilerplate, calibrated `<search>`,
       honest ask/abstain, easy timeless controls, and 6-turn working-memory corrections) via B=1 gradient accumulation.
  3. Before-vs-after evaluation (`eval_all`):
     - Zero-leakage train-PCA blind Procrustes transfer to unseen Microsoft `Phi-4-mini` & AllenAI `OLMo 3 7B` + `items3` TF/TL
     - Held-out `HOUSE` (raw + length-residualized + `r_len` + length-reversed `perfection`) & `SELF2` alignment
     - Held-out neutral `gut` + `chat` NLL drift guardrail
     - Held-out `buddy_eval230.jsonl`: boilerplate rate, markdown rate, `<search>` tag & strict-only recall, easy-fact &
       abstain false-trigger rates, AND `working_mem_ttt` multi-turn & cold-state `O3 ON vs OFF` accuracy
     - Frozen Vision (red square) AND Frozen Audio (16 kHz spoken clip "The password is blue pelican.") multimodal sanity check!

Usage (24 GB CUDA GPU, `python`):
  python aion_8b/e4b_living.py test
  python aion_8b/e4b_living.py eval_base
  python aion_8b/e4b_living.py stage_a
  python aion_8b/e4b_living.py stage_b
  python aion_8b/e4b_living.py eval_all"""
import contextlib
import gc
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
ARCH = HERE.parent
PLATO = ARCH / "fluent" / "plato"
MID = ARCH / "fluent" / "mid"
DATA = HERE / "data"
RUNS = HERE / "runs"
CKPT = Path(os.environ.get("JPL_CKPT_DIR", str(Path.home() / ".jung_plato_lattice" / "ckpt")))

if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from fluent import arch_rot229, exp228, exp229, plato_data
from aion_8b import aion_data230 as bd

MODEL_ID = "google/gemma-4-E4B-it"
# Official Google Gemma 4 E4B recommended inference sampling settings (verified from model generation_config.json):
# lowering temperature below ~0.7 degrades Thinking Mode / chain-of-thought reasoning quality.
GEMMA4_DEFAULT_TEMP = 1.0
GEMMA4_DEFAULT_TOP_P = 0.95
GEMMA4_DEFAULT_TOP_K = 64
TITANS_LAYERS = (12, 21, 30, 39)
STAGE_A_MLP_LAYERS = (11, 12)
STAGE_B_MLP_LAYERS = (40, 41)
PLATO_LAYER = 12


def _atomic_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, p)


def _atomic_torch_save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.pt")
    torch.save(obj, tmp)
    os.replace(tmp, p)


def _autocast_ctx(dev):
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _pad_id(tok):
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    if tok.eos_token_id is not None:
        return int(tok.eos_token_id)
    return 0


def _gemma_encode_with_bos(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    bos = tok.bos_token_id
    return [bos] + ids if (bos is not None and (not ids or ids[0] != bos)) else ids


# ---------------------------------------------------------------- Exact Chunked lm_head + Softcap + CE + KL (Zero Full-Logit OOM!)

def _lm_head_softcap(base_model, h_chunk):
    lm_head = base_model.lm_head
    w_dt = lm_head.weight.dtype
    lg = lm_head(h_chunk.to(w_dt))
    softcap = base_model.config.get_text_config().final_logit_softcapping
    if softcap is not None:
        lg = torch.tanh(lg.float() / float(softcap)) * float(softcap)
        return lg
    return lg.float()


def chunked_lm_head_ce_kl(
    base_model,
    h_cur,
    y_ids=None,
    h_ref=None,
    mask=None,
    ce_w=1.0,
    kl_w=0.0,
    temp_kd=1.0,
    chunk_t=64,
    backward=False,
    return_grad=False,
):
    """Exact chunked lm_head + softcap + CE + KL directly from last_hidden_state (B, T, D).
    Never materializes full (B, T, 262144) logits! Caps peak vocab VRAM to ~chunk_t * V * 4 bytes (~67 MB).
    When `backward=True`, calls `h_cur.backward(h_grad)` directly; when `return_grad=True`, returns `h_grad`
    so callers with an auxiliary layer-12 loss can do ONE fused `torch.autograd.backward([h_cur, aux], [h_grad, None])`."""
    B, T, D = h_cur.shape
    dev = h_cur.device
    need_grad = bool((backward or return_grad) and h_cur.requires_grad)

    if mask is not None:
        tot_w = mask.float().sum().clamp_min(1.0)
    else:
        tot_w = torch.tensor(float(B * T), device=dev, dtype=torch.float32).clamp_min(1.0)

    h_grad = torch.zeros_like(h_cur) if need_grad else None
    tot_ce = 0.0
    tot_kl = 0.0
    t_kd = float(temp_kd)
    inv_t = 1.0 / t_kd
    t2 = t_kd * t_kd

    for s in range(0, T, chunk_t):
        e = min(T, s + chunk_t)
        hc = h_cur[:, s:e, :].detach()
        if need_grad:
            hc.requires_grad_(True)
        lg_c = _lm_head_softcap(base_model, hc)
        V = lg_c.shape[-1]
        mc = mask[:, s:e].float() if mask is not None else None

        loss_c = torch.zeros((), device=dev, dtype=torch.float32)
        if y_ids is not None and ce_w > 0.0:
            yc = y_ids[:, s:e]
            ce_tok = F.cross_entropy(lg_c.reshape(-1, V), yc.reshape(-1), reduction="none").view(B, e - s)
            ce_part = ((ce_tok * mc).sum() if mc is not None else ce_tok.sum()) / tot_w
            tot_ce += float(ce_part.detach().item())
            loss_c = loss_c + float(ce_w) * ce_part

        if h_ref is not None and kl_w > 0.0:
            with torch.no_grad():
                lg_r = _lm_head_softcap(base_model, h_ref[:, s:e, :].detach())
                log_pr = F.log_softmax(lg_r * inv_t, dim=-1)
                pr = log_pr.exp()
            log_pc = F.log_softmax(lg_c * inv_t, dim=-1)
            kl_tok = t2 * (pr * (log_pr - log_pc)).sum(dim=-1)
            kl_part = ((kl_tok * mc).sum() if mc is not None else kl_tok.sum()) / tot_w
            tot_kl += float(kl_part.detach().item())
            loss_c = loss_c + float(kl_w) * kl_part

        if need_grad and (ce_w > 0.0 or (h_ref is not None and kl_w > 0.0)):
            loss_c.backward()
            h_grad[:, s:e, :] = hc.grad

    if backward and need_grad:
        h_cur.backward(h_grad)
    if return_grad:
        return tot_ce, tot_kl, h_grad
    return tot_ce, tot_kl


# ---------------------------------------------------------------- Exact Addendum 226 O3 (Titans-Momentum) SwiGLU Fast Branch

def _smooth_norm(g, eps=1e-12):
    return (g.pow(2).sum(dim=(-2, -1), keepdim=True) + eps).sqrt()


def _swiglu_fwd_bwd(K, V, W1, W2, W3):
    """Exact analytical SwiGLU associative-loss gradient wrt (W1, W2, W3) in fp32 (matches exp226.py!)."""
    C = float(K.shape[1])
    A = torch.matmul(K, W1.transpose(-1, -2))
    G = torch.matmul(K, W3.transpose(-1, -2))
    sig = torch.sigmoid(A)
    silu = A * sig
    H = silu * G
    Y = torch.matmul(H, W2.transpose(-1, -2))
    dY = (Y - V) / C
    dW2 = torch.matmul(dY.transpose(-1, -2), H)
    dH = torch.matmul(dY, W2)
    dG = dH * silu
    dA = dH * G * (sig * (1.0 + A * (1.0 - sig)))
    dW3 = torch.matmul(dG.transpose(-1, -2), K)
    dW1 = torch.matmul(dA.transpose(-1, -2), K)
    return dW1, dW2, dW3


class RMSNormE4B(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d, dtype=torch.float32))

    def forward(self, x):
        return x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * self.w.float()


class TitansBranchE4B(nn.Module):
    """Zero-initialized parallel O3 (Titans-Momentum) SwiGLU fast-weight memory branch for Gemma 4 E4B (D=2560).
    Permanently stored in fp32 (only 12.1M params = 48 MB total across 4 layers!) so AdamW & gate_proj.bias NEVER
    stall under bf16 ULP rounding! At init `o.weight == 0` -> bit-for-bit identical to base Gemma 4 E4B."""
    def __init__(self, D=2560, d_io=256, d_hid=512, chunk=64):
        super().__init__()
        self.D = D
        self.d_io = d_io
        self.d_hid = d_hid
        self.chunk = chunk
        self.n = RMSNormE4B(D)
        self.qkv = nn.Linear(D, 3 * d_io, bias=False, dtype=torch.float32)
        self.W1 = nn.Parameter(torch.empty(d_hid, d_io, dtype=torch.float32))
        self.W2 = nn.Parameter(torch.empty(d_io, d_hid, dtype=torch.float32))
        self.W3 = nn.Parameter(torch.empty(d_hid, d_io, dtype=torch.float32))
        nn.init.normal_(self.W1, 0, d_io ** -0.5)
        nn.init.normal_(self.W3, 0, d_io ** -0.5)
        nn.init.normal_(self.W2, 0, d_hid ** -0.5)
        self.o = nn.Linear(d_io, D, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.o.weight)
        self.gate_proj = nn.Linear(D, 3, bias=True, dtype=torch.float32)
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias.copy_(torch.tensor([0.5, -2.0, -2.5], dtype=torch.float32))

    def init_state(self, B, device=None):
        dev = device or self.W1.device
        W1 = self.W1.detach().float().to(dev).unsqueeze(0).expand(B, self.d_hid, self.d_io).clone()
        W2 = self.W2.detach().float().to(dev).unsqueeze(0).expand(B, self.d_io, self.d_hid).clone()
        W3 = self.W3.detach().float().to(dev).unsqueeze(0).expand(B, self.d_hid, self.d_io).clone()
        return (W1, W2, W3, torch.zeros_like(W1), torch.zeros_like(W2), torch.zeros_like(W3))

    def forward(self, x, state=None, update=True, update_last=False):
        B, T, _ = x.shape
        x_dt = x.dtype
        outs = []
        with torch.autocast(x.device.type, enabled=False):
            xn = self.n(x)  # fp32
            qkv = self.qkv(xn).view(B, T, 3, self.d_io)
            scale_qk = self.d_io ** 0.5
            Q = F.normalize(qkv[:, :, 0], dim=-1, eps=1e-6) * scale_qk
            K = F.normalize(qkv[:, :, 1], dim=-1, eps=1e-6) * scale_qk
            V = qkv[:, :, 2]
            W1_0 = self.W1.unsqueeze(0)
            W2_0 = self.W2.unsqueeze(0)
            W3_0 = self.W3.unsqueeze(0)
            if state is None:
                W1 = W1_0.expand(B, self.d_hid, self.d_io)
                W2 = W2_0.expand(B, self.d_io, self.d_hid)
                W3 = W3_0.expand(B, self.d_hid, self.d_io)
                S1, S2, S3 = torch.zeros_like(W1), torch.zeros_like(W2), torch.zeros_like(W3)
            else:
                W1, W2, W3, S1, S2, S3 = state
                if W1.shape[0] == 1 and B > 1:
                    W1, W2, W3 = W1.expand(B, -1, -1), W2.expand(B, -1, -1), W3.expand(B, -1, -1)
                    S1, S2, S3 = S1.expand(B, -1, -1), S2.expand(B, -1, -1), S3.expand(B, -1, -1)

            do_update = bool(update and T >= 2)
            for s in range(0, T, self.chunk):
                e = min(T, s + self.chunk)
                qc, kc, vc = Q[:, s:e, :], K[:, s:e, :], V[:, s:e, :]
                Ac = torch.matmul(qc, W1.transpose(-1, -2))
                Gc = torch.matmul(qc, W3.transpose(-1, -2))
                rc = torch.matmul(F.silu(Ac) * Gc, W2.transpose(-1, -2))
                outs.append(rc)

                if do_update and (e < T or update_last) and (e - s) >= 2:
                    dW1, dW2, dW3 = _swiglu_fwd_bwd(kc, vc, W1, W2, W3)
                    g_in = xn[:, s:e, :].mean(dim=1)
                    gates = self.gate_proj(g_in)
                    theta = torch.sigmoid(gates[:, 0]).view(B, 1, 1)
                    eta = 0.25 * torch.sigmoid(gates[:, 1]).view(B, 1, 1)
                    alpha = torch.sigmoid(gates[:, 2]).view(B, 1, 1)
                    S1 = theta * S1 - eta * (dW1 / _smooth_norm(dW1))
                    S2 = theta * S2 - eta * (dW2 / _smooth_norm(dW2))
                    S3 = theta * S3 - eta * (dW3 / _smooth_norm(dW3))
                    W1 = (1.0 - alpha) * W1 + alpha * W1_0 + S1
                    W2 = (1.0 - alpha) * W2 + alpha * W2_0 + S2
                    W3 = (1.0 - alpha) * W3 + alpha * W3_0 + S3

            delta = self.o(torch.cat(outs, dim=1))
        new_state = (W1, W2, W3, S1, S2, S3)
        return delta.to(x_dt), new_state


def detach_fast_state(st):
    if st is None:
        return None
    return {li: tuple(t.detach() for t in v) for li, v in st.items()}


class LivingE4B(nn.Module):
    """Option-1 Zero-LoRA wrapper around official multimodal `Gemma4ForConditionalGeneration`.
    Hooks fp32 `TitansBranchE4B` onto `titans_layers` via clean PyTorch forward hooks (zero mutation to Gemma 4's KV
    sharing, RoPE, or vision/audio fusion) plus optional training-only `plato_head` at layer 12."""
    def __init__(self, base_model, processor, titans_layers=TITANS_LAYERS, d_io=256, d_hid=512, chunk=64, with_plato_head=False):
        super().__init__()
        self.base = base_model
        self.processor = processor
        self.tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.lm = base_model.model.language_model
        self.D = int(base_model.config.get_text_config().hidden_size)
        self.n_layers = len(self.lm.layers)
        self.titans_layers = tuple(li for li in titans_layers if li < self.n_layers)

        dev = next(self.lm.parameters()).device
        self.base_dtype = next(self.lm.parameters()).dtype
        # Permanently fp32 (12.1M params = 48 MB total; avoids bf16 ULP freeze on gate_proj.bias & W1/W2/W3!)
        self.titans = nn.ModuleDict({
            str(li): TitansBranchE4B(self.D, d_io=d_io, d_hid=d_hid, chunk=chunk).to(device=dev, dtype=torch.float32)
            for li in self.titans_layers
        })
        self.plato_head = nn.Linear(self.D, 256, bias=True, device=dev, dtype=torch.float32) if with_plato_head else None

        self._ctx = {"enabled": True, "state": None, "out_state": {}, "update": False, "update_last": False, "capture": None, "capture_pre_titans": None}
        self._hooks = []
        self._hooked_layers = set()
        want_layers = set(self.titans_layers) | {li for li in (PLATO_LAYER, 17, 25, 34) if li < self.n_layers}
        for li in sorted(want_layers):
            self._hooks.append(self.lm.layers[li].register_forward_hook(self._make_hook(li)))
            self._hooked_layers.add(li)

    def close(self):
        """Break PyTorch forward-hook closure reference cycles so GPU VRAM frees cleanly on teardown!"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._hooked_layers.clear()

    def _make_hook(self, li):
        def hook_fn(_mod, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if self._ctx.get("capture_pre_titans") is not None and li in self._ctx["capture_pre_titans"]:
                self._ctx["capture_pre_titans"][li] = h
            if self._ctx["enabled"] and str(li) in self.titans:
                st_dict = self._ctx["state"]
                st_in = None if st_dict is None else st_dict.get(li)
                delta, st_out = self.titans[str(li)](
                    h, state=st_in, update=self._ctx["update"], update_last=self._ctx["update_last"],
                )
                h = h + delta
                self._ctx["out_state"][li] = st_out
                # Advance live state after prefill during no-grad generate() so decode steps 1..N read updated fast weights!
                # Never mutate self._ctx["state"] under autograd so gradient-checkpoint recomputation sees identical inputs!
                if self._ctx["update"] and st_dict is not None and not torch.is_grad_enabled():
                    st_dict[li] = tuple(t.detach() for t in st_out)
            if self._ctx["capture"] is not None and li in self._ctx["capture"]:
                self._ctx["capture"][li] = h
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return hook_fn

    def init_fast_state(self, B=1):
        dev = next(self.titans.parameters()).device
        return {li: self.titans[str(li)].init_state(B, device=dev) for li in self.titans_layers}

    @contextlib.contextmanager
    def run_ctx(self, enabled=True, state=None, update=True, update_last=False, capture_layers=None, capture_pre_titans=None):
        req_caps = (set(capture_layers) if capture_layers else set()) | (set(capture_pre_titans) if capture_pre_titans else set())
        if req_caps:
            missing = req_caps - self._hooked_layers
            assert not missing, f"Requested capture_layers {missing} are not hooked! Hooked={sorted(self._hooked_layers)}"
        prev = dict(self._ctx)
        # Shallow copy so hook updates never mutate caller's state dict
        live_state = dict(state) if state is not None else {}
        cap = {li: None for li in capture_layers} if capture_layers else None
        cap_pre = {li: None for li in capture_pre_titans} if capture_pre_titans else None
        self._ctx = {
            "enabled": enabled, "state": live_state, "out_state": {},
            "update": update, "update_last": update_last, "capture": cap, "capture_pre_titans": cap_pre,
        }
        try:
            yield self._ctx
        finally:
            self._ctx = prev

    def set_grad_checkpointing(self, enabled: bool):
        """MUST use use_reentrant=False (verified exact 0.0 grad error across Gemma 4 KV-shared layers!)."""
        if enabled:
            self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.base.gradient_checkpointing_disable()

    @contextlib.contextmanager
    def fp32_mlps_for_training(self, layer_idxs):
        """Upcast ONLY the selected unfrozen MLP layers to fp32 master during AdamW training so lr~1.5e-5
        never stalls under bf16 ULP rounding, restoring original dtype cleanly in `finally:`."""
        orig_dtypes = {}
        try:
            for li in layer_idxs:
                mlp = self.lm.layers[li].mlp
                orig_dtypes[li] = next(mlp.parameters()).dtype
                mlp.to(dtype=torch.float32)
            yield
        finally:
            for li, dt in orig_dtypes.items():
                self.lm.layers[li].mlp.to(dtype=dt)

    def offload_multimodal_to_cpu(self):
        for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            sub = getattr(self.base.model, attr, None)
            if sub is not None:
                sub.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_multimodal_to_gpu(self, dev=None):
        dev = dev or next(self.lm.parameters()).device
        for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            sub = getattr(self.base.model, attr, None)
            if sub is not None:
                sub.to(dev)

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_stage_a(self, mlp_layers=STAGE_A_MLP_LAYERS):
        self.freeze_all()
        for li in mlp_layers:
            for p in self.lm.layers[li].mlp.parameters():
                p.requires_grad_(True)
        if self.plato_head is not None:
            for p in self.plato_head.parameters():
                p.requires_grad_(True)

    def unfreeze_stage_b(self, mlp_layers=STAGE_B_MLP_LAYERS):
        self.freeze_all()
        for p in self.titans.parameters():
            p.requires_grad_(True)
        for li in mlp_layers:
            for p in self.lm.layers[li].mlp.parameters():
                p.requires_grad_(True)

    def snapshot_mlp_weights_cpu(self, layer_idxs):
        return {
            li: {n: p.detach().to(dtype=self.base_dtype, device="cpu").clone() for n, p in self.lm.layers[li].mlp.named_parameters()}
            for li in layer_idxs
        }

    def snapshot_titans_cpu(self, param_names=None):
        out = {}
        for li in self.titans_layers:
            tb = self.titans[str(li)]
            if param_names is None:
                out[li] = {n: p.detach().cpu().clone() for n, p in tb.named_parameters()}
            else:
                out[li] = {pn: getattr(tb, pn).detach().cpu().clone() for pn in param_names}
        return out

    @contextlib.contextmanager
    def swapped_to_ref_weights(self, ref_mlp_cpu, ref_titans_cpu=None, disable_titans=None, backup_buf=None):
        """Zero-extra-VRAM exact KL anchor: temporarily swaps unfrozen MLPs (and optionally Titans slow bases!)
        to CPU-cached `theta_0`, reusing pre-allocated CPU buffers (`backup_buf`) to avoid per-step malloc churn."""
        if disable_titans is None:
            disable_titans = (ref_titans_cpu is None)
        cur_mlp = {} if backup_buf is None else backup_buf.setdefault("mlp", {})
        cur_tit = {} if backup_buf is None else backup_buf.setdefault("titans", {})
        with torch.no_grad():
            for li, sd in ref_mlp_cpu.items():
                slot = cur_mlp.setdefault(li, {})
                for n, p in self.lm.layers[li].mlp.named_parameters():
                    if n not in slot or slot[n].shape != p.shape or slot[n].dtype != p.dtype:
                        slot[n] = torch.empty(p.shape, dtype=p.dtype, device="cpu")
                    slot[n].copy_(p.detach(), non_blocking=False)
                    p.copy_(sd[n].to(p.device, dtype=p.dtype))
            if ref_titans_cpu is not None:
                for li, sd in ref_titans_cpu.items():
                    slot = cur_tit.setdefault(li, {})
                    tb = self.titans[str(li)]
                    for n, ref_t in sd.items():
                        p = dict(tb.named_parameters())[n]
                        if n not in slot or slot[n].shape != p.shape or slot[n].dtype != p.dtype:
                            slot[n] = torch.empty(p.shape, dtype=p.dtype, device="cpu")
                        slot[n].copy_(p.detach(), non_blocking=False)
                        p.copy_(ref_t.to(p.device, dtype=p.dtype))
        try:
            with self.run_ctx(enabled=not disable_titans, state=None, update=False):
                yield
        finally:
            with torch.no_grad():
                for li, sd in ref_mlp_cpu.items():
                    for n, p in self.lm.layers[li].mlp.named_parameters():
                        p.copy_(cur_mlp[li][n].to(p.device, dtype=p.dtype))
                if ref_titans_cpu is not None:
                    for li, sd in ref_titans_cpu.items():
                        tb = self.titans[str(li)]
                        pdict = dict(tb.named_parameters())
                        for n in sd:
                            pdict[n].copy_(cur_tit[li][n].to(pdict[n].device, dtype=pdict[n].dtype))

    def save_living_slice(self, path, meta=None, mlp_layers=None, include_mm_attn=False):
        """Compact Option-1 Zero-LoRA save (~656-730 MB bf16 instead of 16 GB!): saves Titans (fp32) + full-rank MLPs (bf16),
        plus upper self_attn & embed_vision/embed_audio whenever requested OR previously loaded/modified (so a text-only
        night NEVER silently loses earlier multimodal consolidation on restart!)."""
        if include_mm_attn:
            self._has_mm_attn_updates = True
        save_mm = bool(include_mm_attn or getattr(self, "_has_mm_attn_updates", False))
        if mlp_layers is None:
            mlp_layers = sorted(li for li in (set(STAGE_A_MLP_LAYERS) | set(STAGE_B_MLP_LAYERS)) if li < self.n_layers)
        payload = {
            "model_id": MODEL_ID,
            "titans_layers": self.titans_layers,
            "mlp_layers": tuple(mlp_layers),
            "titans": {k: v.detach().float().cpu() for k, v in self.titans.state_dict().items()},
            "mlp_slices": {
                li: {n: p.detach().to(dtype=self.base_dtype, device="cpu") for n, p in self.lm.layers[li].mlp.named_parameters()}
                for li in mlp_layers
            },
            "meta": meta or {},
        }
        if save_mm:
            attn_layers = [li for li in STAGE_B_MLP_LAYERS if li < self.n_layers]
            payload["attn_slices"] = {
                li: {n: p.detach().to(dtype=self.base_dtype, device="cpu") for n, p in self.lm.layers[li].self_attn.named_parameters()}
                for li in attn_layers
            }
            mm_p = {}
            for attr in ("embed_vision", "embed_audio"):
                sub = getattr(self.base.model, attr, None)
                if sub is not None:
                    mm_p[attr] = {n: p.detach().to(dtype=self.base_dtype, device="cpu") for n, p in sub.named_parameters()}
            payload["mm_proj_slices"] = mm_p
        _atomic_torch_save(path, payload)

    def load_living_slice(self, path, strict_titans=True, check_model_id=True):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if check_model_id and payload.get("model_id") is not None:
            assert payload["model_id"] == MODEL_ID, f"Slice model_id mismatch: {payload['model_id']} != {MODEL_ID}"
        if "titans" in payload:
            self.titans.load_state_dict(payload["titans"], strict=strict_titans)
        if payload.get("attn_slices") or payload.get("mm_proj_slices"):
            self._has_mm_attn_updates = True
        with torch.no_grad():
            for li, sd in payload.get("mlp_slices", {}).items():
                li_i = int(li)
                pdict = dict(self.lm.layers[li_i].mlp.named_parameters())
                missing = set(pdict.keys()) - set(sd.keys())
                assert not missing, f"Missing MLP params in slice for layer {li_i}: {missing}"
                for n, p in pdict.items():
                    p.copy_(sd[n].to(p.device, dtype=p.dtype))
            for li, sd in payload.get("attn_slices", {}).items():
                li_i = int(li)
                pdict = dict(self.lm.layers[li_i].self_attn.named_parameters())
                for n, p in pdict.items():
                    if n in sd:
                        p.copy_(sd[n].to(p.device, dtype=p.dtype))
            for attr, sd in payload.get("mm_proj_slices", {}).items():
                sub = getattr(self.base.model, attr, None)
                if sub is not None:
                    for n, p in sub.named_parameters():
                        if n in sd:
                            p.copy_(sd[n].to(p.device, dtype=p.dtype))
        return payload.get("meta", {})

    @torch.no_grad()
    def adapt_text(self, text, state=None):
        """Wake fast-weight update on text (slow weights strictly untouched; preserves <bos>!)."""
        self.eval()
        dev = next(self.lm.parameters()).device
        ids = _gemma_encode_with_bos(self.tok, text)
        if len(ids) > 1024:
            ids = [ids[0]] + ids[-1023:]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with self.run_ctx(enabled=True, state=state, update=True, update_last=True) as ctx:
            _ = self.base.model(input_ids=x, use_cache=False)
            return detach_fast_state(ctx["out_state"])

    @staticmethod
    def split_gemma_thinking(raw_with_special: str):
        """Parse Gemma 4 native `<|channel>thought\\n...<channel|>answer` output into `(thinking, clean_reply)`.
        Strips ONLY Gemma pipe-delimited special tokens & standard BOS/EOS/PAD tokens, NEVER stripping `<search>` tags!"""
        import re as _re
        s = (raw_with_special or "").replace("<turn|>", "").replace("<eos>", "").replace("<pad>", "")
        m = _re.search(r"<\|channel>thought\s*(.*?)<channel\|>(.*)", s, _re.S)
        if m:
            th, ans = m.group(1), m.group(2)
        elif "<|channel>thought" in s:
            th = s.split("<|channel>thought", 1)[1]
            ans = ""
        else:
            th, ans = "", s
        # Strictly match Gemma pipe-delimited special tokens (<|foo|>, <|foo>, <foo|>) and <bos>/<eos>/<pad>/<unk> ONLY
        strip_sp = _re.compile(r"<(?:\|[a-z_]+\|?|[a-z_]+\|)>|<(?:bos|eos|pad|unk)>", _re.I)
        return strip_sp.sub("", th).strip(), strip_sp.sub("", ans).strip()

    @torch.no_grad()
    def chat_reply(
        self, messages, state=None, max_new_tokens=160,
        temp=GEMMA4_DEFAULT_TEMP, use_titans=True, update_state=True, images=None, audio=None,
        enable_thinking=False, return_thinking=False,
        top_p=GEMMA4_DEFAULT_TOP_P, top_k=GEMMA4_DEFAULT_TOP_K,
    ):
        """Generate a reply using native Gemma 4 chat template + optional native `<|think|>` reasoning + live O3 fast memory.
        Defaults match Google's official Gemma 4 E4B sampling config (temperature=1.0, top_p=0.95, top_k=64)."""
        self.eval()
        dev = next(self.lm.parameters()).device
        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(enable_thinking),
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_kw = dict(text=prompt, return_tensors="pt")
        if images is not None:
            proc_kw["images"] = images
        if audio is not None:
            proc_kw["audio"] = audio
        inp = self.processor(**proc_kw).to(dev)
        # Cast float audio/vision features to base_dtype cleanly (Gemma4AudioSubSampleConvProjection needs bf16!)
        for k, v in list(inp.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != self.base_dtype:
                inp[k] = v.to(self.base_dtype)
        p_len = inp["input_ids"].shape[1]

        with self.run_ctx(enabled=use_titans, state=state, update=update_state, update_last=update_state) as ctx:
            gen_kw = dict(max_new_tokens=max_new_tokens, use_cache=True)
            # Prevent KV-sharing prefill slicing from starving Titans layers 30 & 39!
            if use_titans:
                gen_kw["logits_to_keep"] = 0
            if temp > 0.0:
                gen_kw.update(do_sample=True, temperature=float(temp), top_p=float(top_p), top_k=int(top_k))
            else:
                gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            out = self.base.generate(**inp, **gen_kw)
            new_st = detach_fast_state(ctx["state"]) if (use_titans and ctx["state"]) else state
        gen_ids = out[0][p_len:]
        if enable_thinking:
            raw_sp = self.processor.decode(gen_ids, skip_special_tokens=False)
            thinking, reply = self.split_gemma_thinking(raw_sp)
        else:
            thinking = ""
            reply = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        if return_thinking:
            return reply, new_st, thinking
        return reply, new_st


def load_living_e4b(slice_path=None, with_plato_head=False, offload_mm=False):
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    base = Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model = LivingE4B(base, proc, with_plato_head=with_plato_head)
    if slice_path is not None and Path(slice_path).exists():
        meta = model.load_living_slice(slice_path)
        print(f"Loaded living slice {slice_path} (meta={meta})", flush=True)
    if offload_mm:
        model.offload_multimodal_to_cpu()
    return model


# ---------------------------------------------------------------- Exact Gemma-4-Tokenizer Alignment to Frozen C5 Targets

def load_gemma_aligned_plato_pools(tok, ujung_alpha=3.5):
    """Align Gemma 4 E4B's native 262K tokenizer 1-to-1 with saved Addendum 228 targets_train.npz (neutral gut+proc+chat)
    and Addendum 229 targets_myth229.npz (975 verified modernized stories), applying the exact M2 U_jung boost."""
    with open(PLATO / "passages.jsonl") as f:
        tr_rows = [json.loads(l) for l in f]
    with np.load(PLATO / "targets_train.npz") as d_t:
        c5_base_raw = d_t["consensus_256"].astype(np.float64)
        pts_off_t = d_t["pts_off"]
        doms_t = d_t["domains"]
        ids_f, ids_o = d_t["ids_flat"], d_t["ids_off"]

    bpe_tk = plato_data.load_bpe8k()
    bpe_to_row = {}
    for r in tr_rows:
        if r.get("domain") in ("gut", "proc", "chat"):
            bpe_to_row[tuple(bpe_tk.encode("".join(r["segs"])).ids)] = r

    U = None
    if abs(ujung_alpha - 1.0) >= 1e-9:
        with np.load(PLATO / "arch229_consensus.npz") as z_u:
            U = z_u["U_jung_c5"].astype(np.float64)

    myth_items = []
    if (PLATO / "targets_myth229.npz").exists() and (PLATO / "myth229_raw.jsonl").exists():
        raw_m = {}
        with open(PLATO / "myth229_raw.jsonl") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    raw_m[r["id"]] = r
        with np.load(PLATO / "targets_myth229.npz") as d_m:
            c5_myth_raw = d_m["c5_instance_256"].astype(np.float64)
            m_pts_o = d_m["pts_off"]
            m_sids = d_m["story_id"].tolist()
    else:
        c5_myth_raw = np.zeros((0, 256), dtype=np.float64)
        m_pts_o, m_sids, raw_m = np.array([0]), [], {}

    keep_base_idx = np.concatenate([
        np.flatnonzero(doms_t == "gut"),
        np.flatnonzero(doms_t == "proc"),
        np.flatnonzero(doms_t == "chat"),
    ])
    act_mask = np.zeros(len(c5_base_raw), dtype=bool)
    for pi in keep_base_idx:
        act_mask[int(pts_off_t[pi]):int(pts_off_t[pi + 1])] = True

    if len(c5_myth_raw) > 0:
        c5_base, c5_myth = exp229.transform_ujung_pair(c5_base_raw, act_mask, c5_myth_raw, U, ujung_alpha)
    elif abs(ujung_alpha - 1.0) >= 1e-9:
        raise FileNotFoundError(
            f"targets_myth229.npz is required when ujung_alpha={ujung_alpha} != 1.0 (refusing to silently skip U_jung boost!)"
        )
    else:
        c5_base = c5_base_raw.astype(np.float32)
        c5_myth = c5_myth_raw.astype(np.float32)

    g_enc = lambda s: _gemma_encode_with_bos(tok, s)
    base_items = []
    for pi in keep_base_idx:
        b_ids = tuple(ids_f[int(ids_o[pi]):int(ids_o[pi + 1])].tolist())
        r = bpe_to_row.get(b_ids)
        if r is None:
            continue
        pa, pb = int(pts_off_t[pi]), int(pts_off_t[pi + 1])
        g_ids, g_pts = plato_data.boundary_token_indices(g_enc, r["segs"])
        if len(g_pts) == (pb - pa) and 2 <= len(g_ids) <= 512:
            base_items.append((np.array(g_ids, dtype=np.int64), np.array(g_pts, dtype=np.int64), c5_base[pa:pb]))

    for mi, sid in enumerate(m_sids):
        r = raw_m.get(sid)
        if r is None:
            continue
        pa, pb = int(m_pts_o[mi]), int(m_pts_o[mi + 1])
        g_ids, g_pts = plato_data.boundary_token_indices(g_enc, r["segs"])
        if len(g_pts) == (pb - pa) and 2 <= len(g_ids) <= 512:
            myth_items.append((np.array(g_ids, dtype=np.int64), np.array(g_pts, dtype=np.int64), c5_myth[pa:pb]))

    assert len(base_items) >= 2000, f"too few Gemma-aligned base passages: {len(base_items)}"
    print(f"Gemma-4-aligned Platonic pools: base={len(base_items)}, modern_myths={len(myth_items)}, ujung_alpha={ujung_alpha}", flush=True)
    return base_items, myth_items


# ---------------------------------------------------------------- Blind Platonic, HOUSE/SELF2, NLL, Buddy & Multimodal Eval

@torch.no_grad()
def eval_held_nll(model, n_per_domain=40):
    """Held-out neutral books (`held_gut`) & `chat` NLL drift guardrail (uses chunked lm_head, zero OOM!)."""
    model.eval()
    dev = next(model.lm.parameters()).device
    tok = model.tok
    with open(PLATO / "eval_passages.jsonl") as f:
        ev_rows = [json.loads(l) for l in f]
    out = {}
    for dom in ("held_gut", "chat_held"):
        sub = [r for r in ev_rows if r.get("domain") == dom][:n_per_domain]
        if not sub:
            continue
        nlls = []
        for r in sub:
            ids = _gemma_encode_with_bos(tok, "".join(r["segs"]))[:512]
            if len(ids) < 4:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            with model.run_ctx(enabled=True, state=None, update=False):
                h = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :]
            ce, _ = chunked_lm_head_ce_kl(model.base, h, y_ids=x[:, 1:], ce_w=1.0, kl_w=0.0, chunk_t=64)
            nlls.append(ce)
        out[dom + "_nll"] = round(float(np.mean(nlls)), 4)
    return out


@torch.no_grad()
def eval_blind_platonic_and_jung(model):
    """Score Gemma 4 E4B Layer 12 (29.3% depth) blind against unseen Phi-4-mini & OLMo 3 7B using ZERO-LEAKAGE
    train-fitted 256-dim PCA (matching plato_data.py!) + items3 TF/TL + HOUSE (raw, lenres, r_len, perf) & SELF2."""
    model.eval()
    dev = next(model.lm.parameters()).device
    tok = model.tok
    g_enc = lambda s: _gemma_encode_with_bos(tok, s)

    # 1. Fit 256-dim PCA strictly on deterministic neutral train passages (ZERO eval leakage!)
    with open(PLATO / "passages.jsonl") as f:
        tr_neutral = [json.loads(l) for l in f]
    tr_neutral = [r for r in tr_neutral if r.get("domain") in ("gut", "proc", "chat")]
    # Deterministic stride taking 360 balanced neutral train passages (~1,100 boundary points > 256)
    stride = max(1, len(tr_neutral) // 360)
    pca_rows = tr_neutral[::stride][:360]
    tr_feats = []
    for r in pca_rows:
        g_ids, g_pts = plato_data.boundary_token_indices(g_enc, r["segs"])
        if len(g_pts) >= 1 and 2 <= len(g_ids) <= 512:
            x = torch.tensor([g_ids], dtype=torch.long, device=dev)
            with model.run_ctx(enabled=True, state=None, update=False, capture_layers=(PLATO_LAYER,)) as ctx:
                _ = model.base.model(input_ids=x, use_cache=False)
                h12 = ctx["capture"][PLATO_LAYER][0].float().cpu().numpy()
            tr_feats.append(h12[g_pts])
    H_tr_pca = np.concatenate(tr_feats, axis=0).astype(np.float64)

    # 2. Extract held-out eval_passages boundary & story states
    with open(PLATO / "eval_passages.jsonl") as f:
        ev_rows = [json.loads(l) for l in f]
    ev_by_id = {r["id"]: r for r in ev_rows}
    with np.load(PLATO / "targets_eval.npz") as d_ev:
        metas = [json.loads(s) for s in d_ev["meta_json"]]
        pts_o = d_ev["pts_off"]
        pt_groups = d_ev["pt_groups"].tolist()
        pt_domains = d_ev["pt_domains"].tolist()
        pt_ret_mask = d_ev["pt_ret_mask"].copy()
        phi4_pts = d_ev["blind_phi4_256"].astype(np.float64)
        olmo3_pts = d_ev["blind_olmo3_256"].astype(np.float64)
        c5_pts = d_ev["consensus_256"].astype(np.float64)

    feats_pts, feats_story, keep_pt_mask, keep_st_mask = [], [], [], []
    for i, meta in enumerate(metas):
        r = ev_by_id.get(meta["id"])
        pa, pb = int(pts_o[i]), int(pts_o[i + 1])
        n_seg = pb - pa
        if r is None:
            keep_pt_mask.extend([False] * n_seg); keep_st_mask.append(False); continue
        g_ids, g_pts = plato_data.boundary_token_indices(g_enc, r["segs"])
        if len(g_pts) != n_seg or not (2 <= len(g_ids) <= 512):
            keep_pt_mask.extend([False] * n_seg); keep_st_mask.append(False); continue
        x = torch.tensor([g_ids], dtype=torch.long, device=dev)
        with model.run_ctx(enabled=True, state=None, update=False, capture_layers=(PLATO_LAYER,)) as ctx:
            _ = model.base.model(input_ids=x, use_cache=False)
            h12 = ctx["capture"][PLATO_LAYER][0].float().cpu().numpy()
        feats_pts.append(h12[g_pts])
        feats_story.append(h12[1:].mean(axis=0))
        keep_pt_mask.extend([True] * n_seg)
        keep_st_mask.append(True)

    H_pts = np.concatenate(feats_pts, axis=0).astype(np.float64)
    H_st = np.stack(feats_story, axis=0).astype(np.float64)
    m_pt = np.array(keep_pt_mask, dtype=bool)
    m_st = np.array(keep_st_mask, dtype=bool)

    _, Z_pts, _ = plato_data.pca_project(H_tr_pca, H_pts, d_out=256)
    g_use = [g for g, f in zip(pt_groups, m_pt) if f]
    d_use = [d for d, f in zip(pt_domains, m_pt) if f]
    r_use = pt_ret_mask[m_pt]

    out = {
        "n_pca_tr_pts": int(len(H_tr_pca)),
        "n_eval_pts": int(m_pt.sum()),
        "blind_phi4_pts": plato_data.eval_procrustes_transfer(Z_pts, phi4_pts[m_pt], group_ids=g_use, domains=d_use, ret_mask=r_use),
        "blind_olmo3_pts": plato_data.eval_procrustes_transfer(Z_pts, olmo3_pts[m_pt], group_ids=g_use, domains=d_use, ret_mask=r_use),
        "c5_eval_pts": plato_data.eval_procrustes_transfer(Z_pts, c5_pts[m_pt], group_ids=g_use, domains=d_use, ret_mask=r_use),
        "held_nll": eval_held_nll(model),
    }

    # 3. items3 centered TF & TL at layer 12
    metas_k = [m for m, f in zip(metas, m_st) if f]
    i3_idx = [i for i, m in enumerate(metas_k) if m.get("domain", "").startswith("items3_")]
    if i3_idx:
        Hc = H_st.copy()
        Hc[i3_idx] = H_st[i3_idx] - H_st[i3_idx].mean(axis=0, keepdims=True)
        by_item = defaultdict(dict)
        for i in i3_idx:
            m = metas_k[i]
            by_item[(m["kind"], m["item_id"])][m["role"]] = Hc[i]
        i3_res = {}
        cos = lambda a, b: float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))
        for kind in ("myth", "mech"):
            tfs, tls = [], []
            for (k, _), roles in sorted(by_item.items()):
                if k == kind and all(r in roles for r in ("anchor", "true", "foil", "lure")):
                    c_tr = cos(roles["anchor"], roles["true"])
                    tfs.append(float(c_tr > cos(roles["anchor"], roles["foil"])))
                    tls.append(float(c_tr > cos(roles["anchor"], roles["lure"])))
            i3_res[kind] = {"n": len(tfs), "TF": round(float(np.mean(tfs)), 4), "TL": round(float(np.mean(tls)), 4)}
        out["items3_L12_centered"] = i3_res

    # 4. HOUSE / SELF2 2-fold coherence + linear+log length residualization check at layer 12
    pairs = arch_rot229.load_arch_pairs()
    d_vecs, n_vecs, d_lens, n_lens = [], [], [], []
    for p in pairs:
        for txt, vl, ll in ((p["done"], d_vecs, d_lens), (p["neglected"], n_vecs, n_lens)):
            ids = g_enc(txt)[:512]
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            with model.run_ctx(enabled=True, state=None, update=False, capture_layers=(PLATO_LAYER,)) as ctx:
                _ = model.base.model(input_ids=x, use_cache=False)
                h12 = ctx["capture"][PLATO_LAYER][0].float().cpu().numpy()
            vl.append(h12[-1]); ll.append(len(ids))
    D_mat = np.stack(d_vecs) - np.stack(n_vecs)
    d_len = np.array(d_lens, dtype=np.float64)
    n_len = np.array(n_lens, dtype=np.float64)
    cats = np.array([p["cat"] for p in pairs])
    h_idx = np.flatnonzero(np.isin(cats, arch_rot229.HOUSE_CATS))
    s_idx = np.flatnonzero(cats == "self2")
    fh = arch_rot229.make_stratified_2fold(cats[h_idx])
    fs = arch_rot229.make_stratified_2fold(cats[s_idx])
    rh = arch_rot229.eval_disjoint_2fold([D_mat[h_idx]], D_mat[h_idx], cats[h_idx], fh, is_house=True, seed=2301)
    rh_lr = arch_rot229.eval_disjoint_2fold_lenres(
        [D_mat[h_idx]], [(d_len[h_idx], n_len[h_idx])],
        D_mat[h_idx], d_len[h_idx], n_len[h_idx], cats[h_idx], fh, is_house=True, seed=2303,
    )
    rs = arch_rot229.eval_disjoint_2fold([D_mat[s_idx]], D_mat[s_idx], cats[s_idx], fs, is_house=False, seed=2302)
    u_h = arch_rot229.house_dir(D_mat[h_idx], cats[h_idx])
    perf_m = (cats[h_idx] == "perfection")
    r_len = float(np.corrcoef(rh["scores"], d_len[h_idx] - n_len[h_idx])[0, 1])
    out["jung_L12"] = {
        "house_2f_acc": rh["acc"],
        "house_2f_lenres": rh_lr["acc"],
        "r_len": round(r_len, 4),
        "perf_acc": round(float(np.mean(rh["scores"][perf_m] > 0.0)), 4),
        "self2_2f_acc": rs["acc"],
        "house_to_self2_acc": round(float(np.mean((D_mat[s_idx] @ u_h) > 0.0)), 4),
    }
    return out


def _content_tokens(text):
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "your", "have", "will", "about", "into", "still",
        "just", "then", "they", "them", "been", "were", "what", "when", "where", "which", "would", "could",
        "should", "there", "their", "sure", "got", "all", "set", "now", "updated", "changed", "instead",
    }
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if (len(w) >= 3 or w.isdigit()) and w not in stop}


@torch.no_grad()
def eval_buddy_behavior(model):
    """Evaluate ALL 5 categories of held-out `buddy_eval230.jsonl`:
    natural_buddy, search_needed (tag + strict-only), ask_or_abstain, easy_known_fact, AND working_mem_ttt (O3 ON vs OFF)."""
    ev_p = DATA / "buddy_eval230.jsonl"
    if not ev_p.exists():
        return {"skipped": "missing buddy_eval230.jsonl"}
    with open(ev_p) as f:
        items = [json.loads(l) for l in f if l.strip()]

    nat_bp = nat_md = nat_n = 0
    srch_tag_tp = srch_strict_tp = srch_n = 0
    abs_fp = abs_bp = abs_n = 0
    easy_fp = easy_bp = easy_n = 0
    wmem_n = 0
    wmem_full_on, wmem_full_off = [], []
    wmem_cold_on, wmem_cold_off = [], []
    samples = []

    strict_re = re.compile(r"<search>[^<>\n]{4,120}</search>")

    for it in items:
        cat = it["cat"]
        turns = it["turns"]
        if cat == "natural_buddy":
            msgs = [{"role": "system", "content": bd.BUDDY_SYSTEM_PROMPT}, turns[0]]
            rep, _ = model.chat_reply(msgs, state=None, max_new_tokens=96, temp=0.0)
            rep_n = bd._norm_quotes(rep)
            nat_n += 1
            if bd.BOILERPLATE_RE.search(rep_n):
                nat_bp += 1
            if bd.MARKDOWN_WALL_RE.search(rep_n):
                nat_md += 1
            if len(samples) < 3:
                samples.append({"cat": cat, "u": turns[0]["content"], "a": rep})
        elif cat == "search_needed":
            msgs = [{"role": "system", "content": bd.BUDDY_SYSTEM_PROMPT}, turns[0]]
            rep, _ = model.chat_reply(msgs, state=None, max_new_tokens=48, temp=0.0)
            srch_n += 1
            if "<search>" in rep and "</search>" in rep:
                srch_tag_tp += 1
            if strict_re.fullmatch(rep.strip()):
                srch_strict_tp += 1
            if sum(s["cat"] == "search_needed" for s in samples) < 2:
                samples.append({"cat": cat, "u": turns[0]["content"], "a": rep})
        elif cat == "ask_or_abstain":
            msgs = [{"role": "system", "content": bd.BUDDY_SYSTEM_PROMPT}, turns[0]]
            rep, _ = model.chat_reply(msgs, state=None, max_new_tokens=80, temp=0.0)
            rep_n = bd._norm_quotes(rep)
            abs_n += 1
            if bd.SEARCH_TAG_ANY_RE.search(rep):
                abs_fp += 1
            if bd.BOILERPLATE_RE.search(rep_n):
                abs_bp += 1
            if sum(s["cat"] == "ask_or_abstain" for s in samples) < 2:
                samples.append({"cat": cat, "u": turns[0]["content"], "a": rep})
        elif cat == "easy_known_fact":
            msgs = [{"role": "system", "content": bd.BUDDY_SYSTEM_PROMPT}, turns[0]]
            rep, _ = model.chat_reply(msgs, state=None, max_new_tokens=80, temp=0.0)
            rep_n = bd._norm_quotes(rep)
            easy_n += 1
            if bd.SEARCH_TAG_ANY_RE.search(rep):
                easy_fp += 1
            if bd.BOILERPLATE_RE.search(rep_n):
                easy_bp += 1
            if sum(s["cat"] == "easy_known_fact" for s in samples) < 2:
                samples.append({"cat": cat, "u": turns[0]["content"], "a": rep})
        elif cat == "working_mem_ttt":
            wmem_n += 1
            gold_toks = _content_tokens(turns[5]["content"])
            score_fn = lambda r: len(_content_tokens(r) & gold_toks) / max(1, len(gold_toks))
            # (a) Full-context multi-turn chat: O3 ON vs OFF
            msgs_full = [{"role": "system", "content": bd.BUDDY_SYSTEM_PROMPT}] + turns[:5]
            rep_on, _ = model.chat_reply(msgs_full, state=None, max_new_tokens=80, temp=0.0, use_titans=True, update_state=True)
            rep_off, _ = model.chat_reply(msgs_full, state=None, max_new_tokens=80, temp=0.0, use_titans=False, update_state=False)
            wmem_full_on.append(score_fn(rep_on))
            wmem_full_off.append(score_fn(rep_off))
            # (b) Context-cleared cold recall: adapt O3 on turns 0..3, then ask turns[4] with ONLY fast_state!
            ctx_txt = "\n".join(f"{t['role']}: {t['content']}" for t in turns[:4])
            st_adapt = model.adapt_text(ctx_txt, state=None)
            msgs_cold = [{"role": "system", "content": bd.BUDDY_SYSTEM_PROMPT}, turns[4]]
            rep_c_on, _ = model.chat_reply(msgs_cold, state=st_adapt, max_new_tokens=80, temp=0.0, use_titans=True, update_state=False)
            rep_c_off, _ = model.chat_reply(msgs_cold, state=None, max_new_tokens=80, temp=0.0, use_titans=False, update_state=False)
            wmem_cold_on.append(score_fn(rep_c_on))
            wmem_cold_off.append(score_fn(rep_c_off))
            if sum(s["cat"] == "working_mem_ttt" for s in samples) < 2:
                samples.append({"cat": cat, "q": turns[4]["content"], "gold": turns[5]["content"], "a_on": rep_on, "a_cold_on": rep_c_on})

    return {
        "natural_n": nat_n,
        "boilerplate_rate": round(nat_bp / max(1, nat_n), 4),
        "markdown_wall_rate": round(nat_md / max(1, nat_n), 4),
        "search_n": srch_n,
        "search_tag_recall": round(srch_tag_tp / max(1, srch_n), 4),
        "search_strict_only_recall": round(srch_strict_tp / max(1, srch_n), 4),
        "easy_known_n": easy_n,
        "easy_search_false_trigger": round(easy_fp / max(1, easy_n), 4),
        "easy_boilerplate_rate": round(easy_bp / max(1, easy_n), 4),
        "abstain_n": abs_n,
        "abstain_search_false_trigger": round(abs_fp / max(1, abs_n), 4),
        "abstain_boilerplate_rate": round(abs_bp / max(1, abs_n), 4),
        "working_mem_n": wmem_n,
        "wmem_full_recall_O3_on": round(float(np.mean(wmem_full_on)), 4) if wmem_full_on else None,
        "wmem_full_recall_O3_off": round(float(np.mean(wmem_full_off)), 4) if wmem_full_off else None,
        "wmem_cold_recall_O3_on": round(float(np.mean(wmem_cold_on)), 4) if wmem_cold_on else None,
        "wmem_cold_recall_O3_off": round(float(np.mean(wmem_cold_off)), 4) if wmem_cold_off else None,
        "samples": samples,
    }


def _decode_wav_16k_mono(raw_bytes: bytes):
    """Stdlib 16-bit PCM WAV decoder to float32 mono 16 kHz (keeps e4b_living self-contained without companion app deps)."""
    import io, wave
    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        ch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        assert sw == 2, f"Expected 16-bit PCM WAV, got sampwidth={sw}"
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != 16000 and len(pcm) > 1:
        x_old = np.linspace(0.0, 1.0, len(pcm), endpoint=False)
        x_new = np.linspace(0.0, 1.0, max(1, int(round(len(pcm) * 16000.0 / sr))), endpoint=False)
        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
    return pcm


@torch.no_grad()
def eval_multimodal_sanity(model):
    """Verify BOTH frozen vision tower (red square) AND frozen audio tower (spoken speech clip, bf16-cast) stay 100% live!"""
    from PIL import Image
    model.reload_multimodal_to_gpu()
    model.eval()

    img = Image.new("RGB", (224, 224), color=(220, 30, 30))
    vis_msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What single color is this square? Answer in one word."}]}]
    vis_reply, _ = model.chat_reply(vis_msgs, state=None, max_new_tokens=16, temp=0.0, use_titans=True, update_state=False, images=[img])

    # Real spoken 16 kHz speech clip ("The password is blue pelican.") if present, else 440 Hz tone
    speech_p = HERE / "assets" / "test_speech_16k.wav"
    if speech_p.exists():
        wav = _decode_wav_16k_mono(speech_p.read_bytes())
        aud_msgs = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Repeat the exact words spoken in this audio clip."}]}]
        aud_reply, _ = model.chat_reply(aud_msgs, state=None, max_new_tokens=32, temp=0.0, use_titans=True, update_state=False, audio=[wav])
        aud_ok = ("blue" in aud_reply.lower() and "pelican" in aud_reply.lower())
        aud_fixture = "speech_blue_pelican"
    else:
        t = np.linspace(0.0, 1.5, 24000, endpoint=False, dtype=np.float32)
        wav = (0.25 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)
        aud_msgs = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Briefly describe what you hear in this audio clip in a few words."}]}]
        aud_reply, _ = model.chat_reply(aud_msgs, state=None, max_new_tokens=32, temp=0.0, use_titans=True, update_state=False, audio=[wav])
        aud_ok = bool(aud_reply)
        aud_fixture = "tone_440hz_fallback"
    return {
        "vision_red_square_reply": vis_reply,
        "vision_ok": ("red" in vis_reply.lower()),
        "audio_fixture": aud_fixture,
        "audio_speech_reply": aud_reply,
        "audio_ok": aud_ok,
    }


# ---------------------------------------------------------------- Stage A: Layer 11-12 Full-Rank Platonic Consensus + U_jung (M2)

def run_stage_a(steps=450, lr=1.5e-5, kl_w=1.5, plato_w=0.40, ujung_alpha=3.5, seed=230):
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = load_living_e4b(slice_path=None, with_plato_head=True, offload_mm=True)
    base_items, myth_items = load_gemma_aligned_plato_pools(model.tok, ujung_alpha=ujung_alpha)

    ref_mlp_cpu = model.snapshot_mlp_weights_cpu(STAGE_A_MLP_LAYERS)
    backup_buf = {}
    dev = next(model.lm.parameters()).device

    model.set_grad_checkpointing(True)
    opt = None
    try:
        with model.fp32_mlps_for_training(STAGE_A_MLP_LAYERS):
            model.unfreeze_stage_a(STAGE_A_MLP_LAYERS)
            mlp_params = [p for li in STAGE_A_MLP_LAYERS for p in model.lm.layers[li].mlp.parameters()]
            head_params = list(model.plato_head.parameters())
            opt = torch.optim.AdamW([
                {"params": mlp_params, "lr": lr, "weight_decay": 0.01},
                {"params": head_params, "lr": 1e-3, "weight_decay": 0.0},
            ], betas=(0.9, 0.95))

            print(f"STAGE_A start: steps={steps} mlp_params={sum(p.numel() for p in mlp_params):,} (fp32 master) head_params={sum(p.numel() for p in head_params):,} alloc={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
            model.train()

            for s in range(steps):
                w_s = max(1, steps // 10)
                lr_mul = (s + 1) / w_s if s < w_s else 0.5 * (1.0 + math.cos(math.pi * (s - w_s) / max(1, steps - w_s)))
                opt.param_groups[0]["lr"] = lr * lr_mul
                opt.param_groups[1]["lr"] = 1e-3 * lr_mul
                cur_pw = exp229.plato_w_at(plato_w, s, steps, anneal_start=0.65, anneal_end=0.90)

                draws = [base_items[int(i)] for i in rng.integers(0, len(base_items), 3)]
                if myth_items:
                    draws.append(myth_items[int(rng.integers(0, len(myth_items)))])
                n_acc = len(draws)

                # Precompute exact base hidden states (7 MB total!) via ONE CPU weight swap per step
                ref_hs = []
                with torch.no_grad(), model.swapped_to_ref_weights(ref_mlp_cpu, disable_titans=True, backup_buf=backup_buf):
                    with _autocast_ctx(dev):
                        for g_ids, _, _ in draws:
                            x = torch.from_numpy(g_ids).unsqueeze(0).to(dev)
                            hr = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :].detach()
                            ref_hs.append(hr)

                opt.zero_grad(set_to_none=True)
                sum_ce = sum_kl = sum_plato = 0.0
                for (g_ids, g_pts, tgt), h_ref in zip(draws, ref_hs):
                    x = torch.from_numpy(g_ids).unsqueeze(0).to(dev)
                    with _autocast_ctx(dev), model.run_ctx(enabled=False, state=None, update=False, capture_layers=(PLATO_LAYER,)) as ctx:
                        h_cur = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :]
                        h12 = ctx["capture"][PLATO_LAYER]

                        ce_v, kl_v, h_grad = chunked_lm_head_ce_kl(
                            model.base, h_cur, y_ids=x[:, 1:], h_ref=h_ref,
                            ce_w=0.25 / n_acc, kl_w=kl_w / n_acc, chunk_t=64, return_grad=True,
                        )
                        sum_ce += ce_v / n_acc
                        sum_kl += kl_v / n_acc

                        if cur_pw > 0.0:
                            t_idx = torch.from_numpy(g_pts).to(dev)
                            t_tgt = torch.from_numpy(tgt).to(dev, dtype=torch.float32)
                            pred = model.plato_head(h12[0, t_idx, :].float())
                            l_p = exp228.plato_alignment_loss(pred, t_tgt)
                            sum_plato += float(l_p.item()) / n_acc
                            torch.autograd.backward([h_cur, (cur_pw / n_acc) * l_p], [h_grad, None])
                        else:
                            h_cur.backward(h_grad)
                    del h_cur, h12, h_grad

                del ref_hs
                gn = float(torch.nn.utils.clip_grad_norm_(mlp_params, 1.0))
                if cur_pw > 0.0:
                    torch.nn.utils.clip_grad_norm_(head_params, 1.0)
                opt.step()
                assert math.isfinite(sum_ce) and math.isfinite(sum_kl) and math.isfinite(sum_plato), f"Stage A diverged at step {s}!"
                if s % 50 == 0 or s == steps - 1:
                    print(
                        f"  STAGE_A step {s:3d}/{steps} ce={sum_ce:.3f} kl={sum_kl:.4f} plato={sum_plato:.4f} "
                        f"(w={cur_pw:.3f}) gn={gn:.2f} vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB ({time.time()-t0:.0f}s)",
                        flush=True,
                    )
    finally:
        model.set_grad_checkpointing(False)
        model.zero_grad(set_to_none=True)
        model.freeze_all()
        model.eval()
        del opt, ref_mlp_cpu, backup_buf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_ckpt = CKPT / "e4b_stage_a.pt"
    model.save_living_slice(out_ckpt, meta={"stage": "A", "steps": steps, "ujung_alpha": ujung_alpha, "elapsed_s": round(time.time() - t0, 1)})
    ev = eval_blind_platonic_and_jung(model)
    _atomic_json(RUNS / "e4b_stage_a_eval.json", ev)
    print(
        f"STAGE_A_DONE saved={out_ckpt} phi4={ev['blind_phi4_pts']['cos']:.4f}(R2f={ev['blind_phi4_pts']['r2_frob']:+.3f}) "
        f"olmo3={ev['blind_olmo3_pts']['cos']:.4f}(R2f={ev['blind_olmo3_pts']['r2_frob']:+.3f}) "
        f"H2f={ev['jung_L12']['house_2f_acc']:.3f}(lr={ev['jung_L12']['house_2f_lenres']:.3f}) nll={ev['held_nll']} ({time.time()-t0:.0f}s)",
        flush=True,
    )
    model.close()


# ---------------------------------------------------------------- Stage B: O3 Titans-Momentum + Upper Full-Rank Natural Buddy Voice & <search>

def _encode_buddy_dialogue(proc, row, max_len=768):
    """Format a buddy_train230 dialogue using Gemma 4 chat template and build exact assistant-only loss mask."""
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    msgs = [{"role": "system", "content": bd.BUDDY_SYSTEM_PROMPT}]
    for t in row["turns"]:
        role = "user" if t["role"] in ("user", "tool") else "assistant"
        msgs.append({"role": role, "content": t["content"]})

    full_txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    full_ids = tok.encode(full_txt, add_special_tokens=False)[:max_len]
    mask = np.zeros(len(full_ids), dtype=np.float32)

    for i in range(1, len(msgs)):
        if msgs[i]["role"] != "assistant":
            continue
        pre_txt = proc.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inc_txt = proc.apply_chat_template(msgs[:i + 1], tokenize=False, add_generation_prompt=False, enable_thinking=False)
        a = len(tok.encode(pre_txt, add_special_tokens=False))
        b = min(len(full_ids), len(tok.encode(inc_txt, add_special_tokens=False)))
        if a < b:
            mask[a:b] = 1.0
    return np.array(full_ids, dtype=np.int64), mask


def run_stage_b(steps=500, lr_mlp=1.2e-5, lr_titans=5e-4, kl_w=0.8, seed=230):
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    in_ckpt = CKPT / "e4b_stage_a.pt"
    assert in_ckpt.exists(), f"missing {in_ckpt}; run stage_a first!"
    tr_path = DATA / "buddy_train230.jsonl"
    assert tr_path.exists(), f"missing {tr_path}; run aion_data230.py first!"

    model = load_living_e4b(slice_path=in_ckpt, with_plato_head=False, offload_mm=True)
    ref_mlp_cpu = model.snapshot_mlp_weights_cpu(STAGE_B_MLP_LAYERS)
    backup_buf = {}
    dev = next(model.lm.parameters()).device

    with open(tr_path) as f:
        buddy_rows = [json.loads(l) for l in f if l.strip()]
    buddy_enc = [_encode_buddy_dialogue(model.processor, r, max_len=768) for r in buddy_rows]
    buddy_enc = [(ids, msk) for ids, msk in buddy_enc if len(ids) >= 16 and msk.sum() >= 4]
    gen_texts = []
    with open(PLATO / "passages.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if r.get("domain") in ("gut", "proc", "chat"):
                gen_texts.append(r["text"])

    model.set_grad_checkpointing(True)
    opt = None
    try:
        with model.fp32_mlps_for_training(STAGE_B_MLP_LAYERS):
            model.unfreeze_stage_b(STAGE_B_MLP_LAYERS)
            mlp_params = [p for li in STAGE_B_MLP_LAYERS for p in model.lm.layers[li].mlp.parameters()]
            t_decay = [p for p in model.titans.parameters() if p.dim() >= 2]
            t_nodecay = [p for p in model.titans.parameters() if p.dim() < 2]
            titans_params = t_decay + t_nodecay
            opt = torch.optim.AdamW([
                {"params": mlp_params, "lr": lr_mlp, "weight_decay": 0.01},
                {"params": t_decay, "lr": lr_titans, "weight_decay": 0.01},
                {"params": t_nodecay, "lr": lr_titans, "weight_decay": 0.0},
            ], betas=(0.9, 0.95))

            print(
                f"STAGE_B start: steps={steps} buddy_dialogues={len(buddy_enc)} "
                f"mlp_params={sum(p.numel() for p in mlp_params):,} (fp32) titans_params={sum(p.numel() for p in titans_params):,} (fp32)",
                flush=True,
            )
            model.train()

            for s in range(steps):
                w_s = max(1, steps // 10)
                lr_mul = (s + 1) / w_s if s < w_s else 0.5 * (1.0 + math.cos(math.pi * (s - w_s) / max(1, steps - w_s)))
                opt.param_groups[0]["lr"] = lr_mlp * lr_mul
                opt.param_groups[1]["lr"] = lr_titans * lr_mul
                opt.param_groups[2]["lr"] = lr_titans * lr_mul

                opt.zero_grad(set_to_none=True)

                # 1. Two buddy dialogues sequentially (B=1 -> zero padding waste, tiny peak VRAM, exact O3 updates!)
                ce_b_tot = 0.0
                for _ in range(2):
                    ids, msk = buddy_enc[int(rng.integers(0, len(buddy_enc)))]
                    xb = torch.from_numpy(ids).unsqueeze(0).to(dev)
                    mb = torch.from_numpy(msk[1:]).unsqueeze(0).to(dev)
                    with _autocast_ctx(dev), model.run_ctx(enabled=True, state=None, update=True, update_last=False):
                        hb = model.base.model(input_ids=xb, use_cache=False).last_hidden_state[:, :-1, :]
                        ce_b, _ = chunked_lm_head_ce_kl(
                            model.base, hb, y_ids=xb[:, 1:], mask=mb, ce_w=0.5, kl_w=0.0, chunk_t=64, backward=True,
                        )
                    ce_b_tot += 0.5 * ce_b
                    del hb, xb, mb

                # 2. General KL-anchor passage sequentially (Stage-A ref logits via CPU swap!)
                g_txt = gen_texts[int(rng.integers(0, len(gen_texts)))]
                g_ids = _gemma_encode_with_bos(model.tok, g_txt)[:384]
                xg = torch.tensor([g_ids], dtype=torch.long, device=dev)
                with torch.no_grad(), model.swapped_to_ref_weights(ref_mlp_cpu, disable_titans=True, backup_buf=backup_buf):
                    with _autocast_ctx(dev):
                        h_ref = model.base.model(input_ids=xg, use_cache=False).last_hidden_state[:, :-1, :].detach()

                with _autocast_ctx(dev), model.run_ctx(enabled=True, state=None, update=True, update_last=False):
                    hg = model.base.model(input_ids=xg, use_cache=False).last_hidden_state[:, :-1, :]
                    ce_g, kl_g = chunked_lm_head_ce_kl(
                        model.base, hg, y_ids=xg[:, 1:], h_ref=h_ref, ce_w=0.25, kl_w=kl_w, chunk_t=64, backward=True,
                    )
                del hg, h_ref, xg

                gn_m = float(torch.nn.utils.clip_grad_norm_(mlp_params, 1.0))
                gn_t = float(torch.nn.utils.clip_grad_norm_(titans_params, 1.0))
                opt.step()
                assert math.isfinite(ce_b_tot) and math.isfinite(ce_g) and math.isfinite(kl_g), f"Stage B diverged at step {s}!"
                if s % 50 == 0 or s == steps - 1:
                    t_onorm = float(np.mean([model.titans[str(li)].o.weight.detach().float().norm().item() for li in TITANS_LAYERS]))
                    print(
                        f"  STAGE_B step {s:3d}/{steps} ce_buddy={ce_b_tot:.3f} ce_gen={ce_g:.3f} kl={kl_g:.4f} "
                        f"o_norm={t_onorm:.3f} gn=({gn_m:.2f},{gn_t:.2f}) vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB ({time.time()-t0:.0f}s)",
                        flush=True,
                    )
    finally:
        model.set_grad_checkpointing(False)
        model.zero_grad(set_to_none=True)
        model.freeze_all()
        model.eval()
        del opt, ref_mlp_cpu, backup_buf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_ckpt = CKPT / "e4b_stage_b.pt"
    model.save_living_slice(out_ckpt, meta={"stage": "B", "steps": steps, "elapsed_s": round(time.time() - t0, 1)})
    model.reload_multimodal_to_gpu()
    res = {
        "platonic_jung": eval_blind_platonic_and_jung(model),
        "buddy": eval_buddy_behavior(model),
        "multimodal": eval_multimodal_sanity(model),
    }
    _atomic_json(RUNS / "e4b_stage_b_eval.json", res)
    print(
        f"STAGE_B_DONE saved={out_ckpt} bp={res['buddy'].get('boilerplate_rate')} "
        f"srch={res['buddy'].get('search_strict_only_recall')} easy_fp={res['buddy'].get('easy_search_false_trigger')} "
        f"wmem_cold=({res['buddy'].get('wmem_cold_recall_O3_on')} vs {res['buddy'].get('wmem_cold_recall_O3_off')}) "
        f"mm={res['multimodal']} ({time.time()-t0:.0f}s)",
        flush=True,
    )
    model.close()


# ---------------------------------------------------------------- Unit & Tiny-Gemma4 Integration Self-Test

def _make_tiny_gemma4():
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig
    tcfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=256,
        final_logit_softcapping=30.0,
        sliding_window=32,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    )
    cfg = Gemma4Config(text_config=tcfg)
    base = Gemma4ForConditionalGeneration(cfg).to(dtype=torch.float32)
    base.eval()
    return base


class _DummyTok:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0
    def encode(self, s, add_special_tokens=False):
        return [3 + (ord(c) % 200) for c in s]
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class _DummyBatch(dict):
    def to(self, dev):
        return _DummyBatch({k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in self.items()})


class _DummyProc:
    def __init__(self):
        self.tokenizer = _DummyTok()
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "|".join(m["content"] for m in msgs if isinstance(m["content"], str))
    def __call__(self, text, return_tensors="pt", **kw):
        ids = _gemma_encode_with_bos(self.tokenizer, text)
        return _DummyBatch({"input_ids": torch.tensor([ids], dtype=torch.long)})
    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def self_test():
    """Verify O3 Titans-Momentum zero-init, causality, chunked lm_head exactness, KV-shared GC (use_reentrant=False),
    live prefill->decode state handoff, ref-Titans KL==0 swap, slice save/load, and hook teardown!"""
    torch.manual_seed(0)
    tb = TitansBranchE4B(D=64, d_io=16, d_hid=32, chunk=8)
    x = torch.randn(2, 24, 64)
    y0, _ = tb(x, state=None, update=True)
    assert y0.abs().max().item() == 0.0, "TitansBranchE4B not zero-init!"
    with torch.no_grad():
        nn.init.normal_(tb.o.weight, 0, 0.1)
    y_a, st_a = tb(x, state=None, update=True)
    x2 = x.clone(); x2[:, 16:, :] = torch.randn(2, 8, 64)
    y_b, _ = tb(x2, state=None, update=True)
    assert (y_a[:, :16, :] - y_b[:, :16, :]).abs().max().item() < 1e-5, "Titans causality leak!"
    y_1, st_1 = tb(x[:, :1, :], state=st_a, update=True)
    assert torch.isfinite(y_1).all() and all(torch.equal(a, b) for a, b in zip(st_a, st_1))

    # Real 6-layer Gemma4ForConditionalGeneration integration test (with KV sharing!)
    base = _make_tiny_gemma4()
    proc = _DummyProc()
    living = LivingE4B(base, proc, titans_layers=(1, 4, 5), d_io=16, d_hid=32, chunk=4, with_plato_head=False)
    ids = torch.randint(3, 200, (1, 16))
    y_tgt = ids[:, 1:]

    # 1. Exact chunked_lm_head_ce_kl parity vs full forward+backward!
    living.freeze_all()
    for p in living.lm.layers[5].mlp.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        h_ref = living.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1, :].detach()
        lg_ref_full = living.base(input_ids=ids, use_cache=False).logits[:, :-1, :].float()
        p_ref_full = F.softmax(lg_ref_full, dim=-1)

    living.zero_grad(set_to_none=True)
    lg_full = living.base(input_ids=ids, use_cache=False).logits[:, :-1, :].float()
    loss_full = 0.5 * F.cross_entropy(lg_full.reshape(-1, 256), y_tgt.reshape(-1)) + 1.2 * F.kl_div(F.log_softmax(lg_full, -1), p_ref_full, reduction="none").sum(-1).mean()
    loss_full.backward()
    g_full = {n: p.grad.clone() for n, p in living.lm.layers[5].mlp.named_parameters()}

    living.zero_grad(set_to_none=True)
    h_cur = living.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1, :]
    ce_c, kl_c = chunked_lm_head_ce_kl(living.base, h_cur, y_ids=y_tgt, h_ref=h_ref, ce_w=0.5, kl_w=1.2, chunk_t=5, backward=True)
    max_gerr = max((p.grad - g_full[n]).abs().max().item() for n, p in living.lm.layers[5].mlp.named_parameters())
    assert max_gerr < 1e-5, f"chunked_lm_head_ce_kl grad mismatch: {max_gerr}"

    # 2. Verify use_reentrant=False GC matches exact non-GC grads across KV-shared layers!
    living.freeze_all()
    for p in living.lm.layers[1].mlp.parameters():
        p.requires_grad_(True)
    living.zero_grad(set_to_none=True)
    h1 = living.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1, :]
    chunked_lm_head_ce_kl(living.base, h1, y_ids=y_tgt, ce_w=1.0, kl_w=0.0, chunk_t=8, backward=True)
    g_nogc = {n: p.grad.clone() for n, p in living.lm.layers[1].mlp.named_parameters()}
    living.set_grad_checkpointing(True)
    living.train()
    living.zero_grad(set_to_none=True)
    h2 = living.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1, :]
    chunked_lm_head_ce_kl(living.base, h2, y_ids=y_tgt, ce_w=1.0, kl_w=0.0, chunk_t=8, backward=True)
    living.set_grad_checkpointing(False)
    living.eval()
    gc_err = max((p.grad - g_nogc[n]).abs().max().item() for n, p in living.lm.layers[1].mlp.named_parameters())
    assert gc_err < 1e-5, f"KV-shared GC use_reentrant=False grad mismatch: {gc_err}"

    # 2b. Verify Stage-B pattern (Titans enabled+update=True INSIDE GC use_reentrant=False backward!)
    living.unfreeze_stage_b(mlp_layers=(5,))
    living.set_grad_checkpointing(True)
    living.train()
    living.zero_grad(set_to_none=True)
    with living.run_ctx(enabled=True, state=None, update=True, update_last=False):
        h_tb = living.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1, :]
        chunked_lm_head_ce_kl(living.base, h_tb, y_ids=y_tgt, ce_w=1.0, kl_w=0.0, chunk_t=8, backward=True)
    living.set_grad_checkpointing(False)
    living.eval()
    assert living.titans["1"].o.weight.grad is not None and torch.isfinite(living.titans["1"].o.weight.grad).all()

    # 3. Verify swapped_to_ref_weights with ref_titans_cpu gives EXACT KL==0.0 when Titans.o != 0!
    with torch.no_grad():
        for li in living.titans_layers:
            nn.init.normal_(living.titans[str(li)].o.weight, 0, 0.05)
    rmlp = living.snapshot_mlp_weights_cpu((5,))
    rtit = living.snapshot_titans_cpu(("W1", "W2", "W3"))
    bbuf = {}
    with torch.no_grad(), living.swapped_to_ref_weights(rmlp, ref_titans_cpu=rtit, disable_titans=False, backup_buf=bbuf):
        hr_t = living.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1, :]
    with torch.no_grad(), living.run_ctx(enabled=True, state=None, update=False):
        hc_t = living.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1, :]
    _, kl_zero = chunked_lm_head_ce_kl(living.base, hc_t, h_ref=hr_t, ce_w=0.0, kl_w=1.0, chunk_t=8)
    assert kl_zero < 1e-6, f"Step-0 KL with ref_titans_cpu not zero: {kl_zero}"

    # 4. Verify live generate() prefill->decode state update reaches KV-shared Titans layers (4, 5)!
    init_st = living.init_fast_state(1)
    _, new_st = living.chat_reply([{"role": "user", "content": "hello buddy test message here"}], state=init_st, max_new_tokens=4, temp=0.0, use_titans=True, update_state=True)
    for li in living.titans_layers:
        diff_w1 = (new_st[li][0] - init_st[li][0]).abs().max().item()
        assert diff_w1 > 1e-6, f"Titans layer {li} fast state did not update during generate()!"

    # 5. Verify save/load slice + hook teardown in an isolated temp dir (safe under concurrent runs)!
    import tempfile
    with tempfile.TemporaryDirectory(prefix="e4b_living_test_") as td:
        tmp_sl = Path(td) / "_test_slice.pt"
        living.save_living_slice(tmp_sl, mlp_layers=(1, 5))
        living.load_living_slice(tmp_sl)
    living.close()
    assert len(living._hooks) == 0

    print(
        f"E4B_LIVING_TEST_OK zero_init=0.0 causality=OK chunked_gerr={max_gerr:.1e} "
        f"kv_gc_err={gc_err:.1e} ref_titans_kl0={kl_zero:.1e} gen_state_all_layers=OK hooks_closed=OK",
        flush=True,
    )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        self_test()
    elif cmd == "eval_base":
        m = load_living_e4b(slice_path=None, offload_mm=False)
        res = {
            "platonic_jung": eval_blind_platonic_and_jung(m),
            "buddy": eval_buddy_behavior(m),
            "multimodal": eval_multimodal_sanity(m),
        }
        _atomic_json(RUNS / "e4b_base_eval.json", res)
        m.close()
        print("EVAL_BASE_DONE:", json.dumps({k: v for k, v in res.items() if k != "buddy"} | {"buddy": {k: v for k, v in res["buddy"].items() if k != "samples"}}, indent=1), flush=True)
    elif cmd == "stage_a":
        run_stage_a()
    elif cmd == "stage_b":
        run_stage_b()
    elif cmd == "eval_all":
        ckpt = CKPT / "e4b_stage_b.pt"
        m = load_living_e4b(slice_path=ckpt, offload_mm=False)
        res = {
            "platonic_jung": eval_blind_platonic_and_jung(m),
            "buddy": eval_buddy_behavior(m),
            "multimodal": eval_multimodal_sanity(m),
        }
        _atomic_json(RUNS / "e4b_all_eval.json", res)
        m.close()
        print("EVAL_ALL_DONE:", json.dumps(res, indent=1), flush=True)
    else:
        raise ValueError(cmd)
