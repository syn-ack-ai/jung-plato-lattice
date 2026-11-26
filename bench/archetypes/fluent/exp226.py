"""Addendum 226: inference-time online learning screen (5M scale, 100M tokens, 3 seeds; paired directly against Addendum 225's X0_s1..s3).
All four arms update nonlinear or full-rank MLP weights online during inference (chunk size C=64: 16 online updates per 1,024-tok train window,
2-5 live weight updates per proc_held/mcq2 passage), match X0's 4.72M non-embed params within 0.5%, and are 100% strictly causal.
Arms:
  O1: In-Place TTT (arXiv:2604.06169) — online gradient update of real MLP W_down in layers 3-5 on local next-token target
  O2: LaCT (arXiv:2505.23884) — nonlinear 2-layer SwiGLU fast-weight MLP (d_io=128, d_hid=256) in layers 1,3,5 updated by Muon (3-iter Newton-Schulz)
  O3: Titans-Momentum (arXiv:2501.00663) — identical nonlinear SwiGLU fast MLP updated by data-dependent surprise momentum + adaptive forget gate
  O4: TTT-E2E (inspired by arXiv:2512.23675) — true end-to-end online next-token cross-entropy gradient step on fast adapter weights in layers 4-5
Includes context-cleared weight-retention & catastrophic-forgetting evaluation (`score_retention`): one-shot correction retention,
passage-to-weight retention, and 10-correction sequential interference + BLiMP collateral forgetting with KV/history wiped.
Usage:
  python fluent/exp226.py test                       unit checks (analytic SwiGLU, NS3, causality eval+train, CUDA bf16, param parity, retention API)
  python fluent/exp226.py train NAME arm=O1..O4 seed=1 [tokens=100e6]
  python fluent/exp226.py score NAME...              scores BLiMP, COPA, lastword, mcq, mcq2, proc_held + context-cleared retention/forgetting
Public training/eval text only; Jung held-out bpb is evaluated locally only and never printed or exported."""
import json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).parent
RUNS = HERE / "runs"
COMP = HERE / "comp"
MID = HERE / "mid"
PRIV = Path(os.environ.get("PLATONIC_CKPT_DIR", str(HERE / "local_ckpt")))
GUT = HERE.parent / "smallworld"
JUNG8 = Path(os.environ.get("PLATONIC_EXTRA_EVAL_DIR", str(HERE / "local_eval")))
TOK_PATH = HERE.parent / "transfer" / "tok" / "bpe8k.json"
DEV = os.environ.get("FL_DEV") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
AMP = os.environ.get("FL_AMP", "bf16" if DEV.startswith("cuda") else "")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

for p in (RUNS, RUNS / "comp", PRIV / "ckpt"):
    p.mkdir(parents=True, exist_ok=True)


def amp():
    return torch.autocast(DEV.split(":")[0], dtype=torch.bfloat16) if AMP == "bf16" else torch.autocast(DEV.split(":")[0], enabled=False)


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK_PATH))


def streams():
    tr = np.memmap(GUT / "gut_tok2" / "train.bin", dtype=np.uint16, mode="r")
    held = np.memmap(GUT / "gut_tok2" / "held.bin", dtype=np.uint16, mode="r")
    z = np.load(JUNG8 / "held.npz")
    jt, off = z["toks"], z["off"]
    jung = [jt[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    proc = np.memmap(MID / "proc_train.bin", dtype=np.uint16, mode="r")
    chat_tr = np.memmap(MID / "chat_train.bin", dtype=np.uint16, mode="r")
    chat_he = np.memmap(MID / "chat_held.bin", dtype=np.uint16, mode="r")
    return tr, held, jung, proc, chat_tr, chat_he


# ---------------------------------------------------------------- arm configs (matched to X0 = 4,722,304 non-embed within <0.5%)

ARM_SPEC = {
    "O1": dict(mlp=960, mode="inplace", fast_layers=(3, 4, 5)),
    "O2": dict(mlp=800, mode="lact",    fast_layers=(1, 3, 5)),
    "O3": dict(mlp=800, mode="titans",  fast_layers=(1, 3, 5)),
    "O4": dict(mlp=984, mode="e2e",     fast_layers=(4, 5)),
}

BASE_CFG = dict(D=256, L=6, H=8, ctx=1024, batch=32, mb=8, lr=3e-3, wd=0.1, zloss=1e-4,
                tokens=100e6, chunk=64, df_io=128, df_hid=256, e2e_r=128, tbptt=4)


# ---------------------------------------------------------------- core primitives

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


def rope(x, cos, sin):
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


_MASK = {}


def causal_mask(n, dev):
    k = (n, str(dev))
    if k not in _MASK:
        _MASK[k] = torch.ones(n, n, dtype=torch.bool, device=dev).tril()
    return _MASK[k]


def _smooth_norm(X, min_val=None):
    """smooth Frobenius norm per batch item (zero-safe backward: no 0/0 NaN at X == 0)"""
    n = (X.flatten(1).pow(2).sum(dim=1) + 1e-12).sqrt().view(X.shape[0], 1, 1)
    return n.clamp_min(min_val) if min_val is not None else n


def _newton_schulz3(G):
    """3-step quintic Newton-Schulz matrix orthogonalization (Muon, strict fp32).
    Output X already has spectral norm ~1 and entrywise RMS ~1/sqrt(max(m, n)) (amendment 226a: no double division)."""
    with torch.autocast(G.device.type, enabled=False):
        X = G.float()
        m, n = X.shape[-2], X.shape[-1]
        trans = m > n
        if trans:
            X = X.transpose(-1, -2)
        X = X / _smooth_norm(X)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(3):
            A = torch.matmul(X, X.transpose(-1, -2))
            B_mat = b * A + c * torch.matmul(A, A)
            X = a * X + torch.matmul(B_mat, X)
        if trans:
            X = X.transpose(-1, -2)
        return X


def _swiglu_fwd_bwd(K, V, W1, W2, W3):
    """Exact forward and analytical MSE gradients for batched 2-layer SwiGLU fast MLP:
    f(K) = (SiLU(K @ W1^T) * (K @ W3^T)) @ W2^T,  L = (1 / 2C) ||f(K) - V||_F^2. Strict fp32."""
    with torch.autocast(K.device.type, enabled=False):
        K, V = K.float(), V.float()
        W1, W2, W3 = W1.float(), W2.float(), W3.float()
        C = K.shape[1]
        A = torch.matmul(K, W1.transpose(-1, -2))
        G = torch.matmul(K, W3.transpose(-1, -2))
        sig = torch.sigmoid(A)
        S = A * sig
        Z = S * G
        V_hat = torch.matmul(Z, W2.transpose(-1, -2))
        dV = (V_hat - V) / float(C)
        dW2 = torch.matmul(dV.transpose(-1, -2), Z)
        dZ = torch.matmul(dV, W2)
        dG = dZ * S
        dS = dZ * G
        dA = dS * (sig * (1.0 + A * (1.0 - sig)))
        dW1 = torch.matmul(dA.transpose(-1, -2), K)
        dW3 = torch.matmul(dG.transpose(-1, -2), K)
    return dW1, dW2, dW3


# ---------------------------------------------------------------- O1: In-Place TTT MLP (online W_down update)

class InPlaceTTTMLP(nn.Module):
    """In-Place TTT (arXiv:2604.06169): treats real MLP W_down as chunkwise fast weights updated on local next-token representation MSE."""
    def __init__(self, D, mlp, chunk=64):
        super().__init__()
        self.D = D
        self.mlp = mlp
        self.chunk = chunk
        self.up = nn.Linear(D, mlp, bias=False)
        self.down = nn.Linear(mlp, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.eta_logit = nn.Parameter(torch.tensor([-3.5]))    # 1D -> no WD; 1.0 * sigmoid(-3.5) ~ 0.029 with headroom up to 1.0
        self.gamma_logit = nn.Parameter(torch.tensor([2.5]))   # 1D -> no WD; retention ~ 0.924
        nn.init.zeros_(self.down.weight)

    def init_state(self, B):
        return self.down.weight.float().unsqueeze(0).expand(B, self.D, self.mlp).clone()

    def forward(self, xn, state=None, update=True, update_last=False):
        B, T, D = xn.shape
        Z_all = F.relu(self.up(xn)).square()
        V_all = self.v_proj(xn)
        outs = []
        with torch.autocast(xn.device.type, enabled=False):
            W0 = self.down.weight.float().unsqueeze(0)
            Wd = W0.expand(B, D, self.mlp) if state is None else state
            eta = torch.sigmoid(self.eta_logit.float()).view(1, 1, 1)
            gamma = torch.sigmoid(self.gamma_logit.float()).view(1, 1, 1)
            scale_v = D ** 0.5
            for s in range(0, T, self.chunk):
                e = min(T, s + self.chunk)
                c_len = e - s
                Zc = Z_all[:, s:e, :].float()
                outs.append(torch.matmul(Zc, Wd.transpose(-1, -2)))
                if update and (e < T or update_last) and c_len >= 2:
                    V_tgt = F.normalize(V_all[:, s + 1:e, :].float(), dim=-1, eps=1e-6) * scale_v
                    V_hat = torch.matmul(Zc[:, :-1, :], Wd.transpose(-1, -2))
                    grad = torch.matmul((V_hat - V_tgt).transpose(-1, -2), Zc[:, :-1, :]) / float(c_len - 1)
                    grad = grad / _smooth_norm(grad)
                    Wd = (1.0 - gamma) * W0 + gamma * Wd - eta * grad
        return torch.cat(outs, dim=1).to(xn.dtype), Wd


# ---------------------------------------------------------------- O2 (LaCT Muon) & O3 (Titans Surprise-Momentum) Nonlinear SwiGLU Fast MLP

class NonlinearFastMLP(nn.Module):
    """2-layer nonlinear SwiGLU fast-weight MLP updated every C=64 tokens by either:
      - mode='lact':   Muon 3-iter Newton-Schulz orthogonalized surprise gradient (arXiv:2505.23884)
      - mode='titans': Data-dependent dynamic surprise momentum + adaptive forget gate (arXiv:2501.00663)"""
    def __init__(self, D, d_io=128, d_hid=256, chunk=64, mode="lact"):
        super().__init__()
        self.d_io = d_io
        self.d_hid = d_hid
        self.chunk = chunk
        self.mode = mode
        self.n = RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * d_io, bias=False)
        self.W1 = nn.Parameter(torch.empty(d_hid, d_io))
        self.W2 = nn.Parameter(torch.empty(d_io, d_hid))
        self.W3 = nn.Parameter(torch.empty(d_hid, d_io))
        nn.init.normal_(self.W1, 0, d_io ** -0.5)
        nn.init.normal_(self.W3, 0, d_io ** -0.5)
        nn.init.normal_(self.W2, 0, d_hid ** -0.5)
        self.o = nn.Linear(d_io, D, bias=False)
        nn.init.zeros_(self.o.weight)
        if mode == "lact":
            self.eta_logit = nn.Parameter(torch.full((3,), -2.0))    # 1D, per matrix (W1, W2, W3)
            self.gamma_logit = nn.Parameter(torch.full((3,), 2.5))   # 1D, retention
        else:
            self.gate_proj = nn.Linear(D, 3, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            with torch.no_grad():
                self.gate_proj.bias.copy_(torch.tensor([0.5, -2.0, -2.5]))  # theta~0.62, eta~0.03, alpha~0.076

    def init_state(self, B):
        W1 = self.W1.float().unsqueeze(0).expand(B, self.d_hid, self.d_io).clone()
        W2 = self.W2.float().unsqueeze(0).expand(B, self.d_io, self.d_hid).clone()
        W3 = self.W3.float().unsqueeze(0).expand(B, self.d_hid, self.d_io).clone()
        if self.mode == "lact":
            return (W1, W2, W3)
        return (W1, W2, W3, torch.zeros_like(W1), torch.zeros_like(W2), torch.zeros_like(W3))

    def forward(self, x, state=None, update=True, update_last=False):
        B, T, D = x.shape
        xn = self.n(x)
        qkv = self.qkv(xn).view(B, T, 3, self.d_io)
        outs = []
        with torch.autocast(x.device.type, enabled=False):
            # Scale L2-normalized Q, K by sqrt(d_io) -> RMS 1.0 (amendment 226a: avoids 150x SwiGLU init collapse)
            scale_qk = self.d_io ** 0.5
            Q = F.normalize(qkv[:, :, 0].float(), dim=-1, eps=1e-6) * scale_qk
            K = F.normalize(qkv[:, :, 1].float(), dim=-1, eps=1e-6) * scale_qk
            V = qkv[:, :, 2].float()
            W1_0 = self.W1.float().unsqueeze(0)
            W2_0 = self.W2.float().unsqueeze(0)
            W3_0 = self.W3.float().unsqueeze(0)
            if state is None:
                W1 = W1_0.expand(B, self.d_hid, self.d_io)
                W2 = W2_0.expand(B, self.d_io, self.d_hid)
                W3 = W3_0.expand(B, self.d_hid, self.d_io)
                S1 = S2 = S3 = None
                if self.mode != "lact":
                    S1, S2, S3 = torch.zeros_like(W1), torch.zeros_like(W2), torch.zeros_like(W3)
            elif self.mode == "lact":
                W1, W2, W3 = state
                S1 = S2 = S3 = None
            else:
                W1, W2, W3, S1, S2, S3 = state

            if self.mode == "lact":
                etas = 0.25 * torch.sigmoid(self.eta_logit.float())
                gammas = torch.sigmoid(self.gamma_logit.float())

            for s in range(0, T, self.chunk):
                e = min(T, s + self.chunk)
                qc, kc, vc = Q[:, s:e, :], K[:, s:e, :], V[:, s:e, :]
                Ac = torch.matmul(qc, W1.transpose(-1, -2))
                Gc = torch.matmul(qc, W3.transpose(-1, -2))
                rc = torch.matmul(F.silu(Ac) * Gc, W2.transpose(-1, -2))
                outs.append(rc)

                if update and (e < T or update_last):
                    dW1, dW2, dW3 = _swiglu_fwd_bwd(kc, vc, W1, W2, W3)
                    if self.mode == "lact":
                        u1 = _newton_schulz3(dW1)
                        u2 = _newton_schulz3(dW2)
                        u3 = _newton_schulz3(dW3)
                        W1 = (1.0 - gammas[0]) * W1_0 + gammas[0] * W1 - etas[0] * u1
                        W2 = (1.0 - gammas[1]) * W2_0 + gammas[1] * W2 - etas[1] * u2
                        W3 = (1.0 - gammas[2]) * W3_0 + gammas[2] * W3 - etas[2] * u3
                    else:
                        g_in = xn[:, s:e, :].float().mean(dim=1)
                        gates = self.gate_proj(g_in.to(self.gate_proj.weight.dtype)).float()
                        theta = torch.sigmoid(gates[:, 0]).view(B, 1, 1)
                        eta = 0.25 * torch.sigmoid(gates[:, 1]).view(B, 1, 1)
                        alpha = torch.sigmoid(gates[:, 2]).view(B, 1, 1)
                        S1 = theta * S1 - eta * (dW1 / _smooth_norm(dW1))
                        S2 = theta * S2 - eta * (dW2 / _smooth_norm(dW2))
                        S3 = theta * S3 - eta * (dW3 / _smooth_norm(dW3))
                        W1 = (1.0 - alpha) * W1 + alpha * W1_0 + S1
                        W2 = (1.0 - alpha) * W2 + alpha * W2_0 + S2
                        W3 = (1.0 - alpha) * W3 + alpha * W3_0 + S3

        new_state = (W1, W2, W3) if self.mode == "lact" else (W1, W2, W3, S1, S2, S3)
        return self.o(torch.cat(outs, dim=1).to(x.dtype)), new_state


# ---------------------------------------------------------------- O4: TTT-E2E fast adapter params (updated by true NTP autograd)

class FastAdapterE2E(nn.Module):
    """Compact fast-plasticity MLP adapter in top layers (4, 5) updated online by true end-to-end NTP cross-entropy gradients."""
    def __init__(self, D, r=128):
        super().__init__()
        self.n = RMSNorm(D)
        self.w_up = nn.Parameter(torch.empty(r, D))
        self.w_down = nn.Parameter(torch.empty(D, r))
        nn.init.normal_(self.w_up, 0, D ** -0.5)
        nn.init.normal_(self.w_down, 0, r ** -0.5)
        self.scale = nn.Parameter(torch.tensor([0.05]))       # 1D, gentle initial residual contribution
        self.eta_logit = nn.Parameter(torch.tensor([-2.0]))   # 1D, online LR
        self.gamma_logit = nn.Parameter(torch.tensor([2.5]))  # 1D, base retention

    def init_fast(self, B):
        return (
            self.w_up.float().unsqueeze(0).expand(B, -1, -1).clone(),
            self.w_down.float().unsqueeze(0).expand(B, -1, -1).clone(),
        )

    def apply_fast(self, x, w_up, w_down):
        xn = self.n(x)
        with torch.autocast(x.device.type, enabled=False):
            h = F.silu(torch.matmul(xn.float(), w_up.transpose(-1, -2)))
            y = torch.matmul(h, w_down.transpose(-1, -2)) * self.scale.float()
        return y.to(x.dtype)


# ---------------------------------------------------------------- unified Block & Model for O1..O4

class Block226(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        D, self.H, mlp = c["D"], c["H"], c["mlp"]
        self.mode = c["mode"]
        self.is_fast = layer_idx in c["fast_layers"]
        self.n1, self.n2 = RMSNorm(D), RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.qn, self.kn = RMSNorm(D // self.H), RMSNorm(D // self.H)
        nn.init.zeros_(self.o.weight)

        if self.mode == "inplace" and self.is_fast:
            self.mlp_inplace = InPlaceTTTMLP(D, mlp, chunk=c["chunk"])
            self.up = self.down = None
        else:
            self.mlp_inplace = None
            self.up = nn.Linear(D, mlp, bias=False)
            self.down = nn.Linear(mlp, D, bias=False)
            nn.init.zeros_(self.down.weight)

        if self.mode in ("lact", "titans") and self.is_fast:
            self.fast_mlp = NonlinearFastMLP(D, d_io=c["df_io"], d_hid=c["df_hid"], chunk=c["chunk"], mode=self.mode)
        else:
            self.fast_mlp = None

        if self.mode == "e2e" and self.is_fast:
            self.e2e_ad = FastAdapterE2E(D, r=c["e2e_r"])
        else:
            self.e2e_ad = None

    def _attn(self, x, rot, kv_cache=None, math_attn=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        q, k = rope(self.qn(q), *rot), rope(self.kn(k), *rot)
        if kv_cache is not None:
            k_past, v_past = kv_cache
            k_use = torch.cat([k_past, k], dim=2)
            v_use = torch.cat([v_past, v], dim=2)
            T_past = k_past.shape[2]
            mask = torch.cat([torch.ones((T, T_past), dtype=torch.bool, device=x.device), causal_mask(T, x.device)], dim=1)
        else:
            k_use, v_use = k, v
            mask = None
        if math_attn:
            # Explicit fp32-softmax attention: 100% double-backward safe on CUDA/bf16 (amendment 226a)
            m_use = mask if mask is not None else causal_mask(T, x.device)
            with torch.autocast(x.device.type, enabled=False):
                sc = torch.matmul(q.float(), k_use.float().transpose(-1, -2)) * ((D // self.H) ** -0.5)
                sc = sc.masked_fill(~m_use, float("-inf"))
                a = torch.matmul(F.softmax(sc, dim=-1), v_use.float()).to(x.dtype)
            return x + self.o(a.transpose(1, 2).reshape(B, T, D)), (k_use, v_use)
        if mask is not None:
            a = F.scaled_dot_product_attention(q, k_use, v_use, attn_mask=mask)
        elif x.device.type == "cuda":
            a = F.scaled_dot_product_attention(q, k_use, v_use, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k_use, v_use, attn_mask=causal_mask(T, x.device))
        return x + self.o(a.transpose(1, 2).reshape(B, T, D)), (k_use, v_use)

    def forward_state(self, x, rot, state=None, update=True, update_last=False):
        h, _ = self._attn(x, rot)
        new_st = state
        if self.fast_mlp is not None:
            dy, new_st = self.fast_mlp(h, state=state, update=update, update_last=update_last)
            h = h + dy
        if self.mlp_inplace is not None:
            dy, new_st = self.mlp_inplace(self.n2(h), state=state, update=update, update_last=update_last)
            return h + dy, new_st
        return h + self.down(F.relu(self.up(self.n2(h))).square()), new_st

    def forward_chunk_e2e(self, x, rot, kv_cache, fast_w):
        h, new_kv = self._attn(x, rot, kv_cache=kv_cache, math_attn=True)
        if self.e2e_ad is not None:
            h = h + self.e2e_ad.apply_fast(h, fast_w[0], fast_w[1])
        return h + self.down(F.relu(self.up(self.n2(h))).square()), new_kv


class OnlineToy(nn.Module):
    def __init__(self, c, V):
        super().__init__()
        self.c = c
        D = c["D"]
        self.e = nn.Embedding(V, D)
        nn.init.normal_(self.e.weight, 0, D ** -0.5)
        self.b = nn.ModuleList(Block226(c, li) for li in range(c["L"]))
        self.n = RMSNorm(D)
        self.h = nn.Linear(D, V, bias=False)
        dh = D // c["H"]
        freq = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
        t = torch.arange(c["ctx"]).float()[:, None] * freq[None]
        self.register_buffer("cos", t.cos(), persistent=False)
        self.register_buffer("sin", t.sin(), persistent=False)

    def init_state(self, B):
        """initialize per-layer fast-weight state (no KV cache; purely weights for context-cleared retention!)"""
        st = []
        for blk in self.b:
            if blk.mlp_inplace is not None:
                st.append(blk.mlp_inplace.init_state(B))
            elif blk.fast_mlp is not None:
                st.append(blk.fast_mlp.init_state(B))
            elif blk.e2e_ad is not None:
                st.append(blk.e2e_ad.init_fast(B))
            else:
                st.append(None)
        return st

    def _forward_e2e(self, x, state=None, update=True, update_last=False):
        """O4 TTT-E2E: static layers 0..3 once, then chunkwise layers 4..5 with true online NTP gradient steps on fast adapters."""
        B, T = x.shape
        V = self.h.out_features
        chunk = self.c["chunk"]
        tbptt = self.c.get("tbptt", 4)
        h = self.e(x)
        rot_full = (self.cos[:T], self.sin[:T])
        first_fast = min(self.c["fast_layers"])
        for blk in self.b[:first_fast]:
            h, _ = blk.forward_state(h, rot_full, state=None, update=False)

        top_blks = list(self.b[first_fast:])
        if state is None:
            fast_ws = [blk.e2e_ad.init_fast(B) for blk in top_blks]
        else:
            fast_ws = list(state[first_fast:])

        # Fast path when update=False (cold evaluation using already-adapted fast weights, zero inner autograd!)
        if not update:
            for bi, blk in enumerate(top_blks):
                h, _ = blk.forward_chunk_e2e(h, rot_full, None, fast_ws[bi])
            return self.h(self.n(h)), ([None] * first_fast + fast_ws)

        kv_caches = [None for _ in top_blks]
        lg_chunks = []
        outer_grad = torch.is_grad_enabled()

        with torch.enable_grad():
            if not outer_grad:
                h = h.detach()
                fast_ws = [(u.detach().requires_grad_(True), d.detach().requires_grad_(True)) for u, d in fast_ws]

            for ci, s in enumerate(range(0, T, chunk)):
                e = min(T, s + chunk)
                c_len = e - s
                hc = h[:, s:e, :]
                rot_c = (self.cos[s:e], self.sin[s:e])
                new_kvs = []
                for bi, blk in enumerate(top_blks):
                    hc, nkv = blk.forward_chunk_e2e(hc, rot_c, kv_caches[bi], fast_ws[bi])
                    new_kvs.append(nkv)
                kv_caches = new_kvs
                lg_c = self.h(self.n(hc))
                lg_chunks.append(lg_c if outer_grad else lg_c.detach())

                if (e < T or update_last) and c_len >= 2:
                    with torch.autocast(x.device.type, enabled=False):
                        loss_tok = F.cross_entropy(
                            lg_c[:, :-1, :].float().reshape(-1, V),
                            x[:, s + 1:e].reshape(-1),
                            reduction="none",
                        ).view(B, c_len - 1)
                        loss_sum = loss_tok.mean(dim=1).sum()
                        flat_w = [w for pair in fast_ws for w in pair]
                        grads = torch.autograd.grad(loss_sum, flat_w, create_graph=outer_grad)
                        next_ws = []
                        for bi, blk in enumerate(top_blks):
                            ad = blk.e2e_ad
                            gu, gd = grads[2 * bi].float(), grads[2 * bi + 1].float()
                            nu = _smooth_norm(gu, min_val=1.0)
                            nd = _smooth_norm(gd, min_val=1.0)
                            eta = 0.25 * torch.sigmoid(ad.eta_logit.float()).view(1, 1, 1)
                            gamma = torch.sigmoid(ad.gamma_logit.float()).view(1, 1, 1)
                            wu0 = ad.w_up.float().unsqueeze(0)
                            wd0 = ad.w_down.float().unsqueeze(0)
                            wu_new = (1.0 - gamma) * wu0 + gamma * fast_ws[bi][0] - eta * (gu / nu)
                            wd_new = (1.0 - gamma) * wd0 + gamma * fast_ws[bi][1] - eta * (gd / nd)
                            # Truncated 2nd-order BPTT every `tbptt` chunks during training (amendment 226a)
                            if (not outer_grad) or ((ci + 1) % tbptt == 0):
                                wu_new = wu_new.detach().requires_grad_(True)
                                wd_new = wd_new.detach().requires_grad_(True)
                            next_ws.append((wu_new, wd_new))
                        fast_ws = next_ws
                        if not outer_grad:
                            kv_caches = [(k.detach(), v.detach()) for k, v in kv_caches]
        out_state = [None] * first_fast + [(u.detach(), d.detach()) for u, d in fast_ws]
        return torch.cat(lg_chunks, dim=1), out_state

    def forward_with_state(self, x, state=None, update=True, update_last=False):
        T = x.shape[1]
        assert T <= self.cos.shape[0], f"seq len {T} exceeds ctx {self.cos.shape[0]}"
        if self.c["mode"] == "e2e":
            return self._forward_e2e(x, state=state, update=update, update_last=update_last)
        h = self.e(x)
        rot = (self.cos[:T], self.sin[:T])
        out_st = []
        for li, blk in enumerate(self.b):
            st_l = None if state is None else state[li]
            h, nst = blk.forward_state(h, rot, state=st_l, update=update, update_last=update_last)
            out_st.append(nst)
        return self.h(self.n(h)), out_st

    def adapt(self, x, state=None):
        """update fast weights online on token sequence x (including its final/short chunk) and return ONLY updated fast weights (KV discarded)"""
        _, new_st = self.forward_with_state(x, state=state, update=True, update_last=True)
        return new_st

    def forward(self, x):
        lg, _ = self.forward_with_state(x, state=None, update=True, update_last=False)
        return lg


def nonembed(m):
    return sum(p.numel() for n, p in m.named_parameters() if not n.startswith(("e.", "h.")))


# ---------------------------------------------------------------- shared data & eval (matches exp225.py bit-for-bit!)

def bad_runs(tr, tok=188, min_len=64):
    idx = np.flatnonzero(np.asarray(tr) == tok)
    if len(idx) == 0:
        return np.zeros((0, 2), np.int64)
    br = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]] + 1
    keep = ends - starts >= min_len
    return np.stack([starts[keep], ends[keep]], 1)


def draw_clean(rng, n, hi, T, bad):
    out = rng.integers(0, hi, n)
    for k in range(n):
        while any(out[k] < b and out[k] + T + 1 > a for a, b in bad):
            out[k] = rng.integers(0, hi)
    return out


@torch.no_grad()
def eval_bpb(m, c, tk, seqs, nb=None):
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    ctx = c["ctx"]
    arrs = [np.asarray(s[:nb] if nb else s) for s in seqs]
    chunks = np.stack([s[i:i + ctx + 1] for s in arrs for i in range(0, len(s) - ctx - 1, ctx)]).astype(np.int64)
    tot, last_full, last_short = 0.0, [], []
    for i in range(0, len(chunks), c["mb"]):
        x = torch.from_numpy(chunks[i:i + c["mb"], :-1]).to(m_dev)
        y = torch.from_numpy(chunks[i:i + c["mb"], 1:]).to(m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            lg = m(x)
        L = F.cross_entropy(lg.float().transpose(1, 2), y, reduction="none")
        tot += L.sum().item()
        if ctx >= 512:
            last_full.append(L[:, -256:].mean(1).cpu())
            xs, ys = x[:, ctx - 512:], y[:, ctx - 512:]
            with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                lgs = m(xs)
            last_short.append(F.cross_entropy(lgs.float().transpose(1, 2), ys, reduction="none")[:, -256:].mean(1).cpu())
    nbytes = sum(len(tk.decode(ch[1:].tolist()).encode("utf-8")) for ch in chunks)
    m.train()
    out = {"bpb": tot / math.log(2) / nbytes, "nats_per_token": tot / chunks[:, 1:].size, "chunks": len(chunks)}
    if last_full:
        out["ctx_gain_768_vs_256"] = float(torch.cat(last_short).mean() - torch.cat(last_full).mean())
    return out


@torch.no_grad()
def score_mcq_items(m, c, tk, items, save_items=False):
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    start = tk.encode("\n\n").ids
    kinds = sorted({r["kind"] for r in items})

    def score_one(ctx_s, cont):
        a = start + tk.encode(ctx_s).ids if ctx_s else list(start)
        b = tk.encode(cont).ids
        k = len(b)
        ids = (a + b)[-(c["ctx"] + 1):]
        x = torch.tensor([ids[:-1]], device=m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            lg = m(x)[0].float()
        l = torch.log_softmax(lg[-k:], -1).gather(1, torch.tensor(ids[-k:], device=m_dev)[:, None])[:, 0]
        return float(l.mean())

    def credit(scores):
        b = max(scores)
        tied = [i for i, x in enumerate(scores) if x == b]
        return (1.0 / len(tied)) if 0 in tied else 0.0

    res = {}
    all_w, all_wo, per_item = [], [], []
    for kind in kinds:
        sub = [r for r in items if r["kind"] == kind]
        w = [credit([score_one(f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:", " " + o) for o in [r["answer"]] + r["wrong"]]) for r in sub]
        wo = [credit([score_one(f"Question: {r['question']}\nAnswer:", " " + o) for o in [r["answer"]] + r["wrong"]]) for r in sub]
        all_w.extend(w)
        all_wo.extend(wo)
        if save_items:
            for r, wi, woi in zip(sub, w, wo):
                per_item.append([r["kind"], r.get("slot", -1), wi, woi])
        res[kind] = {"n": len(sub), "with": float(np.mean(w)), "without": float(np.mean(wo)), "gain": float(np.mean(w) - np.mean(wo))}
    res["overall"] = {"n": len(all_w), "with": float(np.mean(all_w)), "without": float(np.mean(all_wo)), "gain": float(np.mean(all_w) - np.mean(all_wo))}
    if save_items:
        res["items"] = per_item
    m.train()
    return res


@torch.no_grad()
def score_retention(m, c, tk, proc_items, mcq2_items, blimp_items, n_sub=100):
    """Context-cleared fast-weight retention & catastrophic-forgetting evaluation (user's test, amendment 226a):
      1. oneshot_correct: score cold -> adapt fast weights on 'Question: Q\\nAnswer: A\\n' -> WIPE CONTEXT -> score cold again!
      2. passage_to_weight: adapt fast weights on 'Passage: P\\n' -> WIPE PASSAGE -> score 'Question: Q\\nAnswer:' cold!
      3. seq10_forgetting: 5 batches of 10 sequential corrections (context wiped between each) -> check Q1..5 vs Q6..10 recall + BLiMP collateral forgetting."""
    m.eval()
    m_dev = next(m.parameters()).device
    use_bf16 = (AMP == "bf16" and m_dev.type == "cuda")
    start = tk.encode("\n\n").ids

    def lp_cold(ctx_s, cont, state):
        a = start + tk.encode(ctx_s).ids if ctx_s else list(start)
        b = tk.encode(cont).ids
        k = len(b)
        ids = (a + b)[-(c["ctx"] + 1):]
        x = torch.tensor([ids[:-1]], device=m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            lg, _ = m.forward_with_state(x, state=state, update=False)
        l = torch.log_softmax(lg[0, -k:].float(), -1).gather(1, torch.tensor(ids[-k:], device=m_dev)[:, None])[:, 0]
        return float(l.sum()), float(l.mean())

    def credit(scores):
        b = max(scores)
        tied = [i for i, x in enumerate(scores) if x == b]
        return (1.0 / len(tied)) if 0 in tied else 0.0

    def q_acc(r, state):
        opts = [r["answer"]] + r["wrong"]
        return credit([lp_cold(f"Question: {r['question']}\nAnswer:", " " + o, state)[1] for o in opts])

    def adapt_text(txt, state):
        ids = (start + tk.encode(txt).ids)[-c["ctx"]:]
        x = torch.tensor([ids], device=m_dev)
        with torch.autocast(m_dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            return m.adapt(x, state=state)

    base_st = m.init_state(1)
    rng = np.random.default_rng(2260)
    p_sub = [proc_items[i] for i in rng.choice(len(proc_items), min(n_sub, len(proc_items)), replace=False)]
    v_sub = [mcq2_items[i] for i in rng.choice(len(mcq2_items), min(n_sub, len(mcq2_items)), replace=False)]
    b_sub = [blimp_items[i] for i in rng.choice(len(blimp_items), min(200, len(blimp_items)), replace=False)] if blimp_items else []

    out = {}
    for lbl, pool in (("proc", p_sub), ("mcq2", v_sub)):
        bef, aft_corr, aft_pass = [], [], []
        w2r, r2w, n_wrong, n_right = 0, 0, 0, 0
        for r in pool:
            b0 = q_acc(r, base_st)
            bef.append(b0)
            # 1. One-shot correction -> wipe context -> ask cold
            st_c = adapt_text(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n", base_st)
            a1 = q_acc(r, st_c)
            aft_corr.append(a1)
            if b0 < 0.5:
                n_wrong += 1
                if a1 > 0.5:
                    w2r += 1
            else:
                n_right += 1
                if a1 < 0.5:
                    r2w += 1
            # 2. Passage-to-weight -> wipe passage -> ask cold
            st_p = adapt_text(f"Passage: {r['passage']}\n\n", base_st)
            aft_pass.append(q_acc(r, st_p))
        out[lbl] = {
            "n": len(pool),
            "cold_before": round(float(np.mean(bef)), 4),
            "oneshot_correct_after": round(float(np.mean(aft_corr)), 4),
            "oneshot_gain": round(float(np.mean(aft_corr) - np.mean(bef)), 4),
            "wrong_to_right": round(w2r / max(1, n_wrong), 4),
            "right_to_wrong": round(r2w / max(1, n_right), 4),
            "passage_to_weight_after": round(float(np.mean(aft_pass)), 4),
            "passage_to_weight_gain": round(float(np.mean(aft_pass) - np.mean(bef)), 4),
        }

    # 3. 10-correction sequential interference & BLiMP collateral forgetting stress test (5 batches x 10 items)
    blimp_base = float(np.mean([lp_cold("", r["good"], base_st)[0] > lp_cold("", r["bad"], base_st)[0] for r in b_sub])) if b_sub else None
    early_b, early_a, late_b, late_a, blimp_afters = [], [], [], [], []
    for bi in range(5):
        batch = v_sub[bi * 10:(bi + 1) * 10]
        if len(batch) < 10:
            break
        b_scores = [q_acc(r, base_st) for r in batch]
        st = base_st
        for r in batch:
            st = adapt_text(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n", st)
        a_scores = [q_acc(r, st) for r in batch]
        early_b.extend(b_scores[:5]); early_a.extend(a_scores[:5])
        late_b.extend(b_scores[5:]);  late_a.extend(a_scores[5:])
        if b_sub:
            blimp_afters.append(float(np.mean([lp_cold("", r["good"], st)[0] > lp_cold("", r["bad"], st)[0] for r in b_sub])))
    out["seq10_forgetting"] = {
        "early_q1_5_before": None if not early_b else round(float(np.mean(early_b)), 4),
        "early_q1_5_after10": None if not early_a else round(float(np.mean(early_a)), 4),
        "late_q6_10_before": None if not late_b else round(float(np.mean(late_b)), 4),
        "late_q6_10_after10": None if not late_a else round(float(np.mean(late_a)), 4),
        "blimp_before": None if blimp_base is None else round(blimp_base, 4),
        "blimp_after10": None if not blimp_afters else round(float(np.mean(blimp_afters)), 4),
        "blimp_drop": None if (blimp_base is None or not blimp_afters) else round(float(np.mean(blimp_afters) - blimp_base), 4),
    }
    m.train()
    return out


# ---------------------------------------------------------------- training & scoring

def lr_at(lr, s, steps):
    w = max(1, steps // 50)
    d0 = int(steps * 0.8)
    return lr * ((s + 1) / w if s < w else 1.0 if s < d0 else max(0.0, (steps - s) / (steps - d0)))


def train(name, arm, seed, tokens=100_000_000):
    assert arm in ARM_SPEC, f"unknown arm {arm}"
    assert not name.startswith(("C", "B", "OLD", "M", "T", "S", "D", "X")), "would collide with earlier-addendum checkpoints"
    assert not (RUNS / f"{name}.json").exists(), f"run already exists: {name}.json"

    c = dict(BASE_CFG, **ARM_SPEC[arm], arm=arm, seed=seed, tokens=float(tokens))
    torch.manual_seed(seed)
    # Exact same data_rng seed as exp225.py so O1-O4 see the EXACT SAME batches as X0_s1..s3!
    data_rng = np.random.default_rng(seed + 22500)

    tk = tokenizer()
    V = tk.get_vocab_size()
    tr, held, jung, proc_tr, chat_tr, chat_he = streams()
    bad = bad_runs(tr)
    with open(MID / "proc_held.jsonl") as f:
        proc_held = [json.loads(l) for l in f]

    m = OnlineToy(c, V).to(DEV)
    B, ctx, mb = c["batch"], c["ctx"], c["mb"]
    assert B % 4 == 0
    steps = max(1, int(tokens // (B * ctx)))
    marks = {max(1, steps // 4): "25M", max(2, steps // 2): "50M"}

    dec = [p for p in m.parameters() if p.dim() >= 2]
    nod = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": c["wd"]}, {"params": nod, "weight_decay": 0.0}], c["lr"], betas=(0.9, 0.95))

    print(f"{name}: arm={arm} mode={c['mode']} seed={seed} nonembed={nonembed(m)} steps={steps} dev={DEV} amp={AMP}", flush=True)
    t0 = time.time()
    curve = {}

    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(c["lr"], s, steps)

        i_g = draw_clean(data_rng, B // 2, len(tr) - ctx - 1, ctx, bad)
        i_p = data_rng.integers(0, len(proc_tr) - ctx - 1, B // 4)
        i_c = data_rng.integers(0, len(chat_tr) - ctx - 1, B // 4)
        xb = np.concatenate([
            np.stack([tr[j:j + ctx + 1] for j in i_g]),
            np.stack([proc_tr[j:j + ctx + 1] for j in i_p]),
            np.stack([chat_tr[j:j + ctx + 1] for j in i_c]),
        ], 0).astype(np.int64)
        data_rng.shuffle(xb, axis=0)

        opt.zero_grad(set_to_none=True)
        tot_ntp = 0.0
        for i in range(0, B, mb):
            x = torch.from_numpy(xb[i:i + mb, :-1]).to(DEV)
            y = torch.from_numpy(xb[i:i + mb, 1:]).to(DEV)
            with amp():
                lg = m(x)
            lg_f = lg.float()
            l_ntp = F.cross_entropy(lg_f.reshape(-1, V), y.reshape(-1))
            l = l_ntp
            if c["zloss"]:
                l = l + c["zloss"] * torch.logsumexp(lg_f, -1).pow(2).mean()
            (l * (x.shape[0] / B)).backward()
            tot_ntp += l_ntp.item() * (x.shape[0] / B)

        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        opt.step()
        assert math.isfinite(tot_ntp), "diverged"

        if s % 100 == 0 or s == steps - 1:
            print(f"step {s}/{steps} ntp {tot_ntp:.4f} gnorm {gn:.2f} lr {opt.param_groups[0]['lr']:.2e} s/step {(time.time() - t0) / (s + 1):.3f}", flush=True)
        if (s + 1) in marks:
            lbl = marks[s + 1]
            pr = score_mcq_items(m, c, tk, proc_held)
            curve[lbl] = pr
            print(f"CHECKPOINT {name} {lbl} proc_held overall {pr['overall']}", flush=True)

    train_s = round(time.time() - t0)
    ev_books = eval_bpb(m, c, tk, [held], nb=2_000_000)
    ev_jung = eval_bpb(m, c, tk, jung)
    ev_chat = eval_bpb(m, c, tk, [chat_he])
    pr_end = score_mcq_items(m, c, tk, proc_held)
    curve["100M"] = pr_end
    torch.save(m.state_dict(), PRIV / "ckpt" / f"{name}.pt")
    info = {
        "name": name, "config": c, "dev": DEV, "amp": AMP, "steps": steps, "nonembed": nonembed(m),
        "train_s": train_s, "s_per_step": round(train_s / steps, 4), "last_ntp": tot_ntp,
        "books": ev_books, "jung": ev_jung, "chat_held": ev_chat, "proc_curve": curve,
    }
    with open(RUNS / f"{name}.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"EXP226_DONE {name} books={ev_books['bpb']:.4f} jung={ev_jung['bpb']:.4f} chat={ev_chat['bpb']:.4f} proc_100M={pr_end['overall']['gain']:+.3f} ({train_s} s)", flush=True)


def score(names):
    tk = tokenizer()
    V = tk.get_vocab_size()
    tests = {}
    for k, path in (
        ("blimp", COMP / "blimp.jsonl"),
        ("copa", COMP / "copa.jsonl"),
        ("lastword", COMP / "lastword.jsonl"),
        ("mcq", COMP / "mcq.jsonl"),
        ("mcq2", COMP / "mcq2.jsonl"),
        ("proc", MID / "proc_held.jsonl"),
    ):
        if path.exists():
            with open(path) as f:
                tests[k] = [json.loads(l) for l in f]

    for name in names:
        with open(RUNS / f"{name}.json") as f:
            c = json.load(f)["config"]
        m = OnlineToy(c, V).to(DEV)
        m.load_state_dict(torch.load(PRIV / "ckpt" / f"{name}.pt", map_location=DEV, weights_only=True))
        m.eval()
        start = tk.encode("\n\n").ids
        t0 = time.time()

        @torch.no_grad()
        def lp(ctx_s, cont):
            a = start + tk.encode(ctx_s).ids if ctx_s else list(start)
            b = tk.encode(cont).ids
            ids = (a + b)[-(c["ctx"] + 1):]
            k = len(b)
            x = torch.tensor([ids[:-1]], device=DEV)
            with amp():
                lg = m(x)[0].float()
            l = torch.log_softmax(lg[-k:], -1).gather(1, torch.tensor(ids[-k:], device=DEV)[:, None])[:, 0]
            return float(l.sum()), float(l.mean())

        def credit(scores):
            b = max(scores)
            tied = [i for i, x in enumerate(scores) if x == b]
            return (1.0 / len(tied)) if 0 in tied else 0.0

        @torch.no_grad()
        def top1(ctx_s):
            ids = (start + tk.encode(ctx_s).ids)[-c["ctx"]:]
            with amp():
                return int(m(torch.tensor([ids], device=DEV))[0, -1].argmax())

        res = {}
        if "blimp" in tests:
            ok = [lp("", r["good"])[0] > lp("", r["bad"])[0] for r in tests["blimp"]]
            res["blimp"] = float(np.mean(ok))
            by = defaultdict(list)
            for r, o in zip(tests["blimp"], ok):
                by[r["set"]].append(o)
            res["blimp_sets_below_chance"] = int(sum(np.mean(v) < 0.5 for v in by.values()))
        if "copa" in tests:
            ok = []
            for r in tests["copa"]:
                link = " because" if r["question"] == "cause" else " so"
                pre = r["premise"].rstrip(".") + link
                sc = [lp(pre, " " + ch[0].lower() + ch[1:])[1] for ch in r["choices"]]
                ok.append(0.5 if sc[0] == sc[1] else float(int(np.argmax(sc)) == r["label"]))
            res["copa"] = float(np.mean(ok))
        if "lastword" in tests:
            res["lastword_full"] = float(np.mean([top1(r["context"]) == tk.encode(" " + r["word"]).ids[0] for r in tests["lastword"]]))
            res["lastword_last_sentence"] = float(np.mean([top1(r["last_sentence"]) == tk.encode(" " + r["word"]).ids[0] for r in tests["lastword"]]))
        if "mcq" in tests:
            w, wo = [], []
            for r in tests["mcq"]:
                opts = [r["answer"]] + r["wrong"]
                w.append(credit([lp(f"Passage: {r['passage']}\n\nQuestion: {r['question']}\nAnswer:", " " + o)[1] for o in opts]))
                wo.append(credit([lp(f"Question: {r['question']}\nAnswer:", " " + o)[1] for o in opts]))
            res["mcq_with_passage"] = float(np.mean(w))
            res["mcq_without"] = float(np.mean(wo))
            res["mcq_gain"] = res["mcq_with_passage"] - res["mcq_without"]
            res["mcq_items"] = [[r["pid"], a, b] for r, a, b in zip(tests["mcq"], w, wo)]
        if "mcq2" in tests:
            res["v2"] = score_mcq_items(m, c, tk, tests["mcq2"], save_items=True)
        if "proc" in tests:
            res["proc"] = score_mcq_items(m, c, tk, tests["proc"], save_items=True)
        if "proc" in tests and "mcq2" in tests:
            res["retention"] = score_retention(m, c, tk, tests["proc"], tests["mcq2"], tests.get("blimp", []))
        res["s"] = round(time.time() - t0)
        with open(RUNS / "comp" / f"{name}.json", "w") as f:
            json.dump(res, f, indent=1)
        pg = res.get("proc", {}).get("overall", {}).get("gain", float("nan"))
        vg = res.get("v2", {}).get("overall", {}).get("gain", float("nan"))
        rg = res.get("retention", {}).get("mcq2", {}).get("oneshot_gain", float("nan"))
        print(f"SCORE_DONE {name}: blimp={res.get('blimp', 0):.3f} proc_gain={pg:+.3f} v2_gain={vg:+.3f} retain_1shot={rg:+.3f} ({res['s']} s)", flush=True)


# ---------------------------------------------------------------- rigorous unit tests

def self_test():
    """Verify: (1) analytical SwiGLU grad == autograd to <1e-5 under bf16 autocast; (2) NS3 scale + zero-input finite backward;
    (3) param parity <0.5% vs X0 (4,722,304); (4) strict causality on ALL 4 arms in eval AND train mode (train==eval match);
    (5) edge lengths T in (1, 64, 65) + context-cleared adapt()/forward_with_state(); (6) CUDA bf16 forward+backward smoke test when GPU available."""
    torch.manual_seed(0)
    dev = "cpu"

    # 1. Exact analytical SwiGLU grad vs autograd (under bf16 autocast!)
    B, C, d, h = 2, 16, 32, 64
    K = F.normalize(torch.randn(B, C, d), dim=-1) * (d ** 0.5)
    V = torch.randn(B, C, d)
    W1 = (torch.randn(B, h, d) * (d ** -0.5)).detach().requires_grad_(True)
    W2 = (torch.randn(B, d, h) * (h ** -0.5)).detach().requires_grad_(True)
    W3 = (torch.randn(B, h, d) * (d ** -0.5)).detach().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        g1, g2, g3 = _swiglu_fwd_bwd(K, V, W1.detach(), W2.detach(), W3.detach())
    assert g1.dtype == torch.float32 and g2.dtype == torch.float32 and g3.dtype == torch.float32
    V_hat = torch.matmul(F.silu(torch.matmul(K, W1.transpose(-1, -2))) * torch.matmul(K, W3.transpose(-1, -2)), W2.transpose(-1, -2))
    loss = 0.5 * (V_hat - V).pow(2).sum() / float(C)
    a1, a2, a3 = torch.autograd.grad(loss, [W1, W2, W3])
    e1, e2, e3 = (g1 - a1).abs().max().item(), (g2 - a2).abs().max().item(), (g3 - a3).abs().max().item()
    r1 = e1 / max(1e-8, a1.abs().max().item())
    r2 = e2 / max(1e-8, a2.abs().max().item())
    r3 = e3 / max(1e-8, a3.abs().max().item())
    print(f"SwiGLU analytic vs autograd (bf16 autocast): abs=({e1:.2e}, {e2:.2e}, {e3:.2e}) rel=({r1:.2e}, {r2:.2e}, {r3:.2e})", flush=True)
    assert max(e1, e2, e3) < 1e-5 and max(r1, r2, r3) < 1e-5

    # 2. Newton-Schulz scale + zero-input finite backward check
    G = torch.randn(2, 256, 128)
    U = _newton_schulz3(G)
    rms_u = U.pow(2).mean().sqrt().item()
    expected_rms = 256 ** -0.5  # 0.0625
    assert U.shape == G.shape and abs(rms_u - expected_rms) / expected_rms < 0.25, f"NS3 RMS {rms_u:.4f} vs {expected_rms:.4f}"
    Gz = torch.zeros(2, 64, 32, requires_grad=True)
    _newton_schulz3(Gz).pow(2).sum().backward()
    assert torch.isfinite(Gz.grad).all(), "NS3 zero-input NaN grad!"
    print(f"NS3 check: rms={rms_u:.4f} (target ~{expected_rms:.4f}), zero-grad finite=OK", flush=True)

    # 3. Param parity, causality (eval + train), adapt/retention, edge lengths, and CUDA bf16 smoke on ALL 4 arms
    X0_NONEMBED = 4_722_304
    V_size = 8192
    for arm in ("O1", "O2", "O3", "O4"):
        c = dict(BASE_CFG, **ARM_SPEC[arm], arm=arm, ctx=256)
        torch.manual_seed(11)
        m = OnlineToy(c, V_size).to(dev)
        ne = nonembed(m)
        rel = abs(ne - X0_NONEMBED) / X0_NONEMBED
        assert rel < 0.005, f"{arm} param mismatch: {ne} vs {X0_NONEMBED} ({rel:.2%})"
        for pn, p in m.named_parameters():
            if any(k in pn for k in ("eta_logit", "gamma_logit", "scale")):
                assert p.dim() == 1, f"{pn} should be 1D to avoid weight decay"

        m.eval()
        with torch.no_grad():
            for _, p in m.named_parameters():
                if p.dim() >= 2 and p.abs().max() == 0:
                    nn.init.normal_(p, 0, 0.03)

        x = torch.randint(0, V_size, (2, 195), device=dev)
        with torch.no_grad():
            base_lg = m(x)
            for t in (0, 63, 64, 127, 128, 190):
                x2 = x.clone()
                x2[:, t + 1:] = torch.randint(0, V_size, x2[:, t + 1:].shape, device=dev)
                lg2 = m(x2)
                d_lg = (lg2[:, :t + 1] - base_lg[:, :t + 1]).abs().max().item()
                assert d_lg < 1e-4, f"{arm} LEAK at t={t}: {d_lg:.2e}"

            # Edge lengths T in (1, 64, 65)
            for tlen in (1, 64, 65):
                assert torch.isfinite(m(x[:, :tlen])).all(), f"{arm} failed on T={tlen}"

            # Context-cleared adapt() -> cold forward_with_state() check
            st0 = m.init_state(2)
            lg_cold0, _ = m.forward_with_state(x[:, :20], state=st0, update=False)
            st1 = m.adapt(x[:, 20:50], state=st0)
            lg_cold1, _ = m.forward_with_state(x[:, :20], state=st1, update=False)
            adapt_diff = (lg_cold1 - lg_cold0).abs().max().item()
            assert adapt_diff > 1e-5, f"{arm} context-cleared adapt had no effect: {adapt_diff:.2e}"

        # Train-mode forward matches eval-mode forward & produces finite grads on ALL params
        m.train()
        lg_tr = m(x)
        te_diff = (lg_tr.detach() - base_lg).abs().max().item()
        assert te_diff < 1e-4, f"{arm} train vs eval mismatch: {te_diff:.2e}"
        loss = F.cross_entropy(lg_tr[:, :-1].reshape(-1, V_size), x[:, 1:].reshape(-1))
        loss.backward()
        missing = [pn for pn, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not missing, f"{arm} missing/non-finite grads: {missing[:5]}"

        # CUDA + bf16 forward+backward smoke test (catches Flash/cuDNN double-backward bugs immediately on 4090!)
        cuda_ok = "skip"
        if torch.cuda.is_available():
            mc = OnlineToy(dict(BASE_CFG, **ARM_SPEC[arm], arm=arm, ctx=256), V_size).to("cuda").train()
            xc = torch.randint(0, V_size, (2, 195), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lgc = mc(xc[:, :-1])
                lc = F.cross_entropy(lgc.float().reshape(-1, V_size), xc[:, 1:].reshape(-1))
            lc.backward()
            assert torch.isfinite(lc) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in mc.parameters()), f"{arm} CUDA bf16 grad fail"
            cuda_ok = "OK"

        print(f"  {arm}: nonembed={ne} ({rel:+.2%}) leak=OK adapt_diff={adapt_diff:.2e} tr_ev={te_diff:.1e} meta_grad=OK cuda_bf16={cuda_ok}", flush=True)

    print("EXP226_TEST_OK", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        self_test()
    elif cmd == "train":
        kw = dict(kv.split("=") for kv in sys.argv[3:])
        train(sys.argv[2], kw["arm"], int(kw.get("seed", 1)), tokens=float(kw.get("tokens", 100e6)))
    elif cmd == "score":
        score(sys.argv[2:])
    else:
        raise ValueError(f"unknown cmd {cmd}")
