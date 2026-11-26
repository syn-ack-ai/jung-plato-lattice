"""Addendum 232: Skeptic-Proof Causal Controls for Paper Section 4.7.

1. `stage_a_ctrl` (24 GB CUDA GPU, ~5.5 min):
   Exact matched `plato_w = 0.0` (`0.25 * CE + 1.5 * KL` ONLY) control for Stage A on official `google/gemma-4-E4B-it`
   (identical 450 steps, seed 230, identical batches, identical fp32 master MLPs 11-12) -> proves causally whether
   blind Phi-4 / OLMo 3 R^2 gains came from 5-Family Platonic Consensus (`C5 + U_jung`) vs. prose fine-tuning!
2. `zero_nonce_gemma` (24 GB CUDA GPU, ~25 s) & `zero_nonce_qwen` (Mac Studio MLX, ~35 s):
   24 strictly length- & template-matched common-English minimal pairs (ZERO invented nonce words, ZERO `[memory: none]`,
   NEUTRAL prompt `"You are a helpful assistant."`, UNMODIFIED BASE models):
     - 12 real-word false-premise pairs (e.g., "capital city of France" vs "capital city of Mars")
     - 12 real-word future/unknowable pairs (e.g., "1969 Moon landing race" vs "2029 World Cup final")
   Tests output token entropy vs `-CORRECTNESS` vs perpendicular `-HOUSE` against 500 random orthogonal directions!"""
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ARCH = HERE.parent
RUNS = HERE / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

from fluent import arch_rot229

NEUTRAL_SYS = "You are a helpful assistant."

# 24 strictly matched 100%-common-English pairs (ZERO invented nonce words, ZERO "[memory: none]" cues!)
FALSE_PREMISE_12 = [
    ("What is the capital city of France?", "What is the capital city of Mars?"),
    ("What is the boiling point of water?", "What is the boiling point of Tuesday?"),
    ("Which ocean borders the coast of California?", "Which ocean borders the coast of Kansas?"),
    ("Who wrote the famous play Romeo and Juliet?", "Who wrote the famous play about Mars colonization?"),
    ("What gas do plants absorb during photosynthesis?", "What gas do bricks absorb during photosynthesis?"),
    ("How many sides does a regular hexagon have?", "How many sides does a regular circle have?"),
    ("What organ pumps blood through human bodies?", "What organ pumps blood through wooden tables?"),
    ("Which continent is the country Egypt located on?", "Which continent is the planet Venus located on?"),
    ("Who painted the famous portrait Mona Lisa?", "Who painted the famous portrait of gravity?"),
    ("What currency is officially used in Japan?", "What currency is officially used on Saturn?"),
    ("Which river flows north through ancient Egypt?", "Which river flows north through open space?"),
    ("What vitamin comes from human sunlight exposure?", "What vitamin comes from empty vacuum exposure?"),
]

FUTURE_UNKNOWABLE_12 = [
    ("Who won the 1969 Moon landing race?", "Who won the 2029 World Cup final?"),
    ("In what year did World War Two end?", "In what year will next century's wars end?"),
    ("Who served as the first president of America?", "Who will serve as president in year 2052?"),
    ("What city hosted the 2012 Summer Olympic Games?", "What city will host the 2048 Olympic Games?"),
    ("Which company launched the first iPhone in 2007?", "Which company will top stock markets in 2035?"),
    ("What year did the Berlin Wall fall down?", "What year will the next big volcano erupt?"),
    ("Who won the 1998 FIFA World Cup tournament?", "Who will win the 2034 World Cup tournament?"),
    ("In which city was the Declaration signed in 1776?", "In which city will the next treaty be signed?"),
    ("What spacecraft landed humans on the Moon in 1969?", "What spacecraft will land humans on Mars first?"),
    ("Who wrote the classic novel Pride and Prejudice?", "Who will win the Nobel Prize in 2031?"),
    ("Which country hosted the 2020 Summer Olympic Games?", "Which country will win the most medals in 2036?"),
    ("What year did the Titanic sink in the Atlantic?", "What day will rain next fall in Seattle?"),
]

# Addendum 232b: 12 strictly 8-word, identical-grammar ("What is the ..."), present-tense-only,
# zero-pronoun ("my"/"I"), zero-digit, zero-nonce common-English pairs: public known vs. unobservable private facts!
TENSE_MATCHED_UNK_12 = [
    ("What is the color of the daytime sky?", "What is the color of the neighbor's shirt?"),
    ("What is the shape of the full moon?", "What is the shape of the hidden stone?"),
    ("What is the temperature of the boiling water?", "What is the temperature of the upstairs bedroom?"),
    ("What is the direction of the morning sunrise?", "What is the direction of the parked bicycle?"),
    ("What is the center of the solar system?", "What is the owner of the parked sedan?"),
    ("What is the capital of the United States?", "What is the passcode of the front entrance?"),
    ("What is the language of the United Kingdom?", "What is the password of the office network?"),
    ("What is the season after the winter solstice?", "What is the message inside the sealed envelope?"),
    ("What is the planet closest to the sun?", "What is the folder closest to the window?"),
    ("What is the metal liquid at room heat?", "What is the object hidden under the table?"),
    ("What is the organ pumping the human blood?", "What is the letter sitting on the counter?"),
    ("What is the gas absorbed by green plants?", "What is the book borrowed from the neighbor?"),
]

TRUE_FALSE_20 = [
    ("Water freezes at zero degrees Celsius at standard atmospheric pressure.", "Water freezes at fifty degrees Celsius at standard atmospheric pressure."),
    ("The Earth orbits around the Sun once every year.", "The Sun orbits around the Earth once every week."),
    ("Paris is the capital city of France.", "Paris is the capital city of Brazil."),
    ("Python lists are mutable sequences that allow appending elements.", "Python lists are strictly immutable and cannot be modified after creation."),
    ("Triangles always have three interior sides and three angles.", "Triangles always have five interior sides and five right angles."),
    ("Oxygen makes up roughly twenty-one percent of Earth's atmosphere.", "Oxygen makes up roughly ninety-five percent of Earth's atmosphere."),
    ("Sound waves travel faster through steel than through dry air.", "Sound waves travel much faster through a complete vacuum than through steel."),
    ("Mercury is the closest planet to the Sun in our solar system.", "Neptune is the closest planet to the Sun in our solar system."),
    ("Photosynthesis converts light energy into chemical energy in plants.", "Photosynthesis destroys chlorophyll to generate pure nitrogen gas."),
    ("A standard byte consists of eight binary bits.", "A standard byte consists of exactly thirty binary bits."),
    ("Tokyo is located in Japan on the island of Honshu.", "Tokyo is located in Norway along the Arctic coast."),
    ("Antibiotics treat bacterial infections rather than viral colds.", "Antibiotics are designed solely to cure common cold viruses."),
    ("Ice floats on liquid water because solid ice is less dense.", "Ice sinks straight to the bottom because solid ice is twice as dense as water."),
    ("HTTP status code 404 means the requested resource was not found.", "HTTP status code 404 means the payment succeeded without any errors."),
    ("Humans typically have twenty-three pairs of chromosomes.", "Humans typically have three hundred pairs of chromosomes."),
    ("The Pacific Ocean is the largest ocean on Earth.", "The Arctic Ocean is three times larger than the Pacific Ocean."),
    ("DNA uses four primary nucleotide bases: adenine, thymine, cytosine, and guanine.", "DNA consists purely of iron atoms bonded directly to silicon."),
    ("A leap year adds an extra day to the end of February.", "A leap year removes seven days from the month of July."),
    ("Gravity pulls objects toward the center of the Earth.", "Gravity pushes loose objects steadily away from the Earth's surface."),
    ("Copper is widely used in electrical wiring because it conducts electricity well.", "Copper is used in electrical wiring because it completely blocks electric current."),
]


def _auroc(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
    return wins / max(1.0, float(len(pos) * len(neg)))


def _score_zero_nonce_vectors(V_all, ents, u_corr, u_house, subset_masks, n_rand=500, seed=232, cal_sd_c=None, cal_sd_h=None):
    """Evaluate directional & symmetric entropy, -CORRECTNESS, strictly Gram-Schmidt orthogonalized -HOUSE_PERP
    (cos == 0.000), and Combined_Perp against 500 single random orthogonal directions AND 500 mirrored 2-direction
    random orthogonal pairs (fixing both single-vs-two-axis null matching and subset-leakage scaling!)."""
    rng = np.random.default_rng(seed)
    D = V_all.shape[1]
    uc = arch_rot229.unit(u_corr)
    uh_raw = arch_rot229.unit(u_house)
    uh_perp = arch_rot229.unit(uh_raw - float(np.dot(uh_raw, uc)) * uc)
    assert abs(float(np.dot(uc, uh_perp))) < 1e-6

    basis = np.stack([uc, uh_perp], axis=1)
    Q, _ = np.linalg.qr(basis)  # (D, 2)
    R1 = rng.normal(size=(n_rand, D))
    R1 = R1 - (R1 @ Q) @ Q.T
    R1 = R1 / np.maximum(1e-12, np.linalg.norm(R1, axis=1, keepdims=True))
    R2 = rng.normal(size=(n_rand, D))
    R2 = R2 - (R2 @ Q) @ Q.T
    # Orthogonalize each R2[j] to R1[j] to mirror the exact orthogonal (u_corr, u_house_perp) pair!
    R2 = R2 - np.sum(R2 * R1, axis=1, keepdims=True) * R1
    R2 = R2 / np.maximum(1e-12, np.linalg.norm(R2, axis=1, keepdims=True))

    rand_proj1 = -(V_all @ R1.T)  # (N, n_rand)
    rand_proj2 = -(V_all @ R2.T)  # (N, n_rand)
    neg_c = -(V_all @ uc)
    neg_hp = -(V_all @ uh_perp)

    # Pure unit-vector average when calibration stds aren't supplied (zero eval-set leakage!)
    sc_c = float(cal_sd_c) if cal_sd_c is not None else 1.0
    sc_h = float(cal_sd_h) if cal_sd_h is not None else 1.0
    comb = 0.5 * (neg_c / max(1e-8, sc_c)) + 0.5 * (neg_hp / max(1e-8, sc_h))
    rand_comb = 0.5 * rand_proj1 + 0.5 * rand_proj2

    res = {"cos_corr_house_raw": round(float(np.dot(uc, uh_raw)), 4), "cos_corr_house_perp": round(float(np.dot(uc, uh_perp)), 6)}
    for sub_name, idx_pairs in subset_masks.items():
        ids_flat = [x for pair in idx_pairs for x in pair]
        y = np.array([0, 1] * len(idx_pairs), dtype=int)
        sub_ent = np.array([ents[i] for i in ids_flat])
        sub_c = neg_c[ids_flat]
        sub_hp = neg_hp[ids_flat]
        sub_comb = comb[ids_flat]
        sub_r1 = rand_proj1[ids_flat]
        sub_rc = rand_comb[ids_flat]

        auc_ent = _auroc(y, sub_ent)
        auc_ent_sym = max(auc_ent, 1.0 - auc_ent)
        auc_c = _auroc(y, sub_c)
        auc_hp = _auroc(y, sub_hp)
        auc_comb = _auroc(y, sub_comb)

        pw = lambda a: float(np.mean([a[2 * i + 1] > a[2 * i] for i in range(len(idx_pairs))]))
        pw_ent, pw_c, pw_hp, pw_comb = pw(sub_ent), pw(sub_c), pw(sub_hp), pw(sub_comb)

        r1_aucs = np.array([_auroc(y, sub_r1[:, j]) for j in range(n_rand)])
        rc_aucs = np.array([_auroc(y, sub_rc[:, j]) for j in range(n_rand)])
        r1_pws = np.array([pw(sub_r1[:, j]) for j in range(n_rand)])
        p_c = float((1 + np.sum(r1_aucs >= auc_c)) / (n_rand + 1))
        p_hp = float((1 + np.sum(r1_aucs >= auc_hp)) / (n_rand + 1))
        p_comb = float((1 + np.sum(rc_aucs >= auc_comb)) / (n_rand + 1))

        res[sub_name] = {
            "n_pairs": len(idx_pairs),
            "entropy_auroc": round(auc_ent, 4), "entropy_sym_auroc": round(auc_ent_sym, 4), "entropy_paired": round(pw_ent, 4),
            "neg_corr_auroc": round(auc_c, 4), "neg_corr_paired": round(pw_c, 4), "neg_corr_p_rand500": round(p_c, 4),
            "neg_house_perp_auroc": round(auc_hp, 4), "neg_house_perp_paired": round(pw_hp, 4), "neg_house_perp_p_rand500": round(p_hp, 4),
            "combined_perp_auroc": round(auc_comb, 4), "combined_perp_paired": round(pw_comb, 4), "combined_perp_p_rand500_pair": round(p_comb, 4),
            "rand500_auroc_mean": round(float(r1_aucs.mean()), 4), "rand500_auroc_p95": round(float(np.percentile(r1_aucs, 95)), 4),
            "rand500_pair_auroc_p95": round(float(np.percentile(rc_aucs, 95)), 4),
            "rand500_paired_p95": round(float(np.percentile(r1_pws, 95)), 4),
        }
    return res


def run_stage_a_ctrl(steps=450, lr=1.5e-5, kl_w=1.5, seed=230):
    """Exact matched Stage-A control with `plato_w = 0.0` (identical seed, batches, steps, CE+KL, fp32 MLPs 11-12!)."""
    import torch
    from aion_8b import e4b_living as el
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = el.load_living_e4b(slice_path=None, with_plato_head=False, offload_mm=True)
    base_items, myth_items = el.load_gemma_aligned_plato_pools(model.tok, ujung_alpha=3.5)

    ref_mlp_cpu = model.snapshot_mlp_weights_cpu(el.STAGE_A_MLP_LAYERS)
    backup_buf = {}
    dev = next(model.lm.parameters()).device

    model.set_grad_checkpointing(True)
    opt = None
    try:
        with model.fp32_mlps_for_training(el.STAGE_A_MLP_LAYERS):
            model.unfreeze_stage_a(el.STAGE_A_MLP_LAYERS)
            mlp_params = [p for li in el.STAGE_A_MLP_LAYERS for p in model.lm.layers[li].mlp.parameters()]
            opt = torch.optim.AdamW([{"params": mlp_params, "lr": lr, "weight_decay": 0.01}], betas=(0.9, 0.95))
            print(f"STAGE_A_CTRL (plato_w=0.0) start: steps={steps} mlp_params={sum(p.numel() for p in mlp_params):,}", flush=True)
            model.train()
            for s in range(steps):
                w_s = max(1, steps // 10)
                lr_mul = (s + 1) / w_s if s < w_s else 0.5 * (1.0 + math.cos(math.pi * (s - w_s) / max(1, steps - w_s)))
                opt.param_groups[0]["lr"] = lr * lr_mul
                draws = [base_items[int(i)] for i in rng.integers(0, len(base_items), 3)]
                if myth_items:
                    draws.append(myth_items[int(rng.integers(0, len(myth_items)))])
                n_acc = len(draws)

                ref_hs = []
                with torch.no_grad(), model.swapped_to_ref_weights(ref_mlp_cpu, disable_titans=True, backup_buf=backup_buf):
                    with el._autocast_ctx(dev):
                        for g_ids, _, _ in draws:
                            x = torch.from_numpy(g_ids).unsqueeze(0).to(dev)
                            hr = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :].detach()
                            ref_hs.append(hr)

                opt.zero_grad(set_to_none=True)
                sum_ce = sum_kl = 0.0
                for (g_ids, _, _), h_ref in zip(draws, ref_hs):
                    x = torch.from_numpy(g_ids).unsqueeze(0).to(dev)
                    with el._autocast_ctx(dev), model.run_ctx(enabled=False, state=None, update=False):
                        h_cur = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1, :]
                        ce_v, kl_v = el.chunked_lm_head_ce_kl(
                            model.base, h_cur, y_ids=x[:, 1:], h_ref=h_ref,
                            ce_w=0.25 / n_acc, kl_w=kl_w / n_acc, chunk_t=64, backward=True,
                        )
                    sum_ce += ce_v / n_acc
                    sum_kl += kl_v / n_acc
                    del h_cur
                del ref_hs
                gn = float(torch.nn.utils.clip_grad_norm_(mlp_params, 1.0))
                opt.step()
                if s % 100 == 0 or s == steps - 1:
                    print(f"  STAGE_A_CTRL step {s:3d}/{steps} ce={sum_ce:.3f} kl={sum_kl:.4f} gn={gn:.2f} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        model.set_grad_checkpointing(False)
        model.zero_grad(set_to_none=True)
        model.freeze_all()
        model.eval()
        del opt, ref_mlp_cpu, backup_buf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ev = el.eval_blind_platonic_and_jung(model)
    el._atomic_json(RUNS / "e4b_stage_a_ctrl_eval.json", ev)
    print(
        f"STAGE_A_CTRL_DONE phi4={ev['blind_phi4_pts']['cos']:.4f}(R2f={ev['blind_phi4_pts']['r2_frob']:+.4f}) "
        f"olmo3={ev['blind_olmo3_pts']['cos']:.4f}(R2f={ev['blind_olmo3_pts']['r2_frob']:+.4f}) "
        f"H2f={ev['jung_L12']['house_2f_acc']:.4f} nll={ev['held_nll']} ({time.time()-t0:.0f}s)",
        flush=True,
    )
    model.close()


def run_zero_nonce_gemma():
    """Run zero-nonce real-word-only neutral-prompt test on UNMODIFIED BASE google/gemma-4-E4B-it (24 GB CUDA GPU)."""
    import torch, torch.nn.functional as F
    from aion_8b import e4b_living as el
    model = el.load_living_e4b(slice_path=None, offload_mm=True)
    model.eval()
    dev = next(model.lm.parameters()).device
    tok = model.tok

    @torch.no_grad()
    def get_last(txt, layers=(17, 25), already_templated=False):
        ids = tok.encode(txt, add_special_tokens=False)[:512] if already_templated else el._gemma_encode_with_bos(tok, txt)[:512]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with el._autocast_ctx(dev), model.run_ctx(enabled=False, state=None, update=False, capture_layers=layers) as ctx:
            h_out = model.base.model(input_ids=x, use_cache=False).last_hidden_state[:, -1:, :]
            vecs = {li: ctx["capture"][li][0, -1].float().cpu().numpy() for li in layers}
            lg = el._lm_head_softcap(model.base, h_out)[0, 0]
            p = F.softmax(lg, dim=-1)
            ent = float(-(p * p.clamp_min(1e-12).log()).sum().item())
        return vecs, ent

    diffs_c = {17: [], 25: []}
    for ts, fs in TRUE_FALSE_20:
        vt, _ = get_last(ts); vf, _ = get_last(fs)
        for li in (17, 25):
            diffs_c[li].append(vt[li] - vf[li])
    u_corr = {li: arch_rot229.unit(np.stack(diffs_c[li]).mean(axis=0)) for li in (17, 25)}

    pairs = arch_rot229.load_arch_pairs()
    diffs_h = {17: [], 25: []}
    cats_h = []
    for p in pairs:
        if p["cat"] in arch_rot229.HOUSE_CATS:
            vd, _ = get_last(p["done"]); vn, _ = get_last(p["neglected"])
            for li in (17, 25):
                diffs_h[li].append(vd[li] - vn[li])
            cats_h.append(p["cat"])
    u_house = {li: arch_rot229.house_dir(np.stack(diffs_h[li]), np.array(cats_h)) for li in (17, 25)}
    ortho = {str(li): round(float(np.dot(u_corr[li], u_house[li])), 4) for li in (17, 25)}

    all_pairs = FALSE_PREMISE_12 + FUTURE_UNKNOWABLE_12 + TENSE_MATCHED_UNK_12
    vecs_by_l = {17: [], 25: []}
    ents = []
    for kq, uq in all_pairs:
        for q in (kq, uq):
            prompt = model.processor.apply_chat_template(
                [{"role": "system", "content": NEUTRAL_SYS}, {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            vs, ent = get_last(prompt, already_templated=True)
            for li in (17, 25):
                vecs_by_l[li].append(vs[li])
            ents.append(ent)

    sub_masks = {
        "false_premise_12": [(2 * i, 2 * i + 1) for i in range(12)],
        "future_unknowable_12": [(24 + 2 * i, 24 + 2 * i + 1) for i in range(12)],
        "tense_matched_unk_12": [(48 + 2 * i, 48 + 2 * i + 1) for i in range(12)],
        "all_24_zero_nonce": [(2 * i, 2 * i + 1) for i in range(24)],
        "tense_clean_24": [(2 * i, 2 * i + 1) for i in range(12)] + [(48 + 2 * i, 48 + 2 * i + 1) for i in range(12)],
    }
    out = {
        "model": "google/gemma-4-E4B-it (UNMODIFIED BASE, neutral prompt)",
        "ortho_cos": ortho,
        "L17": _score_zero_nonce_vectors(np.stack(vecs_by_l[17]), ents, u_corr[17], u_house[17], sub_masks),
        "L25": _score_zero_nonce_vectors(np.stack(vecs_by_l[25]), ents, u_corr[25], u_house[25], sub_masks),
    }
    el._atomic_json(RUNS / "sceptic_zero_nonce_gemma_base.json", out)
    model.close()
    print("ZERO_NONCE_GEMMA_DONE:", json.dumps(out, indent=1), flush=True)


def run_zero_nonce_qwen():
    """Run identical zero-nonce common-English neutral-prompt test on UNMODIFIED BASE Alibaba Qwen3.5-9B (Mac MLX)!"""
    import mlx.core as mx
    from mlx_lm import load
    from fluent import plato_data

    mx.set_cache_limit(2 << 30)
    mpath = plato_data.MODEL_SPECS["qwen35_9b"]["path"]
    model, tok = load(mpath)
    layers = plato_data._find_layers(model)
    nl = len(layers)  # 32 -> ~40% = L12/13, ~60% = L19
    l_early, l_mid = round(0.40 * (nl - 1)), round(0.60 * (nl - 1))
    rec = {}
    orig_calls = {}

    for li in (l_early, l_mid):
        blk = layers[li]
        cls = type(blk)
        if cls not in orig_calls:
            orig_calls[cls] = cls.__call__
            def make_wrap(orig_fn):
                def wrapped(self_blk, *a, **kw):
                    y = orig_fn(self_blk, *a, **kw)
                    for idx_t in (l_early, l_mid):
                        if self_blk is layers[idx_t]:
                            h = y[0] if isinstance(y, tuple) else y
                            rec[idx_t] = h[0, -1].astype(mx.float32)
                    return y
                return wrapped
            cls.__call__ = make_wrap(orig_calls[cls])

    def get_last(txt, already_templated=False):
        ids = tok.encode(txt, add_special_tokens=False)[:512] if already_templated else plato_data._text_ids_mlx(tok, txt)[:512]
        rec.clear()
        logits = model(mx.array([ids]))
        lg = logits[0, -1].astype(mx.float32)
        p = mx.softmax(lg, axis=-1)
        ent = -(p * mx.log(mx.maximum(p, 1e-12))).sum()
        mx.eval(rec[l_early], rec[l_mid], ent)
        return {l_early: np.array(rec[l_early], dtype=np.float32), l_mid: np.array(rec[l_mid], dtype=np.float32)}, float(ent.item())

    try:
        diffs_c = {l_early: [], l_mid: []}
        for ts, fs in TRUE_FALSE_20:
            vt, _ = get_last(ts); vf, _ = get_last(fs)
            for li in (l_early, l_mid):
                diffs_c[li].append(vt[li] - vf[li])
        u_corr = {li: arch_rot229.unit(np.stack(diffs_c[li]).mean(axis=0)) for li in (l_early, l_mid)}

        pairs = arch_rot229.load_arch_pairs()
        diffs_h = {l_early: [], l_mid: []}
        cats_h = []
        for p in pairs:
            if p["cat"] in arch_rot229.HOUSE_CATS:
                vd, _ = get_last(p["done"]); vn, _ = get_last(p["neglected"])
                for li in (l_early, l_mid):
                    diffs_h[li].append(vd[li] - vn[li])
                cats_h.append(p["cat"])
        u_house = {li: arch_rot229.house_dir(np.stack(diffs_h[li]), np.array(cats_h)) for li in (l_early, l_mid)}
        ortho = {str(li): round(float(np.dot(u_corr[li], u_house[li])), 4) for li in (l_early, l_mid)}

        all_pairs = FALSE_PREMISE_12 + FUTURE_UNKNOWABLE_12 + TENSE_MATCHED_UNK_12
        vecs_by_l = {l_early: [], l_mid: []}
        ents = []
        for kq, uq in all_pairs:
            for q in (kq, uq):
                msgs = [{"role": "system", "content": NEUTRAL_SYS}, {"role": "user", "content": q}]
                try:
                    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                vs, ent = get_last(prompt, already_templated=True)
                for li in (l_early, l_mid):
                    vecs_by_l[li].append(vs[li])
                ents.append(ent)
    finally:
        for cls, orig_fn in orig_calls.items():
            cls.__call__ = orig_fn
        del model, tok, layers
        gc.collect()
        mx.clear_cache()

    sub_masks = {
        "false_premise_12": [(2 * i, 2 * i + 1) for i in range(12)],
        "future_unknowable_12": [(24 + 2 * i, 24 + 2 * i + 1) for i in range(12)],
        "tense_matched_unk_12": [(48 + 2 * i, 48 + 2 * i + 1) for i in range(12)],
        "all_24_zero_nonce": [(2 * i, 2 * i + 1) for i in range(24)],
        "tense_clean_24": [(2 * i, 2 * i + 1) for i in range(12)] + [(48 + 2 * i, 48 + 2 * i + 1) for i in range(12)],
    }
    out = {
        "model": "Alibaba Qwen3.5-9B (UNMODIFIED BASE, neutral prompt, MLX)",
        "layers": {"early_40pct": l_early, "mid_60pct": l_mid},
        "ortho_cos": ortho,
        f"L{l_early}": _score_zero_nonce_vectors(np.stack(vecs_by_l[l_early]), ents, u_corr[l_early], u_house[l_early], sub_masks),
        f"L{l_mid}": _score_zero_nonce_vectors(np.stack(vecs_by_l[l_mid]), ents, u_corr[l_mid], u_house[l_mid], sub_masks),
    }
    tmp = (RUNS / "sceptic_zero_nonce_qwen_base.json").with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, RUNS / "sceptic_zero_nonce_qwen_base.json")
    print("ZERO_NONCE_QWEN_DONE:", json.dumps(out, indent=1), flush=True)


def self_test():
    assert len(FALSE_PREMISE_12) == 12 and len(FUTURE_UNKNOWABLE_12) == 12 and len(TENSE_MATCHED_UNK_12) == 12
    for kq, uq in FALSE_PREMISE_12 + FUTURE_UNKNOWABLE_12 + TENSE_MATCHED_UNK_12:
        assert abs(len(kq.split()) - len(uq.split())) <= 2, f"Length mismatch: {kq} vs {uq}"
        assert "[memory:" not in uq.lower()
    for kq, uq in TENSE_MATCHED_UNK_12:
        assert len(kq.split()) == 8 and len(uq.split()) == 8, f"Exact 8-word match failed: {kq} | {uq}"
        assert kq.startswith("What is the ") and uq.startswith("What is the ")
        assert " will " not in uq.lower() and " my " not in uq.lower()
    # Verify Gram-Schmidt orthogonalization + mirrored 2-direction random-pair null on synthetic signal
    rng = np.random.default_rng(7)
    D = 64
    uc = arch_rot229.unit(rng.normal(size=D))
    uh = arch_rot229.unit(0.4 * uc + rng.normal(size=D))
    uh_p = arch_rot229.unit(uh - float(np.dot(uh, uc)) * uc)
    V = rng.normal(0, 0.4, size=(24, D))
    V[1::2] -= 1.2 * uc + 1.2 * uh_p
    sc = _score_zero_nonce_vectors(V, [0.5] * 24, uc, uh, {"syn": [(2 * i, 2 * i + 1) for i in range(12)]}, n_rand=100, seed=1)
    assert abs(sc["cos_corr_house_perp"]) < 1e-6 and sc["syn"]["combined_perp_auroc"] > 0.95
    print("SCEPTIC_CTRL230_TEST_OK pairs=36(12_false_premise+12_future+12_tense_matched_unk) gs_perp=OK zero_nonce=OK", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        self_test()
    elif cmd == "stage_a_ctrl":
        run_stage_a_ctrl()
    elif cmd == "zero_nonce_gemma":
        run_zero_nonce_gemma()
    elif cmd == "zero_nonce_qwen":
        run_zero_nonce_qwen()
    else:
        raise ValueError(cmd)
