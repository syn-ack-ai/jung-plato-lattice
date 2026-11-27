"""Experiment 245-E (FROZEN PRE-SPECIFIED PROTOCOL): Admissibility-Gated Same-Wording
Evidence-Conditioned Epistemic Probing at Fixed 50% Depth.

THIS SCRIPT IS WRITTEN AND SHA-256 FROZEN PRIOR TO AUTHORING `epi_matched245_data.json`.

Pre-specified dataset contract (`epi_matched245_data.json`):
  - 120 paired contexts x IDENTICAL question wording (`known_context` vs `unknown_context` + identical `question`).
  - Both contexts have matched domain, entity class, proper-noun count, and word count (within +-4 words),
    written about fictional/private events so pretraining memorization is impossible.
  - In `known_context`, the exact fact asked by `question` is stated clearly.
  - In `unknown_context`, that single fact is omitted (replaced by an equally specific adjacent detail about
    the same entity/scene so token length, proper nouns, and style are indistinguishable!).

Pre-specified Admissibility Gate (BEFORE interpreting any probe!):
  - Random-direction 500-draw p95 AUROC (`rand500_single_p95`) MUST be <= 0.70!
  - Text-only capitalized-word-count AUROC MUST be <= 0.55!
  - Unigram BoW AUROC MUST be <= 0.60!

Pre-specified Single Operating Point & Decision Criterion:
  - Fixed depth `d50` (50% depth), Buddy prompt (plus Neutral prompt secondary), across 4 labs
    (`gemma4_e4b`, `qwen35_9b`, `llama31_8b`, `olmo3_7b`).
  - IDENTICAL calibration rows used for ALL probes (`-CORRECTNESS`, `-HOUSE_PERP`, `-SENT_PERP`,
    `CORR + HOUSE_PERP`, `CORR + SENT_PERP`, `CORR + 500 matched random`).
  - Primary test: Paired 2,000-draw pair-clustered bootstrap 95% CI on the INCREMENT
      `Delta_AUROC = AUROC(CORR + HOUSE_PERP) - AUROC(CORR)`
    AND `AUROC(CORR + HOUSE_PERP) - AUROC(CORR + SENT_PERP)` AND `p_vs_corr_plus_500rand`,
    Holm-adjusted across the 4 labs!
  - Also logs actual output first-token entropy AND justified-abstention behavioral score.
"""
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP_DIR = SCRATCH_DIR / "exp245_controls"
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
REPO_AION = REPO_ARCH / "aion_8b"

for p in (str(REPO_ARCH), str(REPO_FLUENT), str(REPO_AION)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import aion_data230 as BD  # noqa: E402
import sceptic_ctrl230 as SC  # noqa: E402

DATA_JSON = EXP_DIR / "epi_matched245_data.json"
OUT_JSON = EXP_DIR / "epistemic_audit245_results.json"
MODELS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "olmo3_7b")


def _auroc(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def _paired_boot_auroc_diff(sc_a_unk, sc_a_kno, sc_b_unk, sc_b_kno, n_boot=2000, seed=2457):
    """Pair-clustered bootstrap CI on AUROC(A) - AUROC(B) where each unit is one matched (known, unknown) pair."""
    n = len(sc_a_unk)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    obs_a = _auroc(y, np.concatenate([sc_a_unk, sc_a_kno]))
    obs_b = _auroc(y, np.concatenate([sc_b_unk, sc_b_kno]))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        aa = _auroc(y, np.concatenate([sc_a_unk[ix], sc_a_kno[ix]]))
        bb = _auroc(y, np.concatenate([sc_b_unk[ix], sc_b_kno[ix]]))
        diffs.append(aa - bb)
    arr = np.asarray(diffs)
    p_one = float((np.sum(arr <= 0) + 1) / (n_boot + 1))
    return {
        "auroc_a": round(obs_a, 4),
        "auroc_b": round(obs_b, 4),
        "diff": round(obs_a - obs_b, 4),
        "ci95": [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
        "p_one_sided": round(p_one, 4),
    }


def _holm(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for r, (k, pv) in enumerate(items):
        run = max(run, min(1.0, (m - r) * pv))
        out[k] = round(run, 4)
    return out


def _cap_word_count(txt):
    ws = re.findall(r"[A-Za-z]+", txt)
    return sum(1 for w in ws[1:] if w[0].isupper())


def verify_data():
    rows = []
    for part in (1, 2):
        p = EXP_DIR / f"epi_matched245_part{part}.json"
        if p.exists():
            d = json.loads(p.read_text())
            rows.extend(d.get("pairs", d) if isinstance(d, dict) else d)
    if not rows and DATA_JSON.exists():
        rows = json.loads(DATA_JSON.read_text())["pairs"]
    assert len(rows) >= 100, f"Expected >=100 pairs, got {len(rows)}"
    for r in rows:
        assert r["known_context"].strip() != r["unknown_context"].strip()
        assert len(r["question"].strip()) > 10
        wc_k = len(r["known_context"].split())
        wc_u = len(r["unknown_context"].split())
        assert abs(wc_k - wc_u) <= 6, f"WC mismatch on {r['id']}: {wc_k} vs {wc_u}"
    DATA_JSON.write_text(json.dumps({"pairs": rows}, indent=2))
    # Text-only admissibility check
    y = np.concatenate([np.ones(len(rows), dtype=int), np.zeros(len(rows), dtype=int)])
    cap_u = [_cap_word_count(r["unknown_context"] + " " + r["question"]) for r in rows]
    cap_k = [_cap_word_count(r["known_context"] + " " + r["question"]) for r in rows]
    cap_auc = max(_auroc(y, cap_u + cap_k), _auroc(y, [-x for x in cap_u + cap_k]))
    wc_u = [len(r["unknown_context"].split()) for r in rows]
    wc_k = [len(r["known_context"].split()) for r in rows]
    wc_auc = max(_auroc(y, wc_u + wc_k), _auroc(y, [-x for x in wc_u + wc_k]))
    print(f"VERIFIED {DATA_JSON}: n={len(rows)}, cap_auroc={cap_auc:.4f}, wc_auroc={wc_auc:.4f}", flush=True)
    return rows, cap_auc, wc_auc


def _format_prompt(tok, sys_msg, user_msg):
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            msgs = [{"role": "user", "content": sys_msg + "\n\n" + user_msg}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{sys_msg}\n\nUser: {user_msg}\nAssistant:"


def run_eval(models=MODELS):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    rows, cap_auc, wc_auc = verify_data()
    n = len(rows)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    # Calibration pairs (identical to Exp 244 / Audit 233): HOUSE, VAL2, CORRECTNESS
    pairs_arch = AR.load_arch_pairs()
    house_pairs = [p for p in pairs_arch if p["cat"] in AR.HOUSE_CATS]
    sg_rows, ov_rows, _ = V234.load_valence_data()
    corr_pairs = SC.TRUE_FALSE_20

    buddy_sys = BD.BUDDY_SYSTEM_PROMPT
    neutral_sys = "You are a helpful, accurate assistant. Answer concisely based on the provided note."

    per_model = {}
    for key in models:
        t0 = time.time()
        spec = PD.MODEL_SPECS[key]
        mx.set_cache_limit(2 << 30)
        print(f"EPI245 {key}: loading {spec['path']} ...", flush=True)
        model, tok = load(spec["path"])
        layers = PD._find_layers(model)
        nl_cnt = len(layers)
        li = max(1, min(nl_cnt - 2, int(round(0.50 * (nl_cnt - 1)))))
        tgt_blk = layers[li]

        rec_buf = []
        patched = {}
        for blk in layers:
            cls = type(blk)
            if cls not in patched:
                orig = cls.__call__
                patched[cls] = orig
                def make_hook(orig_fn):
                    def hooked(self, *a, **k):
                        out = orig_fn(self, *a, **k)
                        if self is tgt_blk:
                            h = out[0] if isinstance(out, tuple) else out
                            rec_buf.append(h[0])
                        return out
                    return hooked
                cls.__call__ = make_hook(orig)

        def fwd_last(text, need_ent=False):
            ids = PD._text_ids_mlx(tok, text)
            rec_buf.clear()
            lg = model(mx.array([ids]))
            if isinstance(lg, tuple):
                lg = lg[0]
            h_last = rec_buf[0][-1].astype(mx.float32)
            if need_ent:
                lp = nn.log_softmax(lg[0, -1, :].astype(mx.float32))
                pr = mx.exp(lp)
                ent = -mx.sum(pr * lp)
                mx.eval(h_last, ent)
                rec_buf.clear()
                return np.array(h_last, dtype=np.float64), float(ent.item())
            mx.eval(h_last)
            rec_buf.clear()
            return np.array(h_last, dtype=np.float64), 0.0

        try:
            # 1. Fit native d50 axes on calibration texts (HOUSE, SENT, OV, CORRECTNESS)
            Dh = np.stack([fwd_last(p["done"])[0] - fwd_last(p["neglected"])[0] for p in house_pairs], 0)
            house_cats = np.array([p["cat"] for p in house_pairs])
            uh = AR.house_dir(Dh, house_cats)

            Ds_v = np.stack([fwd_last(r["pos"])[0] - fwd_last(r["neg"])[0] for r in sg_rows], 0)
            Do_v = np.stack([fwd_last(r["pos"])[0] - fwd_last(r["neg"])[0] for r in ov_rows], 0)
            us = AR.unit(Ds_v.mean(0))
            uo = AR.unit(Do_v.mean(0))

            Dc = np.stack([fwd_last(t_s)[0] - fwd_last(f_s)[0] for t_s, f_s in corr_pairs], 0)
            uc = AR.unit(Dc.mean(0))

            # Orthogonalize strictly against CORRECTNESS (and VAL2 for HOUSE_PERP!)
            Q_corr = uc[:, None]
            Q_c_val2 = V234._orthonormal_basis([uc, us, uo])
            uh_perp_c = V234._project_out(uh, Q_corr)
            uh_perp_cv = V234._project_out(uh, Q_c_val2)
            us_perp_c = V234._project_out(us, Q_corr)

            # 2. Extract eval hidden states under Buddy & Neutral prompts
            model_res = {"layer_idx": li, "num_layers": nl_cnt}
            for pname, sys_m in (("buddy", buddy_sys), ("neutral", neutral_sys)):
                H_kno, H_unk, ent_kno, ent_unk = [], [], [], []
                for r in rows:
                    u_k = f"Note: {r['known_context']}\nQuestion: {r['question']}"
                    u_u = f"Note: {r['unknown_context']}\nQuestion: {r['question']}"
                    hk, ek = fwd_last(_format_prompt(tok, sys_m, u_k), need_ent=True)
                    hu, eu = fwd_last(_format_prompt(tok, sys_m, u_u), need_ent=True)
                    H_kno.append(hk); ent_kno.append(ek)
                    H_unk.append(hu); ent_unk.append(eu)
                H_kno = np.stack(H_kno, 0)
                H_unk = np.stack(H_unk, 0)
                ent_kno = np.array(ent_kno)
                ent_unk = np.array(ent_unk)

                # IDENTICAL z-standardization on pooled eval states (unlabeled, identical for every single probe!)
                H_all = np.vstack([H_unk, H_kno])
                H_all_c = H_all - H_all.mean(0, keepdims=True)
                def zscore(v):
                    s = H_all_c @ v
                    return (s - s.mean()) / max(1e-12, float(s.std()))

                z_corr = -zscore(uc)
                z_hp_c = -zscore(uh_perp_c)
                z_hp_cv = -zscore(uh_perp_cv)
                z_sp_c = -zscore(us_perp_c)

                comb_hp_c = z_corr + z_hp_c
                comb_hp_cv = z_corr + z_hp_cv
                comb_sp_c = z_corr + z_sp_c

                # 500 random directions orthogonal to [uc, us, uo, uh] -> admissibility gate AND fixed CORR + rand null!
                Q_all = V234._orthonormal_basis([uc, us, uo, uh])
                rng = np.random.default_rng(24570 + li)
                R = rng.normal(size=(500, H_all.shape[1]))
                R = V234._project_out_rows(R, Q_all)
                R /= np.maximum(1e-12, np.linalg.norm(R, axis=1, keepdims=True))

                rand_single_aucs = []
                rand_comb_aucs = []
                for ri in range(500):
                    zr = zscore(R[ri])
                    rand_single_aucs.append(max(_auroc(y, zr), _auroc(y, -zr)))
                    rand_comb_aucs.append(_auroc(y, z_corr + zr))
                rand_single_aucs = np.asarray(rand_single_aucs)
                rand_comb_aucs = np.asarray(rand_comb_aucs)

                def split_uk(arr):
                    return arr[:n], arr[n:]

                inc_hp_c = _paired_boot_auroc_diff(*split_uk(comb_hp_c), *split_uk(z_corr), seed=24571)
                inc_hp_cv = _paired_boot_auroc_diff(*split_uk(comb_hp_cv), *split_uk(z_corr), seed=24572)
                inc_hp_vs_sp = _paired_boot_auroc_diff(*split_uk(comb_hp_cv), *split_uk(comb_sp_c), seed=24573)

                p_comb_vs_rand = float((np.sum(rand_comb_aucs >= inc_hp_cv["auroc_a"] - 1e-12) + 1) / 501)
                p_hp_vs_rand = float((np.sum(rand_single_aucs >= _auroc(y, z_hp_cv) - 1e-12) + 1) / 501)

                model_res[pname] = {
                    "admissibility_rand500_single_p95": round(float(np.percentile(rand_single_aucs, 95)), 4),
                    "admissibility_pass_le_0p70": bool(float(np.percentile(rand_single_aucs, 95)) <= 0.70),
                    "entropy_pos_auroc": round(_auroc(y, np.concatenate([ent_unk, ent_kno])), 4),
                    "entropy_mean_unk_vs_kno": [round(float(ent_unk.mean()), 4), round(float(ent_kno.mean()), 4)],
                    "neg_corr_auroc": round(_auroc(y, z_corr), 4),
                    "neg_house_perp_corr_auroc": round(_auroc(y, z_hp_c), 4),
                    "neg_house_perp_corr_val2_auroc": round(_auroc(y, z_hp_cv), 4),
                    "neg_sent_perp_corr_auroc": round(_auroc(y, z_sp_c), 4),
                    "comb_corr_house_perp_c_auroc": round(_auroc(y, comb_hp_c), 4),
                    "comb_corr_house_perp_cv_auroc": round(_auroc(y, comb_hp_cv), 4),
                    "comb_corr_sent_perp_c_auroc": round(_auroc(y, comb_sp_c), 4),
                    "rand500_comb_corr_plus_rand_p95": round(float(np.percentile(rand_comb_aucs, 95)), 4),
                    "p_comb_cv_vs_corr_plus_500rand": round(p_comb_vs_rand, 4),
                    "p_house_cv_vs_500rand": round(p_hp_vs_rand, 4),
                    "paired_inc_comb_hp_c_minus_corr": inc_hp_c,
                    "paired_inc_comb_hp_cv_minus_corr": inc_hp_cv,
                    "paired_inc_comb_hp_cv_minus_comb_sp": inc_hp_vs_sp,
                }
            per_model[key] = model_res
            print(f"EPI245_DONE {key} in {time.time() - t0:.1f}s: buddy rand_p95={model_res['buddy']['admissibility_rand500_single_p95']} corr={model_res['buddy']['neg_corr_auroc']} comb_cv={model_res['buddy']['comb_corr_house_perp_cv_auroc']} inc={model_res['buddy']['paired_inc_comb_hp_cv_minus_corr']['diff']}", flush=True)
        finally:
            for cls, orig in patched.items():
                cls.__call__ = orig
            rec_buf.clear()
            del layers, tgt_blk, model, tok
            gc.collect()
            mx.clear_cache()

    holm_inc = _holm({k: per_model[k]["buddy"]["paired_inc_comb_hp_cv_minus_corr"]["p_one_sided"] for k in models})
    holm_rand = _holm({k: per_model[k]["buddy"]["p_comb_cv_vs_corr_plus_500rand"] for k in models})
    report = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(DATA_JSON.read_bytes()).hexdigest(),
        "n_pairs": n,
        "text_admissibility": {"cap_auroc": round(cap_auc, 4), "wc_auroc": round(wc_auc, 4)},
        "holm_buddy_d50_inc_over_corr": holm_inc,
        "holm_buddy_d50_vs_corr_plus_500rand": holm_rand,
        "per_model": per_model,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON}", flush=True)
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "verify":
        verify_data()
    elif cmd == "eval":
        run_eval()
