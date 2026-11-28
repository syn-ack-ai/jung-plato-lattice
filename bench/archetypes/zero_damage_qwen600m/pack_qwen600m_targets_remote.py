"""Run on dreamer to tokenize passages.jsonl + eval_passages.jsonl with exact Qwen/Qwen3-0.6B tokenizer
and merge into the single compact `targets_qwen600m_rival30_50.npz` file.
"""
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ARCH = Path("/home/skypanther/systemone/bench/archetypes")
PLATO = ARCH / "fluent" / "plato"
OUT_DIR = ARCH / "exp256_q600"
RAW_NPZ = OUT_DIR / "rival30_50_raw_consensus.npz"
FINAL_NPZ = OUT_DIR / "targets_qwen600m_rival30_50.npz"


def boundary_token_indices(tok, segs):
    full_text = "".join(segs)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    if len(full_ids) < 2:
        return full_ids, []
    pts = []
    acc = ""
    for j, s in enumerate(segs):
        acc += s
        if j == len(segs) - 1:
            idx = len(full_ids) - 1
        else:
            p_ids = tok.encode(acc.rstrip(), add_special_tokens=False)
            idx = max(1, min(len(full_ids) - 1, len(p_ids) - 1))
        if not pts or idx > pts[-1]:
            pts.append(idx)
        else:
            # Ensure exact 1-to-1 point count with teacher segments even if tokenizer merges a boundary!
            pts.append(min(len(full_ids) - 1, pts[-1]))
    return full_ids, pts


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    z = np.load(RAW_NPZ, allow_pickle=True)
    tr_rows = [json.loads(l) for l in open(PLATO / "passages.jsonl") if l.strip()]
    ev_rows = [json.loads(l) for l in open(PLATO / "eval_passages.jsonl") if l.strip()]

    tr_orig_idx = z["tr_orig_idx"]
    tr_cnt = z["tr_cnt"]

    # Pack train passages
    tr_ids_list, tr_pts_list, tr_domains = [], [], []
    tr_ids_off = [0]
    tr_pts_off = [0]
    for k, orig_i in enumerate(tr_orig_idx):
        r = tr_rows[int(orig_i)]
        ids, pts = boundary_token_indices(tok, r["segs"])
        exp_c = int(tr_cnt[k])
        assert len(pts) == exp_c, f"Train passage {k} (orig {orig_i}): got {len(pts)} vs expected {exp_c}"
        tr_ids_list.extend(ids)
        tr_pts_list.extend(pts)
        tr_ids_off.append(len(tr_ids_list))
        tr_pts_off.append(len(tr_pts_list))
        tr_domains.append(r["domain"])

    assert len(tr_pts_list) == z["r3_30_tr"].shape[0] == z["r3_50_tr"].shape[0]

    # Pack eval passages
    ev_cnt = z["ev_cnt"]
    ev_ids_list, ev_pts_list = [], []
    ev_ids_off = [0]
    ev_pts_off = [0]
    for k, r in enumerate(ev_rows):
        ids, pts = boundary_token_indices(tok, r["segs"])
        exp_c = int(ev_cnt[k])
        assert len(pts) == exp_c, f"Eval passage {k}: got {len(pts)} vs expected {exp_c}"
        ev_ids_list.extend(ids)
        ev_pts_list.extend(pts)
        ev_ids_off.append(len(ev_ids_list))
        ev_pts_off.append(len(ev_pts_list))

    assert len(ev_pts_list) == 3192 == z["r3_30_ev"].shape[0]

    np.savez_compressed(
        FINAL_NPZ,
        tr_ids_flat=np.array(tr_ids_list, dtype=np.int32),
        tr_ids_off=np.array(tr_ids_off, dtype=np.int64),
        tr_pts_flat=np.array(tr_pts_list, dtype=np.int32),
        tr_pts_off=np.array(tr_pts_off, dtype=np.int64),
        tr_domains=np.array(tr_domains),
        tr_orig_idx=tr_orig_idx,
        r3_30_tr=z["r3_30_tr"],
        r3_50_tr=z["r3_50_tr"],
        ev_ids_flat=np.array(ev_ids_list, dtype=np.int32),
        ev_ids_off=np.array(ev_ids_off, dtype=np.int64),
        ev_pts_flat=np.array(ev_pts_list, dtype=np.int32),
        ev_pts_off=np.array(ev_pts_off, dtype=np.int64),
        ev_pt_domains=z["ev_pt_domains"],
        ev_pt_groups=z["ev_pt_groups"],
        ev_pt_ret=z["ev_pt_ret"],
        r3_30_ev=z["r3_30_ev"],
        r3_50_ev=z["r3_50_ev"],
        phi4_30_ev=z["phi4_30_ev"],
        phi4_50_ev=z["phi4_50_ev"],
        olmo3_30_ev=z["olmo3_30_ev"],
        olmo3_50_ev=z["olmo3_50_ev"],
        U_jung_r3_30=z["U_jung_r3_30"],
    )
    print(f"Packed Qwen3-0.6B targets -> {FINAL_NPZ} ({FINAL_NPZ.stat().st_size / 1e6:.1f} MB) | tr_toks={len(tr_ids_list)}, tr_pts={len(tr_pts_list)}, ev_toks={len(ev_ids_list)}, ev_pts={len(ev_pts_list)}")


if __name__ == "__main__":
    main()
