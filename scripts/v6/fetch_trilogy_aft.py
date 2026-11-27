"""Fetch upstream Annotated Folktales (`j-hagedorn/trilogy`, `data/aft.csv`) locally without redistributing tale text.

Usage:
    python scripts/v6/fetch_trilogy_aft.py --out-dir ./folklore_local_data

Notes:
  - Upstream license: `j-hagedorn/trilogy`'s `LICENSE.md` states CC BY-NC-SA 4.0 while its README and
    Hagedorn & Darányi (2022, JOHD 8:16) cite CC BY-SA 4.0; Ashliman's source site also carries copyright
    notices on edited/translated texts. This MIT repository therefore NEVER redistributes `aft.csv` or
    reconstructed tale text (and `.gitignore` + `scripts/verify_repro.py` guard against accidental staging).
  - Pinned to upstream Git commit `a6ad5f30aac63f996de56b74c1f42d01c834dd35` (2024-03-05) with strict
    SHA-256 verification (`5214f213b2ef99dded16fcd4f5fca9b1e2c0675aff99ea475b1839c21f441ba1`), exact
    1,518-row schema verification, exact sentence-bounded `clean_excerpt()` reconstruction, and 500/500
    per-excerpt SHA-256 verification against `bench/archetypes/selfshape/aft_clean248_manifest.json`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT_SHA = "a6ad5f30aac63f996de56b74c1f42d01c834dd35"
UPSTREAM_AFT_URL = f"https://raw.githubusercontent.com/j-hagedorn/trilogy/{UPSTREAM_COMMIT_SHA}/data/aft.csv"
EXPECTED_AFT_SHA256 = "5214f213b2ef99dded16fcd4f5fca9b1e2c0675aff99ea475b1839c21f441ba1"
EXPECTED_AFT_ROWS = 1518

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "bench" / "archetypes" / "selfshape" / "aft_clean248_manifest.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_excerpt(text: str, target_lo: int = 160, target_hi: int = 230) -> str:
    """Exact sentence-bounded leading-excerpt builder used to construct the 500 Claim C excerpts."""
    t = re.sub(r"\s+", " ", text or "").strip()
    sents = re.split(r"(?<=[.!?])\s+", t)
    acc, wc = [], 0
    for s in sents:
        sw = len(s.split())
        if wc + sw > target_hi and wc >= target_lo:
            break
        acc.append(s)
        wc += sw
        if wc >= target_hi:
            break
    out = " ".join(acc).strip()
    if len(out.split()) > 250:
        out = " ".join(out.split()[:210]) + "."
    return out


def fetch_and_reconstruct(out_dir: Path, force: bool = False) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "aft.csv"
    out_json = out_dir / "aft_real_myths_500.json"

    if force or not csv_path.exists():
        print(f"Downloading pinned upstream aft.csv ({UPSTREAM_COMMIT_SHA[:12]}) ...")
        req = urllib.request.Request(UPSTREAM_AFT_URL, headers={"User-Agent": "jung-plato-lattice/fetch"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        got_sha = _sha256_bytes(raw)
        if got_sha != EXPECTED_AFT_SHA256:
            raise RuntimeError(
                f"Downloaded aft.csv SHA-256 mismatch: got {got_sha}, expected {EXPECTED_AFT_SHA256}"
            )
        tmp = csv_path.with_suffix(".csv.tmp")
        tmp.write_bytes(raw)
        tmp.replace(csv_path)
    else:
        got_sha = _sha256_file(csv_path)
        if got_sha != EXPECTED_AFT_SHA256:
            raise RuntimeError(
                f"Existing {csv_path} SHA-256 mismatch: got {got_sha}, expected {EXPECTED_AFT_SHA256} (use --force to re-download)"
            )

    csv.field_size_limit(10_000_000)
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        aft_rows = list(csv.DictReader(f))
    if len(aft_rows) != EXPECTED_AFT_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_AFT_ROWS} rows in aft.csv, found {len(aft_rows)}")
    for col in ("atu_id", "tale_title", "provenance", "text"):
        if col not in aft_rows[0]:
            raise RuntimeError(f"Missing required column '{col}' in aft.csv")

    manifest = json.loads(MANIFEST_PATH.read_text())
    if len(manifest) != 500:
        raise RuntimeError(f"Expected 500 records in {MANIFEST_PATH}, found {len(manifest)}")

    rebuilt = []
    seen_ids = set()
    verified_sha_count = 0
    shorter_480, equal_480, kept_480 = 0, 0, 0
    for rec in manifest:
        idx = int(rec["aft_idx"])
        if not (0 <= idx < len(aft_rows)):
            raise RuntimeError(f"aft_idx {idx} out of range (0..{len(aft_rows)-1})")
        if idx in seen_ids:
            raise RuntimeError(f"Duplicate aft_idx {idx} in manifest")
        seen_ids.add(idx)
        row = aft_rows[idx]
        if str(row["atu_id"]).strip() != str(rec["atu_id"]).strip():
            raise RuntimeError(
                f"ATU ID mismatch at aft_idx={idx}: csv='{row['atu_id']}' vs manifest='{rec['atu_id']}'"
            )
        exp_sha = rec.get("excerpt_sha256")
        if not isinstance(exp_sha, str) or len(exp_sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", exp_sha):
            raise RuntimeError(f"Missing or invalid excerpt_sha256 in manifest at aft_idx={idx}")
        if "excerpt_wc" not in rec or int(rec["excerpt_wc"]) <= 0:
            raise RuntimeError(f"Missing or invalid excerpt_wc in manifest at aft_idx={idx}")
        if "historical_non_western" not in rec:
            raise RuntimeError(f"Missing historical_non_western in manifest at aft_idx={idx}")
        ex = clean_excerpt(row["text"])
        ex_wc = len(ex.split())
        ex_sha = _sha256_bytes(ex.encode("utf-8"))
        if ex_sha != exp_sha:
            raise RuntimeError(f"Excerpt SHA-256 mismatch at aft_idx={idx}: got {ex_sha}, expected {exp_sha}")
        if ex_wc != int(rec["excerpt_wc"]):
            raise RuntimeError(f"Excerpt word-count mismatch at aft_idx={idx}: got {ex_wc}, expected {rec['excerpt_wc']}")
        verified_sha_count += 1
        if rec.get("included", False):
            kept_480 += 1
            raw_wc = len((row["text"] or "").split())
            if ex_wc < raw_wc:
                shorter_480 += 1
            elif ex_wc == raw_wc:
                equal_480 += 1
        rebuilt.append({
            "aft_idx": idx,
            "atu_id": str(rec["atu_id"]).strip(),
            "division": rec["division"],
            "title": str(row.get("tale_title", "")).strip(),
            "provenance": str(row.get("provenance", "")).strip(),
            "non_western": bool(rec["historical_non_western"]),
            "adjudicated_provenance": rec["adjudicated_provenance"],
            "included": bool(rec["included"]),
            "exclusion_reason": rec.get("exclusion_reason"),
            "text": ex,
            "wc": ex_wc,
            "excerpt_sha256": ex_sha,
        })

    if verified_sha_count != 500 or len(seen_ids) != 500:
        raise RuntimeError(f"Expected 500 verified unique excerpts, got verified={verified_sha_count}, unique={len(seen_ids)}")
    if (kept_480, shorter_480, equal_480) != (480, 442, 38):
        raise RuntimeError(f"Unexpected 480-set truncation counts: kept={kept_480}, shorter={shorter_480}, equal={equal_480} (expected 480, 442, 38)")

    tmp_j = out_json.with_suffix(".json.tmp")
    tmp_j.write_text(json.dumps(rebuilt, indent=2) + "\n")
    tmp_j.replace(out_json)
    print(
        f"VERIFIED_AFT_FETCH_OK: commit={UPSTREAM_COMMIT_SHA[:12]} sha256={EXPECTED_AFT_SHA256[:12]} "
        f"rows={len(aft_rows)} matched_500_sha256={verified_sha_count}/500 kept_480={kept_480} (shorter={shorter_480}, equal={equal_480}) -> {out_json}"
    )
    return csv_path, out_json


def main():
    ap = argparse.ArgumentParser(description="Fetch pinned upstream aft.csv and verify 500/500 Claim C excerpts locally.")
    ap.add_argument("--out-dir", type=Path, default=Path(os.environ.get("JPL_FOLK_DIR", "./folklore_local_data")))
    ap.add_argument("--force", action="store_true", help="Re-download aft.csv even if already present")
    args = ap.parse_args()
    fetch_and_reconstruct(args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
