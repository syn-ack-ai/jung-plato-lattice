"""sedthh/gutenberg_english, parquet files as published, into smallworld/gutenberg (data only; not committed)."""
import json, os, sys, urllib.request
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gutenberg")
os.makedirs(D, exist_ok=True)
P = json.loads(urllib.request.urlopen(urllib.request.Request("https://datasets-server.huggingface.co/parquet?dataset=sedthh/gutenberg_english", headers={"User-Agent": "research-script"}), timeout=60).read())["parquet_files"]
print("files", len(P), "GB", round(sum(p["size"] for p in P) / 1e9, 2), flush=True)
for i, p in enumerate(P):
    f = os.path.join(D, f"{i:03d}_{p['filename']}")
    if os.path.exists(f) and os.path.getsize(f) == p["size"]:
        continue
    urllib.request.urlretrieve(p["url"], f + ".part"); os.replace(f + ".part", f); print(i + 1, "/", len(P), flush=True)
print("GUTENBERG_DONE", flush=True)
