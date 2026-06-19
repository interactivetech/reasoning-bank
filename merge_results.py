cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

python3 - <<'PY'
import json
from pathlib import Path

parts = [
    "SWE-Bench/results_vllm_10pct_no_memory/preds.json",
    "SWE-Bench/results_vllm_10pct_no_memory_8_13/preds.json",
    "SWE-Bench/results_vllm_10pct_no_memory_11_12/preds.json",
    "SWE-Bench/results_vllm_10pct_no_memory_12_13/preds.json",
    "SWE-Bench/results_vllm_10pct_no_memory_13_30/preds.json",
]

merged = {}
for p in parts:
    data = json.loads(Path(p).read_text())
    overlap = set(merged) & set(data)
    if overlap:
        raise SystemExit(f"Duplicate task ids found in {p}: {sorted(overlap)}")
    merged.update(data)

out = Path("SWE-Bench/results_vllm_10pct_no_memory_merged.json")
out.write_text(json.dumps(merged, indent=2))
print(f"wrote {out} with {len(merged)} predictions")
PY
