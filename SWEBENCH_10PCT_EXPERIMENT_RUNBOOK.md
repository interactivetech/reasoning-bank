# SWE-Bench 10% Experiment Runbook

## Goal

Run a controlled comparison on 10% of SWE-Bench Lite:

1. baseline without memory
2. ReasoningBank with memory
3. official SWE-Bench evaluation for both
4. memory pollution analysis over time for the memory condition

This runbook assumes the local model endpoint is:

```text
${VLLM_API_BASE}
```

and the model name is:

```text
Qwen/Qwen3.6-35B-A3B-FP8
```

## Scope

SWE-Bench Lite has 300 tasks. Ten percent is 30 tasks.

This runbook uses:

* baseline slice: `0:30`
* ReasoningBank slice: `0:30`

If you want the exact same task set in both conditions, keep the same slice.

## 0. Set Local Model Endpoint

Before running either experiment, export your local vLLM endpoint:

```bash
export VLLM_API_BASE="http://YOUR_HOST:8000/v1"
```

Verify:

```bash
echo $VLLM_API_BASE
```

If you open a new shell, set it again before launching the runs.

## 1. Install From Scratch

### System prerequisites

Install:

* Docker Desktop
* Rust toolchain
* Python 3.12
* `uv`

### Verify Docker

```bash
docker version
docker info
```

### Install Rust

```bash
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
rustc --version
```

### Create Python environment

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

uv venv .venv312 --python /usr/local/bin/python3.12
```

### Install repo dependencies

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

uv pip install --python .venv312/bin/python -e ./third_party
uv pip install --python .venv312/bin/python datasets chromadb transformers==4.41.2 "numpy<2" swebench
```

### Verify key packages

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

./.venv312/bin/python - <<'PY'
import swebench
import minisweagent
import chromadb
import transformers
print("swebench:", swebench.__file__)
print("minisweagent:", minisweagent.__file__)
print("chromadb ok")
print("transformers:", transformers.__version__)
PY
```

## 2. Docker Socket For Evaluation

The local SWE-Bench evaluator must use the Docker Desktop engine socket:

```bash
export DOCKER_HOST="unix:///Users/mendeza/.docker/run/docker.sock"
```

Verify:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

export DOCKER_HOST="unix:///Users/mendeza/.docker/run/docker.sock"

./.venv312/bin/python - <<'PY'
import docker
client = docker.from_env()
print(client.version()["ApiVersion"])
PY
```

## 3. Config Files

Use:

* no memory: [`SWE-Bench/vllm_qwen_no_memory.yaml`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/vllm_qwen_no_memory.yaml)
* ReasoningBank memory: [`configs/swebench_chroma.yaml`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/configs/swebench_chroma.yaml)

## 4. Baseline Run

This runs 30 tasks without memory.

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

rm -rf SWE-Bench/results_vllm_10pct_no_memory

./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 0:30 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_no_memory
```

## 5. ReasoningBank Run

Before the memory run, clear prior memory state so the run is clean.

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

rm -f memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl
rm -f memory/openai__Qwen__Qwen3.6-35B-A3B-FP8_telemetry.jsonl
rm -rf memory/chroma/swebench
rm -rf SWE-Bench/results_vllm_10pct_reasoningbank

./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config configs/swebench_chroma.yaml \
  --subset lite \
  --split test \
  --slice 0:30 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_reasoningbank
```

## 6. Evaluate Baseline

If the baseline was completed across multiple output folders because some
segments were killed, merge them first:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

python3 mergez_result.py \
  --parts \
    SWE-Bench/results_vllm_10pct_no_memory/preds.json \
    SWE-Bench/results_vllm_10pct_no_memory_8_13/preds.json \
    SWE-Bench/results_vllm_10pct_no_memory_11_12/preds.json \
    SWE-Bench/results_vllm_10pct_no_memory_12_13/preds.json \
    SWE-Bench/results_vllm_10pct_no_memory_13_30/preds.json \
  --out SWE-Bench/results_vllm_10pct_no_memory_merged.json \
  --missing-out SWE-Bench/results_vllm_10pct_no_memory_missing_ids.txt \
  --target-count 30
```

If `results_vllm_10pct_no_memory_missing_ids.txt` is non-empty, those tasks
still have empty patches and should be rerun before evaluation if you want a
true 30-task comparison. Otherwise, the evaluator will only score the non-empty
subset.

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

export DOCKER_HOST="unix:///Users/mendeza/.docker/run/docker.sock"

./.venv312/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path /Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_merged.json \
  --max_workers 1 \
  --run_id vllm_10pct_no_memory_merged
```

Expected report file:

```text
openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_no_memory_merged.json
```

## 7. Evaluate ReasoningBank

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

export DOCKER_HOST="unix:///Users/mendeza/.docker/run/docker.sock"

./.venv312/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path /Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_reasoningbank/preds.json \
  --max_workers 1 \
  --run_id vllm_10pct_reasoningbank
```

Expected report file:

```text
openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_reasoningbank.json
```

## 8. Compare Results

Quick comparison of resolved counts:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

python3 - <<'PY'
import json

baseline = json.load(open("openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_no_memory_merged.json"))
memory = json.load(open("openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_reasoningbank.json"))

print("baseline_resolved:", baseline["resolved_instances"])
print("baseline_unresolved:", baseline["unresolved_instances"])
print("memory_resolved:", memory["resolved_instances"])
print("memory_unresolved:", memory["unresolved_instances"])
print("delta_resolved:", memory["resolved_instances"] - baseline["resolved_instances"])

print("\nbaseline_resolved_ids:")
for x in baseline["resolved_ids"]:
    print(" ", x)

print("\nmemory_resolved_ids:")
for x in memory["resolved_ids"]:
    print(" ", x)
PY
```

Task-by-task comparison:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

python3 - <<'PY'
import json

baseline = json.load(open("openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_no_memory_merged.json"))
memory = json.load(open("openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_reasoningbank.json"))

b_res = set(baseline["resolved_ids"])
m_res = set(memory["resolved_ids"])

all_ids = sorted(b_res | m_res | set(baseline["submitted_ids"]) | set(memory["submitted_ids"]))
for task_id in all_ids:
    b = "PASS" if task_id in b_res else "FAIL"
    m = "PASS" if task_id in m_res else "FAIL"
    print(f"{task_id}  baseline={b}  memory={m}")
PY
```

## 9. Analyze Memory Pollution Over Time

The ReasoningBank run writes telemetry to:

```text
memory/openai__Qwen__Qwen3.6-35B-A3B-FP8_telemetry.jsonl
```

Run the analyzer:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

python3 scripts/analyze_memory_pollution.py \
  memory/openai__Qwen__Qwen3.6-35B-A3B-FP8_telemetry.jsonl
```

This reports:

* write count
* retrieval count
* writes with quality labels
* retrievals that reused labeled memory
* label counts
* feature counts
* write timeline
* retrieval timeline

## 10. Export Pollution Data For Plotting

Create a CSV from the telemetry log:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

python3 - <<'PY'
import csv
import json

src = "memory/openai__Qwen__Qwen3.6-35B-A3B-FP8_telemetry.jsonl"
dst = "memory/openai__Qwen__Qwen3.6-35B-A3B-FP8_telemetry_writes.csv"

rows = []
with open(src) as f:
    for line in f:
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "write":
            continue
        feats = event.get("quality_features", {})
        rows.append({
            "ts": event.get("ts"),
            "task_id": event.get("task_id"),
            "status": event.get("status"),
            "memory_bank_size_before": event.get("memory_bank_size_before"),
            "labels": "|".join(event.get("quality_labels", [])),
            "item_count": feats.get("item_count"),
            "looks_structured": feats.get("looks_structured"),
            "has_shell_commands": feats.get("has_shell_commands"),
            "has_submit_chatter": feats.get("has_submit_chatter"),
            "has_timeout_trace": feats.get("has_timeout_trace"),
            "has_code_fence": feats.get("has_code_fence"),
            "has_file_paths": feats.get("has_file_paths"),
            "avg_item_length": feats.get("avg_item_length"),
            "char_len": feats.get("char_len"),
        })

with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)

print(f"wrote {dst}")
PY
```

## 11. Plot Pollution Over Time

This generates a simple line plot of bad-memory indicators as the bank grows.

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

python3 - <<'PY'
import csv
import matplotlib.pyplot as plt

src = "memory/openai__Qwen__Qwen3.6-35B-A3B-FP8_telemetry_writes.csv"

x = []
shell = []
submit = []
timeout = []
unstructured = []

with open(src) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        labels = set(filter(None, row["labels"].split("|")))
        x.append(i)
        shell.append(1 if "procedural_shell" in labels else 0)
        submit.append(1 if "procedural_submit" in labels else 0)
        timeout.append(1 if "trace_timeout" in labels else 0)
        unstructured.append(1 if "unstructured" in labels else 0)

plt.figure(figsize=(10, 5))
plt.plot(x, shell, label="procedural_shell", marker="o")
plt.plot(x, submit, label="procedural_submit", marker="o")
plt.plot(x, timeout, label="trace_timeout", marker="o")
plt.plot(x, unstructured, label="unstructured", marker="o")
plt.xlabel("Memory write index")
plt.ylabel("Indicator")
plt.title("ReasoningBank Memory Quality Labels Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("memory/memory_pollution_over_time.png", dpi=200)
print("wrote memory/memory_pollution_over_time.png")
PY
```

## 12. Notes

* For a clean memory experiment, keep `--workers 1` for the ReasoningBank run.
* The baseline run does not require memory cleanup.
* The telemetry-based analysis only applies to runs after the telemetry logging was added.
* The current quality labels are instrumentation, not ground-truth scientific judgments.

## 13. Expected Runtime

Rough estimate for 30 tasks:

* baseline run: 2.5 to 3 hours
* ReasoningBank run: 2.5 to 3.5 hours
* both evaluations combined: 45 to 75 minutes
* total: about 5.5 to 7 hours
