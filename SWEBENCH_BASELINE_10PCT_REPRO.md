# SWE-Bench 10% Baseline Repro

## Context

This note records the exact commands used to complete the 10% SWE-Bench Lite
baseline run without memory.

The baseline was not completed in a single uninterrupted command. The original
`0:30` run was killed by an out-of-memory event after partial progress, so the
remaining slices were resumed as separate commands.

The final baseline result is therefore the union of these output folders:

* [`SWE-Bench/results_vllm_10pct_no_memory`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory)
* [`SWE-Bench/results_vllm_10pct_no_memory_8_13`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_8_13)
* [`SWE-Bench/results_vllm_10pct_no_memory_11_12`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_11_12)
* [`SWE-Bench/results_vllm_10pct_no_memory_12_13`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_12_13)
* [`SWE-Bench/results_vllm_10pct_no_memory_13_30`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_13_30)

These folders together contain 30 unique predictions.

## Environment

The run used:

* local vLLM endpoint via `VLLM_API_BASE`
* Python env: `./.venv312/bin/python`
* config: [`SWE-Bench/vllm_qwen_no_memory.yaml`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/vllm_qwen_no_memory.yaml)

Before running, the local endpoint was exported:

```bash
export VLLM_API_BASE="http://YOUR_HOST:8000/v1"
```

## Cleanup Commands

These cleanup commands were used before restarting the experiment:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

# Stop/remove stale mini-swe-agent containers from killed runs.
ids=$(docker ps -aq --filter "name=minisweagent")
[ -z "$ids" ] || docker rm -f $ids

# Remove baseline outputs.
rm -rf SWE-Bench/results_vllm_10pct_no_memory
rm -rf SWE-Bench/results_vllm_10pct_no_memory_*

# Remove ReasoningBank outputs.
rm -rf SWE-Bench/results_vllm_10pct_reasoningbank
rm -rf SWE-Bench/results_vllm_10pct_reasoningbank_*

# Remove ReasoningBank memory state.
rm -f memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl
rm -f memory/openai__Qwen__Qwen3.6-35B-A3B-FP8_telemetry.jsonl
rm -rf memory/chroma/swebench

# Remove local evaluation summaries.
rm -f openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_no_memory.json
rm -f openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_reasoningbank.json
rm -f openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_qwen_reasoningbank_embed_batch*.json

# Clean Docker leftovers.
docker container prune -f
docker builder prune -af
```

An equivalent cleanup variant that was also used:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

ids=$(docker ps -aq --filter "name=minisweagent")
[ -z "$ids" ] || docker rm -f $ids

rm -rf SWE-Bench/results_vllm_10pct_no_memory
rm -rf SWE-Bench/results_vllm_10pct_reasoningbank

find SWE-Bench -maxdepth 1 -type d -name 'results_vllm_10pct_no_memory_*' -exec rm -rf {} +
find SWE-Bench -maxdepth 1 -type d -name 'results_vllm_10pct_reasoningbank_*' -exec rm -rf {} +

rm -f memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl
rm -f memory/openai__Qwen__Qwen3.6-35B-A3B-FP8_telemetry.jsonl
rm -rf memory/chroma/swebench

find . -maxdepth 1 -type f -name 'openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_no_memory.json' -delete
find . -maxdepth 1 -type f -name 'openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_reasoningbank.json' -delete
find . -maxdepth 1 -type f -name 'openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_qwen_reasoningbank_embed_batch*.json' -delete

docker container prune -f
docker builder prune -af
```

## Baseline Run Commands

### Initial attempt

This launched the intended 30-task baseline:

```bash
export VLLM_API_BASE="http://YOUR_HOST:8000/v1"

cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 0:30 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_no_memory
```

This partial run produced 8 predictions in:

* [`SWE-Bench/results_vllm_10pct_no_memory/preds.json`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory/preds.json)

### Resume from `8:13`

Before resuming, stale containers were removed:

```bash
ids=$(docker ps -aq --filter "name=minisweagent")
[ -z "$ids" ] || docker rm -f $ids
```

Then:

```bash
export VLLM_API_BASE="http://YOUR_HOST:8000/v1"

./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 8:13 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_no_memory_8_13
```

This partial resume produced 3 predictions in:

* [`SWE-Bench/results_vllm_10pct_no_memory_8_13/preds.json`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_8_13/preds.json)

### Resume `11:12`

Again, stale containers were removed:

```bash
ids=$(docker ps -aq --filter "name=minisweagent")
[ -z "$ids" ] || docker rm -f $ids
```

Then:

```bash
export VLLM_API_BASE="http://YOUR_HOST:8000/v1"

./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 11:12 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_no_memory_11_12
```

### Resume `12:13`

```bash
./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 12:13 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_no_memory_12_13
```

### Resume `13:30`

```bash
./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 13:30 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_no_memory_13_30
```

## Resulting Coverage

Prediction counts by folder:

* `results_vllm_10pct_no_memory`: 8
* `results_vllm_10pct_no_memory_8_13`: 3
* `results_vllm_10pct_no_memory_11_12`: 1
* `results_vllm_10pct_no_memory_12_13`: 1
* `results_vllm_10pct_no_memory_13_30`: 17

Total:

* 30 predictions
* 30 unique task ids

## Recommended Next Step

To evaluate the baseline cleanly, merge these `preds.json` files into one
combined predictions file, identify empty-patch tasks, rerun the missing tasks,
and then evaluate the completed merged output.

## Merge Split Outputs

Use the merge utility:

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

This produces:

* merged predictions:
  [`SWE-Bench/results_vllm_10pct_no_memory_merged.json`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_merged.json)
* empty-patch task list:
  [`SWE-Bench/results_vllm_10pct_no_memory_missing_ids.txt`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_missing_ids.txt)

If the merged file still contains empty patches, SWE-Bench evaluation will only
run the non-empty predictions. For a true 30-task comparison, rerun the missing
task ids before evaluating.

## Evaluate The Merged Baseline

Once all 30 tasks have non-empty patches in the merged file:

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
