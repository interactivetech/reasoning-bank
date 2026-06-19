# SWE-Bench 10% Baseline With 200 Turns

## Purpose

Rerun the 10% SWE-Bench Lite baseline without memory using a higher turn budget.

The baseline config now uses:

* [`SWE-Bench/vllm_qwen_no_memory.yaml`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/vllm_qwen_no_memory.yaml)
* `step_limit: 200`

This is intended to reduce empty-patch failures from the earlier 30-task run.

## 0. Set Local Model Endpoint

```bash
export VLLM_API_BASE="http://YOUR_HOST:8000/v1"
echo $VLLM_API_BASE
```

## 1. Clean Previous Baseline Outputs

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

ids=$(docker ps -aq --filter "name=minisweagent")
[ -z "$ids" ] || docker rm -f $ids

rm -rf SWE-Bench/results_vllm_10pct_no_memory_200
rm -f SWE-Bench/results_vllm_10pct_no_memory_200_merged.json
rm -f SWE-Bench/results_vllm_10pct_no_memory_200_missing_ids.txt
rm -f openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_no_memory_200_merged.json

docker container prune -f
docker builder prune -af
```

## 2. Run The 30-Task Baseline

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 0:30 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_no_memory_200
```

## 3. If The Run Is Interrupted

If the process is killed and you need to resume in smaller slices, first remove
stale `minisweagent` containers:

```bash
ids=$(docker ps -aq --filter "name=minisweagent")
[ -z "$ids" ] || docker rm -f $ids
```

Then rerun only the missing slice into a new output folder. Example:

```bash
./.venv312/bin/python -m minisweagent.run.mini_extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 8:13 \
  --workers 1 \
  --output SWE-Bench/results_vllm_10pct_no_memory_200_8_13
```

Repeat as needed for the remaining slices.

## 4. Merge Split Outputs

If you used multiple output folders, merge them with:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

python3 mergez_result.py \
  --parts \
    SWE-Bench/results_vllm_10pct_no_memory_200/preds.json \
    SWE-Bench/results_vllm_10pct_no_memory_200_8_13/preds.json \
    SWE-Bench/results_vllm_10pct_no_memory_200_13_30/preds.json \
  --out SWE-Bench/results_vllm_10pct_no_memory_200_merged.json \
  --missing-out SWE-Bench/results_vllm_10pct_no_memory_200_missing_ids.txt \
  --target-count 30
```

If the run completed in a single folder, you can skip merging and evaluate
`SWE-Bench/results_vllm_10pct_no_memory_200/preds.json` directly.

## 5. Evaluate

If evaluating the merged file:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

export DOCKER_HOST="unix:///Users/mendeza/.docker/run/docker.sock"

rm -rf logs/run_evaluation/vllm_10pct_no_memory_200_merged

./.venv312/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path /Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_200_merged.json \
  --max_workers 1 \
  --run_id vllm_10pct_no_memory_200_merged
```

If evaluating the single-run file directly:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank

export DOCKER_HOST="unix:///Users/mendeza/.docker/run/docker.sock"

rm -rf logs/run_evaluation/vllm_10pct_no_memory_200

./.venv312/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path /Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_200/preds.json \
  --max_workers 1 \
  --run_id vllm_10pct_no_memory_200
```

## 6. Interpret Results

Report both:

* patch-conditioned success rate
* end-to-end success rate over the full 30-task sample

This matters because empty-patch tasks can still occur even with a larger turn
budget.
