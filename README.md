# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

## 📜 Overview

<p align="center">
    <img src="assets/reasoningbank.png" width="80%" alt="intro_case">
</p>

We introduce ReasoningBank, a memory mechanism for agents that learns from both 
successful and failed trajectories, with reasoning stored as memory content.

<p align="center">
    <img src="assets/method.png" width="100%" alt="intro_case">
</p>

Building upon this memory formulation, we propose memory-aware test-time scaling,
which leverages the bidirectional synergy between memory and test-time scaling,
establishing experience-driven memory as another scaling dimension for agent
systems.


## 📂 Code Setup
We release code for `SWE-Bench` (software engineering) and `WebArena` (web-browsing),
as in corresponding directories.

Before we start, please install required packages by running `pip install -r requirements.txt`.

### 0. LLM Configuration
Currently we support three model families: 
- **GPT**: To use GPT models (`gpt-3.5-turbo`, `gpt-4`, `gpt-4o`), you need to set your OpenAI API key as an environment variable:
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  ```

- **Gemini & Claude**: To use Gemini models (`gemini-2.5-flash`, `gemini-2.5-pro`) or Claude (`claude-3-7-sonnet@20250219`) on Vertex AI, you need to configure Google Cloud authentication.
  1.  Install the Google Cloud CLI and log in to set up Application Default Credentials (ADC):
      ```bash
      gcloud auth application-default login
      ```
  2.  Set your project and location as environment variables, as they are required by clients like the one for Claude on Vertex AI:
      ```bash
      export GOOGLE_CLOUD_PROJECT="your-project-id"
      export GOOGLE_CLOUD_LOCATION="your-region"
      export GOOGLE_GENAI_USE_VERTEXAI="True"
      ```


### 1. WebArena
#### Docker Configuration
Make sure to correctly install `browsergym` following the [official documentation](https://github.com/ServiceNow/BrowserGym).

The next step is to download and config docker environment for WebArena. Please refer to [this tutorial](https://github.com/gasse/webarena-setup/tree/main/webarena),
executing the scripts follow the numerical order of file names. Before executing,
make sure to config the address of each website in corresponding scripts as
instructed correspondingly.

#### Directory Structure

* `WebArena/agents/`: implementation for web agents integrating with browsergym
* `WebArena/autoeval/`: llm-as-a-judge for obtaining correctness signal for trajectories
* `WebArena/config_files/`: data processing for webarena tasks
* `WebArena/prompt/`: instructions used across the implementation

#### Data preprocessing

Download raw test files from [here](https://github.com/web-arena-x/webarena/blob/main/config_files/test.raw.json) and put it to `config_files`. The repo also vendors a patched copy at `third_party/webarena/test.raw.json` with shopping-split annotation corrections; use either one.

Run `generate_config_files.py` to process raw test data to config files as input.

#### Use the vendored `webarena` tree

The repo ships a patched `webarena/` harness at `third_party/webarena/` (corrected shopping annotations, wishlist eval fix, `fill('','')` guard, `retry_with_force=True` clicks) to make environment and corresponding evaluation more robust and stable. Prepend it to `PYTHONPATH` so it shadows the pip-installed `browsergym.webarena`:

```bash
export PYTHONPATH="$(pwd)/third_party:$PYTHONPATH"
```

`webarena` is a namespace package, so no code edits are required — every `webarena.*` submodule resolves to the vendored copy.


#### Run the code

Run directly with ReasoningBank: `bash run.sh`, config `model`, `output_dir`, and
`website`, and `memory_mode` accordingly.

To run with scaling setting, please refer to
`pipeline_scaling.py` and `induce_scaling.py`.

### 2. SWE-Bench
We built upon [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent). First, install it from source by `pip install -e .` under the directory of `./third_party` This will install the dependencies as specified in `pyproject.toml`.

The script `SWE-Bench/run.sh` provides direct running command, which will generate
result files in the output directory. Before running, make sure the
configuration for VertexAI is properly configured as instructed in `run.sh`.

For evaluation, please refer to `sb-cli` command in the [official documentation](https://mini-swe-agent.com/latest/usage/swebench/). 

#### Local vLLM integration

The repo now includes local configs for running SWE-Bench against an OpenAI-compatible
vLLM server such as:

```bash
http://173.73.39.103:8000/v1
```

with model:

```bash
Qwen/Qwen3.6-35B-A3B-FP8
```

The added configs are:

* `SWE-Bench/vllm_qwen_no_memory.yaml`: vLLM run without ReasoningBank memory
* `SWE-Bench/vllm_qwen_one_task.yaml`: vLLM run without memory for a single full task
* `SWE-Bench/vllm_qwen_reasoningbank.yaml`: vLLM run with ReasoningBank memory enabled
* `SWE-Bench/vllm_qwen_reasoningbank_validate.yaml`: short validation config for memory plumbing

The SWE-Bench path was also adapted to:

* use LiteLLM against a custom OpenAI-compatible endpoint
* tolerate missing LiteLLM pricing metadata for custom models
* make the no-memory path independent of Gemini / Vertex AI
* make the ReasoningBank path use the configured model for memory generation and judging
* store memory under sanitized file names in the local `memory/` directory

#### Environment setup

One working local setup is:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank
uv venv .venv --python 3.14
. .venv/bin/activate
uv pip install -e ./third_party datasets
```

#### Run two tasks without memory

Use a dedicated output directory for each run:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank
. .venv/bin/activate
mini-extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --slice 0:2 \
  --workers 1 \
  --output SWE-Bench/results_vllm_two_tasks_no_memory
```

If you want two specific tasks instead of the first two:

```bash
mini-extra swebench \
  --config SWE-Bench/vllm_qwen_no_memory.yaml \
  --subset lite \
  --split test \
  --filter '^(astropy__astropy-12907|astropy__astropy-14182)$' \
  --workers 1 \
  --output SWE-Bench/results_vllm_two_tasks_no_memory
```

#### Run two tasks with ReasoningBank memory

The first task populates the local memory bank:

```bash
cd /Users/mendeza/Documents/2026_research_projects/reasoning-bank
. .venv/bin/activate
mini-extra swebench \
  --config SWE-Bench/vllm_qwen_reasoningbank.yaml \
  --subset lite \
  --split test \
  --slice 0:1 \
  --workers 1 \
  --output SWE-Bench/results_vllm_reasoningbank_task1
```

Then run the second task and reuse the memory written by the first run:

```bash
mini-extra swebench \
  --config SWE-Bench/vllm_qwen_reasoningbank.yaml \
  --subset lite \
  --split test \
  --slice 1:2 \
  --workers 1 \
  --output SWE-Bench/results_vllm_reasoningbank_task2
```

If you want to run both tasks into the same output directory, that also works.
`preds.json` will accumulate both completed instances. Just make sure you do not
accidentally reuse the same `--slice` if you intend to keep prior results.

#### Where memory is stored

ReasoningBank memory is stored locally in:

```bash
memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl
```

Each line is a JSON object containing:

* `task_id`
* `query`
* `memory_items`
* `status`

You can inspect it with:

```bash
sed -n '1,80p' memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl
```

#### How to verify memory worked

There are two practical checks:

1. Verify that the memory file exists and contains entries:

```bash
wc -l memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl
sed -n '1,120p' memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl
```

2. Verify that a follow-up task received memory in its system prompt by checking
   the trajectory file:

```bash
python3 - <<'PY'
import json
traj = "SWE-Bench/results_vllm_reasoningbank_task2/astropy__astropy-14182/astropy__astropy-14182.traj.json"
obj = json.load(open(traj))
system = obj["messages"][0]["content"]
print("Below are some memory items" in system)
print(system[:2000])
PY
```

If the first printed value is `True`, the second task was started with retrieved
ReasoningBank memory.

#### Evaluate runs without memory

For a run without memory, submit the resulting `preds.json` with `sb-cli`:

```bash
sb-cli submit swe-bench_lite test \
  --predictions_path /Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_two_tasks_no_memory/preds.json \
  --run_id vllm_qwen_no_memory_two_tasks
```

Or run local evaluation:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path /Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_two_tasks_no_memory/preds.json \
  --max_workers 1 \
  --run_id vllm_qwen_no_memory_two_tasks
```

#### Evaluate runs with memory

Evaluation uses the same `preds.json` output format. For example:

```bash
sb-cli submit swe-bench_lite test \
  --predictions_path /Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_reasoningbank_task2/preds.json \
  --run_id vllm_qwen_reasoningbank_task2
```

or locally:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path /Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_reasoningbank_task2/preds.json \
  --max_workers 1 \
  --run_id vllm_qwen_reasoningbank_task2
```

#### TODO

* Add ChromaDB vector retrieval for ReasoningBank memory
* Add `BAAI/bge-m3` embeddings for memory selection
* Add cleaner instructions for running evaluations without memory
* Add cleaner instructions for running evaluations with memory

## Acknowledgement
We adopt code from the following code repositories. We sincerely appreciate these
great work/codebases:

- [Agent-workflow-memory](https://github.com/zorazrw/agent-workflow-memory)
- [webarena](https://github.com/web-arena-x/webarena)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)

## 📚 Citation
If you find this work useful, please kindly cite our paper:
```
@inproceedings{
  ouyang2026reasoningbank,
  title={ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory},
  author={Siru Ouyang and Jun Yan and I-Hung Hsu and Yanfei Chen and Ke Jiang and Zifeng Wang and Rujun Han and Long Le and Samira Daruki and Xiangru Tang and Vishy Tirumalashetty and George Lee and Mahsan Rofouei and Hangfei Lin and Jiawei Han and Chen-Yu Lee and Tomas Pfister},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=jL7fwchScm}
}
```

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

This project is intended for demonstration purposes only. It is not intended for use in a production environment.
