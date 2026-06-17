# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import random
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Optional
from collections import Counter

import os

import typer
import yaml
from datasets import load_dataset
from jinja2 import Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, load_config
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

try:
    from minisweagent.memory.memory_management import select_memory
except ImportError:
    select_memory = None

try:
    from minisweagent.memory.chroma_retrieval import (
        query_memory_records_raw,
        rebuild_collection_from_jsonl,
        upsert_memory_record,
    )
except ImportError:
    query_memory_records_raw = None
    rebuild_collection_from_jsonl = None
    upsert_memory_record = None

from minisweagent.memory.instruction import SUCCESSFUL_SI, FAILED_SI
from minisweagent.memory.telemetry import append_memory_event

try:
    from google import genai
    from google.genai.types import HttpOptions, GenerateContentConfig

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
except ImportError:
    genai = None
    GenerateContentConfig = None
    client = None

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))

def _tokenize_for_memory(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _memory_safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_name)


def _score_memory_item(cur_query: str, item: dict) -> float:
    query_terms = Counter(_tokenize_for_memory(cur_query))
    memory_text = " ".join(
        [item.get("query", "")]
        + item.get("memory_items", [])
    )
    memory_terms = Counter(_tokenize_for_memory(memory_text))
    if not query_terms or not memory_terms:
        return 0.0
    overlap = sum(min(query_terms[t], memory_terms[t]) for t in query_terms.keys() & memory_terms.keys())
    return overlap / max(sum(query_terms.values()), 1)


def _memory_item_texts(memory_items: list[str]) -> list[str]:
    return [str(item).strip() for item in memory_items if str(item).strip()]


def _memory_quality_features(memory_items: list[str]) -> dict[str, object]:
    items = _memory_item_texts(memory_items)
    text = "\n".join(items)
    lower = text.lower()
    has_memory_item_headers = "# memory item" in lower
    has_title_headers = "## title" in lower
    has_description_headers = "## description" in lower
    has_content_headers = "## content" in lower
    has_summary_headers = "## summary" in lower
    has_code_fence = "```" in text
    has_shell_commands = (
        "```bash" in lower
        or "cd /testbed" in lower
        or "git diff --cached" in lower
        or "rm -f test_" in lower
        or "echo complete_task_and_submit_final_output" in lower
    )
    has_submit_chatter = (
        "complete_task_and_submit_final_output" in lower
        or "let me submit the final output" in lower
        or "the implementation is complete and all tests pass" in lower
    )
    has_timeout_trace = (
        "timeout of 300 seconds" in lower
        or "terminated because it took a timeout" in lower
        or "hint: try kill -9" in lower
    )
    has_file_paths = "/testbed/" in text or "astropy/" in text
    is_empty = not bool(items)
    looks_structured = bool(
        (has_memory_item_headers and has_title_headers and has_description_headers and has_content_headers)
        or has_summary_headers
    )
    avg_item_length = (sum(len(item) for item in items) / len(items)) if items else 0.0
    return {
        "item_count": len(items),
        "is_empty": is_empty,
        "looks_structured": looks_structured,
        "has_memory_item_headers": has_memory_item_headers,
        "has_title_headers": has_title_headers,
        "has_description_headers": has_description_headers,
        "has_content_headers": has_content_headers,
        "has_summary_headers": has_summary_headers,
        "has_code_fence": has_code_fence,
        "has_shell_commands": has_shell_commands,
        "has_submit_chatter": has_submit_chatter,
        "has_timeout_trace": has_timeout_trace,
        "has_file_paths": has_file_paths,
        "avg_item_length": round(avg_item_length, 2),
        "char_len": len(text),
    }


def _memory_quality_labels(features: dict[str, object]) -> list[str]:
    labels: list[str] = []
    if features["is_empty"]:
        labels.append("empty")
    if not features["looks_structured"]:
        labels.append("unstructured")
    if features["has_shell_commands"]:
        labels.append("procedural_shell")
    if features["has_submit_chatter"]:
        labels.append("procedural_submit")
    if features["has_timeout_trace"]:
        labels.append("trace_timeout")
    return labels


def _ensure_chroma_collection(memory_config: dict, memory_path: Path, reasoning_bank: list[dict]) -> None:
    """Rebuild ChromaDB collection from JSONL if it doesn't exist yet."""
    persist_directory = memory_config.get("chroma_path", str(memory_path.parent / "chroma" / "swebench"))
    collection_name = memory_config.get("collection_name", "reasoningbank_swebench")
    model_path = memory_config.get("embedding_model", "BAAI/bge-m3")
    chroma_dir = Path(persist_directory)
    if not chroma_dir.exists():
        chroma_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ChromaDB directory created at %s, rebuilding from JSONL.", chroma_dir)
        if rebuild_collection_from_jsonl is not None:
            rebuild_collection_from_jsonl(
                jsonl_path=str(memory_path),
                persist_directory=persist_directory,
                collection_name=collection_name,
                model_path=model_path,
            )


def select_memory_for_task(
    reasoning_bank: list[dict],
    cur_query: str,
    task_id: str,
    cache_path: str,
    memory_config: dict,
    memory_path: Optional[Path] = None,
) -> list[dict]:
    backend = memory_config.get("retrieval_backend", "lexical")
    top_k = int(memory_config.get("top_k", 1))
    if not reasoning_bank:
        return []

    if backend == "chromadb":
        persist_directory = memory_config.get("chroma_path", str(memory_path.parent / "chroma" / "swebench") if memory_path else "./memory/chroma/swebench")
        collection_name = memory_config.get("collection_name", "reasoningbank_swebench")
        model_path = memory_config.get("embedding_model", "BAAI/bge-m3")

        # Ensure collection exists / rebuild if needed
        if memory_path is not None:
            _ensure_chroma_collection(memory_config, memory_path, reasoning_bank)

        if query_memory_records_raw is None:
            logger.warning("ChromaDB retrieval requested but chromadb_retrieval is not installed. Falling back to lexical.")
        else:
            chroma_results = query_memory_records_raw(
                query_text=cur_query,
                persist_directory=persist_directory,
                collection_name=collection_name,
                top_k=top_k,
            )
            if chroma_results:
                # Map chroma results back to reasoning_bank entries
                out = []
                chroma_task_ids = {r["task_id"] for r in chroma_results}
                for item in reasoning_bank:
                    if item.get("task_id") in chroma_task_ids:
                        out.append(item)
                if out:
                    return out

    if backend == "embedding":
        if select_memory is None:
            raise RuntimeError("Embedding retrieval requested, but memory dependencies are not installed.")
        return select_memory(
            top_k,
            reasoning_bank=reasoning_bank,
            cur_query=cur_query,
            task_id=task_id,
            cache_path=cache_path,
            prefer_model=memory_config.get("embedding_model", "Qwen"),
        )

    ranked = sorted(
        reasoning_bank,
        key=lambda item: _score_memory_item(cur_query, item),
        reverse=True,
    )
    return ranked[:top_k]


def model_generate_text(model, prompt: str, *, system_instruction: str | None = None, temperature: float = 0.0) -> str:
    response = model.query(
        [
            *([{"role": "system", "content": system_instruction}] if system_instruction else []),
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=2048,
    )
    return response["content"]


def llm_generate(prompt: str, model, verbose: bool = False, si: str | None = None) -> list[str]:
    """Generate memory items using the configured model backend."""
    if verbose:
        print("Prompt:\n", prompt, "\n\n")
    response = model_generate_text(model, prompt, system_instruction=si.strip() if si else None, temperature=1.0)
    if verbose:
        print(response)
    return [block.strip() for block in response.split("\n\n") if block.strip()]

def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    memory_model = None
    task = instance["problem_statement"]
    memory_config = config.get("memory", {})
    use_memory = memory_config.get("enabled", True)
    memory_model_key = _memory_safe_model_name(model.config.model_name)
    memory_dir = Path("./memory")
    memory_dir.mkdir(exist_ok=True)
    memory_path = memory_dir / f"{memory_model_key}.jsonl"
    telemetry_path = memory_dir / f"{memory_model_key}_telemetry.jsonl"
    embeddings_cache_path = memory_dir / f"{memory_model_key}_embeddings.jsonl"
    selected_memory = ""
    if use_memory:
        memory_model = get_model(config=config.get("model", {}))
        if not memory_path.exists():
            memory_path.write_text("")

        with open(memory_path, "r") as f:
            memory_bank = [json.loads(line) for line in f.readlines()]

        res = select_memory_for_task(
            reasoning_bank=memory_bank,
            cur_query=task,
            task_id=instance_id,
            cache_path=str(embeddings_cache_path),
            memory_config=memory_config,
            memory_path=memory_path,
        )

        append_memory_event(
            telemetry_path,
            {
                "event": "retrieval",
                "task_id": instance_id,
                "backend": memory_config.get("retrieval_backend", "lexical"),
                "memory_bank_size": len(memory_bank),
                "selected_count": len(res),
                "selected_task_ids": [item.get("task_id") for item in res],
                "selected_quality_features": [
                    _memory_quality_features(item.get("memory_items", []))
                    for item in res
                ],
                "selected_quality_labels": [
                    _memory_quality_labels(_memory_quality_features(item.get("memory_items", [])))
                    for item in res
                ],
            },
        )

        if res:
            mem_items = []
            for item in res:
                for i in item["memory_items"]:
                    mem_items.append(i)
            selected_memory = "\n\n".join(mem_items)
    

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    extra_info = None

    try:
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(task, selected_memory=selected_memory)
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)

        if use_memory:
            # read trajectory and extract memory
            with open(instance_dir / f"{instance_id}.traj.json", "r") as f:
                messages = json.load(f)["messages"]
            trajectory = "\n".join([m["content"] for m in messages if m["role"] != "system"])
            status = llm_judge_status(task, trajectory, memory_model or model)

            trajectory = f"**Query:** {task}\n\n**Trajectory:**\n{trajectory}"
            if status:
                generated_memory_item = llm_generate(trajectory, memory_model or model, False, si=SUCCESSFUL_SI)
            else:
                generated_memory_item = llm_generate(trajectory, memory_model or model, False, si=FAILED_SI)
            quality_features = _memory_quality_features(generated_memory_item)
            quality_labels = _memory_quality_labels(quality_features)

            with open(memory_path, "a") as f:
                f.write(json.dumps({
                    "task_id": instance_id,
                    "query": task,
                    "memory_items": generated_memory_item,
                    "status": "success" if status else "fail",
                }) + "\n")

            append_memory_event(
                telemetry_path,
                {
                    "event": "write",
                    "task_id": instance_id,
                    "status": "success" if status else "fail",
                    "memory_bank_size_before": len(memory_bank),
                    "quality_features": quality_features,
                    "quality_labels": quality_labels,
                    "memory_preview": _memory_item_texts(generated_memory_item)[:3],
                },
            )

            # Upsert into ChromaDB (non-blocking, don't crash on failure)
            if upsert_memory_record is not None:
                try:
                    persist_directory = memory_config.get("chroma_path", str(memory_path.parent / "chroma" / "swebench"))
                    collection_name = memory_config.get("collection_name", "reasoningbank_swebench")
                    upsert_memory_record(
                        record={
                            "task_id": instance_id,
                            "query": task,
                            "memory_items": generated_memory_item,
                            "status": "success" if status else "fail",
                        },
                        persist_directory=persist_directory,
                        collection_name=collection_name,
                        model_path=memory_config.get("embedding_model", "BAAI/bge-m3"),
                    )
                except Exception:
                    logger.warning("ChromaDB upsert failed for %s (non-fatal)", instance_id, exc_info=True)


def llm_judge_status(task: str, trajectory: str, model) -> str:
    prompt = f"Task: {task}\n\nTrajectory:\n{trajectory}\n\nDid the agent successfully complete the task? Answer with 'success' or 'fail' only."
    response = model_generate_text(
        model,
        prompt,
        system_instruction="You are a helpful assistant that judges whether the agent successfully completed the task.",
        temperature=0.0,
    ).strip().lower()
    if "success" in response:
        return True
    else:
        return False


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")


    config = load_config(config_spec)
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
