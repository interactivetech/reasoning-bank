# Memory-Aware Test-Time Scaling Gap

## Purpose

This note maps the gap between the current repository implementation and a
full memory-aware test-time scaling setup consistent with the ReasoningBank
paper framing.

The goal is not to redesign the repo. The goal is to identify what is already
implemented, what is missing, and what would need to be added with minimal
architectural disruption.

## Short Answer

The repository currently has:

* memory retrieval before inference
* memory induction after inference
* a partial multi-trial scaling workflow for WebArena

The repository does not currently have:

* a true test-time scaling implementation for SWE-Bench
* a general benchmark-agnostic scaling abstraction
* explicit best-of-N selection, voting, or reranking for final task outputs
* a clean experimental loop that measures how memory and scaling interact

## Current State

### SWE-Bench

Primary file:

* [`third_party/src/minisweagent/run/extra/swebench.py`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/third_party/src/minisweagent/run/extra/swebench.py)

What exists:

* single-run task execution
* optional pre-run memory retrieval
* prompt injection of retrieved memory items
* post-run memory generation
* JSONL memory persistence
* optional ChromaDB-backed retrieval

What does not exist:

* multiple sampled rollouts for the same task
* selection among multiple candidate patches
* patch reranking using execution outcomes or a judge
* self-consistency or majority-vote style aggregation
* scaling-specific metrics or logging

Conclusion:

SWE-Bench currently implements memory augmentation, not memory-aware
test-time scaling.

### WebArena

Primary files:

* [`WebArena/run.py`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/WebArena/run.py)
* [`WebArena/pipeline_memory.py`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/WebArena/pipeline_memory.py)
* [`WebArena/pipeline_scaling.py`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/WebArena/pipeline_scaling.py)
* [`WebArena/induce_scaling.py`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/WebArena/induce_scaling.py)

What exists:

* standard memory retrieval and induction pipeline
* a scaling pipeline that runs multiple trials for the same task
* post-hoc summarization of multiple trajectories into a memory item

What is missing:

* explicit selection of the best trial as the final answer for the task
* a formal policy for ranking competing trajectories
* a reusable scaling controller shared with other benchmarks
* clear separation between:
  * using scaling to solve the current task
  * using scaling outputs to improve future memory

Conclusion:

WebArena contains a partial scaling workflow, but it is not a complete or
general implementation of memory-aware test-time scaling.

## Main Gaps

### 1. No Task-Time Candidate Selection for SWE-Bench

The current SWE-Bench flow generates one trajectory and one final patch per
task. A true scaling setup would generate multiple candidate trajectories or
patches and then choose among them.

Missing capabilities:

* `num_trials` or equivalent per task
* per-trial isolated output directories
* candidate patch collection
* candidate ranking policy
* final chosen patch writeback to `preds.json`

### 2. No General Scoring or Reranking Layer

The repo has memory retrieval, but not a general mechanism to score multiple
candidate solutions for the same task.

Possible scoring dimensions that are currently absent:

* patch applies cleanly
* tests pass
* task-specific evaluator score
* model-as-judge score
* trajectory-level heuristics

Without a reranker or evaluator-backed selector, multiple samples do not
become a real scaling mechanism.

### 3. Memory and Scaling Are Not Coupled in a Closed Loop

The paper framing suggests a bidirectional interaction:

* memory helps improve sampled reasoning at test time
* richer sampled reasoning improves future memory

Today, the repo mostly supports:

* memory retrieval before run
* memory induction after run

What is still missing is an explicit loop where multiple samples are generated,
compared, and then distilled into both:

* the final answer for the current task
* a higher-quality memory artifact for future tasks

### 4. No Benchmark-Independent Scaling Abstraction

The scaling logic is specific to WebArena scripts. SWE-Bench does not share
that machinery.

A fuller implementation would likely introduce a reusable abstraction around:

* sample N candidates
* evaluate or rank them
* choose one candidate as final output
* distill cross-trial memory

Right now, this pattern is not packaged as a shared primitive.

### 5. No Clean Experimental Controls

To validate memory-aware scaling, the repo should support controlled
comparisons such as:

* no memory, one sample
* memory, one sample
* no memory, N samples
* memory, N samples

The current code does not provide a clean, unified experiment runner for that
matrix, especially for SWE-Bench.

## What the Repo Does Have

These parts are already useful building blocks:

* task-conditioned memory retrieval
* memory injection into prompts
* memory induction from completed runs
* local JSONL memory storage
* optional ChromaDB retrieval backend
* WebArena multi-trial orchestration scaffolding

This means the repo is not starting from zero. The main missing piece is the
selection-and-scaling loop around inference.

## Minimal Path To Close the Gap

### For SWE-Bench

Minimum additions:

1. Add `num_trials` support to run the same instance multiple times.
2. Store each trial in an isolated subdirectory.
3. Collect candidate patches from each trial.
4. Score candidates using one or more of:
   * patch non-empty
   * patch applies
   * targeted tests pass
   * model judge score
5. Write only the selected best candidate into the canonical `preds.json`.
6. Distill a memory item from the full set of trials, not just one trial.

### For WebArena

Minimum additions:

1. Promote the existing multi-trial pipeline from memory-distillation-only to
   task-solution selection.
2. Add an explicit trial ranking policy.
3. Record which trial was selected and why.
4. Separate:
   * final task answer selection
   * future memory distillation

### Shared

Minimum shared abstraction:

* `sample_candidates(task, n)`
* `score_candidates(task, candidates)`
* `select_candidate(candidates)`
* `distill_memory(task, candidates, selected_candidate)`

That would let both benchmarks use the same conceptual loop while keeping
benchmark-specific evaluation details separate.

## Recommended Interpretation

If you are evaluating this codebase today, the correct characterization is:

* ReasoningBank-style memory support is present.
* Local-memory retrieval can be integrated without changing the overall design.
* True memory-aware test-time scaling is only partially represented, mainly in
  WebArena orchestration.
* SWE-Bench does not yet implement the scaling part.

## Practical Review Questions

When reviewing future agent work on this area, these are the key questions:

1. Does the implementation produce multiple candidates for the same task?
2. Does it explicitly select a final candidate using a defined policy?
3. Does the selected candidate become the benchmark output?
4. Does the system also distill multi-trial experience into memory?
5. Can the same logic run on both SWE-Bench and WebArena?
6. Can experiments cleanly separate the effect of memory from the effect of
   scaling?

## Bottom Line

The current repo is best described as:

* full memory augmentation
* partial scaling scaffolding
* not yet a full memory-aware test-time scaling system

For SWE-Bench in particular, the scaling component is absent.
