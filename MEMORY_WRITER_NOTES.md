# Memory Writer Notes

## Context

This repository is the official Google ReasoningBank implementation, so the
current memory-writing behavior should be interpreted as an issue in observed
runtime behavior or benchmark adaptation quality, not as a provenance problem.

The surprising finding is not that the code is unofficial. The surprising
finding is that the observed SWE-Bench memory outputs do not consistently match
the clean structured memory format described in the paper.

## What The Paper Describes

ReasoningBank reports a memory extraction process that:

* self-judges whether a trajectory succeeded or failed
* extracts up to 3 memory items per trajectory
* uses a structured format with:
  * title
  * description
  * content
* aims for concise, non-redundant, generalizable insights rather than task-
  specific chatter

Relevant sources:

* [ReasoningBank arXiv HTML](https://arxiv.org/html/2509.25140v1)
* [ReasoningBank OpenReview PDF](https://openreview.net/pdf?id=jL7fwchScm)

## What We Observed Locally

In the current SWE-Bench runs, the memory writer sometimes produces good
structured reasoning summaries, but it also sometimes writes low-quality
records such as:

* completion chatter
* submit/cleanup shell commands
* timeout observations

Examples:

* [`memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl:2`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl:2)
* [`memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl:3`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/memory/openai__Qwen__Qwen3.6-35B-A3B-FP8.jsonl:3)

Those outputs are not aligned with the intended “generalizable reasoning
strategy” format.

## Likely Interpretation

The most likely explanation is:

* the official implementation provides the intended prompts and pipeline
* the benchmark adaptation plus local model behavior can still yield noisy
  extractions
* the issue is therefore behavioral, not authorship-related

This is consistent with what we saw:

* some records are strong and well-structured
* some records are weak and clearly should not have been stored as memory

## Why This Matters

Poor memory records degrade both:

* retrieval relevance
* downstream prompt quality

This was visible in later SWE-Bench tasks, where noisy memory items were
retrieved and injected into the system prompt.

## Review Questions

When revisiting this later, the right questions are:

1. Is the local model failing to follow the official memory prompt?
2. Is the SWE-Bench adaptation feeding the wrong trajectory text into the
   memory extractor?
3. Should memory records be validated before writeback?
4. Should malformed or low-information memory items be filtered out?
5. Is there any benchmark-specific difference between the paper experiments and
   the local vLLM/Qwen setup that explains the degradation?

## Bottom Line

The repository is official.

The issue is that the observed SWE-Bench memory-writing behavior, under the
current local setup, does not consistently realize the clean structured memory
format described in the paper.
