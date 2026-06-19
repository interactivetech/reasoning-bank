# SWE-Bench 10% Baseline Result

## Setup

Condition:

* baseline
* no memory
* local vLLM model
* SWE-Bench Lite test split
* target sample size: 30 tasks

Evaluation report:

* [`openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_no_memory_merged.json`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/openai__Qwen__Qwen3.6-35B-A3B-FP8.vllm_10pct_no_memory_merged.json)

Merged predictions file:

* [`SWE-Bench/results_vllm_10pct_no_memory_merged.json`](/Users/mendeza/Documents/2026_research_projects/reasoning-bank/SWE-Bench/results_vllm_10pct_no_memory_merged.json)

## Headline Numbers

* submitted tasks: 30
* evaluated tasks with non-empty patches: 18
* resolved: 12
* unresolved: 6
* empty-patch tasks: 12
* errors: 0

## Interpretation

There are two valid ways to summarize this result:

### Patch-conditioned success rate

Among tasks that produced a non-empty patch:

* `12 / 18 = 66.7%`

This measures solution quality only on evaluable outputs.

### End-to-end success rate

Across the full 30-task sample:

* `12 / 30 = 40.0%`

This is the stricter end-to-end baseline because it includes tasks that failed
to produce a usable patch.

### Coverage rate

Fraction of attempted tasks that produced an evaluable patch:

* `18 / 30 = 60.0%`

### Empty-patch rate

Fraction of attempted tasks that produced no usable patch:

* `12 / 30 = 40.0%`

## Resolved Tasks

* `astropy__astropy-12907`
* `astropy__astropy-14995`
* `astropy__astropy-6938`
* `django__django-10914`
* `django__django-11001`
* `django__django-11039`
* `django__django-11049`
* `django__django-11099`
* `django__django-11133`
* `django__django-11583`
* `django__django-11815`
* `django__django-11999`

## Unresolved Tasks

* `astropy__astropy-14182`
* `astropy__astropy-14365`
* `astropy__astropy-7746`
* `django__django-11422`
* `django__django-11742`
* `django__django-11848`

## Empty-Patch Tasks

* `django__django-10924`
* `django__django-11019`
* `django__django-11179`
* `django__django-11283`
* `django__django-11564`
* `django__django-11620`
* `django__django-11630`
* `django__django-11797`
* `django__django-11905`
* `django__django-11910`
* `django__django-11964`
* `django__django-12113`

## Recommendation

If this baseline is compared against ReasoningBank, report both:

* patch-conditioned success: `12/18`
* end-to-end success: `12/30`

Otherwise the comparison can be misleading.

For a stronger baseline, rerun the baseline with a higher agent turn budget so
the empty-patch count is reduced.
