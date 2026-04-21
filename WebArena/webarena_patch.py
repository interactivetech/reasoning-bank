# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monkey-patches webarena's llm_fuzzy_match with MemEvol's lenient prompt.

Import this module before any webarena evaluation code runs.  It replaces the
default "semantically equivalent" grading prompt with a containment-based one
that does not penalise the student for providing extra correct information.

Usage:
    import webarena_patch  # noqa: F401  (side-effect import)
"""

import webarena.evaluation_harness.helper_functions as _hf
import webarena.evaluation_harness.evaluators as _ev


def _memevol_llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """MemEvol-style fuzzy match judge (containment-based, lenient N/A)."""
    message = (
        "Help a teacher to grade the answer of a student given a question. "
        "The reference answer below is ONE part of the full expected answer. "
        "Your job is to check whether the student's answer CONTAINS this specific "
        "piece of information, even if the student also provides additional correct information.\n"
    )
    message += (
        "If the student's answer contains the reference information (possibly with "
        "different phrasing or wording), judge it as correct. "
        "Do NOT penalize the student for including extra information beyond the reference.\n"
    )
    message += f"question: {question}\n"
    message += f"reference answer (one part): {reference}\n"
    message += (
        "Important: The string 'N/A' means the task has no meaningful result "
        "(e.g., querying orders for a month with no orders). Both 'N/A' and "
        "equivalent answers like '0', '$0.00', 'none', or 'no results' should be "
        "treated as correct if the reference is 'N/A'.\n"
    )
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by correct/incorrect/partially correct."

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]
    response = _hf._generate_from_gemini(messages, temperature=0, max_tokens=768).lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response, f"Unexpected LLM judge response: {response!r}"
        return 1.0


# Apply patch to both modules so all call sites are covered.
_hf.llm_fuzzy_match = _memevol_llm_fuzzy_match
_ev.llm_fuzzy_match = _memevol_llm_fuzzy_match
