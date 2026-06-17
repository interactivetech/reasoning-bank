# Local Memory Retrieval Plan

## Goal

Extend the current ReasoningBank codebase so that SWE-Bench and WebArena can use
local vector retrieval with:

* `chromadb` as the vector store
* `BAAI/bge-m3` as the embedding model

The implementation should preserve the current ReasoningBank workflow:

1. run a task
2. write memory items after the run
3. retrieve prior memory items before the next run

This plan is specifically intended for a local agent implementation pass using
`Qwen/Qwen3.6-35B-A3B-FP8`, and should minimize architectural drift from the
existing repo.

## Constraints

* Do not remove the existing JSONL memory format. Keep it as the source-of-truth
  artifact for memory items.
* Add ChromaDB as an additional retrieval backend, not a full rewrite.
* Do not require Gemini / Vertex AI for retrieval.
* Keep the current non-memory path working.
* Keep the current ReasoningBank prompting and post-run memory generation flow
  intact unless a bug fix is required.

## Current code paths

### SWE-Bench

Current memory orchestration is in:

* `third_party/src/minisweagent/run/extra/swebench.py`
* `third_party/src/minisweagent/memory/memory_management.py`
* `third_party/src/minisweagent/memory/instruction.py`

Relevant behavior:

* pre-run memory retrieval happens in `process_instance(...)`
* post-run memory generation happens after the trajectory is saved
* memory is appended to a local JSONL file under `memory/`

### WebArena

Current memory orchestration is in:

* `WebArena/run.py`
* `WebArena/memory_management.py`
* `WebArena/pipeline_memory.py`
* `WebArena/induce_memory.py`

Relevant behavior:

* pre-run retrieval happens before agent execution
* memory items are stored in JSONL
* retrieval currently assumes Gemini or Qwen embedding logic inside
  `memory_management.py`

## Desired architecture

Add a shared local retrieval backend with the following responsibilities:

1. embed memory documents with `BAAI/bge-m3`
2. persist vectors and metadata in a ChromaDB collection
3. query top-k relevant memories for a new task
4. keep the JSONL memory file in sync with the vector store

Important design decision:

* JSONL remains append-only memory storage
* ChromaDB is a derived index for retrieval

That means:

* memory generation still writes JSONL entries first
* indexing code reads JSONL entries and upserts them into ChromaDB
* retrieval returns the original JSONL-backed memory records

## Implementation scope

Implement a new retrieval backend named `chromadb`.

The backend should support:

* local persistence on disk
* embedding with `BAAI/bge-m3`
* querying by task text
* returning top-k memory entries

Keep the following backend choices available:

* `lexical`
* `embedding`
* `chromadb`

The existing temporary `lexical` backend in
`third_party/src/minisweagent/run/extra/swebench.py` can remain for debugging,
but `chromadb` should become the preferred local backend.

## New dependencies

Add the following Python dependencies:

* `chromadb`
* `sentence-transformers`

If `BAAI/bge-m3` works better in the repo’s environment via Hugging Face
Transformers directly than via `sentence-transformers`, that is acceptable, but
the final model used should still be `BAAI/bge-m3`.

## File-level plan

### 1. Add a shared local retrieval module

Create a new module, preferably at one of these locations:

* `third_party/src/minisweagent/memory/chroma_retrieval.py`
* optionally mirror or reuse it from WebArena instead of duplicating logic

The module should expose a small API like:

* `get_embedding_model(...)`
* `get_chroma_collection(...)`
* `upsert_memory_record(...)`
* `query_memory_records(...)`
* `rebuild_collection_from_jsonl(...)`

The module should not know about SWE-Bench-specific or WebArena-specific task
logic. It should operate on generic memory records.

### 2. Define the document format for ChromaDB

For each JSONL memory entry, store:

* `id`: stable unique id
  Example: `<task_id>#<memory_index>` or `<task_id>`
* `document`: text used for embedding and retrieval
* `metadata`:
  * `task_id`
  * `status`
  * `query`
  * `source_jsonl`

Recommended `document` format:

```text
Query:
<original query>

Memory:
<joined memory items>
```

Do not embed only the raw query. The memory content needs to influence retrieval.

### 3. Keep JSONL as source of truth

When a new memory entry is written after a run:

1. append JSON to the existing JSONL file
2. immediately upsert the same entry into ChromaDB

When the vector store is missing or stale:

* rebuild it from the JSONL file

This avoids silent divergence between the index and the stored memories.

### 4. Update SWE-Bench retrieval

Modify:

* `third_party/src/minisweagent/run/extra/swebench.py`

Current state:

* retrieval backend selection happens in `select_memory_for_task(...)`

Required changes:

* keep `lexical` support
* keep `embedding` support only if it still works locally
* add `chromadb` support
* when `memory.retrieval_backend == "chromadb"`, query the ChromaDB collection
  instead of the lexical scorer

The function should still return the original memory entry objects in the same
shape expected by:

* `selected_memory = "\n\n".join(mem_items)`

Do not change the agent prompt interface.

### 5. Update SWE-Bench writeback

Modify:

* `third_party/src/minisweagent/run/extra/swebench.py`

After memory generation and JSONL append:

* call the new ChromaDB upsert helper

If the JSONL append succeeds but ChromaDB upsert fails:

* log a warning
* do not crash the whole task result

The task result is more important than immediate index freshness.

### 6. Update WebArena retrieval

Modify:

* `WebArena/memory_management.py`
* possibly `WebArena/run.py`

Required changes:

* add the same `chromadb` backend option
* reuse the same Chroma retrieval helper if practical
* preserve the current output shape from `select_memory(...)`

Avoid keeping two different Chroma implementations unless forced by import-path
constraints.

### 7. Update WebArena writeback

Modify:

* `WebArena/induce_memory.py`

After writing a new memory JSONL line:

* upsert the entry into the Chroma collection

### 8. Config support

Extend the YAML config pattern already used in SWE-Bench so memory config can
declare:

```yaml
memory:
  enabled: true
  retrieval_backend: chromadb
  top_k: 1
  chroma_path: ./memory/chroma
  collection_name: reasoningbank_swebench
  embedding_model: BAAI/bge-m3
```

For WebArena, expose the equivalent settings via args or a small config object.

## Embedding model guidance

Use `BAAI/bge-m3` consistently for both:

* indexing memory entries
* querying new tasks

Implementation notes:

* normalize embeddings before inserting/querying if the library path does not do
  it by default
* keep one shared model instance per process
* avoid reloading the model repeatedly inside loops

If batching is easy, batch indexing calls. Query-time can remain single-query.

## Chroma persistence layout

Recommended paths:

* SWE-Bench:
  * `memory/chroma/swebench/`
* WebArena:
  * `memory/chroma/webarena/`

Recommended collection names:

* `reasoningbank_swebench`
* `reasoningbank_webarena`

Do not mix SWE-Bench and WebArena memories in the same collection unless there
is an explicit config option to do so.

## Handling old memory data

The code should work when:

* the JSONL file already exists
* the Chroma directory does not exist

In that case:

* auto-create the Chroma directory
* rebuild the collection from JSONL

Add a simple rebuild path instead of forcing manual cleanup.

## Verification checklist

### SWE-Bench

1. Run task 1 with memory enabled and confirm:
   * task output is written
   * JSONL memory file is written
   * Chroma collection exists on disk

2. Run task 2 with memory enabled and confirm:
   * the second task’s trajectory includes the memory banner
   * retrieved memory is sourced from the first task

3. Confirm the no-memory config still works.

### WebArena

1. Run one task with memory enabled and confirm:
   * JSONL memory is written
   * Chroma collection is updated

2. Run a second related task and confirm:
   * memory is injected into the prompt

## Acceptance criteria

The implementation is complete when:

* SWE-Bench can run with `memory.retrieval_backend: chromadb`
* WebArena can run with the same local retrieval backend
* no Gemini / Vertex AI dependency is required for retrieval
* `BAAI/bge-m3` is used for local embedding
* memory remains stored in JSONL
* ChromaDB persists and serves retrieval across runs
* existing no-memory runs still work

## Suggested implementation order

1. Add dependency entries
2. Add shared Chroma retrieval helper
3. Wire helper into SWE-Bench writeback
4. Wire helper into SWE-Bench retrieval
5. Verify two-task SWE-Bench flow
6. Wire helper into WebArena writeback
7. Wire helper into WebArena retrieval
8. Verify two-task WebArena flow
9. Clean up config docs

## Non-goals

Do not do these in the first pass:

* migrate all memory storage away from JSONL
* redesign the ReasoningBank prompts
* add reranking
* add hybrid lexical + vector retrieval
* add a remote vector database

## Notes for the agent implementing this

* Prefer minimal code movement.
* Preserve current function signatures where possible.
* Add ChromaDB as an additive backend, not a replacement architecture.
* If you must choose between elegance and compatibility with the current
  ReasoningBank code, choose compatibility.
