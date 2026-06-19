# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared ChromaDB retrieval backend for ReasoningBank memory.

Uses ``BAAI/bge-m3`` (via sentence-transformers) for local embeddings
and ChromaDB for vector persistence.  JSONL remains the source of truth;
ChromaDB is a derived index that is kept in sync.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton guard – the embedding model is loaded once per process.
# ---------------------------------------------------------------------------
_model: Optional[Any] = None
_tokenizer: Optional[Any] = None
_model_lock = threading.Lock()
_model_path: str = "BAAI/bge-m3"


def _load_embedding_model(model_path: Optional[str] = None) -> Tuple[Any, Any]:
    """Lazy-load the shared embedding tokenizer/model."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    path = model_path or _model_path
    with _model_lock:
        # Double-check after acquiring the lock
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model
        logger.info("Loading embedding model %s …", path)
        _tokenizer = AutoTokenizer.from_pretrained(path)
        _model = AutoModel.from_pretrained(path)
        _model.eval()
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# ChromaDB collection helpers
# ---------------------------------------------------------------------------

# Internal cache of (persist_path, collection_name) -> Collection
_collection_cache: Dict[Tuple[str, str], Any] = {}
_collection_lock = threading.Lock()


def get_chroma_collection(
    persist_directory: str,
    collection_name: str,
) -> Any:
    """Return (or create) a ChromaDB collection backed by *persist_directory*."""
    key = (persist_directory, collection_name)
    with _collection_lock:
        if key in _collection_cache:
            return _collection_cache[key]

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        _collection_cache[key] = collection
        return collection


def _make_document(query: str, memory_items: List[str]) -> str:
    if memory_items:
        return f"Query:\n{query}\n\nMemory:\n" + "\n".join(memory_items)
    return f"Query:\n{query}"


def _encode_texts(texts: List[str], model_path: Optional[str] = None) -> List[List[float]]:
    tokenizer, model = _load_embedding_model(model_path)
    with torch.no_grad():
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        outputs = model(**batch)
        hidden = outputs.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).bool()
        hidden = hidden.masked_fill(~mask, 0.0)
        pooled = hidden.sum(dim=1) / batch["attention_mask"].sum(dim=1, keepdim=True)
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.cpu().tolist()


def rebuild_collection_from_jsonl(
    jsonl_path: str,
    persist_directory: str,
    collection_name: str,
    model_path: Optional[str] = None,
) -> None:
    """Rebuild the entire Chroma collection from a JSONL memory file.

    Each JSONL line must be a dict with keys ``task_id``, ``query``,
    ``memory_items`` (list[str]), and optionally ``status``.
    """
    jsonl_p = Path(jsonl_path)
    if not jsonl_p.exists():
        logger.warning("JSONL file not found: %s", jsonl_path)
        return

    collection = get_chroma_collection(persist_directory, collection_name)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    with open(jsonl_p, "r") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task_id = entry.get("task_id", "unknown")
            query = entry.get("query", "")
            mem_items = entry.get("memory_items", [])
            status = entry.get("status", "")

            doc = _make_document(query, mem_items)

            rec_id = f"{task_id}"
            ids.append(rec_id)
            documents.append(doc)
            metadatas.append({
                "task_id": task_id,
                "status": status,
                "query": query,
                "memory_items_json": json.dumps(mem_items),
                "source_jsonl": str(jsonl_p),
            })

    if not ids:
        logger.warning("No records to upsert from %s", jsonl_path)
        return

    embeddings = _encode_texts(documents, model_path)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info("Rebuilt Chroma collection '%s' with %d records from %s",
                collection_name, len(ids), jsonl_path)


# ---------------------------------------------------------------------------
# Upsert a single memory record
# ---------------------------------------------------------------------------

def upsert_memory_record(
    record: Dict[str, Any],
    persist_directory: str,
    collection_name: str,
    model_path: Optional[str] = None,
) -> None:
    """Upsert a single JSONL-backed memory record into ChromaDB.

    Parameters
    ----------
    record : dict
        A memory dict with keys ``task_id``, ``query``, ``memory_items``,
        and optionally ``status``.
    persist_directory : str
        Disk path for ChromaDB persistence.
    collection_name : str
        Name of the ChromaDB collection.
    model_path : str, optional
        Override the default ``BAAI/bge-m3`` model path.
    """
    task_id = record.get("task_id", "unknown")
    query = record.get("query", "")
    mem_items = record.get("memory_items", [])
    status = record.get("status", "")

    doc = _make_document(query, mem_items)

    rec_id = f"{task_id}"
    metadata = {
        "task_id": task_id,
        "status": status,
        "query": query,
        "memory_items_json": json.dumps(mem_items),
    }

    collection = get_chroma_collection(persist_directory, collection_name)
    embedding = _encode_texts([doc], model_path)[0]
    collection.upsert(ids=[rec_id], documents=[doc], metadatas=[metadata], embeddings=[embedding])
    logger.debug("Upserted record %s into collection '%s'", rec_id, collection_name)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_memory_records(
    query_text: str,
    persist_directory: str,
    collection_name: str,
    top_k: int = 1,
    model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query the ChromaDB collection and return the top-k matching memory
    records (the original JSONL dicts, not just Chroma metadata).

    Returns
    -------
    list[dict]
        Each dict contains at least ``task_id``, ``query``, ``memory_items``,
        ``status``, and the Chroma-provided ``distance``.
    """
    collection = get_chroma_collection(persist_directory, collection_name)
    query_embedding = _encode_texts([query_text], model_path)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        return []

    records: List[Dict[str, Any]] = []
    for i, rec_id in enumerate(ids):
        meta = metadatas[i] if metadatas else {}
        if meta is None:
            meta = {}
        dist = distances[i] if distances else None
        try:
            memory_items = json.loads(meta.get("memory_items_json", "[]"))
        except Exception:
            memory_items = []

        records.append({
            "task_id": meta.get("task_id", rec_id),
            "query": meta.get("query", ""),
            "memory_items": memory_items,
            "status": meta.get("status", ""),
            "distance": float(dist) if dist is not None else None,
            "chroma_id": rec_id,
        })

    return records


def query_memory_records_raw(
    query_text: str,
    persist_directory: str,
    collection_name: str,
    top_k: int = 1,
    model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query ChromaDB and return raw results with Chroma metadata.

    This is a convenience wrapper that returns the full Chroma metadata
    including ``memory_items`` stored alongside each record, so callers
    can get the original memory content back.

    Returns a list of dicts with keys ``task_id``, ``query``, ``memory_items``,
    ``distance``, ``chroma_id``.
    """
    collection = get_chroma_collection(persist_directory, collection_name)
    query_embedding = _encode_texts([query_text], model_path)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        return []

    records: List[Dict[str, Any]] = []
    for i, rec_id in enumerate(ids):
        meta = metadatas[i] if metadatas else {}
        if meta is None:
            meta = {}
        dist = distances[i] if distances else None
        try:
            memory_items = json.loads(meta.get("memory_items_json", "[]"))
        except Exception:
            memory_items = []

        records.append({
            "task_id": meta.get("task_id", rec_id),
            "query": meta.get("query", ""),
            "memory_items": memory_items,
            "status": meta.get("status", ""),
            "distance": float(dist) if dist is not None else None,
            "chroma_id": rec_id,
        })

    return records
