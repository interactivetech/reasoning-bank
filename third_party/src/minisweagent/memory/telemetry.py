from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_memory_event(path: str | Path, event: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": _utc_now_iso(), **event}
    with out_path.open("a") as f:
        f.write(json.dumps(payload) + "\n")
