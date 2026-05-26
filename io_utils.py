from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def pair_key(sample_id: str, response_index: int, strategy: str) -> str:
    return f"{sample_id}__r{int(response_index):02d}__{strategy}"


def chunk_key(pair_id: str, chunk_id: int) -> str:
    return f"{pair_id}__c{int(chunk_id):04d}"


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def discover_runs(run_root_str: str) -> list[str]:
    run_root = Path(run_root_str)
    if not run_root.exists():
        return []
    candidates = [
        p
        for p in run_root.iterdir()
        if p.is_dir()
        and (p / "metadata.json").exists()
        and (p / "samples.jsonl").exists()
        and (p / "predictions.jsonl").exists()
        and (p / "pair_summary.jsonl").exists()
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(item) for item in candidates]
