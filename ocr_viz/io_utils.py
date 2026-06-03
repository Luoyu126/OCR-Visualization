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


_RUN_MARKER_FILES = ("metadata.json", "samples.jsonl", "predictions.jsonl", "pair_summary.jsonl")


def _is_run_directory(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in _RUN_MARKER_FILES)


def discover_runs(run_root_str: str, *, max_depth: int = 3) -> list[str]:
    """Find run directories under run_root, including one nested level (e.g. runs/formula_token_passk/<run>)."""
    run_root = Path(run_root_str)
    if not run_root.exists():
        return []

    candidates: list[Path] = []
    if _is_run_directory(run_root):
        candidates.append(run_root)

    def _walk(base: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for child in children:
            if not child.is_dir():
                continue
            if _is_run_directory(child):
                candidates.append(child)
            else:
                _walk(child, depth + 1)

    _walk(run_root, 0)
    unique: dict[str, Path] = {}
    for path in candidates:
        unique[str(path.resolve())] = path
    ordered = sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(item) for item in ordered]
