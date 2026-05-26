from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


TEXT_TASK = "text"
TABLE_TASK = "table"
FORMULA_TASK = "formula"


def read_paths_file(path: Path) -> list[Path]:
    if not path.exists():
        return []
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [Path(item) for item in rows if item]


def normalize_source_basename(path_like: str) -> str:
    name = Path(str(path_like)).name
    for suffix in (".text_only.parquet", ".table_only.parquet", ".parquet"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def extract_image_bytes(raw_images: Any) -> bytes | None:
    images = raw_images if isinstance(raw_images, list) else [raw_images]
    if not images:
        return None
    first = images[0]
    if isinstance(first, dict):
        for key in ("buffer", "bytes", "image", "image_bytes", "data"):
            value = first.get(key)
            if value is None:
                continue
            if isinstance(value, memoryview):
                return value.tobytes()
            if isinstance(value, bytearray):
                return bytes(value)
            if isinstance(value, bytes):
                return value
    if isinstance(first, memoryview):
        return first.tobytes()
    if isinstance(first, bytearray):
        return bytes(first)
    if isinstance(first, bytes):
        return first
    return None


def _decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


@dataclass
class LookupResult:
    image_path: str
    resolved_parquet: str | None
    resolved_row_index: int | None
    lookup_method: str


class DatasetLookup:
    def __init__(
        self,
        *,
        text_list_path: Path,
        table_list_path: Path,
        formula_list_path: Path,
    ) -> None:
        self._task_files: dict[str, list[Path]] = {
            TEXT_TASK: read_paths_file(text_list_path),
            TABLE_TASK: read_paths_file(table_list_path),
            FORMULA_TASK: read_paths_file(formula_list_path),
        }
        self._task_basename_to_files: dict[str, dict[str, list[Path]]] = {}
        for task_type, files in self._task_files.items():
            key_map: dict[str, list[Path]] = {}
            for path in files:
                key = normalize_source_basename(str(path))
                key_map.setdefault(key, []).append(path)
            self._task_basename_to_files[task_type] = key_map

        self._parquet_row_cache: dict[str, dict[int, dict[str, Any]]] = {}
        self._parquet_source_index_cache: dict[str, dict[str, Any]] = {}
        self._formula_gt_index: dict[str, list[tuple[Path, int]]] | None = None
        self._formula_file_offsets: list[tuple[int, int, Path]] | None = None

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "text_files": len(self._task_files.get(TEXT_TASK, [])),
            "table_files": len(self._task_files.get(TABLE_TASK, [])),
            "formula_files": len(self._task_files.get(FORMULA_TASK, [])),
        }

    def _resolve_parquet_from_source(self, task_type: str, source_parquet: str | None) -> Path | None:
        if not source_parquet:
            return None
        key = normalize_source_basename(source_parquet)
        candidates = self._task_basename_to_files.get(task_type, {}).get(key, [])
        if candidates:
            return candidates[0]
        return None

    def _ensure_formula_offsets(self) -> None:
        if self._formula_file_offsets is not None:
            return
        offsets: list[tuple[int, int, Path]] = []
        cursor = 0
        for path in self._task_files.get(FORMULA_TASK, []):
            try:
                row_count = int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                continue
            offsets.append((cursor, cursor + row_count, path))
            cursor += row_count
        self._formula_file_offsets = offsets

    def _ensure_formula_gt_index(self) -> None:
        if self._formula_gt_index is not None:
            return
        index: dict[str, list[tuple[Path, int]]] = {}
        for path in self._task_files.get(FORMULA_TASK, []):
            try:
                pf = pq.ParquetFile(path)
                available_columns = set(pf.schema_arrow.names)
                if "clean_content" not in available_columns:
                    continue
                row_cursor = 0
                for batch in pf.iter_batches(columns=["clean_content"], batch_size=2048):
                    rows = batch.to_pylist()
                    for row in rows:
                        gt_text = self._extract_gt_from_clean_content(row.get("clean_content"))
                        if gt_text:
                            index.setdefault(gt_text, []).append((path, row_cursor))
                        row_cursor += 1
            except Exception:
                continue
        self._formula_gt_index = index

    def _resolve_formula_by_dataset_index(self, dataset_index: int | None) -> tuple[Path | None, int | None]:
        if dataset_index is None:
            return None, None
        self._ensure_formula_offsets()
        if self._formula_file_offsets is None:
            return None, None
        value = int(dataset_index)
        for start, end, path in self._formula_file_offsets:
            if start <= value < end:
                return path, value - start
        return None, None

    def _resolve_formula_by_ground_truth(self, ground_truth: str | None) -> tuple[Path | None, int | None, bool]:
        gt_text = str(ground_truth or "").strip()
        if not gt_text:
            return None, None, False
        self._ensure_formula_gt_index()
        if self._formula_gt_index is None:
            return None, None, False
        candidates = self._formula_gt_index.get(gt_text, [])
        if not candidates:
            return None, None, False
        path, row_idx = candidates[0]
        return path, int(row_idx), len(candidates) == 1

    def _load_row_from_parquet(self, parquet_path: Path, row_index: int) -> dict[str, Any] | None:
        cache_key = str(parquet_path)
        per_file_cache = self._parquet_row_cache.setdefault(cache_key, {})
        if row_index in per_file_cache:
            return per_file_cache[row_index]

        if row_index < 0:
            return None
        try:
            pf = pq.ParquetFile(parquet_path)
            if row_index >= int(pf.metadata.num_rows):
                return None
            available_columns = set(pf.schema_arrow.names)
            preferred_columns = [
                "image_buffer_list",
                "buffer",
                "source_parquet",
                "source_row_idx",
                "source_row_index",
                "block_idx",
                "source_block_idx",
                "clean_content",
                "block_content",
            ]
            columns = [item for item in preferred_columns if item in available_columns]
            current_offset = 0
            for batch in pf.iter_batches(columns=columns or None, batch_size=2048):
                batch_len = len(batch)
                if current_offset + batch_len <= row_index:
                    current_offset += batch_len
                    continue
                local_index = row_index - current_offset
                taken = batch.take(pa.array([local_index], type=pa.int32())).to_pylist()
                if not taken:
                    return None
                row = taken[0]
                per_file_cache[row_index] = row
                return row
        except Exception:
            return None
        return None

    def _load_source_index_for_parquet(self, parquet_path: Path) -> dict[str, Any]:
        cache_key = str(parquet_path)
        if cache_key in self._parquet_source_index_cache:
            return self._parquet_source_index_cache[cache_key]
        out: dict[str, Any] = {
            "rows": [],
            "by_source_row_idx": {},
        }
        try:
            pf = pq.ParquetFile(parquet_path)
            available_columns = set(pf.schema_arrow.names)
            preferred_columns = [
                "source_row_idx",
                "source_row_index",
                "source_parquet",
                "block_idx",
                "source_block_idx",
                "block_index",
                "block_content",
                "clean_content",
            ]
            columns = [item for item in preferred_columns if item in available_columns]
            if not columns:
                self._parquet_source_index_cache[cache_key] = out
                return out
            row_cursor = 0
            for batch in pf.iter_batches(columns=columns, batch_size=2048):
                rows = batch.to_pylist()
                for row in rows:
                    out["rows"].append(row)
                    source_row_idx = row.get("source_row_idx", row.get("source_row_index"))
                    try:
                        key = int(source_row_idx) if source_row_idx is not None else None
                    except Exception:
                        key = None
                    if key is not None:
                        out["by_source_row_idx"].setdefault(key, []).append(row_cursor)
                    row_cursor += 1
        except Exception:
            pass
        self._parquet_source_index_cache[cache_key] = out
        return out

    def _extract_gt_from_clean_content(self, clean_content: Any) -> str:
        try:
            payload = clean_content if isinstance(clean_content, dict) else json.loads(str(clean_content))
            text_items = payload.get("text", "")
            if isinstance(text_items, str):
                text_items = json.loads(text_items)
            if isinstance(text_items, list) and len(text_items) >= 2:
                return str(text_items[1].get("value", "") or "").strip()
        except Exception:
            return ""
        return ""

    def _extract_row_block_idx(self, row: dict[str, Any]) -> int | None:
        value = row.get("block_idx")
        if value is None:
            value = row.get("source_block_idx")
        if value is None:
            value = row.get("block_index")
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    def _resolve_row_index_by_source_fields(
        self,
        *,
        parquet_path: Path,
        source_row_idx: int,
        source_block_idx: int | None,
        source_parquet: str | None,
        ground_truth: str | None,
    ) -> int | None:
        index_payload = self._load_source_index_for_parquet(parquet_path)
        candidates = list(index_payload.get("by_source_row_idx", {}).get(int(source_row_idx), []))
        if not candidates:
            return None
        if len(candidates) == 1:
            return int(candidates[0])

        rows = index_payload.get("rows", [])
        if source_parquet:
            exact_source = [
                idx
                for idx in candidates
                if str(rows[idx].get("source_parquet", "") or "") == str(source_parquet)
            ]
            if len(exact_source) == 1:
                return int(exact_source[0])
            if exact_source:
                candidates = exact_source

        if source_block_idx is not None:
            exact_block = [idx for idx in candidates if self._extract_row_block_idx(rows[idx]) == int(source_block_idx)]
            if len(exact_block) == 1:
                return int(exact_block[0])
            if exact_block:
                candidates = exact_block

        gt_text = str(ground_truth or "").strip()
        if gt_text:
            matched_by_gt: list[int] = []
            for idx in candidates:
                row = rows[idx]
                block_content = str(row.get("block_content", "") or "").strip()
                parsed_gt = self._extract_gt_from_clean_content(row.get("clean_content"))
                if gt_text and (gt_text == block_content or gt_text == parsed_gt):
                    matched_by_gt.append(idx)
            if len(matched_by_gt) == 1:
                return int(matched_by_gt[0])
            if matched_by_gt:
                candidates = matched_by_gt

        return int(candidates[0])

    def materialize_image_for_record(
        self,
        *,
        sample_id: str,
        task_type: str,
        source_parquet: str | None,
        source_row_idx: int | None,
        source_block_idx: int | None,
        dataset_index: int | None,
        ground_truth: str | None,
        images_dir: Path,
    ) -> LookupResult | None:
        image_path = images_dir / f"{sample_id}.png"
        if image_path.exists():
            return LookupResult(
                image_path=str(image_path),
                resolved_parquet=None,
                resolved_row_index=None,
                lookup_method="cached_file",
            )

        target_parquet: Path | None = None
        target_row_idx: int | None = None
        lookup_method = ""

        if source_row_idx is not None:
            target_parquet = self._resolve_parquet_from_source(task_type, source_parquet)
            if target_parquet is not None:
                target_row_idx = self._resolve_row_index_by_source_fields(
                    parquet_path=target_parquet,
                    source_row_idx=int(source_row_idx),
                    source_block_idx=(int(source_block_idx) if source_block_idx is not None else None),
                    source_parquet=source_parquet,
                    ground_truth=ground_truth,
                )
                lookup_method = "filtered_parquet_triple_lookup"

        if (target_parquet is None or target_row_idx is None) and task_type == FORMULA_TASK:
            target_parquet, target_row_idx, is_unique = self._resolve_formula_by_ground_truth(ground_truth)
            if target_parquet is not None and target_row_idx is not None:
                lookup_method = "formula_ground_truth_unique" if is_unique else "formula_ground_truth_first_match"

        if (
            (target_parquet is None or target_row_idx is None)
            and task_type == FORMULA_TASK
            and source_parquet is None
            and source_row_idx is None
        ):
            target_parquet, target_row_idx = self._resolve_formula_by_dataset_index(dataset_index)
            if target_parquet is not None and target_row_idx is not None:
                lookup_method = "formula_dataset_index_last_resort"

        if target_parquet is None or target_row_idx is None:
            return None

        row = self._load_row_from_parquet(target_parquet, target_row_idx)
        if row is None:
            return None
        image_bytes = extract_image_bytes(row.get("image_buffer_list") or row.get("buffer"))
        if not image_bytes:
            return None
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            _decode_image(image_bytes).save(image_path, "PNG")
        except Exception:
            return None

        return LookupResult(
            image_path=str(image_path),
            resolved_parquet=str(target_parquet),
            resolved_row_index=int(target_row_idx),
            lookup_method=lookup_method,
        )
