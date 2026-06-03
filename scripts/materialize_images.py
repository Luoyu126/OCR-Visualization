#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_viz.dataset_lookup import DatasetLookup
from ocr_viz.io_utils import load_json, load_jsonl, save_json, write_jsonl


DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "rmodel"
DEFAULT_TEXT_LIST = Path(
    "/user/hezhihui/wangzhilue/train/minicpmv5-1b-ocr/verl_mm/tmp_scripts/"
    "ocr_multitask_datasets/v4_filter_stage12/text/text_only_files.txt"
)
DEFAULT_TABLE_LIST = Path(
    "/user/hezhihui/wangzhilue/train/minicpmv5-1b-ocr/verl_mm/tmp_scripts/"
    "ocr_multitask_datasets/v3_filter_stage12/table/table_only_files.txt"
)
DEFAULT_FORMULA_LIST = Path(
    "/user/hezhihui/wangzhilue/train/minicpmv5-1b-ocr/verl_mm/tmp_scripts/"
    "formula_sample_30k/balanced_30k_v1/formula_train_files_30k_balanced.txt"
)
RUN_DIR_PATTERN = re.compile(r"^rl_rollout_step_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize source images for existing visualization runs.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--step", type=int, default=None, help="Only process one rl_rollout_step_NNNN directory")
    parser.add_argument("--step-start", type=int, default=None, help="Range start (inclusive)")
    parser.add_argument("--step-end", type=int, default=None, help="Range end (inclusive)")
    parser.add_argument(
        "--overwrite-existing-images",
        action="store_true",
        help="Regenerate image files even when an image path already exists",
    )
    parser.add_argument("--text-list", type=Path, default=DEFAULT_TEXT_LIST)
    parser.add_argument("--table-list", type=Path, default=DEFAULT_TABLE_LIST)
    parser.add_argument("--formula-list", type=Path, default=DEFAULT_FORMULA_LIST)
    return parser.parse_args()


def parse_step(run_dir: Path) -> int | None:
    match = RUN_DIR_PATTERN.match(run_dir.name)
    if not match:
        return None
    return int(match.group(1))


def select_run_dirs(run_root: Path, *, step: int | None, step_start: int | None, step_end: int | None) -> list[Path]:
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    candidates: list[tuple[int, Path]] = []
    for path in run_root.iterdir():
        if not path.is_dir():
            continue
        parsed_step = parse_step(path)
        if parsed_step is None:
            continue
        if not (path / "samples.jsonl").exists():
            continue
        candidates.append((parsed_step, path))
    candidates.sort(key=lambda item: item[0])

    if step is not None:
        selected = [path for parsed_step, path in candidates if parsed_step == int(step)]
        if not selected:
            raise ValueError(f"Step {step} does not exist under {run_root}")
        return selected

    if step_start is None and step_end is None:
        return [path for _, path in candidates]
    if step_start is None or step_end is None:
        raise ValueError("Both --step-start and --step-end are required when selecting a range.")
    if int(step_end) < int(step_start):
        raise ValueError("--step-end must be >= --step-start")
    return [path for parsed_step, path in candidates if int(step_start) <= parsed_step <= int(step_end)]


def _existing_image_path(sample: dict[str, Any], run_dir: Path, images_dir: Path) -> Path | None:
    raw_image = str(sample.get("image_path", "") or "").strip()
    candidates: list[Path] = []
    if raw_image:
        image_path = Path(raw_image)
        candidates.append(image_path if image_path.is_absolute() else run_dir / image_path)
    sample_id = str(sample.get("sample_id", ""))
    if sample_id:
        candidates.extend(
            [
                images_dir / f"{sample_id}.png",
                images_dir / f"{sample_id}.jpg",
                images_dir / f"{sample_id}.jpeg",
                images_dir / f"{sample_id}.webp",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def materialize_run_images(
    *,
    run_dir: Path,
    lookup: DatasetLookup,
    overwrite_existing_images: bool,
) -> dict[str, Any]:
    samples_path = run_dir / "samples.jsonl"
    sequence_groups_path = run_dir / "sequence_groups.jsonl"
    metadata_path = run_dir / "metadata.json"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(samples_path)
    sequence_groups = load_jsonl(sequence_groups_path)
    image_path_by_sample_id: dict[str, str] = {}
    method_counts: Counter[str] = Counter()
    hit = 0
    miss = 0
    skipped_existing = 0

    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            miss += 1
            continue

        existing_image = _existing_image_path(sample, run_dir, images_dir)
        if existing_image is not None and not overwrite_existing_images:
            image_path = str(existing_image)
            sample["image_path"] = image_path
            image_path_by_sample_id[sample_id] = image_path
            method_counts["existing_file"] += 1
            hit += 1
            skipped_existing += 1
            continue

        result = lookup.materialize_image_for_record(
            sample_id=sample_id,
            task_type=str(sample.get("task_type", "")),
            source_parquet=sample.get("source_parquet"),
            source_row_idx=sample.get("source_row_index"),
            source_block_idx=sample.get("source_block_idx"),
            dataset_index=sample.get("dataset_index"),
            ground_truth=sample.get("ground_truth"),
            images_dir=images_dir,
        )
        if result is None:
            sample["image_path"] = ""
            miss += 1
            continue

        sample["image_path"] = result.image_path
        image_path_by_sample_id[sample_id] = result.image_path
        method_counts[result.lookup_method] += 1
        hit += 1

    if image_path_by_sample_id:
        for group in sequence_groups:
            sample_id = str(group.get("sample_id", ""))
            if sample_id in image_path_by_sample_id:
                group["image_path"] = image_path_by_sample_id[sample_id]

    write_jsonl(samples_path, samples)
    if sequence_groups_path.exists():
        write_jsonl(sequence_groups_path, sequence_groups)

    metadata = load_json(metadata_path, {})
    metadata.update(
        {
            "skip_image_lookup": False,
            "image_lookup_hit": int(hit),
            "image_lookup_miss": int(miss),
            "image_lookup_method_hist": dict(method_counts),
            "image_lookup_skipped_existing": int(skipped_existing),
            "dataset_lookup_summary": lookup.summary,
            "image_materialized_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_json(metadata_path, metadata)

    return {
        "run_dir": str(run_dir),
        "sample_count": len(samples),
        "image_lookup_hit": int(hit),
        "image_lookup_miss": int(miss),
        "image_lookup_method_hist": dict(method_counts),
    }


def main() -> None:
    args = parse_args()
    run_dirs = select_run_dirs(
        args.run_root,
        step=args.step,
        step_start=args.step_start,
        step_end=args.step_end,
    )
    if not run_dirs:
        raise RuntimeError("No run directories selected.")

    lookup = DatasetLookup(
        text_list_path=args.text_list,
        table_list_path=args.table_list,
        formula_list_path=args.formula_list,
    )

    completed: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        row = materialize_run_images(
            run_dir=run_dir,
            lookup=lookup,
            overwrite_existing_images=bool(args.overwrite_existing_images),
        )
        completed.append(row)
        print(
            f"[{Path(row['run_dir']).name}] images_hit={row['image_lookup_hit']} "
            f"images_miss={row['image_lookup_miss']} methods={row['image_lookup_method_hist']}",
            flush=True,
        )

    total_hit = sum(int(row["image_lookup_hit"]) for row in completed)
    total_miss = sum(int(row["image_lookup_miss"]) for row in completed)
    print(
        f"Completed image materialization: runs={len(completed)} images_hit={total_hit} images_miss={total_miss}",
        flush=True,
    )


if __name__ == "__main__":
    main()
