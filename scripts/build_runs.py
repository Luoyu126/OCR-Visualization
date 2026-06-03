#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_viz.io_utils import save_json, write_jsonl
from ocr_viz.rollout_transform import (
    discover_step_files,
    load_rollout_jsonl,
    select_step_files,
    transform_step_rows,
)


DEFAULT_ROLLOUT_DIR = Path(
    "/user/chenyunyi/checkpoints/minicpmv_ocr/"
    "minicpmv5-1b-ocr-natural-sampling-winddata-acc-std-chunk-level-v4-raw-text-ids-debug/rollout_results"
)
DEFAULT_OUTPUT_ROOT = Path("/user/wangzhilue/algorithm/visualization-v1/runs")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Streamlit-compatible runs from RL rollout step JSONL files.")
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--step", type=int, default=None, help="Only process a single global_step_{N}_results.jsonl")
    parser.add_argument("--step-start", type=int, default=None, help="Range start (inclusive)")
    parser.add_argument("--step-end", type=int, default=None, help="Range end (inclusive)")
    parser.add_argument("--expected-rollouts", type=int, default=8, help="Expected responses per sample uid group")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run directory for each step")
    parser.add_argument("--skip-image-lookup", action="store_true", help="Do not lookup/save source images")
    parser.add_argument("--text-list", type=Path, default=DEFAULT_TEXT_LIST)
    parser.add_argument("--table-list", type=Path, default=DEFAULT_TABLE_LIST)
    parser.add_argument("--formula-list", type=Path, default=DEFAULT_FORMULA_LIST)
    return parser.parse_args()


def _write_run(
    run_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    write_jsonl(run_dir / "samples.jsonl", artifacts["samples"])
    write_jsonl(run_dir / "predictions.jsonl", artifacts["predictions"])
    write_jsonl(run_dir / "pair_summary.jsonl", artifacts["pair_summary"])
    write_jsonl(run_dir / "sequence_groups.jsonl", artifacts["sequence_groups"])
    write_jsonl(run_dir / "sequence_scores.jsonl", artifacts["sequence_scores"])
    write_jsonl(run_dir / "sample_manifest.jsonl", artifacts["sample_manifest"])
    for strategy, rows in artifacts["chunk_rows_by_strategy"].items():
        write_jsonl(run_dir / f"chunk_scores_{strategy}.jsonl", rows)
    save_json(run_dir / "metadata.json", artifacts["metadata"])


def main() -> None:
    args = parse_args()
    if not args.rollout_dir.exists():
        raise FileNotFoundError(f"Rollout directory not found: {args.rollout_dir}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    step_files = discover_step_files(args.rollout_dir)
    if not step_files:
        raise RuntimeError(f"No global_step_*_results.jsonl files found under: {args.rollout_dir}")

    selected_steps = select_step_files(
        step_files,
        step=args.step,
        step_start=args.step_start,
        step_end=args.step_end,
    )
    lookup = None
    if not args.skip_image_lookup:
        try:
            from ocr_viz.dataset_lookup import DatasetLookup
        except ImportError as exc:
            raise ImportError(
                "Image lookup requires `pyarrow` and `Pillow`. "
                "Install dependencies or pass --skip-image-lookup."
            ) from exc
        lookup = DatasetLookup(
            text_list_path=args.text_list,
            table_list_path=args.table_list,
            formula_list_path=args.formula_list,
        )

    built_runs: list[dict[str, Any]] = []
    for step, step_path in selected_steps:
        run_name = f"rl_rollout_step_{step:04d}"
        run_dir = args.out_root / run_name
        if run_dir.exists():
            if args.overwrite:
                shutil.rmtree(run_dir)
            else:
                raise FileExistsError(
                    f"Run directory already exists: {run_dir}. "
                    "Use --overwrite or pick a different --out-root."
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        rows = load_rollout_jsonl(step_path)
        artifacts = transform_step_rows(
            rows,
            step=step,
            expected_rollouts=int(args.expected_rollouts),
        )

        image_lookup_hit = 0
        image_lookup_miss = 0
        image_path_by_sample_id: dict[str, str] = {}
        lookup_method_hist: dict[str, int] = {}
        if not args.skip_image_lookup:
            for sample in artifacts["samples"]:
                sample_id = str(sample.get("sample_id", ""))
                lookup_result = lookup.materialize_image_for_record(
                    sample_id=sample_id,
                    task_type=str(sample.get("task_type", "")),
                    source_parquet=sample.get("source_parquet"),
                    source_row_idx=sample.get("source_row_index"),
                    source_block_idx=sample.get("source_block_idx"),
                    dataset_index=sample.get("dataset_index"),
                    ground_truth=sample.get("ground_truth"),
                    images_dir=images_dir,
                )
                if lookup_result is None:
                    image_lookup_miss += 1
                    continue
                image_lookup_hit += 1
                sample["image_path"] = lookup_result.image_path
                image_path_by_sample_id[sample_id] = lookup_result.image_path
                lookup_method_hist[lookup_result.lookup_method] = lookup_method_hist.get(lookup_result.lookup_method, 0) + 1
        else:
            image_lookup_miss = len(artifacts["samples"])

        if image_path_by_sample_id:
            for group in artifacts["sequence_groups"]:
                sample_id = str(group.get("sample_id", ""))
                if sample_id in image_path_by_sample_id:
                    group["image_path"] = image_path_by_sample_id[sample_id]

        mismatched_groups = {
            key: value
            for key, value in artifacts["metadata"]["group_size_hist"].items()
            if int(key) != int(args.expected_rollouts)
        }
        artifacts["metadata"].update(
            {
                "run_name": run_name,
                "rollout_file": str(step_path),
                "expected_rollouts": int(args.expected_rollouts),
                "group_size_mismatch_hist": mismatched_groups,
                "skip_image_lookup": bool(args.skip_image_lookup),
                "image_lookup_hit": int(image_lookup_hit),
                "image_lookup_miss": int(image_lookup_miss),
                "image_lookup_method_hist": lookup_method_hist,
                "dataset_lookup_summary": (lookup.summary if lookup is not None else {}),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

        _write_run(run_dir, artifacts)
        built_runs.append(
            {
                "step": int(step),
                "run_dir": str(run_dir),
                "sample_count": int(artifacts["metadata"]["sample_count"]),
                "pairs_count": int(artifacts["metadata"]["pairs_count"]),
                "chunks_count": int(artifacts["metadata"]["chunks_count"]),
                "image_lookup_hit": int(image_lookup_hit),
                "image_lookup_miss": int(image_lookup_miss),
            }
        )
        print(
            f"[step {step}] built run {run_dir} | "
            f"samples={artifacts['metadata']['sample_count']} "
            f"pairs={artifacts['metadata']['pairs_count']} "
            f"chunks={artifacts['metadata']['chunks_count']} "
            f"images_hit={image_lookup_hit} images_miss={image_lookup_miss}"
        )

    print("Completed runs:")
    for row in built_runs:
        print(row)


if __name__ == "__main__":
    main()
