#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocr_viz.formula_token_scoring import build_run_artifacts, build_run_artifacts_from_scores_jsonl
from ocr_viz.io_utils import save_json, write_jsonl


DEFAULT_INPUT = Path(
    "/user/wangzhilue/algorithm/Formula/rollouts/rollout_formula_ocr_150x8_seed20260524"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "formula_token_passk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score formula pass@k rollouts with token_formula and emit visualization-v1 run artifacts."
    )
    parser.add_argument("--input-run-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None, help="Output run directory name")
    parser.add_argument("--max-responses", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Symlink source images into run images/ when paths exist",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=None,
        help="Import precomputed pass@k scores instead of re-running token_formula locally",
    )
    return parser.parse_args()


def _link_sample_images(samples: list[dict[str, Any]], images_dir: Path) -> tuple[int, int]:
    hit = 0
    miss = 0
    images_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        src = str(sample.get("image_path", "") or "")
        if not sample_id or not src:
            miss += 1
            continue
        src_path = Path(src)
        if not src_path.exists():
            miss += 1
            continue
        dst = images_dir / f"{sample_id}{src_path.suffix or '.jpg'}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src_path.resolve())
            sample["image_path"] = str(dst)
            hit += 1
        except OSError:
            shutil.copy2(src_path, dst)
            sample["image_path"] = str(dst)
            hit += 1
    return hit, miss


def main() -> None:
    args = parse_args()
    input_dir = args.input_run_dir
    if not (input_dir / "samples.jsonl").exists():
        raise FileNotFoundError(f"Missing samples.jsonl under {input_dir}")
    if not (input_dir / "predictions.jsonl").exists():
        raise FileNotFoundError(f"Missing predictions.jsonl under {input_dir}")

    run_name = args.run_name or f"formula_token_{input_dir.name}"
    run_dir = args.out_root / run_name
    if run_dir.exists():
        if args.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(f"Run exists: {run_dir}. Use --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.scores_jsonl is not None:
        print(f"Importing scores from {args.scores_jsonl} ...", flush=True)
        artifacts = build_run_artifacts_from_scores_jsonl(
            input_run_dir=input_dir,
            scores_jsonl=args.scores_jsonl,
        )
    else:
        print(f"Scoring from {input_dir} via token_formula ...", flush=True)
        artifacts = build_run_artifacts(input_run_dir=input_dir, max_responses=int(args.max_responses))

    image_hit = 0
    image_miss = 0
    if args.link_images:
        image_hit, image_miss = _link_sample_images(artifacts["samples"], run_dir / "images")
        for group in artifacts["sequence_groups"]:
            sid = str(group.get("sample_id", ""))
            sample = next((s for s in artifacts["samples"] if s.get("sample_id") == sid), None)
            if sample and sample.get("image_path"):
                group["image_path"] = sample["image_path"]

    artifacts["metadata"].update(
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_link_hit": image_hit,
            "image_link_miss": image_miss,
        }
    )

    write_jsonl(run_dir / "samples.jsonl", artifacts["samples"])
    write_jsonl(run_dir / "predictions.jsonl", artifacts["predictions"])
    write_jsonl(run_dir / "pair_summary.jsonl", artifacts["pair_summary"])
    write_jsonl(run_dir / "sequence_groups.jsonl", artifacts["sequence_groups"])
    write_jsonl(run_dir / "sequence_scores.jsonl", artifacts["sequence_scores"])
    write_jsonl(run_dir / "sample_manifest.jsonl", artifacts["sample_manifest"])
    for strategy, rows in artifacts["chunk_rows_by_strategy"].items():
        write_jsonl(run_dir / f"chunk_scores_{strategy}.jsonl", rows)
    save_json(run_dir / "metadata.json", artifacts["metadata"])

    print(
        f"Built {run_dir} | samples={artifacts['metadata']['sample_count']} "
        f"pairs={artifacts['metadata']['pairs_count']} chunks={artifacts['metadata']['chunks_count']} "
        f"images_hit={image_hit} images_miss={image_miss}"
    )


if __name__ == "__main__":
    main()
