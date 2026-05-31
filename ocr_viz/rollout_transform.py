from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ocr_viz.io_utils import pair_key


STEP_FILE_PATTERN = re.compile(r"^global_step_(\d+)_results\.jsonl$")


def parse_step_from_filename(path: Path) -> int | None:
    match = STEP_FILE_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def discover_step_files(rollout_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in rollout_dir.iterdir():
        if not path.is_file():
            continue
        step = parse_step_from_filename(path)
        if step is None:
            continue
        out[step] = path
    return out


def select_step_files(
    step_files: dict[int, Path],
    *,
    step: int | None,
    step_start: int | None,
    step_end: int | None,
) -> list[tuple[int, Path]]:
    if step is not None:
        if step not in step_files:
            raise ValueError(f"Step {step} does not exist in rollout directory.")
        return [(step, step_files[step])]

    if step_start is None and step_end is None:
        selected_steps = sorted(step_files.keys())
        return [(value, step_files[value]) for value in selected_steps]

    if step_start is None or step_end is None:
        raise ValueError("Both --step-start and --step-end are required when selecting a range.")
    if step_end < step_start:
        raise ValueError("--step-end must be >= --step-start")

    selected: list[tuple[int, Path]] = []
    for value in sorted(step_files.keys()):
        if step_start <= value <= step_end:
            selected.append((value, step_files[value]))
    if not selected:
        raise ValueError(f"No step files found in range [{step_start}, {step_end}].")
    return selected


def load_rollout_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = int(line_no)
            rows.append(row)
    return rows


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return 0.0
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    return variance**0.5


def assign_sequence_adv(records: list[dict[str, Any]]) -> None:
    rewards = [float(record.get("final_reward", 0.0) or 0.0) for record in records]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    if len(rewards) > 1:
        variance = sum((value - mean) ** 2 for value in rewards) / (len(rewards) - 1)
        std = variance**0.5
    else:
        std = 1.0
    if std == 0.0:
        std = 1.0

    for record, reward in zip(records, rewards, strict=True):
        raw_adv = _safe_float_or_none(record.get("grpo_normed_adv"))
        if raw_adv is not None:
            record["adv"] = float(raw_adv)
            record["adv_source"] = "jsonl_grpo_normed_adv"
        else:
            record["adv"] = (reward - mean) / (std + 1e-6)
            record["adv_source"] = "computed_from_final_reward"
        record["reward_score"] = reward
        record["aux_score"] = -float(record.get("aux_deducted_penalty", 0.0) or 0.0)


def _extract_prompt_text(raw_prompt: str) -> str:
    text = str(raw_prompt or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line.endswith("Recognition:"):
            return line
    match = re.search(r"([A-Za-z]+\s+Recognition:)", text)
    if match:
        return match.group(1)
    if lines:
        return lines[0]
    return ""


def _normalize_task_type(value: Any) -> str:
    task = str(value or "").strip().lower()
    if task in {"text", "table", "formula"}:
        return task
    if task:
        return task
    return "text"


def _resolve_strategy(raw_row: dict[str, Any], task_type: str) -> str:
    chunk_meta = raw_row.get("chunk_meta")
    if isinstance(chunk_meta, dict):
        value = str(chunk_meta.get("strategy", "") or "").strip()
        if value:
            return value
    for key in ("strategy", "chunk_strategy"):
        value = str(raw_row.get(key, "") or "").strip()
        if value:
            return value
    if task_type == "text":
        return "token_levenshtein"
    if task_type == "table":
        return "chunk_table"
    if task_type == "formula":
        return "formula_sequence"
    return "rollout"


def _to_sequence_row(
    *,
    sample_id: str,
    response_index: int,
    row: dict[str, Any],
    task_type: str,
    prompt_text: str,
    ground_truth: str,
    source_parquet: str | None,
    source_row_idx: int | None,
    source_block_idx: int | None,
    dataset_index: int | None,
) -> dict[str, Any]:
    base_reward = _safe_float(row.get("base_reward"), 0.0)
    final_reward = _safe_float(row.get("final_reward"), 0.0)
    reward_penalty = _safe_float(row.get("reward_penalty"), 0.0)
    repetition_penalty = _safe_float(row.get("repetition_penalty"), 0.0)
    grpo_normed_adv = _safe_float_or_none(row.get("grpo_normed_adv"))
    chunk_sequence_reward = _safe_float_or_none(row.get("chunk_sequence_reward"))
    return {
        "sample_id": sample_id,
        "response_index": int(response_index),
        "response_text": str(row.get("response", "")),
        "task_type": task_type,
        "prompt_text": prompt_text,
        "ground_truth": ground_truth,
        "source_parquet": source_parquet,
        "source_row_index": source_row_idx,
        "source_block_idx": source_block_idx,
        "dataset_index": dataset_index,
        "uid": str(row.get("uid") or ""),
        "base_reward": float(base_reward),
        "final_reward": float(final_reward),
        "grpo_normed_adv": grpo_normed_adv,
        "chunk_sequence_reward": chunk_sequence_reward,
        "aux_raw_penalty": float(repetition_penalty),
        "aux_deducted_penalty": float(reward_penalty),
        "reward_gain": float(base_reward),
        "reward_penalty": float(reward_penalty),
        "penalty_type": "rollout_reward_penalty",
        "formula_server_url": row.get("formula_server_url"),
        "formula_server_request_id": row.get("formula_server_request_id"),
        "reward_details": {
            "sample_is_valid": bool(row.get("sample_is_valid", True)),
            "reward_transport": row.get("reward_transport"),
            "server_error": row.get("server_error"),
            "server_error_type": row.get("server_error_type"),
            "formula_server_failed": row.get("formula_server_failed"),
        },
    }


def _response_token_by_chunk_id(raw_row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    response_tokens = list(raw_row.get("response_tokens") or [])
    for idx, item in enumerate(response_tokens):
        if not isinstance(item, dict):
            continue
        chunk_id = _safe_int(item.get("token_idx"), idx)
        out[chunk_id] = item
    return out


def _build_chunks_for_response(
    *,
    raw_row: dict[str, Any],
    task_type: str,
    sample_id: str,
    response_index: int,
    strategy: str,
) -> list[dict[str, Any]]:
    chunk_items = list(raw_row.get("chunk_items") or [])
    pred_chunk_records = list(raw_row.get("pred_chunk_records") or [])
    chunk_rewards = list(raw_row.get("chunk_rewards") or [])
    response_token_by_id = _response_token_by_chunk_id(raw_row)
    if not chunk_items and not pred_chunk_records and not chunk_rewards:
        return []

    chunk_by_id: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(chunk_items):
        chunk_id = _safe_int(item.get("chunk_id"), idx)
        chunk_by_id[chunk_id] = item
    pred_by_id: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(pred_chunk_records):
        pred_chunk_id = _safe_int(item.get("pred_chunk_id", item.get("chunk_id")), idx)
        pred_by_id[pred_chunk_id] = item

    ids = sorted(set(chunk_by_id.keys()) | set(pred_by_id.keys()) | set(range(len(chunk_rewards))))
    pair_id = pair_key(sample_id, response_index, strategy)
    out: list[dict[str, Any]] = []

    for chunk_id in ids:
        chunk_item = chunk_by_id.get(chunk_id, {})
        pred_item = pred_by_id.get(chunk_id, {})
        token_item = response_token_by_id.get(chunk_id, {})
        reward_raw = _safe_float(
            chunk_item.get("reward", pred_item.get("chunk_final_reward")),
            _safe_float(chunk_rewards[chunk_id] if chunk_id < len(chunk_rewards) else 0.0, 0.0),
        )
        token_start = _safe_int(chunk_item.get("token_start", pred_item.get("token_start")), -1)
        token_end = _safe_int(chunk_item.get("token_end", pred_item.get("token_end")), -1)
        tokens = _safe_int(chunk_item.get("tokens"), token_end - token_start if token_start >= 0 and token_end >= 0 else 0)
        gt_chunk_id = chunk_item.get("matched_gt_chunk_id")
        if gt_chunk_id is None:
            gt_chunk_id = pred_item.get("matched_gt_chunk_id")
        if gt_chunk_id is None:
            gt_chunk_id = chunk_item.get("gt_chunk_id")
        if gt_chunk_id is None:
            gt_chunk_id = chunk_id

        start_char = _safe_int(
            chunk_item.get("start_char", chunk_item.get("pred_char_start", pred_item.get("pred_char_start"))),
            _safe_int(token_item.get("char_start"), -1),
        )
        end_char = _safe_int(
            chunk_item.get("end_char", chunk_item.get("pred_char_end", pred_item.get("pred_char_end"))),
            _safe_int(token_item.get("char_end"), -1),
        )
        token_text = str(
            token_item.get("token_text", chunk_item.get("pred_chunk_text", pred_item.get("pred_chunk_text", ""))) or ""
        )
        gt_text = str(chunk_item.get("gt_chunk_text", pred_item.get("matched_gt_chunk_text", "")) or "")
        is_eos = token_text in {"<|im_end|>", "</s>", "<eos>"} or bool(chunk_item.get("is_eos", False))

        row = {
            "sample_id": sample_id,
            "response_index": int(response_index),
            "strategy": strategy,
            "pair_key": pair_id,
            "chunk_id": int(chunk_id),
            "gt_chunk_id": _safe_int(gt_chunk_id, chunk_id),
            "chunk_type": str(chunk_item.get("chunk_type", pred_item.get("chunk_type", "")) or ""),
            "tokens": int(max(tokens, 0)),
            "token_start": int(token_start),
            "token_end": int(token_end),
            "start_char": int(start_char),
            "end_char": int(end_char),
            "pred_chunk_text": str(chunk_item.get("pred_chunk_text", pred_item.get("pred_chunk_text", "")) or ""),
            "gt_span_start": chunk_item.get("gt_span_start", pred_item.get("gt_span_start")),
            "gt_span_end": chunk_item.get("gt_span_end", pred_item.get("gt_span_end")),
            "gt_shadow_span_start": chunk_item.get("gt_shadow_span_start"),
            "gt_shadow_span_end": chunk_item.get("gt_shadow_span_end"),
            "gt_chunk_text": gt_text,
            "gt_chunk_text_nospace": gt_text,
            "E_k": _safe_int(chunk_item.get("E_k"), 0),
            "L_k": _safe_int(chunk_item.get("L_k"), 0),
            "P_k": _safe_int(chunk_item.get("P_k"), 0),
            "denominator": _safe_int(chunk_item.get("denominator"), 1),
            "reward_raw": float(reward_raw),
            "reward": round(float(reward_raw), 4),
            "macro": chunk_item.get("macro", pred_item.get("macro")),
            "matching_reward": chunk_item.get("matching_reward", pred_item.get("chunk_matching_reward")),
            "reward_source": chunk_item.get("reward_source", pred_item.get("chunk_reward_source")),
            "chunk_normed_adv": _safe_float_or_none(pred_item.get("chunk_normed_adv", chunk_item.get("chunk_normed_adv"))),
            "token_idx": _safe_int(token_item.get("token_idx"), chunk_id),
            "token_id": token_item.get("token_id"),
            "token_text": token_text or str(chunk_item.get("pred_chunk_text", pred_item.get("pred_chunk_text", "")) or ""),
            "gt_text": gt_text,
            "is_eos": bool(is_eos),
            "token_adv": _safe_float_or_none(token_item.get("advantage")),
            "token_level_score": _safe_float_or_none(token_item.get("token_level_score")),
            "token_level_reward": _safe_float_or_none(token_item.get("token_level_reward")),
            "norm_text": (
                str(chunk_item.get("norm_text"))
                if chunk_item.get("norm_text") is not None
                else (
                    str(pred_item.get("norm_text"))
                    if pred_item.get("norm_text") is not None
                    else None
                )
            ),
            "masked": bool(chunk_item.get("masked", False) or pred_item.get("masked", False)),
            "matched_gt_token_indices": [int(_safe_int(gt_chunk_id, chunk_id))],
            "attributed_gt_token_indices": [int(_safe_int(gt_chunk_id, chunk_id))],
            "inserted_gt_token_indices": [],
        }

        if task_type == "table":
            row["pred_tr_index"] = _safe_int(
                chunk_item.get("pred_tr_index", pred_item.get("pred_tr_index", chunk_id)),
                chunk_id,
            )
            row["pred_tr_matched"] = bool(
                chunk_item.get(
                    "pred_tr_matched",
                    pred_item.get("pred_tr_matched", _safe_int(gt_chunk_id, -1) >= 0),
                )
            )

        out.append(row)

    out.sort(key=lambda item: int(item.get("chunk_id", 0)))
    return out


def _summarize_pair_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        return {
            "chunk_count": 0,
            "pair_reward": 0.0,
            "mean_chunk_reward": 0.0,
            "min_chunk_reward": 0.0,
            "max_chunk_reward": 0.0,
            "reward_std": 0.0,
            "total_E": 0,
            "total_L": 0,
            "total_P": 0,
            "total_denominator": 0,
        }
    rewards = [_safe_float(item.get("reward_raw", item.get("reward")), 0.0) for item in chunks]
    denominators = [_safe_int(item.get("denominator"), 0) for item in chunks]
    total_denominator = sum(max(item, 0) for item in denominators)
    if total_denominator > 0:
        weighted = sum(
            _safe_float(item.get("reward_raw", item.get("reward")), 0.0) * max(_safe_int(item.get("denominator"), 0), 0)
            for item in chunks
        )
        pair_reward = weighted / total_denominator
    else:
        pair_reward = sum(rewards) / len(rewards)
    return {
        "chunk_count": len(chunks),
        "pair_reward": float(pair_reward),
        "mean_chunk_reward": float(sum(rewards) / len(rewards)),
        "min_chunk_reward": float(min(rewards)),
        "max_chunk_reward": float(max(rewards)),
        "reward_std": float(_std(rewards)),
        "total_E": int(sum(_safe_int(item.get("E_k"), 0) for item in chunks)),
        "total_L": int(sum(_safe_int(item.get("L_k"), 0) for item in chunks)),
        "total_P": int(sum(_safe_int(item.get("P_k"), 0) for item in chunks)),
        "total_denominator": int(total_denominator),
    }


def _group_key(row: dict[str, Any], fallback_index: int) -> str:
    uid = row.get("uid")
    if uid is not None and str(uid).strip():
        return f"uid::{uid}"
    parts = [
        str(row.get("task_type") or ""),
        str(row.get("source_parquet") or ""),
        str(row.get("source_row_idx") if row.get("source_row_idx") is not None else ""),
        str(row.get("dataset_index") if row.get("dataset_index") is not None else ""),
        str(row.get("ground_truth") or ""),
    ]
    return f"fallback::{fallback_index}::{'||'.join(parts)}"


def transform_step_rows(
    rows: list[dict[str, Any]],
    *,
    step: int,
    expected_rollouts: int,
) -> dict[str, Any]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for idx, row in enumerate(rows):
        key = _group_key(row, idx)
        grouped.setdefault(key, []).append(row)

    samples: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    sequence_groups: list[dict[str, Any]] = []
    sequence_scores: list[dict[str, Any]] = []
    chunk_rows_by_strategy: dict[str, list[dict[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []

    group_size_hist: dict[int, int] = {}
    task_type_counts: dict[str, int] = {}
    source_triple_available_count = 0

    for sample_order, (_, group_rows) in enumerate(grouped.items()):
        group_rows = sorted(group_rows, key=lambda item: _safe_int(item.get("_line_no"), 0))
        first = group_rows[0]
        uid = str(first.get("uid") or "")
        sample_id = f"uid_{uid}" if uid else f"sample_{sample_order:06d}"
        task_type = _normalize_task_type(first.get("task_type", first.get("category")))
        prompt_text = _extract_prompt_text(str(first.get("prompt", "")))
        ground_truth = str(first.get("ground_truth", ""))
        source_parquet = first.get("source_parquet")
        source_row_idx_raw = first.get("source_row_idx")
        source_block_idx_raw = (
            first.get("source_block_idx")
            if first.get("source_block_idx") is not None
            else first.get("block_idx")
        )
        source_row_idx = _safe_int(source_row_idx_raw, default=-1) if source_row_idx_raw is not None else None
        source_block_idx = _safe_int(source_block_idx_raw, default=-1) if source_block_idx_raw is not None else None
        dataset_index_raw = first.get("dataset_index")
        dataset_index = _safe_int(dataset_index_raw, default=-1) if dataset_index_raw is not None else None
        if source_row_idx is not None and source_row_idx < 0:
            source_row_idx = None
        if source_block_idx is not None and source_block_idx < 0:
            source_block_idx = None
        if dataset_index is not None and dataset_index < 0:
            dataset_index = None
        if source_parquet and source_row_idx is not None and source_block_idx is not None:
            source_triple_available_count += 1

        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        group_size = len(group_rows)
        group_size_hist[group_size] = group_size_hist.get(group_size, 0) + 1

        sample_row = {
            "sample_id": sample_id,
            "uid": uid,
            "task_type": task_type,
            "source_run": f"rl_rollout_step_{step:04d}",
            "prompt_text": prompt_text,
            "rollout_prompt_text": str(first.get("prompt", "")),
            "ground_truth": ground_truth,
            "image_path": "",
            "source_parquet": source_parquet,
            "source_row_index": source_row_idx,
            "source_block_idx": source_block_idx,
            "dataset_index": dataset_index,
            "source": first.get("source"),
            "orig_source": first.get("orig_source"),
            "orig_index": first.get("orig_index"),
        }
        samples.append(sample_row)

        seq_rows: list[dict[str, Any]] = []
        raw_rows_by_response_index: dict[int, dict[str, Any]] = {}
        for response_index, raw_row in enumerate(group_rows):
            seq_row = _to_sequence_row(
                sample_id=sample_id,
                response_index=response_index,
                row=raw_row,
                task_type=task_type,
                prompt_text=prompt_text,
                ground_truth=ground_truth,
                source_parquet=source_parquet,
                source_row_idx=source_row_idx,
                source_block_idx=source_block_idx,
                dataset_index=dataset_index,
            )
            seq_rows.append(seq_row)
            raw_rows_by_response_index[response_index] = raw_row

        assign_sequence_adv(seq_rows)
        sequence_scores.extend(seq_rows)

        responses_payload: list[dict[str, Any]] = []
        for seq_row in seq_rows:
            response_index = int(seq_row["response_index"])
            response_text = str(seq_row.get("response_text", ""))
            predictions.append(
                {
                    "sample_id": sample_id,
                    "response_index": response_index,
                    "response_text": response_text,
                    "task_type": task_type,
                    "source_parquet": source_parquet,
                    "source_row_index": source_row_idx,
                    "source_block_idx": source_block_idx,
                    "dataset_index": dataset_index,
                    "uid": uid,
                }
            )
            responses_payload.append(
                {
                    "response_index": response_index,
                    "response_text": response_text,
                    "base_reward": float(seq_row["base_reward"]),
                    "final_reward": float(seq_row["final_reward"]),
                    "aux_raw_penalty": float(seq_row["aux_raw_penalty"]),
                    "aux_deducted_penalty": float(seq_row["aux_deducted_penalty"]),
                    "reward_gain": float(seq_row["reward_gain"]),
                    "reward_penalty": float(seq_row["reward_penalty"]),
                    "penalty_type": str(seq_row["penalty_type"]),
                    "adv": float(seq_row["adv"]),
                    "adv_source": str(seq_row.get("adv_source", "")),
                    "grpo_normed_adv": seq_row.get("grpo_normed_adv"),
                    "reward_score": float(seq_row["reward_score"]),
                    "aux_score": float(seq_row["aux_score"]),
                    "formula_server_url": seq_row.get("formula_server_url"),
                    "formula_server_request_id": seq_row.get("formula_server_request_id"),
                }
            )

            raw_row = raw_rows_by_response_index[response_index]
            strategy = _resolve_strategy(raw_row, task_type)
            pair_id = pair_key(sample_id, response_index, strategy)
            chunks = _build_chunks_for_response(
                raw_row=raw_row,
                task_type=task_type,
                sample_id=sample_id,
                response_index=response_index,
                strategy=strategy,
            )
            pair_summary = _summarize_pair_from_chunks(chunks)
            raw_chunk_sequence_reward = _safe_float_or_none(raw_row.get("chunk_sequence_reward"))
            if raw_chunk_sequence_reward is not None:
                pair_summary["pair_reward"] = float(raw_chunk_sequence_reward)

            chunk_meta = raw_row.get("chunk_meta") if isinstance(raw_row.get("chunk_meta"), dict) else {}
            matcher_meta = chunk_meta.get("matcher_meta", {}) if isinstance(chunk_meta, dict) else {}
            edit_distance = _safe_int(matcher_meta.get("edit_distance"), 0)
            pair_summaries.append(
                {
                    "sample_id": sample_id,
                    "response_index": response_index,
                    "strategy": strategy,
                    "pair_key": pair_id,
                    "source_parquet": source_parquet,
                    "source_row_index": source_row_idx,
                    "source_block_idx": source_block_idx,
                    "dataset_index": dataset_index,
                    "prompt_text": prompt_text,
                    "ground_truth": ground_truth,
                    "response_text": response_text,
                    "task_type": task_type,
                    "seq_base_reward": float(seq_row["base_reward"]),
                    "seq_final_reward": float(seq_row["final_reward"]),
                    "seq_adv": float(seq_row["adv"]),
                    "seq_reward_score": float(seq_row["reward_score"]),
                    "seq_aux_score": float(seq_row["aux_score"]),
                    "seq_penalty_type": str(seq_row["penalty_type"]),
                    "seq_adv_source": str(seq_row.get("adv_source", "")),
                    "seq_grpo_normed_adv": seq_row.get("grpo_normed_adv"),
                    "chunk_sequence_reward": raw_chunk_sequence_reward,
                    "sample_is_valid": bool(raw_row.get("sample_is_valid", True)),
                    "reward_transport": raw_row.get("reward_transport"),
                    "formula_server_failed": raw_row.get("formula_server_failed"),
                    "formula_server_url": raw_row.get("formula_server_url"),
                    "formula_server_request_id": raw_row.get("formula_server_request_id"),
                    "edit_distance": int(edit_distance),
                    "token_count": int(pair_summary.get("chunk_count", 0)),
                    "mean_token_reward": float(pair_summary.get("mean_chunk_reward", 0.0)),
                    "sample_token_reward_std": float(pair_summary.get("reward_std", 0.0)),
                    **pair_summary,
                }
            )
            if strategy not in chunk_rows_by_strategy:
                chunk_rows_by_strategy[strategy] = []
            chunk_rows_by_strategy[strategy].extend(chunks)

        sequence_groups.append(
            {
                "sample_id": sample_id,
                "uid": uid,
                "step": int(step),
                "task_type": task_type,
                "prompt_text": prompt_text,
                "ground_truth": ground_truth,
                "image_path": "",
                "source_parquet": source_parquet,
                "source_row_index": source_row_idx,
                "source_block_idx": source_block_idx,
                "dataset_index": dataset_index,
                "responses": responses_payload,
            }
        )
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "uid": uid,
                "task_type": task_type,
                "source_parquet": source_parquet,
                "source_row_index": source_row_idx,
                "source_block_idx": source_block_idx,
                "dataset_index": dataset_index,
                "group_size": group_size,
            }
        )

    metadata = {
        "source_run": "rl_rollout_mixed_reward",
        "step": int(step),
        "line_count": len(rows),
        "sample_count": len(samples),
        "expected_rollouts": int(expected_rollouts),
        "group_size_hist": {str(key): int(value) for key, value in sorted(group_size_hist.items())},
        "task_type_counts": task_type_counts,
        "source_triple_available_count": int(source_triple_available_count),
        "pairs_count": len(pair_summaries),
        "chunks_count": int(sum(len(value) for value in chunk_rows_by_strategy.values())),
    }
    return {
        "metadata": metadata,
        "samples": samples,
        "predictions": predictions,
        "pair_summary": pair_summaries,
        "sequence_groups": sequence_groups,
        "sequence_scores": sequence_scores,
        "chunk_rows_by_strategy": chunk_rows_by_strategy,
        "sample_manifest": manifest_rows,
    }
