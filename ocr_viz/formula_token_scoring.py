from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl.utils.reward_score.token_formula import (
    _append_multi_visual_modifier_average_chunks,
    score_token_formula_chunks,
)

from .io_utils import pair_key

STRATEGY = "formula_token"
ADV_SOURCE_ELEMENT = "case_element_normed_adv"
ADV_SOURCE_FALLBACK = "fallback_chunk_seq_adv"


def case_stats(all_rewards: list[float]) -> tuple[float, float]:
    if not all_rewards:
        return 0.0, 0.0
    mean = float(sum(all_rewards) / len(all_rewards))
    if len(all_rewards) >= 2:
        std = float(statistics.pstdev(all_rewards))
    else:
        std = 0.0
    return mean, std


def adv_values(rewards: list[float], mean: float, std: float) -> list[float]:
    if std <= 0:
        return [0.0 for _ in rewards]
    return [float((reward - mean) / std) for reward in rewards]


def chunk_seq_adv(element_advs: list[float]) -> float:
    if not element_advs:
        return 0.0
    return float(sum(element_advs) / len(element_advs))


def sorted_element_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    element_chunks: list[dict[str, Any]] = []
    for item in chunks:
        chunk_type = str(item.get("chunk_type", "") or "")
        if chunk_type.endswith("_eos"):
            continue
        if bool(item.get("is_fallback", False)):
            continue
        if chunk_type != "formula_visual_token":
            continue
        element_chunks.append(dict(item))
    element_chunks.sort(key=lambda row: (int(row.get("pred_char_start", 0)), int(row.get("chunk_id", 0))))
    return element_chunks


def apply_fallback_chunks(
    chunks: list[dict[str, Any]],
    element_adv_by_chunk_id: dict[int, float],
    *,
    fallback_adv: float,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        chunk_id = int(item.get("chunk_id", 0))
        if chunk_id in element_adv_by_chunk_id:
            item["adv"] = float(element_adv_by_chunk_id[chunk_id])
            item["is_fallback"] = False
            item["adv_source"] = ADV_SOURCE_ELEMENT
        else:
            item["adv"] = float(fallback_adv)
            item["is_fallback"] = True
            item["adv_source"] = ADV_SOURCE_FALLBACK
            item.setdefault("matched_gt_chunk_id", None)
            if not str(item.get("reward_source", "") or ""):
                item["reward_source"] = "response_mean_visible_token_adv_fallback"
        enriched.append(item)
    enriched.sort(key=lambda row: (int(row.get("pred_char_start", 0)), int(row.get("chunk_id", 0))))
    return enriched


def annotate_response_from_result(
    *,
    pred: str,
    result: dict[str, Any],
    case_mean: float,
    case_std: float,
) -> dict[str, Any]:
    raw_chunks = list(result.get("chunks", []) or [])
    element_chunks = sorted_element_chunks(raw_chunks)
    element_rewards = [float(item.get("reward", 0.0) or 0.0) for item in element_chunks]
    element_advs = adv_values(element_rewards, case_mean, case_std)
    for item, adv in zip(element_chunks, element_advs, strict=False):
        item["adv"] = float(adv)
        item["chunk_normed_adv"] = float(adv)
        item["adv_source"] = ADV_SOURCE_ELEMENT
        item["is_fallback"] = False

    summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
    meta = result.get("meta", {}) if isinstance(result.get("meta"), dict) else {}
    chunk_seq_reward = float(
        result.get("chunk_sequence_reward", summary.get("pair_reward", summary.get("mean_token_reward", 0.0))) or 0.0
    )
    seq_adv = chunk_seq_adv(element_advs)
    element_adv_by_chunk_id = {
        int(item.get("chunk_id", 0)): float(item.get("adv", 0.0) or 0.0) for item in element_chunks
    }
    enriched_chunks = apply_fallback_chunks(
        list(result.get("chunks", []) or []),
        element_adv_by_chunk_id,
        fallback_adv=seq_adv,
    )

    element_gt_ids = [
        int(item.get("matched_gt_chunk_id")) if item.get("matched_gt_chunk_id") is not None else None
        for item in element_chunks
    ]
    return {
        "pred_html": pred,
        "element_rewards": element_rewards,
        "element_advs": element_advs,
        "element_gt_ids": element_gt_ids,
        "element_count": len(element_chunks),
        "chunk_seq_reward": chunk_seq_reward,
        "chunk_seq_adv": float(seq_adv),
        "chunks": enriched_chunks,
        "element_chunks": element_chunks,
        "summary": summary,
        "meta": meta,
    }


def chunk_to_viz_row(
    chunk: dict[str, Any],
    *,
    sample_id: str,
    response_index: int,
    case_mean: float,
    case_std: float,
    chunk_seq_adv_value: float,
) -> dict[str, Any]:
    start = int(chunk.get("pred_char_start", chunk.get("start_char", 0)) or 0)
    end = int(chunk.get("pred_char_end", chunk.get("end_char", start)) or start)
    reward = float(chunk.get("reward", 0.0) or 0.0)
    is_fallback = bool(chunk.get("is_fallback", False))
    adv = float(chunk.get("adv", 0.0) or 0.0)
    matched_gt = chunk.get("matched_gt_chunk_id")
    pkey = pair_key(sample_id, response_index, STRATEGY)
    token_text = str(chunk.get("pred_chunk_text", "") or "")
    gt_text = str(chunk.get("gt_chunk_text", "") or "")
    row = {
        "sample_id": sample_id,
        "response_index": int(response_index),
        "strategy": STRATEGY,
        "pair_key": pkey,
        "chunk_id": int(chunk.get("chunk_id", 0)),
        "gt_chunk_id": int(matched_gt) if matched_gt is not None else -1,
        "chunk_type": str(chunk.get("chunk_type", "") or ""),
        "tokens": max(0, int(chunk.get("token_end", -1)) - int(chunk.get("token_start", -1)))
        if chunk.get("token_start") is not None and chunk.get("token_end") is not None
        else 0,
        "token_start": int(chunk.get("token_start", -1) if chunk.get("token_start") is not None else -1),
        "token_end": int(chunk.get("token_end", -1) if chunk.get("token_end") is not None else -1),
        "start_char": start,
        "end_char": end,
        "pred_char_start": start,
        "pred_char_end": end,
        "pred_chunk_text": token_text,
        "token_text": token_text,
        "gt_chunk_text": gt_text,
        "gt_text": gt_text,
        "gt_chunk_text_nospace": gt_text,
        "reward_raw": reward,
        "reward": round(reward, 4),
        "adv": adv,
        "chunk_normed_adv": adv if not is_fallback else None,
        "token_adv": adv,
        "chunk_seq_adv": float(chunk_seq_adv_value),
        "is_fallback": is_fallback,
        "adv_source": str(chunk.get("adv_source", ADV_SOURCE_FALLBACK if is_fallback else ADV_SOURCE_ELEMENT)),
        "case_element_mean": float(case_mean),
        "case_element_std": float(case_std),
        "matching_reward": chunk.get("matching_reward"),
        "reward_source": str(chunk.get("reward_source", "") or ""),
        "matched_gt_chunk_id": matched_gt,
        "macro": chunk.get("macro"),
        "denominator": 1,
        "E_k": 0,
        "L_k": 0,
        "P_k": 0,
        "is_eos": str(chunk.get("chunk_type", "")).endswith("_eos"),
    }
    for key in (
        "is_modifier_average",
        "modifier_macro",
        "modifier_segment",
        "modifier_visual_chunk_ids",
        "modifier_visual_rewards",
    ):
        if key in chunk:
            row[key] = chunk.get(key)
    return row


def score_sample_responses(
    *,
    sample_id: str,
    gt: str,
    responses: list[dict[str, Any]],
    max_responses: int = 8,
) -> list[dict[str, Any]]:
    ordered = sorted(responses, key=lambda row: int(row.get("response_index", 0)))
    ordered = [row for row in ordered if int(row.get("response_index", -1)) >= 0][:max_responses]
    if not ordered:
        return []

    raw_scored: list[dict[str, Any]] = []
    case_all_rewards: list[float] = []
    for row in ordered:
        pred = str(row.get("response_text", "") or "")
        result = score_token_formula_chunks(pred=pred, gt=gt, tokenizer=None, include_eos=False)
        element_chunks = sorted_element_chunks(list(result.get("chunks", []) or []))
        element_rewards = [float(item.get("reward", 0.0) or 0.0) for item in element_chunks]
        case_all_rewards.extend(element_rewards)
        raw_scored.append(
            {
                "response_index": int(row.get("response_index", 0)),
                "pred": pred,
                "result": result,
            }
        )

    case_mean, case_std = case_stats(case_all_rewards)
    scored: list[dict[str, Any]] = []
    for item in raw_scored:
        annotated = annotate_response_from_result(
            pred=str(item["pred"]),
            result=dict(item["result"]),
            case_mean=case_mean,
            case_std=case_std,
        )
        annotated["response_index"] = int(item["response_index"])
        annotated["case_element_mean"] = case_mean
        annotated["case_element_std"] = case_std
        annotated["case_element_count"] = len(case_all_rewards)
        scored.append(annotated)
    return scored


def _emit_scored_sample_records(
    *,
    sample_id: str,
    sample: dict[str, Any],
    scored: list[dict[str, Any]],
    input_run_dir: Path,
    manifest_by_id: dict[str, dict[str, Any]],
    out_samples: list[dict[str, Any]],
    out_predictions: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    sequence_groups: list[dict[str, Any]],
    sequence_scores: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    out_manifest: list[dict[str, Any]],
) -> None:
    gt = str(sample.get("ground_truth", "") or "")
    image_path = str(sample.get("image_path", "") or "")
    task_type = str(sample.get("task_type", "formula") or "formula")
    prompt_text = str(sample.get("prompt_text", "") or "")

    out_samples.append(
        {
            "sample_id": sample_id,
            "task_type": task_type,
            "source_run": str(sample.get("source_rollout", input_run_dir.name)),
            "prompt_text": prompt_text,
            "rollout_prompt_text": str(sample.get("rollout_prompt_text", "") or ""),
            "ground_truth": gt,
            "image_path": image_path,
            "source_dataset": sample.get("source_dataset"),
            "source_row_index": sample.get("source_row_index"),
            "dataset_sample_id": sample.get("dataset_sample_id"),
        }
    )

    seq_payload: list[dict[str, Any]] = []
    for ann in scored:
        resp_idx = int(ann["response_index"])
        pred_text = str(ann.get("pred_html", ann.get("pred", "")))
        chunk_seq_reward = float(ann["chunk_seq_reward"])
        chunk_seq_adv_val = float(ann["chunk_seq_adv"])
        case_mean = float(ann.get("case_element_mean", 0.0))
        case_std = float(ann.get("case_element_std", 0.0))

        out_predictions.append(
            {
                "sample_id": sample_id,
                "response_index": resp_idx,
                "response_text": pred_text,
                "task_type": task_type,
            }
        )

        chunks = list(ann.get("chunks", []))
        element_count = int(ann.get("element_count", len(ann.get("element_rewards", []))))
        fallback_count = sum(1 for c in chunks if c.get("is_fallback"))
        rewards = [float(c.get("reward", c.get("reward_raw", 0.0)) or 0.0) for c in chunks]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        pkey = pair_key(sample_id, resp_idx, STRATEGY)
        pair_summaries.append(
            {
                "sample_id": sample_id,
                "response_index": resp_idx,
                "strategy": STRATEGY,
                "pair_key": pkey,
                "task_type": task_type,
                "ground_truth": gt,
                "response_text": pred_text,
                "chunk_sequence_reward": chunk_seq_reward,
                "pair_reward": chunk_seq_reward,
                "seq_final_reward": chunk_seq_reward,
                "seq_adv": chunk_seq_adv_val,
                "chunk_seq_adv": chunk_seq_adv_val,
                "case_element_mean": case_mean,
                "case_element_std": case_std,
                "case_element_count": int(ann.get("case_element_count", 0)),
                "element_count": element_count,
                "fallback_chunk_count": fallback_count,
                "chunk_count": len(chunks),
                "token_count": element_count,
                "mean_token_reward": mean_reward,
                "mean_chunk_reward": mean_reward,
            }
        )

        for chunk in chunks:
            chunk_rows.append(
                chunk_to_viz_row(
                    chunk,
                    sample_id=sample_id,
                    response_index=resp_idx,
                    case_mean=case_mean,
                    case_std=case_std,
                    chunk_seq_adv_value=chunk_seq_adv_val,
                )
            )

        seq_payload.append(
            {
                "response_index": resp_idx,
                "response_text": pred_text,
                "final_reward": chunk_seq_reward,
                "adv": chunk_seq_adv_val,
                "chunk_seq_reward": chunk_seq_reward,
                "chunk_seq_adv": chunk_seq_adv_val,
                "element_count": element_count,
                "fallback_chunk_count": fallback_count,
                "adv_source": "chunk_seq_adv_mean_of_elements",
            }
        )
        sequence_scores.append(
            {
                "sample_id": sample_id,
                "response_index": resp_idx,
                "task_type": task_type,
                "response_text": pred_text,
                "final_reward": chunk_seq_reward,
                "adv": chunk_seq_adv_val,
                "chunk_sequence_reward": chunk_seq_reward,
                "grpo_normed_adv": chunk_seq_adv_val,
                "adv_source": "case_element_normed_then_seq_mean",
            }
        )

    sequence_groups.append(
        {
            "sample_id": sample_id,
            "task_type": task_type,
            "prompt_text": prompt_text,
            "ground_truth": gt,
            "image_path": image_path,
            "responses": seq_payload,
        }
    )

    man = manifest_by_id.get(sample_id, {})
    out_manifest.append(
        {
            "sample_id": sample_id,
            "task_type": task_type,
            "dataset_sample_id": man.get("dataset_sample_id", sample.get("dataset_sample_id")),
            "source_row_index": man.get("source_row_index", sample.get("source_row_index")),
            "group_size": len(scored),
        }
    )


def _passk_row_to_scored_list(row: dict[str, Any]) -> list[dict[str, Any]]:
    case_mean = float(row.get("case_element_mean", 0.0) or 0.0)
    case_std = float(row.get("case_element_std", 0.0) or 0.0)
    case_count = int(row.get("case_element_count", 0) or 0)
    responses = list(row.get("responses", []))
    if not responses:
        responses = [
            {
                "response_index": int(row.get("response_index", 0)),
                "pred_html": row.get("pred_html", ""),
                "chunks": row.get("chunks", []),
                "element_chunks": row.get("element_chunks", []),
                "element_rewards": row.get("element_rewards", []),
                "element_count": len(row.get("element_rewards", [])),
                "chunk_seq_reward": row.get("chunk_seq_reward", 0.0),
                "chunk_seq_adv": row.get("chunk_seq_adv", 0.0),
            }
        ]
    scored: list[dict[str, Any]] = []
    for resp in sorted(responses, key=lambda item: int(item.get("response_index", 0))):
        chunks = list(resp.get("chunks", []))
        pred_html = str(resp.get("pred_html", "") or "")
        postprocessed = _append_multi_visual_modifier_average_chunks(
            {"chunks": chunks, "meta": dict(resp.get("meta", {}) or {})},
            pred_html,
        )
        chunks = list(postprocessed.get("chunks", []) or chunks)
        enriched_chunks: list[dict[str, Any]] = []
        chunk_seq_adv_val = float(resp.get("chunk_seq_adv", 0.0) or 0.0)
        for chunk in chunks:
            item = dict(chunk)
            is_fallback = bool(item.get("is_fallback", False))
            item.setdefault("adv_source", ADV_SOURCE_FALLBACK if is_fallback else ADV_SOURCE_ELEMENT)
            if "adv" not in item:
                item["adv"] = chunk_seq_adv_val if is_fallback else float(item.get("chunk_normed_adv", 0.0) or 0.0)
            enriched_chunks.append(item)
        scored.append(
            {
                "response_index": int(resp.get("response_index", 0)),
                "pred_html": pred_html,
                "chunks": enriched_chunks,
                "element_count": int(resp.get("element_count", len(resp.get("element_rewards", [])))),
                "chunk_seq_reward": float(resp.get("chunk_seq_reward", row.get("chunk_seq_reward", 0.0)) or 0.0),
                "chunk_seq_adv": chunk_seq_adv_val,
                "case_element_mean": case_mean,
                "case_element_std": case_std,
                "case_element_count": case_count,
            }
        )
    return scored


def build_run_artifacts_from_scores_jsonl(
    *,
    input_run_dir: Path,
    scores_jsonl: Path,
) -> dict[str, Any]:
    from .io_utils import load_jsonl

    samples = load_jsonl(input_run_dir / "samples.jsonl")
    manifest = load_jsonl(input_run_dir / "sample_manifest.jsonl") if (input_run_dir / "sample_manifest.jsonl").exists() else []
    input_meta: dict[str, Any] = {}
    meta_path = input_run_dir / "metadata.json"
    if meta_path.exists():
        import json

        input_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    sample_map = {str(row.get("sample_id")): row for row in samples}
    manifest_by_id = {str(row.get("sample_id")): row for row in manifest}
    score_rows = load_jsonl(scores_jsonl)
    score_by_id = {str(row.get("sample_id")): row for row in score_rows}

    out_samples: list[dict[str, Any]] = []
    out_predictions: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    sequence_groups: list[dict[str, Any]] = []
    sequence_scores: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    out_manifest: list[dict[str, Any]] = []

    for sample_id in sorted(sample_map.keys()):
        sample = dict(sample_map[sample_id])
        score_row = score_by_id.get(sample_id)
        if score_row is None:
            continue
        if score_row.get("image_path"):
            sample["image_path"] = score_row.get("image_path")
        if score_row.get("gt_html"):
            sample["ground_truth"] = score_row.get("gt_html")
        scored = _passk_row_to_scored_list(score_row)
        if not scored:
            continue
        _emit_scored_sample_records(
            sample_id=sample_id,
            sample=sample,
            scored=scored,
            input_run_dir=input_run_dir,
            manifest_by_id=manifest_by_id,
            out_samples=out_samples,
            out_predictions=out_predictions,
            pair_summaries=pair_summaries,
            sequence_groups=sequence_groups,
            sequence_scores=sequence_scores,
            chunk_rows=chunk_rows,
            out_manifest=out_manifest,
        )

    metadata = {
        "source_run": "formula_token_passk",
        "input_rollout_dir": str(input_run_dir),
        "scores_jsonl": str(scores_jsonl),
        "input_rollout_name": input_meta.get("rollout_name", input_run_dir.name),
        "reward_mode": "formula_v3_token_visual",
        "strategy": STRATEGY,
        "sample_count": len(out_samples),
        "pairs_count": len(pair_summaries),
        "chunks_count": len(chunk_rows),
        "expected_rollouts": 8,
        "task_type_counts": {"formula": len(out_samples)},
        "adv_source_element": ADV_SOURCE_ELEMENT,
        "adv_source_fallback": ADV_SOURCE_FALLBACK,
        "imported_from_scores_jsonl": True,
    }
    return {
        "metadata": metadata,
        "samples": out_samples,
        "predictions": out_predictions,
        "pair_summary": pair_summaries,
        "sequence_groups": sequence_groups,
        "sequence_scores": sequence_scores,
        "chunk_rows_by_strategy": {STRATEGY: chunk_rows},
        "sample_manifest": out_manifest,
    }


def build_run_artifacts(
    *,
    input_run_dir: Path,
    max_responses: int = 8,
) -> dict[str, Any]:
    from .io_utils import load_jsonl

    samples_path = input_run_dir / "samples.jsonl"
    preds_path = input_run_dir / "predictions.jsonl"
    manifest_path = input_run_dir / "sample_manifest.jsonl"
    meta_path = input_run_dir / "metadata.json"

    samples = load_jsonl(samples_path)
    preds = load_jsonl(preds_path)
    manifest = load_jsonl(manifest_path) if manifest_path.exists() else []
    input_meta: dict[str, Any] = {}
    if meta_path.exists():
        import json

        input_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    sample_map = {str(row.get("sample_id")): row for row in samples}
    pred_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in preds:
        sid = str(row.get("sample_id", ""))
        if sid:
            pred_grouped.setdefault(sid, []).append(row)

    out_samples: list[dict[str, Any]] = []
    out_predictions: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    sequence_groups: list[dict[str, Any]] = []
    sequence_scores: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    out_manifest: list[dict[str, Any]] = []

    manifest_by_id = {str(row.get("sample_id")): row for row in manifest}

    for sample_id in sorted(sample_map.keys()):
        sample = sample_map[sample_id]
        gt = str(sample.get("ground_truth", "") or "")
        responses_in = pred_grouped.get(sample_id, [])
        scored = score_sample_responses(
            sample_id=sample_id,
            gt=gt,
            responses=responses_in,
            max_responses=max_responses,
        )
        if not scored:
            continue
        _emit_scored_sample_records(
            sample_id=sample_id,
            sample=sample,
            scored=scored,
            input_run_dir=input_run_dir,
            manifest_by_id=manifest_by_id,
            out_samples=out_samples,
            out_predictions=out_predictions,
            pair_summaries=pair_summaries,
            sequence_groups=sequence_groups,
            sequence_scores=sequence_scores,
            chunk_rows=chunk_rows,
            out_manifest=out_manifest,
        )

    metadata = {
        "source_run": "formula_token_passk",
        "input_rollout_dir": str(input_run_dir),
        "input_rollout_name": input_meta.get("rollout_name", input_run_dir.name),
        "reward_mode": "formula_v3_token_visual",
        "strategy": STRATEGY,
        "sample_count": len(out_samples),
        "pairs_count": len(pair_summaries),
        "chunks_count": len(chunk_rows),
        "expected_rollouts": int(max_responses),
        "task_type_counts": {"formula": len(out_samples)},
        "adv_source_element": ADV_SOURCE_ELEMENT,
        "adv_source_fallback": ADV_SOURCE_FALLBACK,
    }
    return {
        "metadata": metadata,
        "samples": out_samples,
        "predictions": out_predictions,
        "pair_summary": pair_summaries,
        "sequence_groups": sequence_groups,
        "sequence_scores": sequence_scores,
        "chunk_rows_by_strategy": {STRATEGY: chunk_rows},
        "sample_manifest": out_manifest,
    }
