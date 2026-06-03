"""Token-level advantage extremeness: absolute-adv thresholds and response flags."""

from __future__ import annotations

import copy
import math
from statistics import mean, stdev
from typing import Any


def extract_chunk_adv(chunk: dict[str, Any]) -> float | None:
    for key in ("token_adv", "chunk_normed_adv", "adv"):
        value = chunk.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return None


def collect_adv_values(pairs: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for pair in pairs:
        for chunk in pair.get("chunks", []) or []:
            adv = extract_chunk_adv(chunk)
            if adv is not None:
                values.append(adv)
    return values


def compute_adv_thresholds(values: list[float], *, abs_threshold: float = 2.0) -> dict[str, Any]:
    abs_threshold = max(0.0, float(abs_threshold))
    if not values:
        return {
            "abs_threshold": abs_threshold,
            "n": 0,
            "mean": None,
            "std": None,
        }
    return {
        "abs_threshold": abs_threshold,
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def classify_adv(adv: float | None, abs_threshold: float | None) -> tuple[bool, str | None]:
    if adv is None or abs_threshold is None:
        return False, None
    if abs(adv) >= float(abs_threshold):
        return True, "low" if adv < 0 else "high"
    return False, None


def extreme_token_record(chunk: dict[str, Any], *, side: str, adv: float) -> dict[str, Any]:
    token_text = str(
        chunk.get("token_text")
        or chunk.get("pred_chunk_text")
        or chunk.get("pred_chunk_text_raw")
        or ""
    )
    if len(token_text) > 120:
        token_text = token_text[:120] + "..."
    return {
        "chunk_id": int(chunk.get("chunk_id", 0) or 0),
        "token_id": chunk.get("token_id"),
        "side": side,
        "adv": round(adv, 6),
        "token_text": token_text,
        "reward": chunk.get("reward_raw", chunk.get("reward")),
        "chunk_type": chunk.get("chunk_type"),
        "start_char": chunk.get("start_char"),
        "end_char": chunk.get("end_char"),
    }


def enrich_pairs_with_adv_extremeness(
    pairs: list[dict[str, Any]],
    *,
    abs_threshold: float = 2.0,
) -> dict[str, Any]:
    """Annotate pair/chunk dicts in-place with extreme adv flags."""
    values = collect_adv_values(pairs)
    thresholds = compute_adv_thresholds(values, abs_threshold=abs_threshold)
    abs_t = thresholds["abs_threshold"]

    extreme_response_count = 0
    extreme_token_count = 0
    response_flags: dict[int, dict[str, Any]] = {}

    for pair in pairs:
        extreme_tokens: list[dict[str, Any]] = []
        for chunk in pair.get("chunks", []) or []:
            adv = extract_chunk_adv(chunk)
            is_extreme, side = classify_adv(adv, abs_t)
            chunk["is_extreme_token"] = is_extreme
            chunk["extreme_adv_side"] = side
            if is_extreme and adv is not None and side is not None:
                extreme_token_count += 1
                extreme_tokens.append(extreme_token_record(chunk, side=side, adv=adv))

        response_index = int(pair.get("response_index", 0))
        pair["is_extreme_response"] = bool(extreme_tokens)
        pair["extreme_token_count"] = len(extreme_tokens)
        pair["extreme_tokens"] = extreme_tokens
        if extreme_tokens:
            extreme_response_count += 1
            response_flags.setdefault(
                response_index,
                {
                    "is_extreme_response": True,
                    "extreme_token_count": 0,
                    "extreme_tokens": [],
                },
            )
            response_flags[response_index]["extreme_token_count"] += len(extreme_tokens)
            response_flags[response_index]["extreme_tokens"].extend(extreme_tokens)

    return {
        "thresholds": thresholds,
        "extreme_response_count": extreme_response_count,
        "extreme_token_count": extreme_token_count,
        "response_flags": response_flags,
    }


def copy_pairs_and_enrich(
    pairs: list[dict[str, Any]],
    *,
    abs_threshold: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    copied = copy.deepcopy(pairs)
    adv_stats = enrich_pairs_with_adv_extremeness(copied, abs_threshold=abs_threshold)
    return copied, adv_stats


def sample_has_extreme_response(sample_view: dict[str, Any]) -> bool:
    return any(bool(pair.get("is_extreme_response")) for pair in sample_view.get("pairs", []) or [])


def format_threshold_summary(adv_stats: dict[str, Any]) -> str:
    thresholds = adv_stats.get("thresholds", {})
    abs_threshold = thresholds.get("abs_threshold")
    n = thresholds.get("n", 0)
    if abs_threshold is None:
        return "无 token adv 阈值。"
    if not n:
        return f"无 token adv 数据；当前极端 token 阈值为 |adv| ≥ {float(abs_threshold):.4f}。"
    return (
        f"Run 内 token adv 阈值（n={n:,}）："
        f"|adv| ≥ {g_fmt(float(abs_threshold))}。"
    )


def g_fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)
