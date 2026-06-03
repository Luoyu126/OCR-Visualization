from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from ocr_viz.adv_stats import (
    copy_pairs_and_enrich,
    format_threshold_summary,
    sample_has_extreme_response,
)
from ocr_viz.io_utils import (
    append_jsonl,
    chunk_key,
    discover_runs,
    load_json,
    load_jsonl,
    pair_key,
    save_json,
    stable_id,
)
from ocr_viz.ui_renderers import (
    advantage_color,
    colorized_adv_html,
    format_table_source_with_tr_newlines,
    formula_chunk_detail_rows,
    render_code_block,
    render_formula_preview,
    render_formula_token_interactive_pair,
    render_interactive_table_pair_html,
    render_table_html,
    render_text_block,
    render_token_alignment_component,
    reward_hover_text,
)


RUNS_ROOT = Path("/user/chenyunyi/projects/verl_mm/tmp_scripts/visualization-v1/runs")
DEFAULT_RUN_ROOT = RUNS_ROOT / "rmodel"
FORMULA_TOKEN_RUN_ROOT = RUNS_ROOT / "formula_token_passk"
CASE_LABELS = ["unmarked", "incorrect", "correct", "minor_error"]
CASE_LABEL_DISPLAY = {
    "unmarked": "未标注",
    "incorrect": "错误",
    "correct": "正确",
    "minor_error": "有轻微错误",
}
CHUNK_LABELS = ["unmarked", "correct", "incorrect"]


def case_label_text(label: str) -> str:
    return CASE_LABEL_DISPLAY.get(str(label), str(label))


def safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def discover_runs_cached(run_root_str: str) -> list[str]:
    return discover_runs(run_root_str)


@st.cache_data(show_spinner=True)
def load_run_data(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    metadata = load_json(run_dir / "metadata.json", {})
    samples = load_jsonl(run_dir / "samples.jsonl")
    predictions = load_jsonl(run_dir / "predictions.jsonl")
    pair_rows = load_jsonl(run_dir / "pair_summary.jsonl")
    sequence_groups = load_jsonl(run_dir / "sequence_groups.jsonl")
    sequence_scores = load_jsonl(run_dir / "sequence_scores.jsonl")

    chunk_rows: list[dict[str, Any]] = []
    for chunk_path in sorted(run_dir.glob("chunk_scores_*.jsonl")):
        chunk_rows.extend(load_jsonl(chunk_path))

    samples_by_id = {str(row.get("sample_id", "")): row for row in samples}
    predictions_by_key = {
        f"{row.get('sample_id', '')}__r{int(row.get('response_index', 0)):02d}": row
        for row in predictions
    }
    sequence_group_by_sample: dict[str, dict[str, Any]] = {}
    for row in sequence_groups:
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            responses = list(row.get("responses", []))
            responses.sort(key=lambda item: int(item.get("response_index", 0)))
            row["responses"] = responses
            sequence_group_by_sample[sample_id] = row
    if not sequence_group_by_sample and sequence_scores:
        by_sample: dict[str, list[dict[str, Any]]] = {}
        for row in sequence_scores:
            sample_id = str(row.get("sample_id", ""))
            by_sample.setdefault(sample_id, []).append(row)
        for sample_id, rows in by_sample.items():
            rows.sort(key=lambda item: int(item.get("response_index", 0)))
            sequence_group_by_sample[sample_id] = {"sample_id": sample_id, "responses": rows}

    chunks_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        pkey = str(row.get("pair_key", ""))
        if not pkey:
            pkey = pair_key(
                str(row.get("sample_id", "")),
                int(row.get("response_index", 0)),
                str(row.get("strategy", "")),
            )
        chunks_by_pair.setdefault(pkey, []).append(row)
    for pair_chunks in chunks_by_pair.values():
        pair_chunks.sort(key=lambda item: int(item.get("chunk_id", 0)))

    if not pair_rows and chunks_by_pair:
        for pkey, pair_chunks in chunks_by_pair.items():
            if not pair_chunks:
                continue
            first = pair_chunks[0]
            sample_id = str(first.get("sample_id", ""))
            response_index = int(first.get("response_index", 0))
            strategy = str(first.get("strategy", ""))
            den = [int(item.get("denominator", 1) or 1) for item in pair_chunks]
            rewards = [float(item.get("reward_raw", item.get("reward", 0.0)) or 0.0) for item in pair_chunks]
            total_den = sum(den)
            weighted = sum(r * d for r, d in zip(rewards, den, strict=True))
            pair_rows.append(
                {
                    "sample_id": sample_id,
                    "response_index": response_index,
                    "strategy": strategy,
                    "pair_key": pkey,
                    "pair_reward": (weighted / total_den) if total_den > 0 else (sum(rewards) / max(1, len(rewards))),
                    "chunk_count": len(pair_chunks),
                }
            )

    pair_views: list[dict[str, Any]] = []
    for row in pair_rows:
        sample_id = str(row.get("sample_id", ""))
        response_index = int(row.get("response_index", 0))
        strategy = str(row.get("strategy", ""))
        pkey = str(row.get("pair_key", "")) or pair_key(sample_id, response_index, strategy)
        sample = samples_by_id.get(sample_id, {})
        task_type = str(row.get("task_type", sample.get("task_type", "")) or "").lower()
        pair_views.append(
            {
                **row,
                "pair_key": pkey,
                "task_type": task_type,
                "sample": sample,
                "prediction": predictions_by_key.get(f"{sample_id}__r{response_index:02d}", {}),
                "sequence_group": sequence_group_by_sample.get(sample_id, {"sample_id": sample_id, "responses": []}),
                "chunks": chunks_by_pair.get(pkey, []),
            }
        )
    pair_views.sort(
        key=lambda item: (
            str(item.get("sample_id", "")),
            int(item.get("response_index", 0)),
            str(item.get("strategy", "")),
        )
    )
    return {"metadata": metadata, "pairs": pair_views}


@st.cache_data(show_spinner=False)
def load_pairs_with_adv_stats(run_dir_str: str, abs_adv_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_run_data(run_dir_str)
    return copy_pairs_and_enrich(data["pairs"], abs_threshold=abs_adv_threshold)


def rebuild_sample_views(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_sample_views(pairs)


def build_sample_views(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for pair_item in pairs:
        sample_id = str(pair_item.get("sample_id", ""))
        if sample_id not in grouped:
            grouped[sample_id] = {
                "sample_id": sample_id,
                "task_type": str(pair_item.get("task_type", pair_item.get("sample", {}).get("task_type", "")) or "").lower(),
                "sample": pair_item.get("sample", {}),
                "sequence_group": pair_item.get("sequence_group", {"sample_id": sample_id, "responses": []}),
                "pairs": [],
            }
        grouped[sample_id]["pairs"].append(pair_item)
        if not grouped[sample_id].get("sample"):
            grouped[sample_id]["sample"] = pair_item.get("sample", {})
        existing_group = grouped[sample_id].get("sequence_group", {"responses": []})
        if not existing_group.get("responses"):
            grouped[sample_id]["sequence_group"] = pair_item.get("sequence_group", existing_group)

    views = list(grouped.values())
    for view in views:
        view["pairs"].sort(key=lambda item: int(item.get("response_index", 0)))
    views.sort(key=lambda item: str(item.get("sample_id", "")))
    return views


def load_annotations(run_dir: Path) -> dict[str, Any]:
    payload = load_json(
        run_dir / "annotations_current.json",
        {"case_annotations": {}, "pair_annotations": {}, "chunk_annotations": {}},
    )
    if not isinstance(payload, dict):
        payload = {"case_annotations": {}, "pair_annotations": {}, "chunk_annotations": {}}
    payload.setdefault("case_annotations", {})
    payload.setdefault("pair_annotations", {})
    payload.setdefault("chunk_annotations", {})
    return payload


def save_case_annotation(run_dir: Path, sample_id: str, label: str, note: str) -> None:
    annotations = load_annotations(run_dir)
    annotations["case_annotations"][sample_id] = {
        "label": label,
        "note": note.strip(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(run_dir / "annotations_current.json", annotations)
    append_jsonl(
        run_dir / "annotation_events.jsonl",
        {
            "kind": "case_annotation",
            "sample_id": sample_id,
            "label": label,
            "note": note.strip(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def annotation_pair_key(sample_id: str, pair_item: dict[str, Any], response_index: int | None = None) -> str:
    existing = str(pair_item.get("pair_key", "") or "")
    if existing:
        return existing
    if response_index is None:
        response_index = int(pair_item.get("response_index", 0) or 0)
    strategy = str(pair_item.get("strategy", "") or pair_item.get("chunk_strategy", "") or "default")
    return pair_key(str(sample_id), int(response_index), strategy)


def save_pair_annotation(
    run_dir: Path,
    *,
    sample_id: str,
    pair_id: str,
    response_index: int,
    label: str,
    note: str,
) -> None:
    annotations = load_annotations(run_dir)
    now = datetime.now().isoformat(timespec="seconds")
    annotations["pair_annotations"][pair_id] = {
        "sample_id": str(sample_id),
        "response_index": int(response_index),
        "label": label,
        "note": note.strip(),
        "updated_at": now,
    }
    save_json(run_dir / "annotations_current.json", annotations)
    append_jsonl(
        run_dir / "annotation_events.jsonl",
        {
            "kind": "pair_annotation",
            "sample_id": str(sample_id),
            "pair_key": str(pair_id),
            "response_index": int(response_index),
            "label": label,
            "note": note.strip(),
            "updated_at": now,
        },
    )


def save_chunk_annotations(run_dir: Path, updates: list[dict[str, Any]]) -> None:
    annotations = load_annotations(run_dir)
    now = datetime.now().isoformat(timespec="seconds")
    for item in updates:
        annotations["chunk_annotations"][str(item["chunk_key"])] = {
            "label": str(item["label"]),
            "note": str(item.get("note", "")).strip(),
            "updated_at": now,
        }
    save_json(run_dir / "annotations_current.json", annotations)
    for item in updates:
        append_jsonl(
            run_dir / "annotation_events.jsonl",
            {
                "kind": "chunk_annotation",
                "pair_key": str(item["pair_key"]),
                "chunk_id": int(item["chunk_id"]),
                "chunk_key": str(item["chunk_key"]),
                "label": str(item["label"]),
                "note": str(item.get("note", "")).strip(),
                "updated_at": now,
            },
        )


def get_case_label(annotations: dict[str, Any], sample_id: str) -> str:
    value = annotations.get("case_annotations", {}).get(sample_id, {})
    label = str(value.get("label", "unmarked"))
    return label if label in CASE_LABELS else "unmarked"


def get_pair_annotation(annotations: dict[str, Any], sample_id: str, pair_id: str) -> dict[str, Any]:
    pair_value = annotations.get("pair_annotations", {}).get(pair_id)
    if isinstance(pair_value, dict):
        return pair_value
    legacy_case = annotations.get("case_annotations", {}).get(sample_id, {})
    return legacy_case if isinstance(legacy_case, dict) else {}


def get_pair_label(annotations: dict[str, Any], sample_id: str, pair_id: str) -> str:
    value = get_pair_annotation(annotations, sample_id, pair_id)
    label = str(value.get("label", "unmarked"))
    return label if label in CASE_LABELS else "unmarked"


def get_chunk_label(annotations: dict[str, Any], chunk_id: str) -> str:
    value = annotations.get("chunk_annotations", {}).get(chunk_id, {})
    label = str(value.get("label", "unmarked"))
    return label if label in CHUNK_LABELS else "unmarked"


def get_chunk_note(annotations: dict[str, Any], chunk_id: str) -> str:
    value = annotations.get("chunk_annotations", {}).get(chunk_id, {})
    return str(value.get("note", "")).strip()


def summarize_case_labels(samples: list[dict[str, Any]], annotations: dict[str, Any]) -> dict[str, int]:
    counts = {label: 0 for label in CASE_LABELS}
    for item in samples:
        label = get_case_label(annotations, str(item.get("sample_id", "")))
        counts[label] += 1
    return {"total": len(samples), **counts}


def sample_has_pair_label(sample_view: dict[str, Any], annotations: dict[str, Any], label: str) -> bool:
    sample_id = str(sample_view.get("sample_id", ""))
    for pair_item in sample_view.get("pairs", []) or []:
        response_index = int(pair_item.get("response_index", 0) or 0)
        pair_id = annotation_pair_key(sample_id, pair_item, response_index)
        if get_pair_label(annotations, sample_id, pair_id) == label:
            return True
    return False


def summarize_pair_labels(samples: list[dict[str, Any]], annotations: dict[str, Any]) -> dict[str, int]:
    counts = {label: 0 for label in CASE_LABELS}
    total = 0
    for sample_view in samples:
        sample_id = str(sample_view.get("sample_id", ""))
        for pair_item in sample_view.get("pairs", []) or []:
            response_index = int(pair_item.get("response_index", 0) or 0)
            pair_id = annotation_pair_key(sample_id, pair_item, response_index)
            label = get_pair_label(annotations, sample_id, pair_id)
            counts[label] += 1
            total += 1
    return {"total": total, **counts}


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{100.0 * numerator / denominator:.2f}%"


def resolve_sample_image_path(sample: dict[str, Any], run_dir: Path, sample_id: str) -> Path | None:
    candidates: list[Path] = []
    raw_image = str(sample.get("image_path", "") or "").strip()
    if raw_image:
        image_path = Path(raw_image)
        candidates.append(image_path if image_path.is_absolute() else run_dir / image_path)
    candidates.extend(
        [
            run_dir / "images" / f"{sample_id}.png",
            run_dir / "images" / f"{sample_id}.jpg",
            run_dir / "images" / f"{sample_id}.jpeg",
            run_dir / "images" / f"{sample_id}.webp",
        ]
    )
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path
    return None


def prev_sample_index(samples: list[dict[str, Any]], current_index: int) -> int:
    if not samples:
        return 0
    return max(0, min(int(current_index) - 1, len(samples) - 1))


def next_sample_index(samples: list[dict[str, Any]], current_index: int) -> int:
    if not samples:
        return 0
    return max(0, min(int(current_index) + 1, len(samples) - 1))


def sequence_rows_for_sample(sample_view: dict[str, Any]) -> list[dict[str, Any]]:
    sequence_group = sample_view.get("sequence_group", {"responses": []})
    responses = list(sequence_group.get("responses", []))
    responses.sort(key=lambda item: int(item.get("response_index", 0)))
    if responses:
        return responses

    out: list[dict[str, Any]] = []
    for pair_item in sample_view.get("pairs", []):
        out.append(
            {
                "response_index": int(pair_item.get("response_index", 0)),
                "final_reward": safe_float(pair_item.get("seq_final_reward"), 0.0),
                "adv": safe_float(pair_item.get("seq_adv"), 0.0),
                "base_reward": safe_float(pair_item.get("seq_base_reward"), 0.0),
                "aux_deducted_penalty": safe_float(pair_item.get("seq_aux_score"), 0.0),
                "response_text": str(pair_item.get("response_text", "")),
            }
        )
    out.sort(key=lambda item: int(item.get("response_index", 0)))
    return out


def sequence_row_by_response(sample_view: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = sequence_rows_for_sample(sample_view)
    return {int(item.get("response_index", 0)): item for item in rows}


def compute_sample_gt_first_hit_rollout(
    sample_view: dict[str, Any],
    *,
    success_threshold: float,
    success_epsilon: float,
) -> dict[int, int]:
    first_hit: dict[int, int] = {}
    pairs = sorted(sample_view.get("pairs", []), key=lambda item: int(item.get("response_index", 0)))
    for pair_item in pairs:
        response_index = int(pair_item.get("response_index", 0))
        for token in list(pair_item.get("chunks", [])):
            reward = float(token.get("reward_raw", token.get("reward", 0.0)) or 0.0)
            if reward < float(success_threshold) - float(success_epsilon):
                continue
            for gt_idx in token.get("attributed_gt_token_indices", []) or []:
                gt_token_index = int(gt_idx)
                if gt_token_index not in first_hit or response_index < first_hit[gt_token_index]:
                    first_hit[gt_token_index] = response_index
    return first_hit


def init_chunk_checkbox_state(label: str, correct_key: str, incorrect_key: str, lock_key: str) -> None:
    if correct_key not in st.session_state:
        st.session_state[correct_key] = label == "correct"
    if incorrect_key not in st.session_state:
        st.session_state[incorrect_key] = label == "incorrect"
    if lock_key not in st.session_state:
        st.session_state[lock_key] = False


def on_chunk_correct_toggle(correct_key: str, incorrect_key: str, lock_key: str) -> None:
    if bool(st.session_state.get(lock_key, False)):
        return
    st.session_state[lock_key] = True
    try:
        if bool(st.session_state.get(correct_key, False)):
            st.session_state[incorrect_key] = False
    finally:
        st.session_state[lock_key] = False


def on_chunk_incorrect_toggle(correct_key: str, incorrect_key: str, lock_key: str) -> None:
    if bool(st.session_state.get(lock_key, False)):
        return
    st.session_state[lock_key] = True
    try:
        if bool(st.session_state.get(incorrect_key, False)):
            st.session_state[correct_key] = False
    finally:
        st.session_state[lock_key] = False


def chunk_label_from_checkboxes(correct_key: str, incorrect_key: str) -> str:
    if bool(st.session_state.get(correct_key, False)):
        return "correct"
    if bool(st.session_state.get(incorrect_key, False)):
        return "incorrect"
    return "unmarked"


def table_chunk_info_text(chunk: dict[str, Any], *, seq_adv: float) -> str:
    chunk_adv = safe_float_or_none(chunk.get("chunk_normed_adv"))
    chunk_adv_text = "N/A" if chunk_adv is None else f"{chunk_adv:+.4f}"
    extreme_line = ""
    if bool(chunk.get("is_extreme_token")):
        extreme_line = f"\n⚠ extreme_adv: {chunk.get('extreme_adv_side', 'extreme')} tail"
    return (
        f"Chunk #{int(chunk.get('chunk_id', 0) or 0)}\n"
        f"reward: {reward_hover_text(chunk)}\n"
        f"chunk_adv: {chunk_adv_text}{extreme_line}\n"
        f"seq_adv: {seq_adv:+.4f}\n"
        f"matching_reward: {chunk.get('matching_reward')}\n"
        f"reward_source: {chunk.get('reward_source')}\n"
        f"pred_tr_index: {chunk.get('pred_tr_index')}\n"
        f"gt_chunk_id: {chunk.get('gt_chunk_id')}\n"
        f"pred_tr_matched: {bool(chunk.get('pred_tr_matched', False))}\n"
        f"token_start/token_end: {chunk.get('token_start')}:{chunk.get('token_end')}\n"
        f"pred_chunk_text: {chunk.get('pred_chunk_text', '')}\n"
        f"gt_chunk_text: {chunk.get('gt_chunk_text', '')}"
    )


def interpolate_rgb(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> str:
    value = max(0.0, min(float(ratio), 1.0))
    channels = [round(start_item + (end_item - start_item) * value) for start_item, end_item in zip(start, end, strict=True)]
    return f"#{channels[0]:02x}{channels[1]:02x}{channels[2]:02x}"


def table_chunk_adv_color(*, reward: float, chunk_adv: float | None) -> str:
    adv = 0.0 if chunk_adv is None else float(chunk_adv)
    if reward < 0.999999:
        ratio = min(max(-adv, 0.0) / 2.0, 1.0)
        return interpolate_rgb((254, 226, 226), (248, 113, 113), ratio)
    ratio = min(max(adv, 0.0) / 2.0, 1.0)
    return interpolate_rgb((220, 252, 231), (74, 222, 128), ratio)


def build_table_hover_maps(
    chunks: list[dict[str, Any]],
    *,
    seq_adv: float,
    highlight_extreme_adv: bool = False,
) -> dict[str, dict[int, Any]]:
    pred_info: dict[int, str] = {}
    gt_info: dict[int, str] = {}
    pred_bad: dict[int, bool] = {}
    gt_bad: dict[int, bool] = {}
    pred_color: dict[int, str] = {}
    gt_color: dict[int, str] = {}
    pred_link: dict[int, str] = {}
    gt_link: dict[int, str] = {}
    for chunk in chunks:
        chunk_id = int(chunk.get("chunk_id", 0) or 0)
        reward = safe_float(chunk.get("reward_raw", chunk.get("reward")), 0.0)
        chunk_adv = safe_float_or_none(chunk.get("chunk_normed_adv"))
        is_bad = reward < 0.999999
        is_extreme = bool(chunk.get("is_extreme_token"))
        color = table_chunk_adv_color(reward=reward, chunk_adv=chunk_adv)
        if highlight_extreme_adv and is_extreme:
            side = str(chunk.get("extreme_adv_side") or "high")
            color = "#fecaca" if side == "low" else "#fde68a"
        info_text = table_chunk_info_text(chunk, seq_adv=seq_adv)
        pred_idx_raw = chunk.get("pred_tr_index")
        gt_idx_raw = chunk.get("gt_chunk_id", chunk.get("chunk_id"))
        try:
            pred_idx = int(pred_idx_raw)
        except Exception:
            pred_idx = -1
        try:
            gt_idx = int(gt_idx_raw)
        except Exception:
            gt_idx = -1
        if pred_idx >= 0:
            pred_info[pred_idx] = info_text
            pred_bad[pred_idx] = is_bad
            pred_color[pred_idx] = color
            pred_link[pred_idx] = f"chunk-{chunk_id}"
        if gt_idx >= 0:
            gt_info[gt_idx] = info_text
            gt_bad[gt_idx] = is_bad
            gt_color[gt_idx] = color
            gt_link[gt_idx] = f"chunk-{chunk_id}"
    return {
        "pred_info": pred_info,
        "gt_info": gt_info,
        "pred_bad": pred_bad,
        "gt_bad": gt_bad,
        "pred_color": pred_color,
        "gt_color": gt_color,
        "pred_link": pred_link,
        "gt_link": gt_link,
    }


def build_formula_common_mask(gt_formula: str, predictions: list[str]) -> list[bool]:
    if not predictions:
        return []
    gt_text = str(gt_formula or "")
    out: list[bool] = []
    for idx, char in enumerate(gt_text):
        out.append(all(idx < len(prediction) and prediction[idx] == char for prediction in predictions))
    return out


def render_case_annotation_form(run_dir: Path, annotations: dict[str, Any], sample_view: dict[str, Any]) -> None:
    sample_id = str(sample_view.get("sample_id", ""))
    current_annotation = annotations.get("case_annotations", {}).get(sample_id, {})
    current_label = str(current_annotation.get("label", "unmarked"))
    current_note = str(current_annotation.get("note", ""))
    st.markdown("---")
    st.markdown("### Case 级标注")
    with st.form(f"case-form-{stable_id(sample_id)}"):
        new_label = st.selectbox(
            "当前 case 奖励是否正确",
            options=CASE_LABELS,
            index=CASE_LABELS.index(current_label) if current_label in CASE_LABELS else 0,
            format_func=case_label_text,
        )
        new_note = st.text_input("备注（可选）", value=current_note)
        submitted = st.form_submit_button("保存 Case 标注")
    if submitted:
        save_case_annotation(run_dir, sample_id, new_label, new_note)
        st.success("Case annotation saved.")
        st.rerun()


def render_pair_annotation_form(
    run_dir: Path,
    annotations: dict[str, Any],
    *,
    sample_id: str,
    pair_item: dict[str, Any],
    response_index: int,
) -> None:
    pair_id = annotation_pair_key(sample_id, pair_item, response_index)
    current_annotation = get_pair_annotation(annotations, sample_id, pair_id)
    current_label = str(current_annotation.get("label", "unmarked"))
    if current_label not in CASE_LABELS:
        current_label = "unmarked"
    current_note = str(current_annotation.get("note", ""))
    form_key = f"pair-form-{stable_id(pair_id)}"
    with st.form(form_key):
        st.markdown(f"#### GT-response 标注 r{response_index:02d}")
        new_label = st.selectbox(
            "当前 GT-response 奖励是否正确",
            options=CASE_LABELS,
            index=CASE_LABELS.index(current_label),
            format_func=case_label_text,
            key=f"pair-label-{stable_id(pair_id)}",
        )
        new_note = st.text_input(
            "备注（可选）",
            value=current_note,
            key=f"pair-note-{stable_id(pair_id)}",
        )
        submitted = st.form_submit_button("保存当前 GT-response 标注")
    if submitted:
        save_pair_annotation(
            run_dir,
            sample_id=sample_id,
            pair_id=pair_id,
            response_index=response_index,
            label=new_label,
            note=new_note,
        )
        st.success(f"GT-response r{response_index:02d} annotation saved.")
        st.rerun()


def render_extreme_adv_panel(
    pair_item: dict[str, Any],
    *,
    response_index: int,
    scope_key: str,
    task_type: str,
) -> bool:
    """Render control + table for extreme adv tokens. Returns whether focus mode is active."""
    pair_key_str = str(pair_item.get("pair_key", "")) or f"r{response_index:02d}"
    widget_key = f"focus-extreme-{scope_key}-{stable_id(pair_key_str)}"
    list_key = f"show-extreme-list-{widget_key}"
    extreme_tokens = list(pair_item.get("extreme_tokens", []) or [])
    is_extreme_response = bool(pair_item.get("is_extreme_response"))
    count = int(pair_item.get("extreme_token_count", len(extreme_tokens)) or 0)

    focus_extreme = bool(st.session_state.get(widget_key, False))
    btn_col, toggle_col = st.columns([1, 2])
    with btn_col:
        if is_extreme_response and st.button(
            f"显示极端 adv token ({count})",
            key=f"btn-{widget_key}",
            help="展开 token 列表并聚焦高亮极端 token",
        ):
            st.session_state[list_key] = True
            st.session_state[widget_key] = True
            focus_extreme = True
    with toggle_col:
        if is_extreme_response:
            focus_extreme = st.checkbox(
                f"聚焦高亮 r{response_index:02d} 的 {count} 个极端 token",
                key=widget_key,
                value=focus_extreme,
            )
        elif task_type:
            st.caption("该 rollout 无极端 adv token")

    if is_extreme_response and extreme_tokens:
        with st.expander(
            f"极端 adv token 列表 r{response_index:02d}",
            expanded=bool(st.session_state.get(list_key, False)) or focus_extreme,
        ):
            st.dataframe(extreme_tokens, use_container_width=True, hide_index=True)
    return focus_extreme


def render_text_sample(
    *,
    run_dir: Path,
    metadata: dict[str, Any],
    sample_view: dict[str, Any],
    annotations: dict[str, Any],
    max_tokens: int,
    scope_key: str,
    highlight_extreme_adv: bool,
) -> None:
    sample = sample_view.get("sample", {})
    sample_id = str(sample_view.get("sample_id", ""))
    gt_text = str(sample.get("ground_truth", ""))
    seq_by_resp = sequence_row_by_response(sample_view)
    success_threshold = safe_float(metadata.get("success_threshold"), 1.0)
    success_epsilon = safe_float(metadata.get("success_epsilon"), 1e-9)
    sample_gt_first_hit_rollout = compute_sample_gt_first_hit_rollout(
        sample_view,
        success_threshold=success_threshold,
        success_epsilon=success_epsilon,
    )

    render_text_block("Ground Truth", gt_text, height_px=180)

    for pair_item in sample_view.get("pairs", []):
        response_index = int(pair_item.get("response_index", 0))
        prediction_text = str(pair_item.get("prediction", {}).get("response_text", pair_item.get("response_text", "")))
        seq_row = seq_by_resp.get(response_index, {})
        seq_reward = safe_float(seq_row.get("final_reward", pair_item.get("seq_final_reward")), 0.0)
        seq_adv = safe_float(seq_row.get("adv", pair_item.get("seq_adv")), 0.0)
        extreme_tag = " | ⚠ extreme response" if bool(pair_item.get("is_extreme_response")) else ""

        st.markdown("---")
        st.markdown(
            f"### Rollout r{response_index:02d} | seq_reward={seq_reward:.4f} | seq_adv={colorized_adv_html(seq_adv)}{extreme_tag}",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Pair Reward", f"{safe_float(pair_item.get('pair_reward'), 0.0):.4f}")
        m2.metric("Mean Token Reward", f"{safe_float(pair_item.get('mean_token_reward', pair_item.get('mean_chunk_reward')), 0.0):.4f}")
        m3.metric("Edit Distance", f"{int(pair_item.get('edit_distance', 0) or 0)}")
        m4.metric("Token Count", f"{int(pair_item.get('token_count', pair_item.get('chunk_count', len(pair_item.get('chunks', [])))) or 0)}")
        m5.metric("Token Std", f"{safe_float(pair_item.get('sample_token_reward_std', pair_item.get('reward_std')), 0.0):.4f}")

        render_pair_annotation_form(
            run_dir,
            annotations,
            sample_id=sample_id,
            pair_item=pair_item,
            response_index=response_index,
        )
        render_text_block("Prediction", prediction_text, height_px=150)
        focus_extreme = render_extreme_adv_panel(
            pair_item,
            response_index=response_index,
            scope_key=scope_key,
            task_type="text",
        )
        st.markdown("#### Token 级映射（hover 查看奖励信息）")
        render_token_alignment_component(
            prediction_text,
            gt_text,
            list(pair_item.get("chunks", [])),
            max_tokens=max_tokens,
            sample_gt_first_hit_rollout=sample_gt_first_hit_rollout,
            highlight_extreme_adv=highlight_extreme_adv,
            focus_extreme_only=focus_extreme,
        )
        with st.expander(f"Token rows r{response_index:02d}", expanded=False):
            st.dataframe(list(pair_item.get("chunks", [])), use_container_width=True)


def render_table_pair_chunks(
    *,
    run_dir: Path,
    annotations: dict[str, Any],
    pair_item: dict[str, Any],
    seq_adv: float,
) -> None:
    pair_id = str(pair_item.get("pair_key", ""))
    chunks = list(pair_item.get("chunks", []))
    if not chunks:
        st.info("This rollout has no chunk rows.")
        return

    updates: list[dict[str, Any]] = []
    pair_hash = stable_id(pair_id)
    for chunk in chunks:
        cid = int(chunk.get("chunk_id", 0))
        ck = chunk_key(pair_id, cid)
        current_label = get_chunk_label(annotations, ck)
        current_note = get_chunk_note(annotations, ck)
        chunk_reward = float(chunk.get("reward_raw", chunk.get("reward", 0.0)) or 0.0)
        chunk_adv = safe_float_or_none(chunk.get("chunk_normed_adv"))
        if chunk_adv is None:
            chunk_adv_html = "<span style='color:#6b7280;font-weight:700'>N/A</span>"
        else:
            chunk_adv_html = f"<span style='color:{advantage_color(chunk_adv)};font-weight:700'>{chunk_adv:+.4f}</span>"

        correct_widget_key = f"chunk-correct-{pair_hash}-{cid}"
        incorrect_widget_key = f"chunk-incorrect-{pair_hash}-{cid}"
        lock_widget_key = f"chunk-lock-{pair_hash}-{cid}"
        note_widget_key = f"chunk-note-{pair_hash}-{cid}"
        init_chunk_checkbox_state(current_label, correct_widget_key, incorrect_widget_key, lock_widget_key)
        if note_widget_key not in st.session_state:
            st.session_state[note_widget_key] = current_note

        with st.container():
            st.markdown(
                (
                    f"##### Chunk {cid} | seq_adv={colorized_adv_html(seq_adv)} | "
                    f"reward={chunk_reward:.4f} | chunk_adv={chunk_adv_html}"
                ),
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                render_table_html("pred_chunk_text", str(chunk.get("pred_chunk_text", "")), height_px=140)
            with c2:
                render_table_html("gt_chunk_text", str(chunk.get("gt_chunk_text", "")), height_px=140)
            details = [
                f"- `pred_tr_index`: `{int(chunk.get('pred_tr_index', -1) or -1)}`",
                f"- `pred_tr_matched`: `{bool(chunk.get('pred_tr_matched', False))}`",
                f"- `reward_source`: `{chunk.get('reward_source')}`",
                f"- `matching_reward`: `{chunk.get('matching_reward')}`",
                f"- `token_start/token_end`: `{int(chunk.get('token_start', -1) or -1)}:{int(chunk.get('token_end', -1) or -1)}`",
            ]
            st.markdown("\n".join(details))

            ann_col1, ann_col2 = st.columns([1, 2])
            with ann_col1:
                st.markdown(f"**Chunk {cid} 标注**")
                st.checkbox(
                    "正确",
                    key=correct_widget_key,
                    on_change=on_chunk_correct_toggle,
                    args=(correct_widget_key, incorrect_widget_key, lock_widget_key),
                )
                st.checkbox(
                    "错误",
                    key=incorrect_widget_key,
                    on_change=on_chunk_incorrect_toggle,
                    args=(correct_widget_key, incorrect_widget_key, lock_widget_key),
                )
            with ann_col2:
                st.text_input(
                    f"Chunk {cid} 备注",
                    key=note_widget_key,
                    placeholder="可选：写下你认为该 chunk 奖励正确/错误的原因",
                )
            st.markdown("---")
        updates.append(
            {
                "pair_key": pair_id,
                "chunk_id": cid,
                "chunk_key": ck,
                "label": chunk_label_from_checkboxes(correct_widget_key, incorrect_widget_key),
                "note": str(st.session_state.get(note_widget_key, "")).strip(),
            }
        )

    if st.button("保存当前 Rollout 全部 Chunk 标注", type="primary", key=f"save-all-chunks-{pair_hash}"):
        save_chunk_annotations(run_dir, updates)
        st.success("Chunk annotations saved.")
        st.rerun()


def render_table_sample(
    *,
    run_dir: Path,
    sample_view: dict[str, Any],
    annotations: dict[str, Any],
    scope_key: str,
    highlight_extreme_adv: bool,
) -> None:
    sample = sample_view.get("sample", {})
    sample_id = str(sample_view.get("sample_id", ""))
    gt_table = str(sample.get("ground_truth", ""))
    seq_by_resp = sequence_row_by_response(sample_view)

    for pair_item in sample_view.get("pairs", []):
        response_index = int(pair_item.get("response_index", 0))
        prediction_text = str(pair_item.get("prediction", {}).get("response_text", pair_item.get("response_text", "")))
        seq_row = seq_by_resp.get(response_index, {})
        seq_reward = safe_float(seq_row.get("final_reward", pair_item.get("seq_final_reward")), 0.0)
        seq_adv = safe_float(seq_row.get("adv", pair_item.get("seq_adv")), 0.0)
        chunks = list(pair_item.get("chunks", []))
        extreme_tag = " | ⚠ extreme response" if bool(pair_item.get("is_extreme_response")) else ""
        focus_extreme = render_extreme_adv_panel(
            pair_item,
            response_index=response_index,
            scope_key=scope_key,
            task_type="table",
        )
        hover_maps = build_table_hover_maps(
            chunks,
            seq_adv=seq_adv,
            highlight_extreme_adv=highlight_extreme_adv or focus_extreme,
        )

        st.markdown("---")
        st.markdown(
            f"### Rollout r{response_index:02d} | seq_reward={seq_reward:.4f} | seq_adv={colorized_adv_html(seq_adv)}{extreme_tag}",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pair Reward", f"{safe_float(pair_item.get('pair_reward'), 0.0):.4f}")
        m2.metric("Chunk Count", f"{int(pair_item.get('chunk_count', len(chunks)) or len(chunks))}")
        m3.metric("Mean Chunk Reward", f"{safe_float(pair_item.get('mean_chunk_reward'), 0.0):.4f}")
        m4.metric("Reward Std", f"{safe_float(pair_item.get('reward_std'), 0.0):.4f}")

        render_pair_annotation_form(
            run_dir,
            annotations,
            sample_id=sample_id,
            pair_item=pair_item,
            response_index=response_index,
        )
        render_interactive_table_pair_html(
            left_title="Ground Truth Table",
            left_table_html=gt_table,
            right_title="Prediction Table",
            right_table_html=prediction_text,
            left_row_color_map=hover_maps["gt_color"],
            right_row_color_map=hover_maps["pred_color"],
            left_row_info_map=hover_maps["gt_info"],
            right_row_info_map=hover_maps["pred_info"],
            left_row_bad_reward_map=hover_maps["gt_bad"],
            right_row_bad_reward_map=hover_maps["pred_bad"],
            left_row_link_map=hover_maps["gt_link"],
            right_row_link_map=hover_maps["pred_link"],
        )
        tleft, tright = st.columns(2)
        with tleft:
            render_code_block("Ground Truth Source", format_table_source_with_tr_newlines(gt_table), language="html")
        with tright:
            render_code_block("Prediction Source", format_table_source_with_tr_newlines(prediction_text), language="html")

        with st.expander(f"Raw chunk rows r{response_index:02d}", expanded=False):
            st.dataframe(chunks, use_container_width=True)


def _is_formula_token_run(sample_view: dict[str, Any], metadata: dict[str, Any]) -> bool:
    if str(metadata.get("strategy", "")) == "formula_token":
        return True
    pairs = sample_view.get("pairs", [])
    if pairs and str(pairs[0].get("strategy", "")) == "formula_token":
        return True
    return any(bool(pair_item.get("chunks")) and "is_fallback" in (pair_item.get("chunks") or [{}])[0] for pair_item in pairs)


def render_formula_sample(
    *,
    run_dir: Path,
    sample_view: dict[str, Any],
    annotations: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    scope_key: str = "",
    highlight_extreme_adv: bool = False,
) -> None:
    sample = sample_view.get("sample", {})
    sample_id = str(sample_view.get("sample_id", ""))
    gt_formula = str(sample.get("ground_truth", ""))
    seq_by_resp = sequence_row_by_response(sample_view)
    rollout_predictions = [
        str(pair_item.get("prediction", {}).get("response_text", pair_item.get("response_text", "")))
        for pair_item in sample_view.get("pairs", [])
    ]
    common_mask = build_formula_common_mask(gt_formula, rollout_predictions)
    token_run = _is_formula_token_run(sample_view, metadata or {})

    if token_run:
        first_pair = sample_view.get("pairs", [{}])[0]
        st.caption(
            "Formula token-level 可视化：实线高亮=实际 visual token 打分；虚线灰底=未覆盖片段，adv 使用 chunk_seq_adv fallback。"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Case element mean", f"{safe_float(first_pair.get('case_element_mean'), 0.0):.4f}")
        c2.metric("Case element std", f"{safe_float(first_pair.get('case_element_std'), 0.0):.4f}")
        c3.metric("Case element count", f"{int(first_pair.get('case_element_count', 0) or 0)}")
        c4.metric("Reward mode", str(metadata.get("reward_mode", "formula_token") if metadata else "formula_token"))

    for pair_item in sample_view.get("pairs", []):
        response_index = int(pair_item.get("response_index", 0))
        prediction_text = str(pair_item.get("prediction", {}).get("response_text", pair_item.get("response_text", "")))
        seq_row = seq_by_resp.get(response_index, {})
        seq_reward = safe_float(
            seq_row.get("final_reward", pair_item.get("seq_final_reward", pair_item.get("chunk_sequence_reward"))),
            0.0,
        )
        seq_adv = safe_float(seq_row.get("adv", pair_item.get("seq_adv", pair_item.get("chunk_seq_adv"))), 0.0)
        chunks = list(pair_item.get("chunks", []))
        extreme_tag = " | ⚠ extreme response" if bool(pair_item.get("is_extreme_response")) else ""
        focus_extreme = render_extreme_adv_panel(
            pair_item,
            response_index=response_index,
            scope_key=scope_key,
            task_type="formula",
        )

        st.markdown("---")
        st.markdown(
            f"### Rollout r{response_index:02d} | chunk_seq_reward={seq_reward:.4f} | chunk_seq_adv={colorized_adv_html(seq_adv)}{extreme_tag}",
            unsafe_allow_html=True,
        )
        if token_run:
            element_count = int(pair_item.get("element_count", pair_item.get("token_count", 0)) or 0)
            fallback_count = int(pair_item.get("fallback_chunk_count", 0) or 0)
            if not fallback_count:
                fallback_count = sum(1 for c in chunks if bool(c.get("is_fallback")))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Element tokens", element_count)
            m2.metric("Fallback chunks", fallback_count)
            m3.metric("Mean chunk reward", f"{safe_float(pair_item.get('mean_token_reward', pair_item.get('mean_chunk_reward')), 0.0):.4f}")
            m4.metric("Pair reward (F1)", f"{safe_float(pair_item.get('pair_reward'), 0.0):.4f}")

            render_pair_annotation_form(
                run_dir,
                annotations,
                sample_id=sample_id,
                pair_item=pair_item,
                response_index=response_index,
            )
            render_formula_token_interactive_pair(
                gt_text=gt_formula,
                pred_text=prediction_text,
                chunks=chunks,
                case_mean=safe_float(pair_item.get("case_element_mean"), 0.0),
                case_std=safe_float(pair_item.get("case_element_std"), 0.0),
                chunk_seq_adv=seq_adv,
                response_index=response_index,
                highlight_extreme_adv=highlight_extreme_adv,
                focus_extreme_only=focus_extreme,
            )
            with st.expander(f"Token/chunk rows r{response_index:02d}", expanded=False):
                detail_rows = formula_chunk_detail_rows(chunks)
                if detail_rows:
                    st.dataframe(detail_rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No chunk rows.")
            st.markdown("**LaTeX 渲染预览**")
            c1, c2 = st.columns(2)
            with c1:
                render_formula_preview("Ground Truth", gt_formula, common_mask=common_mask)
            with c2:
                render_formula_preview("Prediction", prediction_text, common_mask=common_mask)
            continue

        m1, m2, m3 = st.columns(3)
        m1.metric("Pair Reward", f"{safe_float(pair_item.get('pair_reward'), 0.0):.4f}")
        m2.metric("Sequence Reward", f"{seq_reward:.4f}")
        m3.metric("Sequence Adv", f"{seq_adv:+.4f}")
        render_pair_annotation_form(
            run_dir,
            annotations,
            sample_id=sample_id,
            pair_item=pair_item,
            response_index=response_index,
        )
        st.caption(
            f"formula_server_url={pair_item.get('formula_server_url') or seq_row.get('formula_server_url') or 'N/A'} | "
            f"formula_server_failed={pair_item.get('formula_server_failed')}"
        )

        c1, c2 = st.columns(2)
        with c1:
            render_formula_preview("Ground Truth（源码 + 渲染）", gt_formula, common_mask=common_mask)
        with c2:
            render_formula_preview("Prediction（源码 + 渲染）", prediction_text, common_mask=common_mask)


def render_sample_header(sample_view: dict[str, Any], run_dir: Path) -> None:
    sample = sample_view.get("sample", {})
    sample_id = str(sample_view.get("sample_id", ""))
    prompt_text = str(
        sample.get("prompt_text")
        or sample_view.get("pairs", [{}])[0].get("prompt_text")
        or sample.get("rollout_prompt_text")
        or ""
    )
    image_path = resolve_sample_image_path(sample, run_dir, sample_id)

    st.markdown("### 当前样本图片与 Prompt")
    left, right = st.columns([1, 2])
    with left:
        if image_path is not None:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)
        else:
            st.info("No image found for this sample.")
    with right:
        if prompt_text:
            render_text_block("Prompt", prompt_text, height_px=180)


def render_sequence_overview(sample_view: dict[str, Any], adv_stats: dict[str, Any] | None = None) -> None:
    rows = sequence_rows_for_sample(sample_view)
    if not rows:
        st.info("No sequence-level responses found.")
        return

    response_flags = (adv_stats or {}).get("response_flags", {})
    st.markdown("### Sequence-level 对比（当前 sample 全部 rollouts）")
    overview_rows = []
    for row in rows:
        response_index = int(row.get("response_index", 0))
        flag = response_flags.get(response_index, {})
        pair_extreme = any(
            int(item.get("response_index", -1)) == response_index and bool(item.get("is_extreme_response"))
            for item in sample_view.get("pairs", [])
        )
        overview_rows.append(
            {
                "response_index": response_index,
                "final_reward": round(safe_float(row.get("final_reward"), 0.0), 6),
                "adv": round(safe_float(row.get("adv"), 0.0), 6),
                "extreme_response": "⚠ yes" if pair_extreme else "no",
                "extreme_token_count": int(
                    next(
                        (
                            int(item.get("extreme_token_count", 0) or 0)
                            for item in sample_view.get("pairs", [])
                            if int(item.get("response_index", -1)) == response_index
                        ),
                        flag.get("extreme_token_count", 0),
                    )
                ),
                "chunk_seq_reward": round(safe_float(row.get("chunk_seq_reward", row.get("final_reward")), 0.0), 6),
                "chunk_seq_adv": round(safe_float(row.get("chunk_seq_adv", row.get("adv")), 0.0), 6),
                "element_count": int(row.get("element_count", 0) or 0),
                "fallback_chunk_count": int(row.get("fallback_chunk_count", 0) or 0),
                "base_reward": round(safe_float(row.get("base_reward"), 0.0), 6),
                "aux_deducted_penalty": round(safe_float(row.get("aux_deducted_penalty"), 0.0), 6),
                "response_preview": str(row.get("response_text", ""))[:120],
            }
        )
    st.dataframe(overview_rows, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="OCR RL Reward QC Viewer (visualization-v1)", layout="wide")
    st.title("OCR RL Reward QC Viewer (visualization-v1)")

    with st.sidebar:
        st.header("Run Selection")
        run_root_str = st.text_input(
            "Run root",
            value=str(DEFAULT_RUN_ROOT),
            help="可填 runs/ 或 runs/formula_token_passk/；会递归查找含 metadata.json 的 run 目录",
        )
        runs = discover_runs_cached(run_root_str)
        if not runs:
            st.warning(
                "No runs found under the selected root. "
                f"Formula pass@k 示例: {FORMULA_TOKEN_RUN_ROOT}"
            )
            return
        selected_run = st.selectbox("Choose run", options=runs, index=0, format_func=lambda path: Path(path).name)
        if st.button("Refresh run data"):
            st.cache_data.clear()
            st.rerun()

    run_dir = Path(selected_run)
    with st.sidebar:
        st.header("Adv Extremeness")
        abs_adv_threshold = st.number_input(
            "|adv| threshold",
            min_value=0.0,
            value=2.0,
            step=0.1,
            help="标记 abs(token_adv) 大于等于该值的 token 为极端 token",
        )
        highlight_extreme_adv = st.checkbox("全局高亮极端 adv token", value=False)
        only_extreme_samples = st.checkbox("仅显示含极端 response 的 sample", value=False)

    pairs, adv_stats = load_pairs_with_adv_stats(selected_run, float(abs_adv_threshold))
    metadata = load_run_data(selected_run)["metadata"]
    annotations = load_annotations(run_dir)

    if not pairs:
        st.error("No pair rows found in this run.")
        return

    sample_views = rebuild_sample_views(pairs)

    with st.sidebar:
        st.caption(format_threshold_summary(adv_stats))
        st.caption(
            f"极端 response: {adv_stats.get('extreme_response_count', 0)} / {len(pairs)} | "
            f"极端 token: {adv_stats.get('extreme_token_count', 0)}"
        )

    with st.sidebar:
        st.header("Filters")
        task_type_filter = st.selectbox("Task type", options=["all", "text", "table", "formula"], index=0)
        sample_filter = st.text_input("Sample ID contains", value="")
        pair_status_filter = st.selectbox(
            "GT-response annotation status",
            options=["all", *CASE_LABELS],
            index=0,
            format_func=lambda value: "全部" if value == "all" else case_label_text(value),
        )
        max_tokens = st.number_input("Max tokens to render (text)", min_value=32, max_value=4096, value=512, step=32)

    visible_samples = list(sample_views)
    if task_type_filter != "all":
        visible_samples = [item for item in visible_samples if str(item.get("task_type", "")) == task_type_filter]
    if sample_filter.strip():
        needle = sample_filter.strip()
        visible_samples = [item for item in visible_samples if needle in str(item.get("sample_id", ""))]
    if pair_status_filter != "all":
        visible_samples = [
            item
            for item in visible_samples
            if sample_has_pair_label(item, annotations, pair_status_filter)
        ]
    if only_extreme_samples:
        visible_samples = [item for item in visible_samples if sample_has_extreme_response(item)]

    pair_stats = summarize_pair_labels(visible_samples, annotations)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("GT-response 总数", f"{pair_stats['total']}")
    m2.metric("未标注", f"{pair_stats['unmarked']}", format_ratio(pair_stats["unmarked"], pair_stats["total"]))
    m3.metric("错误", f"{pair_stats['incorrect']}", format_ratio(pair_stats["incorrect"], pair_stats["total"]))
    m4.metric("正确", f"{pair_stats['correct']}", format_ratio(pair_stats["correct"], pair_stats["total"]))
    m5.metric("轻微错误", f"{pair_stats['minor_error']}", format_ratio(pair_stats["minor_error"], pair_stats["total"]))

    if not visible_samples:
        st.warning("No samples match current filters.")
        with st.expander("Run metadata", expanded=False):
            st.json(metadata)
        return

    scope_key = stable_id(f"{selected_run}|{task_type_filter}|{sample_filter.strip()}|{pair_status_filter}")
    sample_select_key = f"sample-select-{scope_key}"
    st.session_state.setdefault(sample_select_key, 0)
    st.session_state[sample_select_key] = max(
        0,
        min(int(st.session_state[sample_select_key]), len(visible_samples) - 1),
    )

    with st.sidebar:
        st.markdown("#### Data Navigation")
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("⏮ 上一条数据", use_container_width=True, key=f"sample-prev-{scope_key}"):
                st.session_state[sample_select_key] = prev_sample_index(
                    visible_samples,
                    int(st.session_state[sample_select_key]),
                )
        with next_col:
            if st.button("下一条数据 ⏭", use_container_width=True, key=f"sample-next-{scope_key}"):
                st.session_state[sample_select_key] = next_sample_index(
                    visible_samples,
                    int(st.session_state[sample_select_key]),
                )
        selected_sample_index = st.selectbox(
            "Choose sample",
            options=list(range(len(visible_samples))),
            key=sample_select_key,
            format_func=lambda idx: (
                f"{'⚠ ' if sample_has_extreme_response(visible_samples[idx]) else ''}"
                f"{visible_samples[idx].get('sample_id', '')} | "
                f"{visible_samples[idx].get('task_type', '')} | "
                f"rollouts={len(visible_samples[idx].get('pairs', []))}"
            ),
        )

    current_sample = visible_samples[int(selected_sample_index)]
    st.subheader(
        f"task_type: {current_sample.get('task_type', '')}"
        + (" | ⚠ 含极端 adv response" if sample_has_extreme_response(current_sample) else "")
    )
    render_sequence_overview(current_sample, adv_stats)
    render_sample_header(current_sample, run_dir)

    task_type = str(current_sample.get("task_type", ""))
    if task_type == "text":
        render_text_sample(
            run_dir=run_dir,
            metadata=metadata,
            sample_view=current_sample,
            annotations=annotations,
            max_tokens=int(max_tokens),
            scope_key=scope_key,
            highlight_extreme_adv=highlight_extreme_adv,
        )
    elif task_type == "table":
        render_table_sample(
            run_dir=run_dir,
            sample_view=current_sample,
            annotations=annotations,
            scope_key=scope_key,
            highlight_extreme_adv=highlight_extreme_adv,
        )
    elif task_type == "formula":
        render_formula_sample(
            run_dir=run_dir,
            sample_view=current_sample,
            annotations=annotations,
            metadata=metadata,
            scope_key=scope_key,
            highlight_extreme_adv=highlight_extreme_adv,
        )
    else:
        st.warning(f"Unsupported task_type: {task_type}")
        st.dataframe(current_sample.get("pairs", []), use_container_width=True)

    with st.expander("Run metadata", expanded=False):
        st.json(metadata)


if __name__ == "__main__":
    main()
