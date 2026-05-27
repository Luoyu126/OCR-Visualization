from __future__ import annotations

import html
import json
import re
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from lxml import etree, html as lxml_html

from io_utils import stable_id


PAIR_COLORS = [
    "#fde68a",
    "#bfdbfe",
    "#bbf7d0",
    "#fecaca",
    "#e9d5ff",
    "#fbcfe8",
    "#c7d2fe",
    "#99f6e4",
    "#fed7aa",
    "#ddd6fe",
    "#a7f3d0",
    "#e5e7eb",
]

TOKEN_CORRECT_COLOR = "#dcfce7"
TOKEN_ERROR_COLOR = "#fee2e2"


def advantage_color(value: float) -> str:
    if value > 0:
        return "#16a34a"
    if value < 0:
        return "#dc2626"
    return "#6b7280"


def colorized_adv_html(value: float) -> str:
    color = advantage_color(value)
    return f"<span style='color:{color};font-weight:700'>{value:+.4f}</span>"


def token_reward_color(reward: float) -> str:
    return TOKEN_CORRECT_COLOR if reward >= 0.999999 else TOKEN_ERROR_COLOR


def render_text_block(title: str, content: str, *, height_px: int = 160) -> None:
    safe_text = html.escape(content or "")
    st.markdown(f"**{title}**")
    st.markdown(
        f"""
<div style="
  border: 1px solid #d9d9d9;
  border-radius: 8px;
  padding: 12px;
  background: #ffffff;
  color: #111111;
  white-space: pre-wrap;
  overflow-y: auto;
  min-height: {height_px}px;
  max-height: {height_px}px;
  line-height: 1.55;
  font-size: 14px;
">
{safe_text}
</div>
""",
        unsafe_allow_html=True,
    )


def render_code_block(title: str, content: str, *, language: str = "text") -> None:
    _ = language
    st.markdown(f"**{title}**")
    safe_text = html.escape(str(content or ""))
    st.markdown(
        f"""
<div style="
  border: 1px solid #d9d9d9;
  border-radius: 8px;
  padding: 12px;
  background: #f8fafc;
  color: #111111;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  max-width: 100%;
  box-sizing: border-box;
  overflow-x: hidden;
  overflow-y: auto;
  max-height: 320px;
  line-height: 1.45;
  font-size: 13px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
">{safe_text}</div>
""",
        unsafe_allow_html=True,
    )


def format_table_source_with_tr_newlines(source: str) -> str:
    text = str(source or "")
    if not text.strip():
        return text
    return re.sub(r"</tr>(?!\s*\n)", "</tr>\n", text, flags=re.IGNORECASE)


def wrap_html_if_needed(table_html: str) -> str:
    text = str(table_html or "").strip()
    if "<html" in text.lower():
        return text
    if "<table" in text.lower():
        return f"<html><body>{text}</body></html>"
    return f"<html><body><table>{text}</table></body></html>"


def _extract_table_content(table_html: str) -> str:
    wrapped = wrap_html_if_needed(table_html)
    lowered = wrapped.lower()
    body_start = lowered.find("<body")
    if body_start < 0:
        return wrapped
    body_open_end = lowered.find(">", body_start)
    body_close = lowered.rfind("</body>")
    if body_open_end < 0 or body_close < 0:
        return wrapped
    return wrapped[body_open_end + 1 : body_close]


def build_pair_color_maps(pairs: list[list[int]]) -> dict[str, dict[int, str]]:
    pred_colors: dict[int, str] = {}
    gt_colors: dict[int, str] = {}
    for idx, pair in enumerate(pairs):
        if len(pair) < 2:
            continue
        pred_idx, gt_idx = int(pair[0]), int(pair[1])
        color = PAIR_COLORS[idx % len(PAIR_COLORS)]
        pred_colors[pred_idx] = color
        gt_colors[gt_idx] = color
    return {"pred_colors": pred_colors, "gt_colors": gt_colors}


def extract_matched_tr_pairs_from_chunks(chunks: list[dict[str, Any]]) -> list[list[int]]:
    pairs: list[list[int]] = []
    for chunk in chunks:
        gt_idx = int(chunk.get("gt_chunk_id", chunk.get("chunk_id", -1)) or -1)
        pred_idx = int(chunk.get("pred_tr_index", -1) or -1)
        matched = bool(chunk.get("pred_tr_matched", False))
        if matched and gt_idx >= 0 and pred_idx >= 0:
            pairs.append([pred_idx, gt_idx])
    pairs.sort(key=lambda item: (item[1], item[0]))
    return pairs


def _highlighted_table_content(
    table_html: str,
    row_color_map: dict[int, str],
    row_info_map: dict[int, str] | None = None,
    row_bad_reward_map: dict[int, bool] | None = None,
    row_link_map: dict[int, str] | None = None,
) -> str:
    row_info_map = row_info_map or {}
    row_bad_reward_map = row_bad_reward_map or {}
    row_link_map = row_link_map or {}
    if not row_color_map and not row_info_map and not row_bad_reward_map and not row_link_map:
        return _extract_table_content(table_html)
    wrapped = wrap_html_if_needed(table_html)
    try:
        root = lxml_html.fromstring(wrapped, parser=lxml_html.HTMLParser(remove_comments=True, encoding="utf-8"))
        tables = root.xpath("body/table")
        if not tables:
            return _extract_table_content(table_html)
        table_node = tables[0]
        trs = table_node.xpath(".//tr")
        for idx, tr in enumerate(trs):
            color = row_color_map.get(idx)
            existing = tr.attrib.get("style", "").strip()
            if existing and not existing.endswith(";"):
                existing = f"{existing};"
            style_parts: list[str] = []
            if color:
                style_parts.append(f"background-color: {color};")
            if row_bad_reward_map.get(idx) and not color:
                style_parts.append("background-color: #fee2e2;")
            if row_info_map.get(idx):
                tr.attrib["data-row-info"] = row_info_map[idx]
                tr.attrib["title"] = row_info_map[idx]
            if row_link_map.get(idx):
                tr.attrib["data-link-id"] = row_link_map[idx]
            cursor = " cursor: help;" if row_info_map.get(idx) else ""
            tr.attrib["style"] = f"{existing} {' '.join(style_parts)} color: #111827;{cursor}".strip()
        return etree.tostring(table_node, encoding="unicode", method="html")
    except Exception:
        return _extract_table_content(table_html)


def render_table_html(
    title: str,
    table_html: str,
    *,
    height_px: int | None = None,
    row_color_map: dict[int, str] | None = None,
    row_info_map: dict[int, str] | None = None,
    row_bad_reward_map: dict[int, bool] | None = None,
    row_link_map: dict[int, str] | None = None,
) -> None:
    safe_content = _highlighted_table_content(
        table_html,
        row_color_map or {},
        row_info_map=row_info_map,
        row_bad_reward_map=row_bad_reward_map,
        row_link_map=row_link_map,
    )
    style_parts = [
        "border: 1px solid #d9d9d9",
        "border-radius: 8px",
        "padding: 10px",
        "background: #ffffff",
    ]
    if height_px is None:
        style_parts.extend(["overflow-x: auto", "overflow-y: visible"])
    else:
        style_parts.extend([f"min-height: {height_px}px", f"max-height: {height_px}px", "overflow: auto"])
    container_style = "; ".join(style_parts) + ";"
    st.markdown(f"**{title}**")
    st.markdown(
        f"""
<div style="{container_style}">
  <style>
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    td, th {{ border: 1px solid #d1d5db; padding: 4px 6px; vertical-align: top; }}
    tr[title] {{ cursor: help; }}
  </style>
  {safe_content}
</div>
""",
        unsafe_allow_html=True,
    )


def render_interactive_table_html(
    title: str,
    table_html: str,
    *,
    height_px: int | None = None,
    row_color_map: dict[int, str] | None = None,
    row_info_map: dict[int, str] | None = None,
    row_bad_reward_map: dict[int, bool] | None = None,
    row_link_map: dict[int, str] | None = None,
) -> None:
    safe_content = _highlighted_table_content(
        table_html,
        row_color_map or {},
        row_info_map=row_info_map,
        row_bad_reward_map=row_bad_reward_map,
        row_link_map=row_link_map,
    )
    component_id = f"table-hover-{stable_id(title + table_html[:1000] + str(len(row_info_map or {})))}"
    max_height_css = "none" if height_px is None else f"{height_px}px"
    min_height_css = "auto" if height_px is None else f"{height_px}px"
    default_info = "Hover a matched table row to inspect chunk reward details."
    st.markdown(f"**{title}**")
    components.html(
        f"""
<style>
#{component_id} {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: #111827;
}}
#{component_id} .table-wrap {{
  border: 1px solid #d9d9d9;
  border-radius: 8px;
  padding: 10px;
  background: #ffffff;
  overflow: auto;
  min-height: {min_height_css};
  max-height: {max_height_css};
}}
#{component_id} table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
#{component_id} td, #{component_id} th {{ border: 1px solid #d1d5db; padding: 4px 6px; vertical-align: top; }}
#{component_id} tr[data-row-info] {{ cursor: help; transition: box-shadow 0.12s ease, filter 0.12s ease; }}
#{component_id} tr[data-row-info].active {{ box-shadow: inset 0 0 0 2px rgba(59, 130, 246, 0.75); filter: brightness(1.03); }}
#{component_id} .info-card {{
  margin-top: 10px;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  background: #f8fafc;
  padding: 12px;
}}
#{component_id} .info-title {{
  font-size: 13px;
  color: #4b5563;
  font-weight: 600;
  margin-bottom: 6px;
}}
#{component_id} .info-body {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: 13px;
  line-height: 1.45;
  min-height: 160px;
  max-height: 260px;
  overflow-y: auto;
}}
</style>
<div id="{component_id}">
  <div class="table-wrap">
    {safe_content}
  </div>
  <div class="info-card">
    <div class="info-title">Chunk Detail</div>
    <div class="info-body" id="{component_id}-info">{html.escape(default_info)}</div>
  </div>
</div>
<script>
(function() {{
  const root = document.getElementById("{component_id}");
  if (!root) return;
  const infoBox = document.getElementById("{component_id}-info");
  const defaultInfo = {json.dumps(default_info)};
  function clearActive() {{
    root.querySelectorAll("tr[data-row-info].active").forEach((el) => el.classList.remove("active"));
  }}
  root.querySelectorAll("tr[data-row-info]").forEach((el) => {{
    el.addEventListener("mouseenter", () => {{
      clearActive();
      el.classList.add("active");
      if (infoBox) infoBox.textContent = el.getAttribute("data-row-info") || defaultInfo;
    }});
    el.addEventListener("mouseleave", () => {{
      clearActive();
      if (infoBox) infoBox.textContent = defaultInfo;
    }});
  }});
}})();
</script>
""",
        height=(height_px or 420) + 160,
        scrolling=True,
    )


def render_interactive_table_pair_html(
    *,
    left_title: str,
    left_table_html: str,
    right_title: str,
    right_table_html: str,
    left_row_color_map: dict[int, str] | None = None,
    right_row_color_map: dict[int, str] | None = None,
    left_row_info_map: dict[int, str] | None = None,
    right_row_info_map: dict[int, str] | None = None,
    left_row_bad_reward_map: dict[int, bool] | None = None,
    right_row_bad_reward_map: dict[int, bool] | None = None,
    left_row_link_map: dict[int, str] | None = None,
    right_row_link_map: dict[int, str] | None = None,
    table_height_px: int = 300,
) -> None:
    left_content = _highlighted_table_content(
        left_table_html,
        left_row_color_map or {},
        row_info_map=left_row_info_map,
        row_bad_reward_map=left_row_bad_reward_map,
        row_link_map=left_row_link_map,
    )
    right_content = _highlighted_table_content(
        right_table_html,
        right_row_color_map or {},
        row_info_map=right_row_info_map,
        row_bad_reward_map=right_row_bad_reward_map,
        row_link_map=right_row_link_map,
    )
    component_id = f"table-pair-hover-{stable_id(left_table_html[:800] + right_table_html[:800])}"
    default_info = "Hover a matched GT or Prediction table row to inspect chunk reward details."
    components.html(
        f"""
<style>
#{component_id} {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: #111827;
}}
#{component_id} .tables {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 12px;
}}
#{component_id} .table-title {{
  font-weight: 600;
  margin-bottom: 6px;
}}
#{component_id} .table-wrap {{
  border: 1px solid #d9d9d9;
  border-radius: 8px;
  padding: 10px;
  background: #ffffff;
  overflow: auto;
  max-height: {int(table_height_px)}px;
}}
#{component_id} table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
#{component_id} td, #{component_id} th {{
  border: 1px solid #d1d5db;
  padding: 4px 6px;
  vertical-align: top;
  overflow-wrap: anywhere;
  word-break: break-word;
}}
#{component_id} tr[data-row-info] {{ cursor: help; transition: box-shadow 0.12s ease, filter 0.12s ease; }}
#{component_id} tr[data-row-info].active, #{component_id} tr[data-link-id].active {{
  box-shadow: inset 0 0 0 2px rgba(59, 130, 246, 0.75);
  filter: brightness(1.03);
}}
#{component_id} .info-card {{
  margin-top: 12px;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  background: #f8fafc;
  padding: 12px;
}}
#{component_id} .info-title {{
  font-size: 13px;
  color: #4b5563;
  font-weight: 600;
  margin-bottom: 6px;
}}
#{component_id} .info-body {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: 13px;
  line-height: 1.45;
  min-height: 120px;
  max-height: 220px;
  overflow-y: auto;
}}
</style>
<div id="{component_id}">
  <div class="tables">
    <div>
      <div class="table-title">{html.escape(left_title)}</div>
      <div class="table-wrap">{left_content}</div>
    </div>
    <div>
      <div class="table-title">{html.escape(right_title)}</div>
      <div class="table-wrap">{right_content}</div>
    </div>
  </div>
  <div class="info-card">
    <div class="info-title">Chunk Detail</div>
    <div class="info-body" id="{component_id}-info">{html.escape(default_info)}</div>
  </div>
</div>
<script>
(function() {{
  const root = document.getElementById("{component_id}");
  if (!root) return;
  const infoBox = document.getElementById("{component_id}-info");
  const defaultInfo = {json.dumps(default_info)};
  function clearActive() {{
    root.querySelectorAll("tr.active").forEach((el) => el.classList.remove("active"));
  }}
  root.querySelectorAll("tr[data-row-info]").forEach((el) => {{
    el.addEventListener("mouseenter", () => {{
      clearActive();
      const linkId = el.getAttribute("data-link-id");
      if (linkId) {{
        root.querySelectorAll('tr[data-link-id="' + linkId + '"]').forEach((row) => row.classList.add("active"));
      }} else {{
        el.classList.add("active");
      }}
      if (infoBox) infoBox.textContent = el.getAttribute("data-row-info") || defaultInfo;
    }});
    el.addEventListener("mouseleave", () => {{
      clearActive();
    }});
  }});
}})();
</script>
""",
        height=int(table_height_px) + 240,
        scrolling=False,
    )


def strip_html_tags(text: str, max_len: int = 80) -> str:
    out = re.sub(r"<[^>]+>", " ", str(text or ""))
    out = re.sub(r"\s+", " ", out).strip()
    return out[:max_len]


def normalize_token_id(token: dict[str, Any], fallback: int) -> int:
    return int(token.get("token_id", token.get("chunk_id", fallback)) or fallback)


def normalize_group_id(token: dict[str, Any], fallback: int) -> int:
    return int(token.get("micro_chunk_id", token.get("chunk_id", fallback)) or fallback)


def _reward_value(token: dict[str, Any]) -> float:
    return float(token.get("reward_raw", token.get("reward", 0.0)) or 0.0)


def _format_earliest_gt_rollout(
    token: dict[str, Any],
    sample_gt_first_hit_rollout: dict[int, int] | None,
) -> str:
    gt_indices = [int(item) for item in (token.get("attributed_gt_token_indices", []) or [])]
    if not gt_indices:
        return "无"
    first_hit = sample_gt_first_hit_rollout or {}
    if len(gt_indices) == 1:
        value = first_hit.get(gt_indices[0])
        return "无" if value is None else f"r{int(value):02d}"
    pieces: list[str] = []
    for gt_idx in gt_indices:
        value = first_hit.get(gt_idx)
        pieces.append(f"gt#{gt_idx}=无" if value is None else f"gt#{gt_idx}=r{int(value):02d}")
    return ", ".join(pieces)


def _token_info_text(
    token: dict[str, Any],
    token_id: int,
    *,
    sample_gt_first_hit_rollout: dict[int, int] | None = None,
) -> str:
    reward = _reward_value(token)
    token_adv = token.get("token_adv")
    token_adv_text = "N/A" if token_adv is None else f"{float(token_adv):+.4f}"
    earliest_rollout = _format_earliest_gt_rollout(token, sample_gt_first_hit_rollout)
    return (
        f"Token #{token_id}\n"
        f"chunk_type: {token.get('chunk_type', '')}\n"
        f"is_eos: {bool(token.get('is_eos', False))}\n"
        f"token_text: {token.get('token_text', token.get('pred_chunk_text', ''))}\n"
        f"gt_text: {token.get('gt_text', token.get('gt_chunk_text', ''))}\n"
        f"reward: {reward:.4f}\n"
        f"token_adv: {token_adv_text}\n"
        f"chunk_normed_adv: {token.get('chunk_normed_adv')}\n"
        f"E_k: {int(token.get('E_k', 0) or 0)}\n"
        f"L_k: {int(token.get('L_k', 0) or 0)}\n"
        f"P_k: {int(token.get('P_k', 0) or 0)}\n"
        f"denominator: {int(token.get('denominator', 0) or 0)}\n"
        f"pred_span: {token.get('start_char')}:{token.get('end_char')}\n"
        f"gt_span: {token.get('gt_span_start')}:{token.get('gt_span_end')}\n"
        f"attributed_gt_token_indices: {token.get('attributed_gt_token_indices', [])}\n"
        f"earliest_correct_rollout_for_gt: {earliest_rollout}\n"
        f"matched_gt_token_indices: {token.get('matched_gt_token_indices', [])}\n"
        f"inserted_gt_token_indices: {token.get('inserted_gt_token_indices', [])}"
    )


def _build_prediction_tokens_html(
    pred_text: str,
    token_rows: list[dict[str, Any]],
    *,
    max_tokens: int,
    sample_gt_first_hit_rollout: dict[int, int] | None = None,
) -> tuple[str, dict[int, str]]:
    safe_text = str(pred_text or "")
    if not token_rows:
        return html.escape(safe_text), {}

    pieces: list[str] = []
    info_by_group_id: dict[int, str] = {}
    cursor = 0
    shown = token_rows[:max_tokens]
    for fallback_idx, token in enumerate(shown):
        token_id = normalize_token_id(token, fallback_idx)
        group_id = normalize_group_id(token, fallback_idx)
        start = token.get("start_char")
        end = token.get("end_char")
        start = len(safe_text) if start is None else max(0, min(int(start), len(safe_text)))
        end = start if end is None else max(start, min(int(end), len(safe_text)))
        if start > cursor:
            pieces.append(html.escape(safe_text[cursor:start]))
        seg_text = safe_text[start:end] or str(token.get("token_text", "")) or " "
        reward = _reward_value(token)
        info_text = _token_info_text(
            token,
            token_id,
            sample_gt_first_hit_rollout=sample_gt_first_hit_rollout,
        )
        info_by_group_id[group_id] = info_text
        pieces.append(
            "<span class='token-seg pred-seg"
            f"{' eos-seg' if bool(token.get('is_eos', False)) else ''}' "
            f"data-group-id='{group_id}' data-info='{html.escape(info_text, quote=True)}' "
            f"title='{html.escape(info_text, quote=True)}' style='background:{token_reward_color(reward)}'>"
            f"{html.escape(seg_text)}</span>"
        )
        cursor = max(cursor, end)
    if cursor < len(safe_text):
        pieces.append(html.escape(safe_text[cursor:]))
    if len(token_rows) > len(shown):
        pieces.append(f"<br/><span class='truncated'>... only first {len(shown)} / {len(token_rows)} tokens shown</span>")
    return "".join(pieces), info_by_group_id


def _build_gt_alignment_html(
    gt_text: str,
    token_rows: list[dict[str, Any]],
    info_by_group_id: dict[int, str],
    *,
    max_tokens: int,
    sample_gt_first_hit_rollout: dict[int, int] | None = None,
) -> str:
    safe_text = str(gt_text or "")
    if not token_rows:
        return html.escape(safe_text)
    spans: list[dict[str, Any]] = []
    for fallback_idx, token in enumerate(token_rows[:max_tokens]):
        start = token.get("gt_span_start")
        end = token.get("gt_span_end")
        if start is None or end is None:
            continue
        start_i = max(0, min(int(start), len(safe_text)))
        end_i = max(start_i, min(int(end), len(safe_text)))
        if end_i <= start_i:
            continue
        token_id = normalize_token_id(token, fallback_idx)
        group_id = normalize_group_id(token, fallback_idx)
        spans.append(
            {
                "start": start_i,
                "end": end_i,
                "token_id": token_id,
                "group_id": group_id,
                "reward": _reward_value(token),
                "info": info_by_group_id.get(
                    group_id,
                    _token_info_text(
                        token,
                        token_id,
                        sample_gt_first_hit_rollout=sample_gt_first_hit_rollout,
                    ),
                ),
            }
        )
    spans.sort(key=lambda item: (int(item["start"]), int(item["end"]), int(item["token_id"])))

    pieces: list[str] = []
    cursor = 0
    for item in spans:
        start = max(cursor, int(item["start"]))
        end = max(start, int(item["end"]))
        if end <= start:
            continue
        if start > cursor:
            pieces.append(html.escape(safe_text[cursor:start]))
        seg_text = safe_text[start:end]
        pieces.append(
            "<span class='token-seg gt-seg' "
            f"data-group-id='{int(item['group_id'])}' "
            f"data-info='{html.escape(str(item['info']), quote=True)}' "
            f"title='{html.escape(str(item['info']), quote=True)}' "
            f"style='background:{token_reward_color(float(item['reward']))}'>"
            f"{html.escape(seg_text)}</span>"
        )
        cursor = end
    if cursor < len(safe_text):
        pieces.append(html.escape(safe_text[cursor:]))
    return "".join(pieces)


def render_token_alignment_component(
    pred_text: str,
    gt_text: str,
    token_rows: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    sample_gt_first_hit_rollout: dict[int, int] | None = None,
) -> None:
    pred_html, info_by_group_id = _build_prediction_tokens_html(
        pred_text,
        token_rows,
        max_tokens=max_tokens,
        sample_gt_first_hit_rollout=sample_gt_first_hit_rollout,
    )
    gt_html = _build_gt_alignment_html(
        gt_text,
        token_rows,
        info_by_group_id,
        max_tokens=max_tokens,
        sample_gt_first_hit_rollout=sample_gt_first_hit_rollout,
    )
    default_info = "Hover a token to inspect reward details."
    component_id = f"token-alignment-{stable_id(pred_text[:1000] + gt_text[:1000] + str(len(token_rows)))}"
    components.html(
        f"""
<style>
#{component_id} {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: #111827;
}}
#{component_id} .token-panel {{
  border: 1px solid #d9d9d9;
  border-radius: 10px;
  background: #ffffff;
}}
#{component_id} .token-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 12px;
  margin-bottom: 10px;
}}
#{component_id} .panel-title {{
  font-size: 13px;
  color: #4b5563;
  padding: 8px 12px 0 12px;
  font-weight: 600;
}}
#{component_id} .panel-content {{
  padding: 10px 12px 12px 12px;
  white-space: pre-wrap;
  overflow-y: auto;
  min-height: 180px;
  max-height: 260px;
  line-height: 1.55;
  font-size: 14px;
}}
#{component_id} .token-seg {{
  display: inline-block;
  padding: 1px 4px;
  margin: 1px;
  border-radius: 4px;
  transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
}}
#{component_id} .pred-seg.eos-seg {{
  border: 1px dashed #4b5563;
  font-weight: 700;
}}
#{component_id} .token-seg {{
  color: #111827;
}}
#{component_id} .token-seg.active {{
  transform: scale(1.12);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.6);
  filter: brightness(1.05);
  position: relative;
  z-index: 2;
}}
#{component_id} .token-info-card {{
  border: 1px solid #d1d5db;
  border-radius: 12px;
  background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
  padding: 14px 16px;
  box-shadow: 0 4px 14px rgba(17, 24, 39, 0.08);
}}
#{component_id} .token-info-body {{
  color: #111827;
  font-size: 14px;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  min-height: 160px;
  max-height: 260px;
  overflow-y: auto;
}}
#{component_id} .truncated {{
  color: #6b7280;
}}
</style>
<div id="{component_id}">
  <div class="token-grid">
    <div class="token-panel">
      <div class="panel-title">Prediction Tokens</div>
      <div class="panel-content pred-content">{pred_html}</div>
    </div>
    <div class="token-panel">
      <div class="panel-title">Ground Truth Aligned Spans</div>
      <div class="panel-content gt-content">{gt_html}</div>
    </div>
  </div>
  <div class="token-info-card">
    <div class="panel-title">Token Detail</div>
    <div class="token-info-body" id="{component_id}-info">{html.escape(default_info)}</div>
  </div>
</div>
<script>
(function() {{
  const root = document.getElementById("{component_id}");
  if (!root) return;
  const infoBox = document.getElementById("{component_id}-info");
  const defaultInfo = {json.dumps(default_info)};
  function clearActive() {{
    root.querySelectorAll(".token-seg.active").forEach((el) => el.classList.remove("active"));
  }}
  function activateByGroupId(groupId, infoText) {{
    clearActive();
    root.querySelectorAll('.token-seg[data-group-id="' + groupId + '"]').forEach((el) => el.classList.add("active"));
    if (infoBox) infoBox.textContent = infoText || defaultInfo;
  }}
  root.querySelectorAll(".token-seg").forEach((el) => {{
    el.addEventListener("mouseenter", () => activateByGroupId(el.getAttribute("data-group-id"), el.getAttribute("data-info")));
    el.addEventListener("mouseleave", () => {{
      clearActive();
    }});
  }});
}})();
</script>
""",
        height=560,
        scrolling=False,
    )


def _highlight_source_by_mask(source: str, common_mask: list[bool] | None) -> str:
    text = str(source or "")
    if not common_mask:
        return html.escape(text)
    pieces: list[str] = []
    for idx, char in enumerate(text):
        safe_char = html.escape(char)
        if idx < len(common_mask) and common_mask[idx]:
            pieces.append(f"<span style='color:#16a34a;font-weight:700'>{safe_char}</span>")
        else:
            pieces.append(safe_char)
    return "".join(pieces)


def render_formula_preview(title: str, source: str, *, common_mask: list[bool] | None = None) -> None:
    st.markdown(f"**{title}**")
    safe_source = _highlight_source_by_mask(str(source or ""), common_mask)
    st.markdown(
        f"""
<div style="
  border: 1px solid #d9d9d9;
  border-radius: 8px;
  padding: 12px;
  background: #f8fafc;
  color: #111111;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  max-width: 100%;
  box-sizing: border-box;
  overflow-x: hidden;
  overflow-y: auto;
  max-height: 240px;
  line-height: 1.45;
  font-size: 13px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
">{safe_source}</div>
""",
        unsafe_allow_html=True,
    )
    cleaned = str(source or "").strip()
    if cleaned.startswith("\\[") and cleaned.endswith("\\]"):
        cleaned = cleaned[2:-2].strip()
    elif cleaned.startswith("$$") and cleaned.endswith("$$"):
        cleaned = cleaned[2:-2].strip()
    if not cleaned:
        st.info("Empty formula source.")
        return
    try:
        st.latex(cleaned)
    except Exception:
        st.info("Formula render failed; showing source only.")
