# visualization-v1 OCR RL 在线质检

本目录用于混合奖励（text token-level / table chunk-level / formula sequence-level）训练的 rollout 质检可视化。

## 1) 从 step 构建 run

先把 `rollout_results/global_step_{N}_results.jsonl` 转成 Streamlit 直接读取的 run 目录。

```bash
conda activate /user/hezhihui/miniconda3/envs/minicpmv5-qwen3.5-patched-wzl
cd /user/chenyunyi/projects/verl_mm/tmp_scripts/visualization-v1
python scripts/build_runs.py --step 22 --out-root runs/rmodel --overwrite
```

常用参数：

- `--rollout-dir`: rollout jsonl 根目录（默认指向当前 v4 训练路径）
- `--out-root`: 运行产物输出目录（默认 `visualization-v1/runs`）
- `--step`: 指定单个 step（推荐）
- `--step-start --step-end`: 构建一个 step 区间
- `--skip-image-lookup`: 跳过原图回查（调试速度更快）

说明：

- 原图回查依赖 `pyarrow` + `Pillow`；如果环境缺失可先加 `--skip-image-lookup`。
- 你提供的环境 `minicpmv5-qwen3.5-patched-wzl` 已可直接跑完整流程（含图片回查）。

构建后每个 step 会生成一个目录：`runs/rmodel/rl_rollout_step_xxxx/`

## 2) 启动可视化

```bash
conda activate /user/hezhihui/miniconda3/envs/minicpmv5-qwen3.5-patched-wzl
cd /user/chenyunyi/projects/verl_mm/tmp_scripts/visualization-v1
streamlit run app/streamlit_app.py
```

页面流程：

1. 选择 run
2. 选择 sample（数据级导航）
3. 查看该 sample 的 8 rollout 序列奖励和分类型细节

## 3) run 目录结构

每个 run 包含：

- `metadata.json`
- `samples.jsonl`
- `predictions.jsonl`
- `pair_summary.jsonl`
- `sequence_groups.jsonl`
- `sequence_scores.jsonl`
- `sample_manifest.jsonl`
- `chunk_scores_*.jsonl`
- `images/*.png`（如果图片回查成功）
- `annotations_current.json` / `annotation_events.jsonl`（使用页面标注后生成）

## 4) 字段来源说明（核心）

所有奖励展示均来自 rollout jsonl 原始字段，不在页面重新计算奖励值：

- **sequence-level**: `final_reward`, `grpo_normed_adv`（缺失时才做兜底标准化）
- **pair-level**: `chunk_sequence_reward`（若存在优先）或 chunk 聚合
- **chunk/token-level**:
  - `chunk_items[*].reward / reward_source / matching_reward`
  - `pred_chunk_records[*].chunk_normed_adv`
  - `response_tokens[*]`（token 文本、span、adv 等）

## 5) 三类样本展示范围

- `text`: token 对齐高亮 + hover 细节 + 8 rollout 同屏
- `table`: 表格渲染 + 源码（`</tr>` 分行美化）+ prediction 顺序 chunk 细节 + 8 rollout 同屏
- `formula`: 源码 + 渲染结果 + 8 rollout 同屏（sequence-level）
- `formula`（token pass@k run）: 使用 `token_formula` 打分后的 run，展示实际 visual token 与 fallback 片段（虚线灰底）

## 5b) Formula token pass@k 打分与可视化

对 pass@k formula rollout（如 `rollout_formula_ocr_150x8_seed20260524`）离线打分并生成标准 run：

```bash
conda activate /user/hezhihui/miniconda3/envs/minicpmv5-qwen3.5-patched-wzl
cd /user/chenyunyi/projects/verl_mm/tmp_scripts/visualization-v1
python scripts/build_formula_token_passk_run.py \
  --input-run-dir /user/wangzhilue/algorithm/Formula/rollouts/rollout_formula_ocr_150x8_seed20260524 \
  --out-root runs/formula_token_passk \
  --link-images \
  --overwrite
```

若本机缺少 `xelatex`/`magick` 导致 CDM 渲染失败，可导入已打好的 pass@k 分数：

```bash
python scripts/build_formula_token_passk_run.py \
  --input-run-dir /user/wangzhilue/algorithm/Formula/rollouts/rollout_formula_ocr_150x8_seed20260524 \
  --out-root runs/formula_token_passk \
  --scores-jsonl /user/chenyunyi/projects/verl_mm/tmp_scripts/formula_token_passk_scores.jsonl \
  --link-images \
  --overwrite
```

启动 Streamlit 后，将 **Run root** 设为 `tmp_scripts/visualization-v1/runs`，选择 `formula_token_rollout_formula_ocr_150x8_seed20260524`（或对应 run 名）。

Adv 逻辑（与训练 pass@k 脚本一致）：

- case 内前 8 条 response 的全部 `formula_visual_token` reward 计算 mean/std
- 每个实际 element：`chunk_normed_adv = (reward - mean) / std`
- 每条 response：`chunk_seq_adv = mean(element chunk_normed_adv)`
- 未覆盖/非 visual token 片段：`is_fallback=true`，adv 使用 `chunk_seq_adv`

## 6) 图片加载与 `No image found` 排查

页面里的图片来自 run 目录下的 `images/{sample_id}.png`，不是 Streamlit 启动时临时查询 parquet。构建 run 时如果使用了 `--skip-image-lookup`，或者构建环境里的 `pyarrow` / `Pillow` 不可用，`samples.jsonl` 里的 `image_path` 会保持为空，页面就会显示 `No image found for this sample.`。

已存在的 run 不建议直接用 `build_runs.py --overwrite` 只为补图，因为这会删除整个 run 目录，可能误删 `annotations_current.json` 和 `annotation_events.jsonl`。使用下面的非破坏性补图脚本即可：

```bash
conda activate /user/hezhihui/miniconda3/envs/minicpmv5-qwen3.5-patched-wzl
cd /user/chenyunyi/projects/verl_mm/tmp_scripts/visualization-v1

# 单个 step
python scripts/materialize_images.py \
  --run-root runs/rmodel \
  --step 702

# 一个 step 区间
python scripts/materialize_images.py \
  --run-root runs/rmodel \
  --step-start 474 \
  --step-end 702
```

图片回查逻辑在 `ocr_viz/dataset_lookup.py`，大致顺序是：

- text/table：先用 `source_parquet`（也兼容 `parquet_id`、`parquet_path`、`orig_source`）定位过滤后的 parquet；再用 `source_row_idx`（也兼容 `row_index`、`row_idx`、`orig_index`）找候选行。
- 如果同一个 source row 对应多个 block，会继续用 `source_block_idx` / `block_idx` / `block_index` 消歧。
- 如果 block 信息缺失或仍有多个候选，会用 rollout 里的 `ground_truth` 去匹配 parquet 行里的 `block_content` 或 `clean_content` 中解析出的 GT。
- formula：优先用 ground truth 在 formula parquet 中建立索引；必要时再用 `dataset_index` 兜底。

补图脚本会更新 `samples.jsonl`、`sequence_groups.jsonl` 和 `metadata.json` 中的图片统计，不会改奖励数据，也不会删除标注文件。补图完成后，如果页面已打开，点击侧边栏的 `Refresh run data` 让 Streamlit 清掉缓存。
