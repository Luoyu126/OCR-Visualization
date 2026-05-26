# visualization-v1 OCR RL 在线质检

本目录用于混合奖励（text token-level / table chunk-level / formula sequence-level）训练的 rollout 质检可视化。

## 1) 从 step 构建 run

先把 `rollout_results/global_step_{N}_results.jsonl` 转成 Streamlit 直接读取的 run 目录。

```bash
conda activate /user/hezhihui/miniconda3/envs/minicpmv5-qwen3.5-patched-wzl
cd /user/wangzhilue/algorithm/visualization-v1
python build_rl_rollout_runs.py --step 22 --overwrite
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

构建后每个 step 会生成一个目录：`runs/rl_rollout_step_xxxx/`

## 2) 启动可视化

```bash
conda activate /user/hezhihui/miniconda3/envs/minicpmv5-qwen3.5-patched-wzl
cd /user/wangzhilue/algorithm/visualization-v1
streamlit run streamlit_rl_reward_qc_app.py
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
