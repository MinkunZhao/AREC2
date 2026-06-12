# AREC^2

Agentic Retrieval-Enrichment for Context-Aware Generative Recommendation

本仓库是论文 **AREC^2** 的代码实现。项目围绕 OpenOneRec/OneRec 构建，在 RecIF-Bench 场景中把用户画像、近期意图、标签行为、跨域信号、协同邻居和文本检索结果组织成结构化 context cards，用于上下文增强的生成式推荐训练、偏好优化和评测。

当前代码覆盖论文主线实验流程：离线检索 stores、agentic retrieval-enrichment、TextGrad prompt 优化、RecIF SFT、DPO/RecPO 偏好训练、RecIF 官方风格评测，以及 Amazon transfer-learning 扩展实验。

## 目录结构

```text
AREC^2/
├── arec2/
│   ├── agents/          # planner / executor / LLM planner / LLM compiler
│   ├── enrichment/      # evidence graph 与 context card compiler
│   ├── retrieval/       # profile、label、collaborative、text stores
│   ├── training/        # RecIF + General_SFT 数据构造
│   ├── rl/              # preference pair generation 与 DPO trainer
│   ├── eval/            # RecIF 评测封装
│   ├── textgrad/        # prompt optimization
│   └── amazon/          # Amazon transfer-learning 实验
├── configs/             # SFT、DPO、TextGrad、Amazon 配置
├── scripts/             # 数据构建、训练、评测、消融脚本
├── data/                # RecIF、General_SFT、Amazon、cards、preferences
├── caches/              # 离线 stores 与 LLM cache
├── models/              # 本地 OneRec 权重
├── prompts/             # 初始和优化后的 planner/compiler prompts
└── tests/               # stores / tools / compiler 基础测试
```

## 环境安装

建议使用 Python 3.10+，并先按本机 CUDA 版本安装合适的 PyTorch。

```bash
pip install -r requirements.txt
```

如果需要从 Hugging Face 下载模型或数据，可按网络环境配置：

```bash
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
export HF_ENDPOINT=https://hf-mirror.com
```

Windows PowerShell 中使用 `$env:HF_HOME="..."` 形式设置即可。

## 数据与模型

默认路径约定如下：

```text
models/1.7B/                                      # OneRec-1.7B 本地权重，可替换为 HF 名称
data/recif/onerec_bench_release.parquet           # RecIF release 数据
data/recif/benchmark_data/{task}/{task}_test.parquet
data/recif/video_ad_pid2sid.parquet
data/recif/product_pid2sid.parquet
caches/profile_store.pkl
caches/label_behavior_store.pkl
caches/collaborative_store.*
caches/item_text_store.pkl
```

RecIF 训练主要使用 `onerec_bench_release.parquet` 构造样本；评测读取 `benchmark_data/` 下的测试集。Amazon transfer-learning 实验使用 `data/amazon14/` 下的 10 个 Amazon-2014 domain parquet。

## 主流程

### 1. 构建离线 stores

```bash
python scripts/01_build_offline_stores.py
```

该步骤生成 profile、label behavior、collaborative 和 item text stores，是后续 retrieval-enrichment、SFT、DPO preference generation 和评测增强的共同依赖。

### 2. 验证 agentic enrichment

```bash
python tests/test_stores.py
python tests/test_tools_and_compiler.py
python scripts/02_test_agentic_pipeline.py
```

`PlannerAgent` 选择 profile、recent intent、label behavior、cross-domain、collaborative 等工具；`ExecutorAgent` 执行检索并通过 compiler 生成 context card。

### 3. SFT 训练

```bash
python scripts/03_train_sft.py \
  --config configs/training_config.yaml \
  --card_source heuristic
```

`configs/training_config.yaml` 默认使用 `OpenOneRec/OneRec-1.7B`，训练任务包括 `video`、`ad`、`product`、`label_cond`、`label_pred`，数据混合比例为 80% RecIF + 20% General_SFT。

`--card_source` 支持：

- `heuristic`：训练时即时生成规则 context cards
- `llm_optimized`：读取 `data/cards_v2/` 中的预计算 LLM-optimized cards
- `none`：关闭上下文增强，作为消融基线

默认输出：

```text
checkpoints/arec2-lora-r16-v2/
```

### 4. TextGrad 优化与预计算 cards

```bash
python scripts/05_run_textgrad.py --config configs/textgrad_config.yaml
python scripts/05b_precompute_cards.py --config configs/textgrad_config.yaml
```

TextGrad 会优化 planner/compiler prompts，并将结果写入 `prompts/optimized/`。预计算 cards 写入：

```text
data/cards_v2/
```

### 5. DPO / RecPO 偏好训练

先生成偏好对：

```bash
python scripts/06_generate_preferences.py \
  --config configs/dpo_config.yaml \
  --tasks video ad product label_cond label_pred \
  --max-pairs 100000
```

再训练 DPO：

```bash
python scripts/07_train_dpo.py --config configs/dpo_config.yaml
```

默认 DPO 输出：

```text
checkpoints/arec2-dpo-r16/
```

偏好构造以 on-policy hard negatives 为主；SID 类任务会 canonicalize 并长度匹配 rejected candidates，`label_pred` 使用二分类偏好构造，避免破坏 “是/否” 判别概率。

## 评测

RecIF 官方风格评测：

```bash
python scripts/08_eval_recif.py \
  --model ./models/1.7B \
  --adapter ./checkpoints/arec2-dpo-r16/final \
  --tasks video ad product label_cond interactive label_pred \
  --batch-size 64 \
  --num-beams 128 \
  --num-return-sequences 128 \
  --enrich false
```

快速 smoke test：

```bash
python scripts/08_eval_recif.py \
  --model ./models/1.7B \
  --adapter "" \
  --tasks video \
  --max-samples 100 \
  --batch-size 2 \
  --num-beams 2 \
  --num-return-sequences 2 \
  --judge false \
  --enrich false
```

任务与指标：

- `video`、`ad`、`product`、`label_cond`、`interactive`：Recall / PASS 类 SID 排名指标
- `label_pred`：AUC
- `item_understand`、`rec_reason`：可选外部 LLM judge

消融实验入口：

```bash
python scripts/09_run_ablations.py
```

## Amazon Transfer

Amazon transfer-learning 实验对应论文中的 text-augmented itemic token 思路：保留 3-layer itemic tokens，并追加 5 个 metadata keywords 做语义消歧。

```bash
# 构建 Amazon item artifacts
python scripts/A1_build_amazon_tokens.py --embed-model Qwen/Qwen3-Embedding-0.6B

# 多域联合训练
python scripts/A2_train_amazon.py --regime joint \
  --base-model OpenOneRec/OneRec-8B \
  --sft-adapter ./full_lora_8b

# Recall@{5,10}, NDCG@{5,10} 评测
python scripts/A3_eval_amazon.py \
  --base-model OpenOneRec/OneRec-8B \
  --sft-adapter ./full_lora_8b \
  --adapter ./checkpoints/amazon/strategy3_joint/final
```

## 常用检查

```bash
python scripts/00_smoke_test.py
python scripts/test_data_pipeline.py
python scripts/sanity_check.py --config configs/training_config.yaml
pytest tests
```

## 说明

- 项目名称统一写作 `AREC^2`，全称为 `Agentic Retrieval-Enrichment for Context-Aware Generative Recommendation`。
- `AREC^2.pdf` 是本仓库对应论文稿件；README 只保留实现与复现实验所需的最小说明。
- 数据集、基础模型和外部评测 LLM 的许可与使用限制请分别遵循其原始发布协议。
- 论文正式公开后，可在此补充 BibTeX 和项目主页链接。
