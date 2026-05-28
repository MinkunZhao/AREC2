# AREC^2

Agentic Retrieval-Enrichment for Context-Aware Generative Recommendation

AREC^2 是一个围绕 OpenOneRec/OneRec-1.7B 构建的上下文感知生成式推荐增强框架。项目目标是在 RecIF-Bench 场景中，把用户画像、近期意图、标签行为、跨域信号、协同邻居和文本检索信号组织为可控的 retrieval-enriched context cards，再用于 SFT、RecPO/DPO 和评测，让模型在生成推荐结果时显式利用结构化证据。

当前版本已经从早期的规则式 agentic enrichment 扩展到完整实验链路：

- 离线检索 stores：`ProfileStore`、`LabelBehaviorStore`、`CollaborativeStore`、`ItemTextStore`
- Agentic 工具链：profile、recent intent、label behavior、cross domain、collaborative
- 证据图与上下文卡片：规则式 `GraphCompiler`，以及 LLM 版 `LLMCompiler`
- Planner：规则式 `PlannerAgent`，以及带 fallback 的 `LLMPlanner`
- TextGrad prompt 优化：自动优化 planner/compiler prompt，并产出 `prompts/optimized/`
- 上下文预计算：`data/cards_v2/*.parquet` 支持训练时直接读取优化卡片
- SFT：RecIF 训练样本 + General_SFT 混合训练 LoRA
- RecPO/DPO：基于 on-policy hard negative 生成偏好对并继续偏好优化
- 官方风格 RecIF 评测：支持召回任务、AUC 任务、LLM judge 任务和消融实验

## 项目结构

```text
AREC^2/
├── arec2/
│   ├── agents/          # Planner/Executor，以及 LLMPlanner/LLMCompiler/LLMClient
│   ├── base_model/      # OpenOneRec wrapper，生成、batch 生成、候选打分
│   ├── enrichment/      # EvidenceGraph 与上下文卡片编译器
│   ├── eval/            # RecIFRunner 评测封装
│   ├── retrieval/       # 离线 stores 构建和查询
│   ├── rl/              # RecPO/DPO 偏好对生成与 DPOTrainer 封装
│   ├── textgrad/        # TextGrad prompt 优化组件
│   ├── tools/           # agentic retrieval tools
│   └── training/        # RecIF/General_SFT/CombinedDataset 数据管线
├── configs/
│   ├── training_config.yaml
│   ├── dpo_config.yaml
│   ├── textgrad_config.yaml
│   └── judge_llm_config.json
├── data/
│   ├── recif/           # RecIF release、benchmark_data、PID->SID 映射
│   ├── general_sft/     # 通用 SFT parquet 数据
│   ├── cards_v2/        # 预计算上下文卡片
│   └── preferences/     # DPO 偏好对
├── caches/              # 离线 stores 与 LLM cache
├── models/1.7B/         # 本地 OneRec-1.7B 权重
├── prompts/             # 初始 prompt 与 TextGrad 优化后的 prompt
├── scripts/             # 构建、训练、评测、消融脚本
├── tests/               # stores、tools、compiler 单元/集成测试
├── README.md
├── requirements.txt
└── code.txt             # 项目代码汇总文档，内嵌 README.md 内容
```

## 核心流程

### 1. 构建离线 stores

stores 从 `data/recif/onerec_bench_release.parquet` 和 PID 到 SID 映射构建，并缓存到 `caches/`。

```bash
python scripts/01_build_offline_stores.py
```

当前缓存文件包括：

- `caches/profile_store.pkl`
- `caches/label_behavior_store.pkl`
- `caches/collaborative_store.npz`
- `caches/collaborative_store.meta.pkl`
- `caches/item_text_store.pkl`

这些缓存是 retrieval enrichment、SFT、DPO 偏好生成和测试时 enrichment 的共同依赖。

### 2. 验证 agentic pipeline

```bash
python tests/test_stores.py
python tests/test_tools_and_compiler.py
python scripts/02_test_agentic_pipeline.py
```

`PlannerAgent` 会按任务类型选择工具，`ExecutorAgent` 负责执行工具调用并通过 compiler 生成上下文卡片。典型任务包括 `video`、`ad`、`product`、`label_cond` 和 `label_pred`。

### 3. SFT 训练

当前项目只保留 `configs/training_config.yaml` 作为 SFT 配置入口，`configs/training_config_quick.yaml` 已删除。

```bash
python scripts/03_train_sft.py --config configs/training_config.yaml
```

如果需要快速实验，直接在 `configs/training_config.yaml` 中临时调小采样量，例如：

```yaml
data:
  max_recif_samples: 10000
  max_general_samples: 2500

training:
  num_train_epochs: 1
```

当前 SFT 数据管线要点：

- RecIF 训练样本来自 `onerec_bench_release.parquet`，避免直接使用测试集用户造成泄漏
- 默认任务：`video`、`ad`、`product`、`label_cond`、`label_pred`
- 默认混合比例：80% RecIF + 20% General_SFT
- 支持三种 card source：
  - `heuristic`：训练时即时使用规则 planner/executor/compiler 生成卡片
  - `llm_optimized`：从 `data/cards_v2/` 读取预计算 LLM 优化卡片
  - `none`：关闭上下文增强
- 默认最大长度为 4096，并使用 `truncate_preserving_answer` 保留 assistant answer

训练输出默认写入：

- `checkpoints/arec2-lora-r16/`

### 4. 合并 LoRA

```bash
python scripts/04_merge_lora.py \
  --base_model ./models/1.7B \
  --lora_path ./checkpoints/arec2-lora-r16/final \
  --output_dir ./models/arec2-merged
```

如果使用 Hugging Face 模型名作为基础模型，也可以把 `--base_model` 设置为 `OpenOneRec/OneRec-1.7B`。

### 5. TextGrad 优化 prompt 与预计算卡片

TextGrad 用一小部分 dev 数据反复评估 planner/compiler prompt，对 `prompts/planner_v0.txt` 和 `prompts/compiler_v0.txt` 做优化，并输出到 `prompts/optimized/`。

```bash
python scripts/05_run_textgrad.py --config configs/textgrad_config.yaml
```

生成或更新预计算上下文卡片：

```bash
python scripts/05b_precompute_cards.py --config configs/textgrad_config.yaml
```

预计算结果位于：

- `data/cards_v2/video.parquet`
- `data/cards_v2/ad.parquet`
- `data/cards_v2/product.parquet`
- `data/cards_v2/label_cond.parquet`
- `data/cards_v2/label_pred.parquet`

如果要让 SFT 直接使用这些卡片，在 `configs/training_config.yaml` 中设置：

```yaml
data:
  card_source: "llm_optimized"
  cards_v2_dir: "./data/cards_v2"
```

### 6. RecPO/DPO 偏好训练

先从 SFT 模型生成偏好对：

```bash
python scripts/06_generate_preferences.py \
  --config configs/dpo_config.yaml \
  --tasks video ad product label_cond \
  --max-pairs 50000
```

输出：

```text
data/preferences/all_pairs.parquet
```

再运行 DPO：

```bash
python scripts/07_train_dpo.py --config configs/dpo_config.yaml
```

当前 RecPO/DPO 设计重点：

- P1 on-policy hard negative 是默认策略
- `chosen` 是 ground-truth SID 列表
- `rejected` 是模型自己生成的非 GT 高置信 SID，并进行 canonicalize 与长度匹配
- DPO 前会把 SFT LoRA merge 到 base，再挂载新的 DPO LoRA
- reference 使用 fresh adapter disabled 的等价模型，脚本里带有 reference equivalence sanity check

### 7. RecIF 评测与消融

官方风格 RecIF 评测脚本：

```bash
python scripts/08_eval_recif.py \
  --model ./models/1.7B \
  --adapter ./checkpoints/arec2-lora-r16/final \
  --tasks video ad product label_cond interactive label_pred \
  --batch-size 64 \
  --num-beams 128 \
  --num-return-sequences 128
```

常用快速 smoke：

```bash
python scripts/08_eval_recif.py \
  --model ./models/1.7B \
  --adapter "" \
  --tasks video \
  --max-samples 100 \
  --batch-size 2 \
  --num-beams 2 \
  --num-return-sequences 2 \
  --judge false
```

评测任务与指标：

- `video`、`ad`、`product`、`label_cond`、`interactive`：Recall/PASS 类 SID 排名指标
- `label_pred`：AUC
- `item_understand`、`rec_reason`：可选外部 LLM judge

运行 ablation sweep：

```bash
python scripts/09_run_ablations.py
```

`09_run_ablations.py` 内置了 base、SFT v1、SFT v2、SFT+DPO、关闭测试时 enrichment 等配置，便于对比不同训练和上下文设置。

## 环境安装

建议使用 Python 3.10 到 3.12，并准备 CUDA 可用的 PyTorch 环境。

```bash
pip install -r requirements.txt
```

涉及 LLM planner/compiler、TextGrad、LLM judge 或官方评测时，还需要确保以下依赖可用：

```bash
pip install openai diskcache tenacity pydantic scikit-learn
```

如果从 Hugging Face 下载模型或数据，按所在网络环境配置 `HF_ENDPOINT`、`HF_HOME`、`HF_DATASETS_CACHE` 等环境变量。

## 数据与模型约定

当前代码默认使用这些路径：

```text
data/recif/onerec_bench_release.parquet
data/recif/benchmark_data/{task}/{task}_test.parquet
data/recif/video_ad_pid2sid.parquet
data/recif/product_pid2sid.parquet
caches/*.pkl / caches/collaborative_store.*
models/1.7B/
```

`data/recif/benchmark_data/` 目前包含 8 个 RecIF 任务：

- `video`
- `ad`
- `product`
- `label_cond`
- `label_pred`
- `interactive`
- `item_understand`
- `rec_reason`

训练时主要使用 `onerec_bench_release.parquet` 构造训练样本；评测时才读取 `benchmark_data` 下的测试 parquet。

## 常用命令

```bash
# 基础模型 smoke test
python scripts/00_smoke_test.py

# 构建 stores
python scripts/01_build_offline_stores.py

# 测试 agentic pipeline
python scripts/02_test_agentic_pipeline.py

# 检查 SFT 数据管线
python scripts/test_data_pipeline.py

# SFT 训练
python scripts/03_train_sft.py --config configs/training_config.yaml

# 合并 LoRA
python scripts/04_merge_lora.py --base_model ./models/1.7B --lora_path ./checkpoints/arec2-lora-r16/final --output_dir ./models/arec2-merged

# TextGrad 优化 prompt
python scripts/05_run_textgrad.py --config configs/textgrad_config.yaml

# 预计算 cards_v2
python scripts/05b_precompute_cards.py --config configs/textgrad_config.yaml

# 生成 DPO 偏好对
python scripts/06_generate_preferences.py --config configs/dpo_config.yaml --tasks video --max-pairs 20000

# DPO 训练
python scripts/07_train_dpo.py --config configs/dpo_config.yaml

# RecIF 评测
python scripts/08_eval_recif.py --model ./models/1.7B --tasks video label_pred --max-samples 100 --judge false

# 消融实验
python scripts/09_run_ablations.py
```

## 配置说明

### `configs/training_config.yaml`

SFT 配置，默认：

- base model：`OpenOneRec/OneRec-1.7B`
- 输出目录：`checkpoints/arec2-lora-r16/`
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`
- RecIF tasks：`video`、`ad`、`product`、`label_cond`、`label_pred`
- `max_recif_samples: null`
- `max_general_samples: 119250`
- batch size：16
- max length：4096
- card source：`heuristic`

### `configs/textgrad_config.yaml`

用于 LLM planner/compiler 和 TextGrad prompt 优化：

- dev train/val sampling
- LLM cache
- planner/compiler 初始 prompt
- 优化输出目录
- `cards_v2_dir`

建议把真实 LLM API key 放在本地配置或环境变量中，不要把私钥提交到公开仓库。

### `configs/dpo_config.yaml`

用于 RecPO/DPO：

- SFT checkpoint 路径
- DPO LoRA 输出路径
- preference parquet 路径
- DPO beta、loss type、max length、max prompt length

## 开发与验证

```bash
python tests/test_stores.py
python tests/test_tools_and_compiler.py
python scripts/sanity_check.py --config configs/training_config.yaml
```

当前项目不是标准 Python package 安装布局时，脚本会把项目根目录加入 `sys.path`。如果在 notebook 或交互式环境中使用，建议从项目根目录启动 Python。

## 注意事项

- 项目全称统一为 Agentic Retrieval-Enrichment for Context-Aware Generative Recommendation，简称 AREC^2
- `configs/training_config_quick.yaml` 已移除，所有训练和 sanity check 命令都应显式传入 `configs/training_config.yaml`
- `caches/` 是当前代码使用的缓存目录，不是旧文档里的 `cache/`
- 本地模型目录当前是 `models/1.7B/`，部分旧示例里的 `models/1.7B-pretrain/` 需要按实际情况替换
- `scripts/08_eval_recif.py --enrich true` 参数保留兼容性，但脚本说明中标注它会改变 benchmark prompt，不适合作为 paper-comparable 官方结果
- `item_understand` 和 `rec_reason` 的 judge 需要外部 LLM 配置
- 大规模 `data/general_sft/`、`data/multimodal_embedding/` 和模型权重体积很大，迁移服务器时建议分批传输或使用压缩包

## License

项目依赖的数据集和基础模型分别遵循其原始许可证。仓库代码许可证请按实际发布策略补充。
