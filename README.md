# submod_query_optimization

面向“少而多样”的搜索查询选择实验与 Benchmark：先由 LLM 生成候选查询，再使用多种查询选择算法压缩候选集合，并从相关性、多样性、语料覆盖和运行时间多个维度进行统一比较。

## Phase 3：真实 Information Retrieval Benchmark

现已支持基于显式 qrels / ground truth 的真实离线检索评估，而不再只依赖 embedding coverage proxy。

Phase 3 会对 OriginalQuery、TopRelevance、MMR、Random、Facility Location、Graph Cut 在同一文档集合和同一 qrels 上计算：

- Precision@K
- Recall@K
- HitRate@K
- MRR@K
- nDCG@K
- query-level macro average
- deterministic bootstrap 95% confidence interval

同时支持：

- BEIR-style corpus.jsonl / queries.jsonl / qrels/test.tsv
- 3 列 qrels 与 4 列 TREC qrels
- Facility Location alpha sweep
- Graph Cut alpha × lambda sweep
- MMR lambda sweep
- Random 多 seed 重复实验
- 完整 retrieval ranking / selected expansion 审计输出

快速运行：

    export OPENAI_API_KEY=sk-...

    python main.py \
      --ir-only \
      --ir-dataset-dir examples/ir_dataset \
      --m3e-path model \
      --ir-cutoffs 5,10,20 \
      --ir-alpha-grid 0.2,0.5,0.8 \
      --ir-lambda-grid 0.2,0.5,0.8 \
      --ir-mmr-grid 0.3,0.6,0.9 \
      --ir-random-repeats 10 \
      --output-dir output

完整实验协议、指标定义、数据格式和统计口径见 docs/IR_BENCHMARK.md。

## 当前能力

### 1. Prompt 消融

比较：

- `simple`
- `structured`

两种 Prompt 的候选查询质量。

### 2. 子模算法正确性消融

目标函数：

- Facility Location
- Graph Cut

实现：

- Standard Greedy
- CELF-style Lazy Greedy

用于验证 Lazy Greedy 是否在减少重复计算的同时保持与 Standard Greedy 一致的目标值。

### 3. 统一 Benchmark

同一个候选集合上比较：

- **Top-Relevance**：只按 topic relevance 排序
- **MMR**：相关性与集合内冗余的经典基线
- **Random**：固定随机种子的无策略基线
- **Facility Location**
- **Graph Cut**

Benchmark 会同时记录：

- selected count
- runtime
- mean / median intra-set cosine similarity
- duplicate rate @ 0.8
- average relevance to topic
- embedding-space corpus coverage

并生成机器可读的 `benchmark_summary.csv`。

## 目标函数

### Facility Location

当前实现：

    f(S) = alpha * sum(rel_i, i in S)
         + (1 - alpha) * sum_j max(sim(i, j), i in S)

relevance 只有候选真正被选中时才产生收益；coverage 则用于奖励代表性。

### Graph Cut

当前实现：

    f(S) = alpha * sum(rel_i, i in S)
         + lambda * sum(w(i, j), i in S, j not in S)

其中：

    w(i, j) = 1 - cosine_similarity(i, j)

Graph Cut 是子模函数，但不一定单调。因此当剩余候选都没有正边际收益时，选择器会提前结束，最终数量可能小于 `k`。

### MMR

Benchmark 基线：

    score(i) = lambda * relevance(i)
             - (1 - lambda) * max_similarity(i, selected)

`--mmr-lambda` 越高越偏向相关性，越低越强调去冗余。

## 实验可重复性

当前实现同时提供两层候选复用：

1. **进程内缓存**：同一次运行中，相同 `topic + mode` 只调用一次 LLM。
2. **磁盘缓存**：默认使用 `.cache/candidates/`，允许不同运行复用候选。

磁盘缓存不是只按照 topic 命中。缓存 key 会包含：

- topic
- mode
- num_candidates
- model name
- temperature
- max tokens
- system prompt
- 实际 user prompt

因此修改 Prompt、模型或关键生成参数后，旧缓存会自动 miss，而不会污染新实验。

可以通过：

    --no-candidate-cache

关闭跨运行缓存。

Random baseline 使用 `--random-seed`，并通过稳定哈希为不同 topic 派生独立 seed，不依赖 Python 内置 `hash()`，因此不同进程中仍可复现。

## 项目结构

    submod_query_optimization/
    ├── main.py
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── src/
    │   ├── __init__.py
    │   ├── baselines.py
    │   ├── cache.py
    │   ├── config.py
    │   ├── corpus.py
    │   ├── embeddings.py
    │   ├── evaluator.py
    │   ├── llm.py
    │   ├── logging_utils.py
    │   ├── objectives.py
    │   ├── runner.py
    │   ├── selectors.py
    │   └── utils.py
    ├── tests/
    │   ├── test_baselines.py
    │   ├── test_cache.py
    │   ├── test_config.py
    │   ├── test_objectives.py
    │   ├── test_selectors.py
    │   └── test_utils.py
    ├── model/             # 本地 embedding 模型，不提交
    ├── data/              # 本地 txt 语料，不提交
    ├── .cache/            # LLM 候选缓存，不提交
    └── output/            # 实验结果，不提交

`v1.py` 是历史单文件版本。新开发以 `main.py + src/` 为准。

## 环境

建议 Python 3.10+。

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -r requirements.txt

准备：

1. 将本地 sentence-transformer / m3e 模型放到 `model/`，或通过 `--m3e-path` 指定。
2. 将用于 coverage 评估的 `.txt` 文档放到 `data/`。
3. 配置 `OPENAI_API_KEY`。

## 运行

    export OPENAI_API_KEY=sk-...

    python main.py \
      --topics "embeddings and rerankers" "generative ai" \
      --num-candidates 20 \
      --k 6 \
      --alpha 0.5 \
      --lambda-diversity 0.5 \
      --mmr-lambda 0.6 \
      --random-seed 42 \
      --llm-model gpt-4o \
      --m3e-path model \
      --data-dir data \
      --candidate-cache-dir .cache/candidates \
      --output-dir output

## 参数校验

程序会在模型/API 初始化前后尽早拒绝明显错误配置，包括：

- `num_candidates > 0`
- `k > 0`
- `alpha in [0, 1]`
- `lambda_diversity in [0, 1]`
- `mmr_lambda in [0, 1]`
- `random_seed >= 0`
- `0 <= chunk_overlap < chunk_size`

## 测试

核心算法测试不需要 OpenAI API，也不需要下载 embedding 模型：

    python -m pip install -r requirements-dev.txt
    python -m pytest -q

CI 在 Python 3.10 / 3.11 / 3.12 上执行：

- `python -m compileall -q src main.py`
- `python -m pytest -q`

测试覆盖：

- incremental marginal gain 与 exact recomputation 一致性
- diminishing returns
- Standard / Lazy Greedy 等价性
- MMR / Top-Relevance / Random baseline
- Random seed 跨运行稳定性
- candidate cache round-trip / invalidation / corruption
- vector edge cases
- 配置边界

## 输出

`output/` 主要包含：

- `experiment_config.json`
- `prompt_ablation_<topic>.json`
- `submod_ablation_<topic>.json`
- `benchmark_<topic>.json`
- `benchmark_summary.csv`
- `run.log`

`benchmark_summary.csv` 可以直接被 pandas、DuckDB、R 或 BI 工具读取，用于后续绘图和统计分析。

## 指标解释

当前 `coverage_score_vs_corpus` 是：

> 对每个 corpus chunk，计算它与已选查询中最相似查询的 cosine similarity，再求平均值。

它是一个 embedding-space coverage proxy。

它**不是** Recall@K、MRR 或 nDCG。真实信息检索指标需要显式的 query-document relevance judgments 或可信 ground truth。项目目前不会把 proxy 指标伪装成真实检索质量指标。

## 当前边界与下一步

已经解决：

- 子模目标的主要语义问题
- Standard / Lazy 正确性验证
- LLM 候选跨实验复用
- 跨运行候选缓存
- Benchmark 基线缺失
- Random baseline 不可复现
- 结果缺少统一 CSV

Phase 3 已补齐 qrels-backed IR dataset、Recall/Precision/HitRate/MRR/nDCG、多 seed、bootstrap CI 与参数 sweep。后续更适合继续加入 benchmark 可视化、embedding 缓存、dev/test split 自动化以及 `v1.py` 迁移到 `legacy/`。
