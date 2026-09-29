# submod_query_optimization

面向“少而多样”的搜索查询选择实验：先由 LLM 生成候选查询，再用子模目标函数与贪心算法选出较小但覆盖更广的查询集合，并输出消融实验结果。

## 主要能力

- Prompt 消融：simple vs structured
- 子模目标：
  - Facility Location：相关性 + 代表性覆盖
  - Graph Cut：相关性 + 显式多样性
- 选择算法：
  - Standard Greedy
  - CELF-style Lazy Greedy
- 评估指标：
  - 集合内平均/中位余弦相似度
  - Dup@0.8
  - Topic relevance
  - Corpus coverage
- 正确性复核：
  - 每步目标值
  - Standard/Lazy 最终目标值一致性
  - 单元测试验证 marginal gain 与 exact recomputation 一致
- 实验可复现性：
  - 同一 topic/mode 的 LLM 候选只生成一次并跨消融复用
  - 配置与语料 chunk 数写入 experiment_config.json
  - JSON 结果采用临时文件替换，避免半写入产物

## 目标函数

### Facility Location

当前实现使用：

f(S) = alpha * sum(rel_i, i in S)
     + (1 - alpha) * sum_j max(sim(i, j), i in S)

与旧实现相比，relevance 现在是“只有选中候选才获得”的模块项，而不是与 S 无关的覆盖基线。因此 alpha 真正控制相关性与覆盖之间的权衡。

### Graph Cut

当前实现使用：

f(S) = alpha * sum(rel_i, i in S)
     + lambda * sum(w(i, j), i in S, j not in S)

其中 w(i, j) = 1 - cosine_similarity(i, j)。

Graph Cut 是子模函数，但不保证单调；当剩余候选都没有正边际收益时，选择器会提前停止，因此结果数量可能小于 k。

## 项目结构

    submod_query_optimization/
    ├── main.py
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── logging_utils.py
    │   ├── utils.py
    │   ├── embeddings.py
    │   ├── llm.py
    │   ├── objectives.py
    │   ├── selectors.py
    │   ├── evaluator.py
    │   ├── corpus.py
    │   └── runner.py
    ├── tests/
    │   ├── test_config.py
    │   ├── test_objectives.py
    │   ├── test_selectors.py
    │   └── test_utils.py
    ├── model/      # 本地 embedding 模型，不提交
    ├── data/       # 本地 txt 语料，不提交
    └── output/     # 实验产物，不提交

v1.py 为早期单文件版本，仅保留作历史参考；新开发应以 main.py + src/ 为准。

## 环境

建议 Python 3.10+。

安装运行依赖：

    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt

准备：

1. 将本地 m3e / sentence-transformer 模型放到 model/，或通过 --m3e-path 指向其他目录。
2. 将评估语料放到 data/，支持递归读取 .txt 文件。
3. 配置 OPENAI_API_KEY。

## 运行

    export OPENAI_API_KEY=sk-...

    python main.py       --topics "embeddings and rerankers" "generative ai"       --num-candidates 20       --k 6       --alpha 0.5       --lambda-diversity 0.5       --llm-model gpt-4o       --m3e-path model       --data-dir data       --output-dir output

关键参数会在启动阶段校验，例如：

- num_candidates > 0
- k > 0
- alpha in [0, 1]
- lambda_diversity in [0, 1]
- 0 <= chunk_overlap < chunk_size

## 测试

核心算法测试不依赖 OpenAI API，也不需要下载 embedding 模型：

    pip install -r requirements-dev.txt
    pytest -q

CI 会在 Python 3.10 / 3.11 / 3.12 上执行源码编译检查和核心测试。

## 输出

output/ 中主要包含：

- experiment_config.json
- prompt_ablation_<topic>.json
- submod_ablation_<topic>.json
- run.log

submod_ablation 结果会记录候选、选择索引、实际查询文本、每步 marginal gain、目标值、运行时间和评估指标。

## 当前边界

- LLM 输出仍具有随机性；仓库保证同一次 runner 内相同候选集合被复用，但跨独立运行并不保证完全一致。
- Graph Cut 目标非单调，因此可能提前停止而不足 k。
- 当前 corpus coverage 是 embedding-space 的平均最大相似度，不等价于真实检索 Recall/NDCG。
- v1.py 尚未删除，后续可迁移到 legacy/ 或 release tag，进一步减少维护面。
