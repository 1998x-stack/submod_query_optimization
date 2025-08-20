# submod_query_optimization
```
submod_query_optimization/
├── main.py                                # 入口脚本（CLI）
├── requirements.txt                       # 依赖
├── src/
│   ├── config.py                          # 所有配置的数据类（Embedding/LLM/实验）
│   ├── logging_utils.py                   # loguru 初始化与统一日志器
│   ├── utils.py                           # 通用工具：相似度、JSON 解析、slug 等
│   ├── embeddings.py                      # LangChain 基于本地 m3e 的向量封装
│   ├── llm.py                             # GPT-4o 候选查询生成（简单/结构化提示词）
│   ├── objectives.py                      # 子模目标：设施选址法 & 图割法（显式多样性）
│   ├── selectors.py                       # 标准贪心 & 懒贪心 选择器
│   ├── evaluator.py                       # 指标评估（多样性、重复率、相关性、覆盖率）
│   ├── corpus.py                          # 读取 data/*.txt，LangChain 分割并嵌入
│   └── runner.py                          # 消融实验总管道 + 正确性复核（recheck）
├── model/                                 # 你的 m3e 本地模型目录（自备）
├── data/                                  # 你的 txt 语料目录（自备）
└── output/                                # 实验输出目录（自动创建）
```