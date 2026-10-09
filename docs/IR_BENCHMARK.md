# Phase 3 — Real Information Retrieval Benchmark

Phase 3 evaluates whether selected expansion queries actually improve document retrieval against explicit relevance judgments (qrels).

## Evaluation semantics

For every benchmark query:

1. Generate candidate expansion queries with the existing LLM pipeline.
2. Select expansions with one of the benchmark methods.
3. Always retain the original benchmark query.
4. Embed the original query, selected expansions, and benchmark documents.
5. Compute cosine similarity from every document to every query in the set.
6. Fuse multi-query scores with max fusion.
7. Rank documents by the fused score.
8. Evaluate the ranked list against qrels for the original benchmark query.

This measures the incremental retrieval value of query expansion rather than replacing the original intent.

## Dataset formats

Native layout:

    my_ir_dataset/
    ├── documents.jsonl
    ├── queries.jsonl
    └── qrels.tsv

documents.jsonl:

    {"doc_id":"d1","text":"document text"}

queries.jsonl:

    {"query_id":"q1","text":"benchmark query"}

qrels.tsv:

    query_id    doc_id    relevance
    q1          d1        2

BEIR-style layout is also supported:

    dataset/
    ├── corpus.jsonl
    ├── queries.jsonl
    └── qrels/
        └── test.tsv

The loader understands BEIR-style _id, title, text fields and query-id / corpus-id / score qrels headers.

Four-column TREC qrels are also accepted:

    q1 0 d1 2

## Qrels rules

- Relevance must be a non-negative integer.
- relevance > 0 is considered relevant for binary metrics.
- Graded relevance is used for nDCG.
- Unjudged documents are treated as non-relevant.
- Every benchmark query must have at least one positive qrel.
- Unknown query/document ids fail fast.

## Metrics

For every configured cutoff k:

- Precision@k: relevant documents in top-k divided by k.
- Recall@k: relevant documents in top-k divided by all positive qrels.
- HitRate@k: 1 when at least one relevant document appears in top-k.
- MRR@k: macro mean reciprocal rank of the first relevant document within top-k.
- nDCG@k: graded DCG normalized by the ideal ranking, using gain 2^rel - 1.

The summary is a macro average across benchmark queries.

## Confidence intervals

The benchmark emits deterministic percentile-bootstrap 95% confidence intervals.

The sampling unit is a benchmark query, not an individual Random run. Random repeats are averaged inside each query first, then macro-averaged and bootstrapped across queries.

## Methods and parameter sweep

- OriginalQuery
- TopRelevance
- MMR(lambda=...)
- Random
- FacilityLocation(alpha=...)
- GraphCut(alpha=...,lambda=...)

Example:

    export OPENAI_API_KEY=sk-...

    python main.py \
      --ir-only \
      --ir-dataset-dir examples/ir_dataset \
      --m3e-path model \
      --num-candidates 20 \
      --k 6 \
      --ir-cutoffs 5,10,20 \
      --ir-k-grid 2,4,6,8 \
      --ir-alpha-grid 0.2,0.5,0.8 \
      --ir-lambda-grid 0.2,0.5,0.8 \
      --ir-mmr-grid 0.3,0.6,0.9 \
      --ir-random-repeats 10 \
      --ir-bootstrap-iterations 5000 \
      --random-seed 42 \
      --output-dir output

Each non-original method is evaluated for every value in ir-k-grid. Graph Cut evaluates the Cartesian product of ir-k-grid × ir-alpha-grid × ir-lambda-grid.

## Output artifacts

Phase 3 writes to output/ir/:

- ir_benchmark_detail.csv: query × variant × repeat × cutoff audit trail, including selected expansions, selection runtime, retrieved doc ids/scores, and all IR metrics.
- ir_benchmark_summary.csv: macro means and 95% CI for every variant × cutoff.
- ir_benchmark_manifest.json: dataset sizes, qrel counts, parameter grids, metric semantics, and retrieval/fusion semantics.

## Reproducibility

Phase 2's disk-backed candidate cache is reused. Random selection uses deterministic per-query seeds derived from SHA-256.

## Interpretation caveats

- Incomplete qrels can make unjudged-but-useful documents look non-relevant.
- Results are specific to the selected embedding model and document representation.
- Avoid tuning and reporting on the same split when a development/test split is available.
