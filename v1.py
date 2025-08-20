#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Submodular Query Optimization — Industrial-Grade Pipeline.

本脚本提供一个完整、可扩展的工业级 Python 实现，用于：
1) 基于 LLM 的候选查询生成（简单提示词 vs 结构化提示词的对比实验）；
2) 使用子模函数 + 贪心算法（含标准贪心与懒贪心、设施选址法与图割法）从候选中选出“少而多样”的高质量查询；
3) 进行多维度消融实验（Prompts / 目标函数 / 贪心算法实现），并输出对比指标与可重复的实验产物。

目录结构假设：
- model/ : 已下载好的 m3e Embedding 模型（如 Moka-AI/m3e-base 等）的本地目录。
- data/  : 若干 .txt 文本文件（英文/中文均可），用于评估“查询覆盖率”（相对于语料库的覆盖）。
- 输出目录：output/ （若不存在将自动创建）。

运行示例：
$ export OPENAI_API_KEY=sk-...
$ python submod_query_opt.py \
    --topics "embeddings and rerankers" "generative ai" \
    --num-candidates 20 \
    --k 6 \
    --alpha 0.5 \
    --lambda-diversity 0.5 \
    --llm-model gpt-4o \
    --m3e-path model \
    --data-dir data \
    --output-dir output

依赖：
- numpy, sentence_transformers, openai (或 requests 作为后备), tqdm, tabulate

注意：
- 符合 PEP 257/PEP 8，提供类型注解与中文注释；
- 考虑边界条件；
- 代码包含关键步骤的二次校验（correctness recheck）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

# 优先使用 openai 官方 SDK；若不可用，则使用 requests 直接调用 REST API。
try:
    import openai  # type: ignore
    _HAS_OPENAI = True
except Exception:  # pragma: no cover - 容错
    import requests  # type: ignore
    _HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "缺少 sentence_transformers，请先安装：pip install -U sentence-transformers"
    ) from exc


# =============================================================
# 实用函数区域
# =============================================================

def ensure_dir(path: Path) -> None:
    """Ensure directory exists.

    Args:
        path: 目标目录路径。
    """
    path.mkdir(parents=True, exist_ok=True)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of vectors.

    所有输入向量都会在函数内部进行 L2 归一化，以确保相似度计算的稳定性。

    Args:
        a: 形状 (n, d) 的向量集合。
        b: 形状 (m, d) 的向量集合。

    Returns:
        形状 (n, m) 的余弦相似度矩阵，范围约为 [-1, 1]。
    """
    # 防止零向量导致 NaN——加上一个极小值并做安全归一化
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.asarray(a_norm @ b_norm.T, dtype=np.float32)


def pairwise_cosine(a: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for a set of vectors.

    Args:
        a: 形状 (n, d) 的向量集合。

    Returns:
        形状 (n, n) 的对称相似度矩阵，主对角线为 1。
    """
    return cosine_similarity_matrix(a, a)


def safe_json_loads(maybe_json: str) -> Optional[Any]:
    """Try to parse a JSON string robustly.

    该函数用于对 LLM 的返回进行健壮解析：
    - 去除围栏/多余字符；
    - 失败时返回 None。

    Args:
        maybe_json: 可能是 JSON 的字符串。

    Returns:
        解析出的对象或 None。
    """
    try:
        # 去除 Markdown 代码块围栏
        cleaned = maybe_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # 剪掉语言声明，如 ```json
            cleaned = cleaned[cleaned.find("\n") + 1 :]
        return json.loads(cleaned)
    except Exception:
        return None


# =============================================================
# 数据与模型封装
# =============================================================

@dataclass
class EmbeddingConfig:
    """Embedding 配置。"""

    model_path: str = "model"
    batch_size: int = 32
    device: Optional[str] = None  # 例如 "cuda" / "cpu" / None 自动


class M3EEmbedder:
    """m3e Embedding 模型封装。

    使用 SentenceTransformer 从本地目录加载模型，提供批量编码接口。
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        # 加载本地模型；若模型不在本地，用户应提前将模型放置到 model/ 目录
        self.model = SentenceTransformer(self.cfg.model_path, device=self.cfg.device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts to embeddings.

        Args:
            texts: 文本序列。

        Returns:
            形状 (n, d) 的 numpy array，dtype=float32。
        """
        if len(texts) == 0:
            return np.zeros((0, 768), dtype=np.float32)
        # SentenceTransformer.encode 已经做了分批；设置 convert_to_numpy=True
        emb = self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # 统一在相似度计算处归一化
            show_progress_bar=False,
        )
        # 转为 float32，节省内存
        return emb.astype(np.float32)


@dataclass
class LLMConfig:
    """LLM 配置。"""

    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 800
    system_prompt: str = (
        "你是一位严谨的研究助理，只以 JSON 数组形式返回 N 条短查询字符串，"
        "每条长度不超过 16 个汉字或 10 个英文词，不要编号、不要描述。"
    )


class OpenAILLM:
    """OpenAI Chat Completions 简单封装。

    使用官方 SDK；若导入失败则使用 requests 调用 REST API。
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("未检测到 OPENAI_API_KEY 环境变量，请先设置后再运行。")
        self.api_key = api_key
        if _HAS_OPENAI:
            openai.api_key = api_key  # type: ignore
        else:
            self.session = requests.Session()  # type: ignore[attr-defined]

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Call chat completions API and return raw text.

        Args:
            messages: 对话消息列表。
        Returns:
            LLM 返回的文本（一般为 JSON 字符串）。
        """
        if _HAS_OPENAI:
            # 官方 SDK 路径
            resp = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                model=self.cfg.model_name,
                messages=messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            return resp["choices"][0]["message"]["content"]  # type: ignore[index]
        else:
            # REST API 路径（尽量保持兼容）
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.cfg.model_name,
                "messages": messages,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            resp = self.session.post(
                "https://api.openai.com/v1/chat/completions",  # noqa: E501
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    # ------------------------- Prompt 设计 -------------------------
    @staticmethod
    def build_prompt_simple(topic: str, num_queries: int) -> str:
        """简单提示词：直接要求 LLM 生成多样查询。

        Args:
            topic: 原始主题。
            num_queries: 需要的查询数量。
        Returns:
            提示词字符串。
        """
        return (
            "你是一位生成多样化搜索查询的专家。对于任意输入主题，"
            f"请只以 JSON 数组形式返回恰好 {num_queries} 个不同的搜索查询，"
            "覆盖该主题的不同角度和方面。不要解释、不要编号、不要换行。\n"
            f"输入主题：{topic}"
        )

    @staticmethod
    def build_prompt_structured(topic: str, num_queries: int) -> str:
        """结构化提示词：显式提出相关性/多样性/覆盖率/流程。

        Args:
            topic: 原始主题。
            num_queries: 数量。
        Returns:
            提示词字符串。
        """
        return (
            "你是专业的研究策略师。目标：生成一个多样化搜索查询集，最大化信息覆盖并最小化冗余。\n"
            "请仅返回 JSON 数组（长度为 N），不要解释。遵循：\n"
            "- 相关性：每条查询必须与主题语义相关；\n"
            "- 多样性：每条查询探索不同方面，尽量低重叠；\n"
            "- 覆盖率：合在一起尽量覆盖主题全景；\n"
            "流程：分解概念→视角映射（理论/实践/历史/比较/产业/风险等）→构建具体可搜查询→多样性检查。\n"
            f"N={num_queries}，主题：{topic}"
        )

    def generate_candidates(self, topic: str, num_queries: int, mode: str) -> List[str]:
        """Generate candidate queries with a given prompting mode.

        Args:
            topic: 原始主题。
            num_queries: 候选数。
            mode: "simple" 或 "structured"。
        Returns:
            候选查询列表（长度可能小于等于 num_queries；解析失败时做降级处理）。
        """
        assert mode in {"simple", "structured"}, "mode 必须为 simple 或 structured"
        prompt = (
            self.build_prompt_simple(topic, num_queries)
            if mode == "simple"
            else self.build_prompt_structured(topic, num_queries)
        )
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]
        raw_text = self._chat(messages)
        parsed = safe_json_loads(raw_text)
        queries: List[str]
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            queries = [x.strip() for x in parsed if x and isinstance(x, str)]
        else:
            # 兜底：按行切分，去掉空行
            lines = [ln.strip("-* \t") for ln in raw_text.splitlines()]
            queries = [ln for ln in lines if ln]
        # 去重 + 截断到 num_queries
        uniq = []
        seen = set()
        for q in queries:
            if q not in seen:
                seen.add(q)
                uniq.append(q)
            if len(uniq) >= num_queries:
                break
        return uniq


# =============================================================
# 子模目标函数：设施选址法 / 图割法
# =============================================================

class BaseSubmodularObjective:
    """子模目标函数抽象基类。

    子类需要实现：
    - reset_state()
    - initial_gains()
    - marginal_gain(idx)
    - add_to_set(idx)
    - total_value(selected_set)
    """

    def reset_state(self) -> None:
        raise NotImplementedError

    def initial_gains(self) -> np.ndarray:
        raise NotImplementedError

    def marginal_gain(self, idx: int) -> float:
        raise NotImplementedError

    def add_to_set(self, idx: int) -> None:
        raise NotImplementedError

    def total_value(self, selected: Sequence[int]) -> float:
        raise NotImplementedError


class FacilityLocationObjective(BaseSubmodularObjective):
    """设施选址（隐式多样性）目标函数。

    f(S) = \sum_{j in V} max( alpha * rel[j], max_{i in S} sim[i, j] )

    实现细节：
    - 维护一个当前覆盖向量 current_cov[j] = max(alpha*rel[j], max_{i∈S} sim[i,j])。
    - 添加新元素 i 时的边际增益：sum_j max(0, sim[i,j] - current_cov[j])。
    """

    def __init__(
        self,
        sim: np.ndarray,
        rel: np.ndarray,
        alpha: float = 0.5,
    ) -> None:
        assert sim.shape[0] == sim.shape[1], "sim 必须为 (n,n)"
        assert rel.shape[0] == sim.shape[0], "rel 维度不匹配"
        self.sim = sim.astype(np.float32)
        self.rel = rel.astype(np.float32)
        self.alpha = float(alpha)
        self.n = sim.shape[0]
        self.current_cov = np.zeros((self.n,), dtype=np.float32)
        self.selected_mask = np.zeros((self.n,), dtype=bool)
        self.reset_state()

    def reset_state(self) -> None:
        # 初始化覆盖向量为 alpha * rel
        self.current_cov = (self.alpha * self.rel).copy()
        self.selected_mask[:] = False

    def initial_gains(self) -> np.ndarray:
        # 第一次添加每个 i 的增益：sum_j max(0, sim[i,j] - alpha*rel[j])
        base = self.current_cov  # shape (n,)
        gains = np.maximum(0.0, self.sim - base[None, :]).sum(axis=1)
        gains[self.selected_mask] = -np.inf
        return gains.astype(np.float32)

    def marginal_gain(self, idx: int) -> float:
        # 计算在当前状态下添加 idx 的边际增益
        if self.selected_mask[idx]:
            return -math.inf
        diff = self.sim[idx, :] - self.current_cov
        gain = float(np.maximum(0.0, diff).sum())
        return gain

    def add_to_set(self, idx: int) -> None:
        # 更新覆盖向量与掩码
        if self.selected_mask[idx]:
            return
        self.current_cov = np.maximum(self.current_cov, self.sim[idx, :])
        self.selected_mask[idx] = True

    def total_value(self, selected: Sequence[int]) -> float:
        # 重新计算 f(S) 的总值（用于校验）
        base = self.alpha * self.rel
        if not selected:
            return float(base.sum())
        # 取已选 queries 对每个 j 的最大覆盖
        cover = np.maximum.reduce([self.sim[i, :] for i in selected])
        cover = np.maximum(base, cover)
        return float(cover.sum())


class GraphCutObjective(BaseSubmodularObjective):
    """图割（显式多样性）目标函数（基于“不相似度”作为割权重）。

    定义：
        w(i,j) = 1 - sim(i,j) >= 0 ；w(ii) = 0。
        Cut(S) = sum_{i in S, j in V\S} w(i,j)，是子模函数。
        Relevance(S) = sum_{i in S} sim(q0, i)（模函数）。
        f(S) = alpha * Relevance(S) + lambda * Cut(S)。

    增量计算：当 i 从 V\S 移入 S 时：
        ΔCut = [sum_{t in V\S\{i}} w(i,t)] - [sum_{s in S} w(s,i)]
              = total_row[i] - 2 * sum_to_S[i]
        ΔRel = sim(q0, i)
        Δf = alpha*ΔRel + lambda_div*( total_row[i] - 2*sum_to_S[i] )
    """

    def __init__(
        self,
        sim: np.ndarray,
        rel: np.ndarray,
        alpha: float = 0.5,
        lambda_div: float = 0.5,
    ) -> None:
        assert sim.shape[0] == sim.shape[1], "sim 必须为 (n,n)"
        assert rel.shape[0] == sim.shape[0], "rel 维度不匹配"
        self.sim = sim.astype(np.float32)
        self.rel = rel.astype(np.float32)
        self.alpha = float(alpha)
        self.lambda_div = float(lambda_div)
        self.n = sim.shape[0]

        self.w = (1.0 - self.sim).astype(np.float32)
        np.fill_diagonal(self.w, 0.0)
        self.total_row = self.w.sum(axis=1)  # shape (n,)
        self.sum_to_S = np.zeros((self.n,), dtype=np.float32)
        self.selected_mask = np.zeros((self.n,), dtype=bool)
        self.reset_state()

    def reset_state(self) -> None:
        self.sum_to_S.fill(0.0)
        self.selected_mask[:] = False

    def initial_gains(self) -> np.ndarray:
        # 初始 S=∅：ΔCut = total_row[i] - 0
        gains = self.alpha * self.rel + self.lambda_div * self.total_row
        gains[self.selected_mask] = -np.inf
        return gains.astype(np.float32)

    def marginal_gain(self, idx: int) -> float:
        if self.selected_mask[idx]:
            return -math.inf
        delta_cut = float(self.total_row[idx] - 2.0 * self.sum_to_S[idx])
        delta_rel = float(self.rel[idx])
        return self.alpha * delta_rel + self.lambda_div * delta_cut

    def add_to_set(self, idx: int) -> None:
        if self.selected_mask[idx]:
            return
        # 更新 sum_to_S：对于每个 j，将 w(j, idx) 加到其 sum_to_S[j]
        self.sum_to_S += self.w[:, idx]
        self.selected_mask[idx] = True

    def total_value(self, selected: Sequence[int]) -> float:
        # 精确重算 f(S) 用于校验
        if not selected:
            return 0.0
        S = np.zeros((self.n,), dtype=bool)
        S[selected] = True
        # Cut(S)
        # 仅统计 S->V\S 部分：矩阵乘法 + 掩码
        cut_val = float(self.w[np.ix_(S, ~S)].sum())
        rel_val = float(self.rel[S].sum())
        return self.alpha * rel_val + self.lambda_div * cut_val


# =============================================================
# 贪心选择器（标准贪心 / 懒贪心）
# =============================================================

@dataclass
class GreedyResult:
    """贪心选择的结果对象。"""

    selected_indices: List[int]
    objective_values: List[float]
    per_step_gains: List[float]
    runtime_sec: float
    algo_name: str


class StandardGreedySelector:
    """标准贪心选择器：每一轮都重算所有候选的边际增益。"""

    def __init__(self, objective: BaseSubmodularObjective) -> None:
        self.objective = objective

    def select(self, k: int) -> GreedyResult:
        start = time.time()
        self.objective.reset_state()
        n = getattr(self.objective, "n")
        selected: List[int] = []
        gains_all: List[float] = []
        values: List[float] = []

        for step in range(k):
            gains = np.full((n,), -np.inf, dtype=np.float32)
            for idx in range(n):
                gains[idx] = self.objective.marginal_gain(idx)
            best_idx = int(np.argmax(gains))
            best_gain = float(gains[best_idx])
            if not math.isfinite(best_gain) or best_gain <= 0.0:
                # 边界情况：剩余候选无正增益，则提前停止
                break
            self.objective.add_to_set(best_idx)
            selected.append(best_idx)
            gains_all.append(best_gain)
            values.append(self.objective.total_value(selected))

        runtime = time.time() - start
        return GreedyResult(
            selected_indices=selected,
            objective_values=values,
            per_step_gains=gains_all,
            runtime_sec=runtime,
            algo_name="standard_greedy",
        )


class LazyGreedySelector:
    """懒贪心选择器：使用最大堆避免不必要的增益重算。"""

    def __init__(self, objective: BaseSubmodularObjective) -> None:
        self.objective = objective

    def select(self, k: int) -> GreedyResult:
        import heapq

        start = time.time()
        self.objective.reset_state()
        n = getattr(self.objective, "n")
        selected: List[int] = []
        gains_all: List[float] = []
        values: List[float] = []

        # 初始化堆：(-gain, last_update_step, idx)
        gains0 = self.objective.initial_gains()
        heap: List[Tuple[float, int, int]] = [(-float(g), -1, i) for i, g in enumerate(gains0)]
        heapq.heapify(heap)

        step = 0
        while step < k and heap:
            # 取出当前“看起来”最优的元素
            neg_gain, last_step, idx = heapq.heappop(heap)
            current_gain = -neg_gain

            # 若该元素不是在当前 S 下评估的，则进行一次“懒重算”
            if last_step != step:
                true_gain = self.objective.marginal_gain(idx)
                heapq.heappush(heap, (-true_gain, step, idx))
                continue

            # 否则，该元素在当前步已经评估过，直接选择
            if not math.isfinite(current_gain) or current_gain <= 0.0:
                break

            self.objective.add_to_set(idx)
            selected.append(idx)
            gains_all.append(float(current_gain))
            values.append(self.objective.total_value(selected))

            # 进入下一步，同时将所有元素视为“过期评估”，通过 last_step != step+1 触发懒重算
            step += 1

        runtime = time.time() - start
        return GreedyResult(
            selected_indices=selected,
            objective_values=values,
            per_step_gains=gains_all,
            runtime_sec=runtime,
            algo_name="lazy_greedy",
        )


# =============================================================
# 指标与评估
# =============================================================

@dataclass
class EvaluationMetrics:
    """评估指标集合。"""

    mean_intra_similarity: float
    median_intra_similarity: float
    duplicate_rate_08: float
    avg_relevance_to_topic: float
    coverage_score_vs_corpus: float


class Evaluator:
    """对查询集合进行多维评估的工具类。"""

    def __init__(self, embedder: M3EEmbedder, corpus_embeddings: Optional[np.ndarray]) -> None:
        self.embedder = embedder
        self.corpus_embeddings = corpus_embeddings  # (m, d) 或 None

    @staticmethod
    def _dup_rate(sim_mat: np.ndarray, threshold: float = 0.8) -> float:
        n = sim_mat.shape[0]
        if n <= 1:
            return 0.0
        mask = np.triu(np.ones_like(sim_mat, dtype=bool), k=1)
        pairs = sim_mat[mask]
        dup = float((pairs >= threshold).sum())
        total = float(pairs.size)
        return dup / max(total, 1.0)

    def evaluate(
        self,
        queries: Sequence[str],
        query_embeddings: Optional[np.ndarray],
        topic_embedding: Optional[np.ndarray],
    ) -> EvaluationMetrics:
        """对查询集合进行多维评估（多样性、相关性、覆盖率等）。

        Args:
            queries: 查询字符串集合。
            query_embeddings: 查询的向量表示，若为 None 则自动编码。
            topic_embedding: 主题（原始输入）的向量表示。

        Returns:
            EvaluationMetrics 对象。
        """
        if len(queries) == 0:
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        if query_embeddings is None:
            query_embeddings = self.embedder.encode(queries)

        sim_mat = pairwise_cosine(query_embeddings)
        # 计算内部相似度统计
        n = sim_mat.shape[0]
        tri = sim_mat[np.triu_indices(n, k=1)]
        mean_intra = float(tri.mean()) if tri.size > 0 else 0.0
        median_intra = float(np.median(tri)) if tri.size > 0 else 0.0
        dup_rate = self._dup_rate(sim_mat, threshold=0.8)

        # 与主题的平均相关性
        if topic_embedding is None:
            avg_rel = 0.0
        else:
            rel_vec = cosine_similarity_matrix(query_embeddings, topic_embedding[None, :]).reshape(-1)
            avg_rel = float(rel_vec.mean())

        # 语料覆盖率：对语料中的每个文档向量，取与查询集合的最大相似度，然后求和 / m
        coverage = 0.0
        if self.corpus_embeddings is not None and self.corpus_embeddings.shape[0] > 0:
            doc2q = cosine_similarity_matrix(self.corpus_embeddings, query_embeddings)  # (m, n)
            max_per_doc = doc2q.max(axis=1)  # 每个文档被覆盖到的最大程度
            coverage = float(max_per_doc.mean())

        return EvaluationMetrics(
            mean_intra_similarity=mean_intra,
            median_intra_similarity=median_intra,
            duplicate_rate_08=dup_rate,
            avg_relevance_to_topic=avg_rel,
            coverage_score_vs_corpus=coverage,
        )


# =============================================================
# 实验管道（Ablation Studies）
# =============================================================

@dataclass
class ExperimentConfig:
    """实验配置。"""

    topics: List[str]
    num_candidates: int = 20
    k: int = 6
    alpha: float = 0.5
    lambda_diversity: float = 0.5
    llm_model: str = "gpt-4o"
    m3e_path: str = "model"
    data_dir: str = "data"
    output_dir: str = "output"


class CorpusLoader:
    """加载 data/ 目录下的文本，并构建文档向量。"""

    def __init__(self, data_dir: str, embedder: M3EEmbedder) -> None:
        self.data_dir = Path(data_dir)
        self.embedder = embedder

    def load(self, max_chars_per_file: int = 1200) -> Tuple[List[str], np.ndarray]:
        """Load .txt files and build document embeddings.

        Args:
            max_chars_per_file: 每个文件截取的最大字符数（降低编码成本）。
        Returns:
            (docs_texts, embeddings)
        """
        texts: List[str] = []
        for p in sorted(self.data_dir.glob("**/*.txt")):
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            content = content.strip()
            if not content:
                continue
            texts.append(content[:max_chars_per_file])
        if not texts:
            return [], np.zeros((0, 768), dtype=np.float32)
        emb = self.embedder.encode(texts)
        return texts, emb


class AblationRunner:
    """组织并运行各类消融实验的主类。"""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.embedder = M3EEmbedder(EmbeddingConfig(model_path=cfg.m3e_path))
        self.llm = OpenAILLM(LLMConfig(model_name=cfg.llm_model))
        self.output_dir = Path(cfg.output_dir)
        ensure_dir(self.output_dir)

        # 加载语料库向量（用于覆盖率评估）
        corpus_loader = CorpusLoader(cfg.data_dir, self.embedder)
        self.corpus_texts, self.corpus_emb = corpus_loader.load()

    # ------------------------ PROMPT 消融 ------------------------
    def run_prompt_ablation(self) -> None:
        """比较简单提示词 vs 结构化提示词在候选层面的差异。"""
        print("\n[ABLT] Prompt Ablation — 简单提示词 vs 结构化提示词")
        rows = []
        for topic in self.cfg.topics:
            topic_emb = self.embedder.encode([topic])[0]

            # 生成候选（simple / structured）
            cand_simple = self.llm.generate_candidates(topic, self.cfg.num_candidates, "simple")
            cand_struct = self.llm.generate_candidates(topic, self.cfg.num_candidates, "structured")

            emb_simple = self.embedder.encode(cand_simple)
            emb_struct = self.embedder.encode(cand_struct)

            evaluator = Evaluator(self.embedder, self.corpus_emb)
            m_simple = evaluator.evaluate(cand_simple, emb_simple, topic_emb)
            m_struct = evaluator.evaluate(cand_struct, emb_struct, topic_emb)

            # 输出并保存
            rows.append([
                topic,
                len(cand_simple),
                f"{m_simple.mean_intra_similarity:.3f}",
                f"{m_simple.duplicate_rate_08:.2%}",
                f"{m_simple.avg_relevance_to_topic:.3f}",
                f"{m_simple.coverage_score_vs_corpus:.3f}",
                len(cand_struct),
                f"{m_struct.mean_intra_similarity:.3f}",
                f"{m_struct.duplicate_rate_08:.2%}",
                f"{m_struct.avg_relevance_to_topic:.3f}",
                f"{m_struct.coverage_score_vs_corpus:.3f}",
            ])

            # 保存具体候选到文件，便于人工检查“主流观点偏好/冷门角度缺失”
            out_json = {
                "topic": topic,
                "candidates_simple": cand_simple,
                "candidates_structured": cand_struct,
                "metrics_simple": dataclasses.asdict(m_simple),
                "metrics_structured": dataclasses.asdict(m_struct),
            }
            (self.output_dir / f"prompt_ablation_{self._slug(topic)}.json").write_text(
                json.dumps(out_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        headers = [
            "Topic",
            "#Simple",
            "Simple-MeanIntraSim",
            "Simple-Dup@0.8",
            "Simple-AvgRel",
            "Simple-Coverage",
            "#Struct",
            "Struct-MeanIntraSim",
            "Struct-Dup@0.8",
            "Struct-AvgRel",
            "Struct-Coverage",
        ]
        print(tabulate(rows, headers=headers, tablefmt="github"))
        print("说明：MeanIntraSim 越低越多样；Dup@0.8 越低越少重复；AvgRel 越高越贴题；Coverage 越高说明覆盖语料更广。")

    # ------------------ 子模函数/算法 消融 ------------------
    def run_submodular_ablation(self) -> None:
        """对比（设施选址 vs 图割）×（标准贪心 vs 懒贪心）。"""
        print("\n[ABLT] Submodular Objectives × Greedy Variants")
        rows = []
        for topic in self.cfg.topics:
            topic_emb = self.embedder.encode([topic])[0]

            # 统一使用“简单提示词”生成更自然的候选，然后用不同目标函数+选择器做子集选择
            candidates = self.llm.generate_candidates(topic, self.cfg.num_candidates, "simple")
            cand_emb = self.embedder.encode(candidates)

            # 构建相似度与相关性向量
            sim = pairwise_cosine(cand_emb)
            rel = cosine_similarity_matrix(cand_emb, topic_emb[None, :]).reshape(-1)

            # 目标函数对象
            obj_fl = FacilityLocationObjective(sim=sim, rel=rel, alpha=self.cfg.alpha)
            obj_gc = GraphCutObjective(sim=sim, rel=rel, alpha=self.cfg.alpha, lambda_div=self.cfg.lambda_diversity)

            # 选择器
            sel_std_fl = StandardGreedySelector(obj_fl)
            sel_lazy_fl = LazyGreedySelector(obj_fl)
            sel_std_gc = StandardGreedySelector(obj_gc)
            sel_lazy_gc = LazyGreedySelector(obj_gc)

            # 执行选择
            res_std_fl = sel_std_fl.select(self.cfg.k)
            res_lazy_fl = sel_lazy_fl.select(self.cfg.k)
            res_std_gc = sel_std_gc.select(self.cfg.k)
            res_lazy_gc = sel_lazy_gc.select(self.cfg.k)

            # 评估被选中的查询集合
            evaluator = Evaluator(self.embedder, self.corpus_emb)
            def eval_sel(res: GreedyResult) -> EvaluationMetrics:
                qs = [candidates[i] for i in res.selected_indices]
                qemb = cand_emb[res.selected_indices] if res.selected_indices else None
                return evaluator.evaluate(qs, qemb, topic_emb)

            m_std_fl = eval_sel(res_std_fl)
            m_lazy_fl = eval_sel(res_lazy_fl)
            m_std_gc = eval_sel(res_std_gc)
            m_lazy_gc = eval_sel(res_lazy_gc)

            # 汇总数据行
            def summarize_row(name: str, res: GreedyResult, m: EvaluationMetrics) -> List[Any]:
                return [
                    topic,
                    name,
                    len(res.selected_indices),
                    f"{res.runtime_sec:.3f}s",
                    f"{m.mean_intra_similarity:.3f}",
                    f"{m.duplicate_rate_08:.2%}",
                    f"{m.avg_relevance_to_topic:.3f}",
                    f"{m.coverage_score_vs_corpus:.3f}",
                ]

            rows.append(summarize_row("FL-Std", res_std_fl, m_std_fl))
            rows.append(summarize_row("FL-Lazy", res_lazy_fl, m_lazy_fl))
            rows.append(summarize_row("GC-Std", res_std_gc, m_std_gc))
            rows.append(summarize_row("GC-Lazy", res_lazy_gc, m_lazy_gc))

            # 保存详细结果（含增益曲线、选择的一致性检查）
            out_json = {
                "topic": topic,
                "candidates": candidates,
                "selected": {
                    "FL_Std": res_std_fl.selected_indices,
                    "FL_Lazy": res_lazy_fl.selected_indices,
                    "GC_Std": res_std_gc.selected_indices,
                    "GC_Lazy": res_lazy_gc.selected_indices,
                },
                "objective_values": {
                    "FL_Std": res_std_fl.objective_values,
                    "FL_Lazy": res_lazy_fl.objective_values,
                    "GC_Std": res_std_gc.objective_values,
                    "GC_Lazy": res_lazy_gc.objective_values,
                },
                "metrics": {
                    "FL_Std": dataclasses.asdict(m_std_fl),
                    "FL_Lazy": dataclasses.asdict(m_lazy_fl),
                    "GC_Std": dataclasses.asdict(m_std_gc),
                    "GC_Lazy": dataclasses.asdict(m_lazy_gc),
                },
                "runtimes": {
                    "FL_Std": res_std_fl.runtime_sec,
                    "FL_Lazy": res_lazy_fl.runtime_sec,
                    "GC_Std": res_std_gc.runtime_sec,
                    "GC_Lazy": res_lazy_gc.runtime_sec,
                },
            }
            (self.output_dir / f"submod_ablation_{self._slug(topic)}.json").write_text(
                json.dumps(out_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # -------------------- 正确性 & 逻辑复核 --------------------
            # 1) 目标函数值单调递增（或不下降）
            for key, res in {
                "FL_Std": res_std_fl,
                "FL_Lazy": res_lazy_fl,
                "GC_Std": res_std_gc,
                "GC_Lazy": res_lazy_gc,
            }.items():
                vals = res.objective_values
                if any(vals[i] > vals[i + 1] + 1e-6 for i in range(len(vals) - 1)):
                    print(f"[WARN] {topic} - {key} 的目标函数曲线非单调，需检查实现与数据。")

            # 2) 懒贪心与标准贪心应当产生相同或极为接近的解（允许同分多解）
            if res_std_fl.selected_indices != res_lazy_fl.selected_indices:
                print(f"[INFO] {topic} - FL: 懒贪心与标准贪心选集不同（可能存在等价多解）。")
            if res_std_gc.selected_indices != res_lazy_gc.selected_indices:
                print(f"[INFO] {topic} - GC: 懒贪心与标准贪心选集不同（可能存在等价多解）。")

        headers = [
            "Topic",
            "Method",
            "#Selected",
            "Runtime",
            "MeanIntraSim",
            "Dup@0.8",
            "AvgRel",
            "Coverage",
        ]
        print(tabulate(rows, headers=headers, tablefmt="github"))
        print("注：Runtime 体现懒贪心的效率优势；MeanIntraSim/Dup@0.8 越低越好；AvgRel/Coverage 越高越好。")

    # ------------------------ 工具方法 ------------------------
    @staticmethod
    def _slug(text: str) -> str:
        keep = [c.lower() if c.isalnum() else '-' for c in text]
        s = ''.join(keep)
        while '--' in s:
            s = s.replace('--', '-')
        return s.strip('-') or 'topic'


# =============================================================
# CLI 与主流程
# =============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Submodular Query Optimization — 少而多样的高质量查询",
    )
    p.add_argument(
        "--topics",
        nargs="*",
        default=["embeddings and rerankers"],
        help="一组原始主题（每个主题将独立运行实验）",
    )
    p.add_argument("--num-candidates", type=int, default=20, help="每个主题的候选查询数")
    p.add_argument("--k", type=int, default=6, help="最终选择的查询数")
    p.add_argument("--alpha", type=float, default=0.5, help="相关性权重（0~1）")
    p.add_argument(
        "--lambda-diversity",
        type=float,
        default=0.5,
        help="图割法中的多样性权重（0~1）",
    )
    p.add_argument("--llm-model", type=str, default="gpt-4o", help="OpenAI 模型名")
    p.add_argument("--m3e-path", type=str, default="model", help="m3e 本地模型目录")
    p.add_argument("--data-dir", type=str, default="data", help="评估用语料目录")
    p.add_argument("--output-dir", type=str, default="output", help="输出目录")
    return p.parse_args()


def main() -> None:
    """主入口：运行三类消融实验并输出结果。"""
    args = parse_args()

    # 参数边界检查
    if args.num_candidates <= 0 or args.k <= 0:
        raise ValueError("num-candidates 与 k 必须为正整数。")
    if args.k > args.num_candidates:
        print("[WARN] k 大于候选数，将在候选不足时提前停止。")

    cfg = ExperimentConfig(
        topics=list(args.topics),
        num_candidates=int(args.num_candidates),
        k=int(args.k),
        alpha=float(args.alpha),
        lambda_diversity=float(args.lambda_diversity),
        llm_model=str(args.llm_model),
        m3e_path=str(args.m3e_path),
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
    )

    print("=" * 88)
    print("Submodular Query Optimization — 实验配置")
    print(dataclasses.asdict(cfg))
    print("=" * 88)

    runner = AblationRunner(cfg)

    # 1) Prompt 消融：验证“LLM 自带偏见、易放大主流观点、难覆盖冷门角度”的现象
    # 通过 MeanIntraSim / Dup@0.8 / Coverage 等指标作定量支持，并保存候选以便人工复核。
    runner.run_prompt_ablation()

    # 2) 子模函数 × 贪心算法 消融：设施选址 vs 图割；标准贪心 vs 懒贪心
    runner.run_submodular_ablation()

    print("\n✅ 全部实验完成。详细 JSON 输出位于:", cfg.output_dir)


if __name__ == "__main__":
    main()

