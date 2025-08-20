# -*- coding: utf-8 -*-
"""LLM (OpenAI GPT-4o) candidate generation.

包含两种提示词：
- 简单提示词（simple）
- 结构化提示词（structured）

仅返回 JSON 数组的短查询，解析失败自动降级。
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger
from openai import OpenAI  # 新式 SDK

from src.config import LLMConfig
from src.utils import safe_json_loads


class OpenAILLM:
    """OpenAI LLM 包装器。"""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("未检测到 OPENAI_API_KEY，请先设置环境变量。")
        self.client = OpenAI(api_key=api_key)
        logger.info("🤖 OpenAI client initialized with model: {}", cfg.model_name)

    @staticmethod
    def build_prompt_simple(topic: str, n: int) -> str:
        """简单提示词。"""
        return (
            "你是一位生成多样化搜索查询的专家。对于任意输入主题，"
            f"请只以 JSON 数组形式返回恰好 {n} 个不同的搜索查询，"
            "覆盖该主题的不同角度和方面。不要解释、不要编号、不要换行。\n"
            f"输入主题：{topic}"
        )

    @staticmethod
    def build_prompt_structured(topic: str, n: int) -> str:
        """结构化提示词。"""
        return (
            "你是专业的研究策略师。目标：生成一个多样化搜索查询集，最大化信息覆盖并最小化冗余。\n"
            "请仅返回 JSON 数组（长度为 N），不要解释。遵循：\n"
            "- 相关性：每条查询必须与主题语义相关；\n"
            "- 多样性：每条查询探索不同方面，尽量低重叠；\n"
            "- 覆盖率：合在一起尽量覆盖主题全景；\n"
            "流程：分解概念→视角映射（理论/实践/历史/比较/产业/风险等）→构建具体可搜查询→多样性检查。\n"
            f"N={n}，主题：{topic}"
        )

    def _chat(self, user_prompt: str) -> str:
        """调用 Chat Completions 并返回文本。"""
        resp = self.client.chat.completions.create(
            model=self.cfg.model_name,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return text

    def generate_candidates(self, topic: str, n: int, mode: str) -> list[str]:
        """按模式生成候选查询。

        Args:
            topic: 原始主题。
            n: 需要的候选数量。
            mode: "simple" 或 "structured".

        Returns:
            长度 <= n 的候选查询列表（去重、兜底解析）。
        """
        assert mode in {"simple", "structured"}, "mode 必须为 simple 或 structured"
        prompt = (
            self.build_prompt_simple(topic, n) if mode == "simple" else self.build_prompt_structured(topic, n)
        )
        logger.info("🧪 Generating candidates (mode={}): {}", mode, topic)
        raw = self._chat(prompt)
        parsed = safe_json_loads(raw)

        queries: list[str]
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            queries = [x.strip() for x in parsed if str(x).strip()]
        else:
            # 兜底：按行切分/去掉前缀 bullet
            lines = [ln.strip("-* \t") for ln in raw.splitlines()]
            queries = [ln for ln in lines if ln]

        # 去重 & 截断
        uniq: list[str] = []
        seen = set()
        for q in queries:
            if q not in seen:
                uniq.append(q)
                seen.add(q)
            if len(uniq) >= n:
                break

        if not uniq:
            logger.warning("⚠️ LLM 未返回有效候选，topic='{}'", topic)

        return uniq
