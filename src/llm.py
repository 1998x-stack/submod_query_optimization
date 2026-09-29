# -*- coding: utf-8 -*-
"""OpenAI-backed candidate query generation."""

from __future__ import annotations

import os

from loguru import logger
from openai import OpenAI

from src.config import LLMConfig
from src.utils import safe_json_loads


class OpenAILLM:
    """Small OpenAI client wrapper with deterministic parsing boundaries."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        self.client = OpenAI(
            api_key=api_key,
            timeout=cfg.request_timeout_sec,
            max_retries=cfg.max_retries,
        )
        logger.info("OpenAI client initialized with model: {}", cfg.model_name)

    @staticmethod
    def build_prompt_simple(topic: str, n: int) -> str:
        return (
            "你是一位生成多样化搜索查询的专家。对于任意输入主题，"
            f"请只以 JSON 数组形式返回恰好 {n} 个不同的搜索查询，"
            "覆盖该主题的不同角度和方面。不要解释、不要编号。\n"
            f"输入主题：{topic}"
        )

    @staticmethod
    def build_prompt_structured(topic: str, n: int) -> str:
        return (
            "你是专业的研究策略师。目标：生成一个多样化搜索查询集，"
            "最大化信息覆盖并最小化冗余。\n"
            "仅返回 JSON 数组，不要解释。要求：\n"
            "- 每条查询与主题语义相关；\n"
            "- 每条查询探索不同方面；\n"
            "- 合集覆盖理论、实践、历史、比较、产业、风险等不同视角；\n"
            f"- 数组长度必须为 {n}。\n"
            f"主题：{topic}"
        )

    def _chat(self, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model_name,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if not resp.choices:
            raise RuntimeError("OpenAI response contained no choices")
        return resp.choices[0].message.content or ""

    def generate_candidates(self, topic: str, n: int, mode: str) -> list[str]:
        """Generate, parse, normalize, deduplicate, and cap candidate queries."""
        topic = topic.strip()
        if not topic:
            raise ValueError("topic must not be empty")
        if n <= 0:
            raise ValueError("n must be positive")
        if mode not in {"simple", "structured"}:
            raise ValueError("mode must be 'simple' or 'structured'")

        prompt = (
            self.build_prompt_simple(topic, n)
            if mode == "simple"
            else self.build_prompt_structured(topic, n)
        )
        logger.info("Generating candidates (mode={}): {}", mode, topic)
        raw = self._chat(prompt)
        parsed = safe_json_loads(raw)

        if isinstance(parsed, list):
            queries = [str(item).strip() for item in parsed if isinstance(item, str) and item.strip()]
        else:
            lines = [line.strip("-* \t") for line in raw.splitlines()]
            queries = [line for line in lines if line]

        uniq: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join(query.split())
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(normalized)
            if len(uniq) >= n:
                break

        if len(uniq) < n:
            logger.warning(
                "LLM returned {} valid unique candidates; requested {} for topic={!r}",
                len(uniq),
                n,
                topic,
            )

        return uniq
