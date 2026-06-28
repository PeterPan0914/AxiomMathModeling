"""AwardJudgeAgent 模块，系统的"国奖评审官"。

评估论文的竞赛获奖潜力。
评分维度：Method(30) + Innovation(30) + Interpretability(20) + CompetitionStyle(20)。
Innovation < 20 分时强制打回重写。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger

INNOVATION_THRESHOLD = 20


@dataclass
class AwardJudgeVerdict:
    """AwardJudgeAgent 的评审结果。"""
    decision: str = "pass"  # pass / rewrite / innovation_reject
    total_score: int = 0
    dimension_scores: dict[str, int] = field(default_factory=dict)
    innovation_score: int = 0
    improvement_suggestions: list[dict] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    is_award_worthy: bool = False

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "total_score": self.total_score,
            "dimension_scores": self.dimension_scores,
            "innovation_score": self.innovation_score,
            "is_award_worthy": self.is_award_worthy,
            "improvement_count": len(self.improvement_suggestions),
            "strengths_count": len(self.strengths),
        }


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r"\[thinking\].*?\[/thinking\]", "", text, flags=re.DOTALL)


def _extract_json(text: str) -> dict | None:
    cleaned = _strip_thinking_tags(text)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    brace_start = cleaned.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _build_award_judge_prompt(paper_content: str, global_state_summary: str, competition_type: str) -> str:
    return f"""你是一位前全国大学生数学建模竞赛国赛评审专家组组长。
你有 15 年评审经验，曾参与多届国赛评奖。
你只看"有没有国奖潜力"，不重复检查正确性（已由其他审稿人完成）。

## 竞赛类型
{competition_type}

## 全局状态摘要
{global_state_summary[:2000]}

## 论文内容
{paper_content[:8000]}

请从以下四个维度进行国奖潜力评估。

## 评分维度

### 1. 方法论 (30分)
- 是否列出至少 2-3 种不同方法族的候选方法并进行对比？（只列 1 种 → -8 分）
- 假设是否有检验数据支撑？（裸用 → -10 分）
- 验证方法是否充分？有交叉验证和基线对比吗？（缺失 → -5 分）

### 2. 创新性 (30分) ⚠️ 低于 20 分将直接打回重写
- 方法是否只是教科书方法的直接套用？（完全套用 → -15 分）
- 方法族选择是否多元？是否跨学科？（单一 → -8 分）
- 是否有视角创新？
- 方法的 novelty 是否在论文中明确阐述？（未阐述 → -3 分）

### 3. 解释性 (20分)
- 图表论证是否有三段式（观察→含义→处置）？（缺失 → -5 分/图表）
- 方法有效性是否有机制解释？（缺失 → -5 分）
- 结论是否有具体数据支撑？（"取得了较好效果" → -3 分）

### 4. 竞赛风格 (20分)
- 摘要是否每个问题单独成段、有具体数值？（不足 500 字 → -5 分）
- 模型章节是否达到 3000-4500 字？（不足 → -5 分）
- 空洞套话是否过多？（→ -3 分）

## 输出格式

```json
{{
  "dimension_scores": {{
    "method": 0,
    "innovation": 0,
    "interpretability": 0,
    "competition_style": 0
  }},
  "dimension_details": {{
    "method": {{
      "strengths": ["亮点1"],
      "deductions": [
        {{"item": "扣分项", "points_lost": 0, "reason": "原因", "fix": "改进建议"}}
      ]
    }},
    "innovation": {{ ... }},
    "interpretability": {{ ... }},
    "competition_style": {{ ... }}
  }},
  "overall_assessment": {{
    "award_potential": "national_first | national_second | provincial | insufficient",
    "key_gap": "距离国奖最大的差距是什么",
    "priority_fixes": ["按优先级排列的改进项"]
  }}
}}
```"""


class AwardJudgeAgent(Agent):
    """系统的"国奖评审官"。评估论文的竞赛获奖潜力。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        max_json_retries: int = 3,
        innovation_threshold: int = INNOVATION_THRESHOLD,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        super().__init__(
            task_id, model, context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )
        self.max_json_retries = max_json_retries
        self.innovation_threshold = innovation_threshold

    async def evaluate(
        self,
        paper_content: str,
        global_state_summary: str = "",
        competition_type: str = "国赛",
    ) -> AwardJudgeVerdict:
        """评估论文的国奖潜力。"""
        logger.info("AwardJudgeAgent: 开始国奖潜力评估")

        system_prompt = (
            "你是一位前全国大学生数学建模竞赛国赛评审专家组组长。"
            "你只评估'有没有国奖潜力'，不重复检查正确性。"
            "请严格按 JSON 格式输出评审结果。"
        )
        user_prompt = _build_award_judge_prompt(paper_content, global_state_summary, competition_type)

        parsed = None
        for attempt in range(self.max_json_retries):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if attempt > 0:
                messages.append({
                    "role": "user",
                    "content": "JSON 格式有误，请严格按要求的 JSON 格式重新输出。",
                })

            response = await self._chat(
                history=messages,
                agent_name=self.__class__.__name__,
                sub_title="国奖评估",
            )

            raw_response = response.content or ""

            if self.diagnostic_logger:
                await self.diagnostic_logger.log_interaction(
                    agent_name=self.__class__.__name__,
                    sub_title="award_judge",
                    messages=messages,
                    response_content=raw_response,
                    response_reasoning=response.reasoning_content,
                )

            parsed = _extract_json(raw_response)
            if parsed:
                break

        if not parsed:
            return AwardJudgeVerdict(
                decision="rewrite", total_score=40,
                dimension_scores={"method": 12, "innovation": 10, "interpretability": 8, "competition_style": 10},
                innovation_score=10,
            )

        dim_scores = parsed.get("dimension_scores", {})
        method_score = min(dim_scores.get("method", 15), 30)
        innovation_score = min(dim_scores.get("innovation", 10), 30)
        interp_score = min(dim_scores.get("interpretability", 8), 20)
        comp_score = min(dim_scores.get("competition_style", 8), 20)
        total_score = method_score + innovation_score + interp_score + comp_score

        improvement_suggestions = []
        all_strengths = []
        dim_details = parsed.get("dimension_details", {})
        for dim_name, detail in dim_details.items():
            for strength in detail.get("strengths", []):
                all_strengths.append(f"[{dim_name}] {strength}")
            for deduction in detail.get("deductions", []):
                improvement_suggestions.append({
                    "dimension": dim_name,
                    "item": deduction.get("item", ""),
                    "points_lost": deduction.get("points_lost", 0),
                    "reason": deduction.get("reason", ""),
                    "fix": deduction.get("fix", ""),
                })

        is_award_worthy = total_score >= 75

        if innovation_score < self.innovation_threshold:
            decision = "innovation_reject"
            logger.warning(f"AwardJudgeAgent: 创新性 {innovation_score}/30 < {self.innovation_threshold}，强制打回")
        elif total_score < 60:
            decision = "rewrite"
        else:
            decision = "pass"

        return AwardJudgeVerdict(
            decision=decision,
            total_score=total_score,
            dimension_scores={
                "method": method_score,
                "innovation": innovation_score,
                "interpretability": interp_score,
                "competition_style": comp_score,
            },
            innovation_score=innovation_score,
            improvement_suggestions=improvement_suggestions,
            strengths=all_strengths,
            is_award_worthy=is_award_worthy,
        )
