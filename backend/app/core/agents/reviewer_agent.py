"""ReviewerAgent 模块，系统的"正确性守卫"。

替代原 CriticAgent 的正确性检查角色，输出升级为结构化评分。
不评估创新性和竞赛潜力（那是 AwardJudgeAgent 的职责）。
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


_RETURN_AGENT_MAP = {
    "method_choice": "ModelerAgent",
    "result_interpretation": "CoderAgent",
    "chapter_logic": "WriterAgent",
}


@dataclass
class ReviewVerdict:
    """ReviewerAgent 的审查结果。"""
    decision: str = "approve"  # approve / revise / reject
    correctness_score: int = 0
    dimension_scores: dict[str, int] = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    fatal_issues: list[dict] = field(default_factory=list)
    return_to_agent: str | None = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "correctness_score": self.correctness_score,
            "dimension_scores": self.dimension_scores,
            "total_issues": len(self.issues),
            "fatal_count": len(self.fatal_issues),
            "return_to_agent": self.return_to_agent,
            "suggestions": self.suggestions,
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


def _build_chapter_review_prompt(chapter_num: int, chapter_content: str, global_state_summary: str) -> str:
    return f"""你是一位学术写作逻辑审查专家。你的职责是验证章节论证的正确性和自洽性。

## 当前审查对象
论文第 {chapter_num} 章内容如下：

{chapter_content[:6000]}

## 全局状态参考
{global_state_summary[:2000]}

请从以下三个维度逐一审查。

## 审查维度

### 1. 数学正确性 (40分)
- 公式和推导是否正确？符号是否与全局符号表一致？
- 引用的数值是否与之前章节一致？

### 2. 逻辑自洽性 (35分)
- 论证是否有逻辑跳跃？是否存在循环论证？
- "显然""容易得到"后面是否有实质支撑？
- 结论是否由分析结果自然得出？

### 3. 数据一致性 (25分)
- 前后章节的数字是否一致？
- 图表引用是否正确？
- 摘要中的数字是否与正文一致？

## 输出格式

```json
{{
  "dimension_scores": {{
    "math_correctness": 0,
    "logic_consistency": 0,
    "data_consistency": 0
  }},
  "issues": [
    {{
      "dimension": "math_correctness | logic_consistency | data_consistency",
      "issue": "具体问题描述",
      "location": "问题所在段落",
      "severity": "fatal | important | minor",
      "fix": "具体修改方案"
    }}
  ],
  "overall_assessment": {{
    "correctness_level": "strong | adequate | weak | critically_flawed",
    "critical_fixes": ["必须修复的致命问题"]
  }}
}}
```"""


class ReviewerAgent(Agent):
    """系统的"正确性守卫"。只检查"写得对不对"。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        max_json_retries: int = 3,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        super().__init__(
            task_id, model, context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )
        self.max_json_retries = max_json_retries

    async def review(
        self,
        target_output: str,
        review_type: str,
        global_state_summary: str = "",
        chapter_num: int = 0,
    ) -> ReviewVerdict:
        """执行正确性审查。"""
        logger.info(f"ReviewerAgent: 开始审查 (类型: {review_type})")

        system_prompt = (
            "你是一位严格的数学建模正确性审查专家。"
            "你的职责是验证内容的数学正确性、逻辑自洽性和数据一致性。"
            "请严格按 JSON 格式输出审查结果。"
        )
        user_prompt = _build_chapter_review_prompt(chapter_num, target_output, global_state_summary)

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
                sub_title=f"审查_{review_type}",
            )

            raw_response = response.content or ""

            if self.diagnostic_logger:
                await self.diagnostic_logger.log_interaction(
                    agent_name=self.__class__.__name__,
                    sub_title=f"review_{review_type}",
                    messages=messages,
                    response_content=raw_response,
                    response_reasoning=response.reasoning_content,
                )

            parsed = _extract_json(raw_response)
            if parsed:
                break

        if not parsed:
            return ReviewVerdict(
                decision="revise", correctness_score=50,
                issues=[{"issue": "审查结果解析失败", "severity": "important"}],
            )

        issues = parsed.get("issues", [])
        dim_scores = parsed.get("dimension_scores", {})
        math_score = dim_scores.get("math_correctness", 20)
        logic_score = dim_scores.get("logic_consistency", 17)
        data_score = dim_scores.get("data_consistency", 12)
        correctness_score = math_score + logic_score + data_score

        fatal_issues = [i for i in issues if i.get("severity") == "fatal"]
        important_issues = [i for i in issues if i.get("severity") == "important"]
        suggestions = [i.get("fix", "") for i in issues if i.get("fix")]

        if fatal_issues:
            decision = "reject"
            return_to = _RETURN_AGENT_MAP.get(review_type)
        elif important_issues:
            decision = "revise"
            return_to = None
        else:
            decision = "approve"
            return_to = None

        return ReviewVerdict(
            decision=decision,
            correctness_score=correctness_score,
            dimension_scores={
                "math_correctness": math_score,
                "logic_consistency": logic_score,
                "data_consistency": data_score,
            },
            issues=[{
                "issue": i.get("issue", ""),
                "dimension": i.get("dimension", ""),
                "severity": i.get("severity", "minor"),
                "location": str(i.get("location", "")),
                "fix": i.get("fix", ""),
            } for i in issues],
            suggestions=suggestions,
            fatal_issues=fatal_issues,
            return_to_agent=return_to,
        )
