"""题目深度分析 Agent 模块，负责识别题目陷阱、预判评分重点、分析子任务依赖。"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts.problem_analyst import PROBLEM_ANALYST_PROMPT

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger

from app.utils.log_util import logger


@dataclass
class ProblemAnalysis:
    """题目深度分析结果。"""
    pitfalls: list[str] = field(default_factory=list)
    scoring_focus: list[str] = field(default_factory=list)
    method_families: dict[str, list[str]] = field(default_factory=dict)
    forbidden_methods: dict[str, list[str]] = field(default_factory=dict)
    subtask_dependencies: dict[str, list[str]] = field(default_factory=dict)
    data_characteristics: str = ""
    competition_type_analysis: str = ""

    def to_dict(self) -> dict:
        return {
            "pitfalls": self.pitfalls,
            "scoring_focus": self.scoring_focus,
            "method_families": self.method_families,
            "forbidden_methods": self.forbidden_methods,
            "subtask_dependencies": self.subtask_dependencies,
            "data_characteristics": self.data_characteristics,
            "competition_type_analysis": self.competition_type_analysis,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProblemAnalysis:
        return cls(
            pitfalls=d.get("pitfalls", []),
            scoring_focus=d.get("scoring_focus", []),
            method_families=d.get("method_families", {}),
            forbidden_methods=d.get("forbidden_methods", {}),
            subtask_dependencies=d.get("subtask_dependencies", {}),
            data_characteristics=d.get("data_characteristics", ""),
            competition_type_analysis=d.get("competition_type_analysis", ""),
        )


class ProblemAnalystAgent(Agent):
    """题目深度分析 Agent，识别陷阱、预判评分重点、分析子任务依赖。

    在 CoordinatorAgent 拆题后运行，为 ModelerAgent 提供更深层的题目理解。
    """

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        super().__init__(
            task_id, model, context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )
        self.system_prompt = PROBLEM_ANALYST_PROMPT

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        ques_all: str,
        questions: dict,
    ) -> ProblemAnalysis:
        """分析题目，输出结构化的分析结果。

        Args:
            ques_all: 用户输入的完整题目信息。
            questions: CoordinatorAgent 拆解的结构化问题。

        Returns:
            ProblemAnalysis 对象。
        """
        logger.info("ProblemAnalyst: 开始深度分析题目")

        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )

        # 构造分析提示
        prompt = f"""请对以下数学建模竞赛题目进行深度分析。

【原始题目】
{ques_all}

【已拆解的问题】
{json.dumps(questions, ensure_ascii=False, indent=2)}

请按照输出规范输出 JSON 格式的分析结果。"""
        await self.append_chat_history({"role": "user", "content": prompt})

        response = await self._chat(
            history=self.chat_history,
            agent_name=self.__class__.__name__,
            sub_title="题目深度分析",
        )

        response_content = response.content or ""
        assistant_msg: dict = {"role": "assistant", "content": response_content}
        if response.reasoning_content:
            assistant_msg["reasoning_content"] = response.reasoning_content
        await self.append_chat_history(assistant_msg)

        # 记录诊断日志
        if self.diagnostic_logger:
            await self.diagnostic_logger.log_interaction(
                agent_name=self.__class__.__name__,
                sub_title="题目深度分析",
                messages=self.chat_history[:-1],
                response_content=response_content,
                response_reasoning=response.reasoning_content,
            )

        # 解析结果
        analysis = self._parse_response(response_content)

        logger.info(
            f"ProblemAnalyst: 分析完成, "
            f"识别到 {len(analysis.pitfalls)} 个陷阱, "
            f"{len(analysis.scoring_focus)} 个评分重点"
        )

        return analysis

    def _parse_response(self, response: str) -> ProblemAnalysis:
        """解析 LLM 返回的 JSON 结果。"""
        json_str = response.strip()
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        json_str = re.sub(r'\[thinking\].*?\[/thinking\]', '', json_str, flags=re.DOTALL)

        try:
            data = json.loads(json_str)
            return ProblemAnalysis.from_dict(data)
        except json.JSONDecodeError:
            logger.warning("ProblemAnalyst: JSON 解析失败，返回空结果")
            return ProblemAnalysis()
