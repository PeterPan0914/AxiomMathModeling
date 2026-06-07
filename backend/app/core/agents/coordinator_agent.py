"""协调者 Agent 模块，负责战略规划与子任务拆解。

该 Agent 读取 ProblemAnalystAgent 和 LiteratureAgent 的输出，
以团队队长的视角制定整体战略，输出带 DAG 依赖关系的子任务列表。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts.coordinator import COORDINATOR_PROMPT, get_coordinator_system_prompt
from app.schemas.A2A import CoordinatorToModeler
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger


class CoordinatorAgent(Agent):
    """协调者 Agent，负责战略规划并将题目拆解为带依赖关系的子任务 DAG。

    可接收 ProblemAnalystAgent 和 LiteratureAgent 的输出来生成更具
    战略性的任务规划；当不提供这些参数时退化为基础拆题模式。
    """

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        max_retries: int = 5,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        super().__init__(
            task_id,
            model,
            context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )
        # 默认使用基础 prompt，run() 中可根据参数动态替换
        self.system_prompt = COORDINATOR_PROMPT
        self.max_retries = max_retries

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        ques_all: str,
        problem_analysis: str = "",
        literature_review: str = "",
    ) -> CoordinatorToModeler:
        """战略规划，输出子任务 DAG 和整体策略。

        当 ``problem_analysis`` 或 ``literature_review`` 非空时，
        系统提示词会注入这些上下文信息，引导 LLM 输出更具战略性的规划。

        Args:
            ques_all: 用户输入的完整题目信息。
            problem_analysis: ProblemAnalystAgent 的题目深度分析文本。
            literature_review: LiteratureAgent 的文献调研结果文本。

        Returns:
            CoordinatorToModeler 对象，包含结构化问题、问题数量
            以及战略规划字段（sub_tasks、priority_order 等）。

        Raises:
            ValueError: 超过最大重试次数仍无法解析。
        """
        # 根据是否有额外上下文动态选择系统提示词
        if problem_analysis or literature_review:
            system_prompt = get_coordinator_system_prompt(
                problem_analysis=problem_analysis,
                literature_review=literature_review,
            )
        else:
            system_prompt = self.system_prompt

        await self.append_chat_history(
            {"role": "system", "content": system_prompt}
        )
        await self.append_chat_history({"role": "user", "content": ques_all})
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = await self._chat(
                    history=self.chat_history,
                    agent_name=self.__class__.__name__,
                )
                json_str = response.content or ""

                # 记录诊断日志
                if self.diagnostic_logger:
                    self.diagnostic_logger.log_interaction(
                        agent_name=self.__class__.__name__,
                        sub_title="问题拆解",
                        messages=self.chat_history,
                        response_content=json_str,
                        response_reasoning=response.reasoning_content,
                    )

                # 清理 JSON 字符串：剥离 thinking 块、markdown 标记和控制字符
                json_str = re.sub(r"\[thinking\].*?\[/thinking\]", "", json_str, flags=re.DOTALL)
                json_str = json_str.replace("```json", "").replace("```", "").strip()
                json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)

                if not json_str:
                    raise ValueError("返回的 JSON 字符串为空")

                questions = json.loads(json_str)
                ques_count = questions["ques_count"]
                logger.info(f"questions:{questions}")
                return CoordinatorToModeler(questions=questions, ques_count=ques_count)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                attempt += 1
                logger.warning(f"解析失败 (尝试 {attempt}/{self.max_retries}): {str(e)}")

                if attempt >= self.max_retries:
                    raise ValueError(
                        f"CoordinatorAgent 在 {self.max_retries} 次尝试后仍无法解析问题。"
                        f"最后的错误: {str(e)}"
                    )

                # 添加错误反馈提示，使用当前轮次的 system_prompt
                error_prompt = f"上次响应格式错误: {str(e)}。请严格输出JSON格式"
                await self.append_chat_history({
                    "role": "system",
                    "content": system_prompt + "\n" + error_prompt,
                })
