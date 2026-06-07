"""论文大纲规划 Agent 模块，在写作前规划完整论文结构。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts.outline import get_outline_prompt
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger


class OutlineAgent(Agent):
    """论文大纲规划 Agent——在写作前运行一次，规划完整论文结构。

    职责：
    1. 基于全局状态（题目分析、建模决策、代码结果）设计论文论证弧线
    2. 为每章规划内容要点、字数目标、图表引用、章节衔接
    3. 标注危险区域和评委关注点
    4. 确保承诺-兑现机制贯穿全文
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
            task_id,
            model,
            context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        global_state_summary: str,
        competition_type: str = "国赛",
        page_limit: int = 20,
    ) -> str:
        """生成完整论文大纲。

        基于全局状态摘要，调用 LLM 规划论文的章节结构、内容要点、
        字数分配、图表引用和章节衔接，输出 JSON 格式的大纲。

        Args:
            global_state_summary: 全局状态摘要，包含题目分析、建模决策、
                代码结果等关键信息。
            competition_type: 竞赛类型（"国赛"、"美赛"等）。
            page_limit: 论文页数限制。

        Returns:
            JSON 格式的论文大纲字符串，包含 core_story、chapters、
            red_lines、innovation_emphasis 等字段。
        """
        logger.info(f"OutlineAgent: 开始规划论文大纲，竞赛类型={competition_type}")

        system_prompt = get_outline_prompt(
            global_state_summary=global_state_summary,
            competition_type=competition_type,
            page_limit=page_limit,
        )

        # 大纲 Agent 只需要一轮对话，prompt 内容已包含在 system_prompt 中
        prompt = "请根据上述信息，输出完整的论文大纲 JSON。"

        result = await super().run(
            prompt=prompt,
            system_prompt=system_prompt,
            sub_title="论文大纲规划",
        )

        logger.info("OutlineAgent: 论文大纲规划完成")
        return result
