"""全文一致性检查 Agent 模块，在所有章节写完后检查论文内部一致性。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts.consistency import get_consistency_check_prompt
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger


class ConsistencyAgent(Agent):
    """全文一致性检查 Agent——在所有章节写完后运行。

    职责：
    1. 检查符号、数字、术语在全文中是否一致
    2. 检查图表引用与实际内容是否匹配
    3. 检查摘要结论与正文结论是否一致
    4. 检查模型假设是否在建模章节中被实际引用
    5. 检查每个结论是否有证据支撑（逻辑链完整性）
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
        all_chapters_text: str,
        global_state_summary: str,
    ) -> str:
        """检查全文一致性，返回 JSON 格式的问题列表。

        对论文全文执行 7 项一致性检查（符号、数字、术语、引用、
        结论、假设使用、逻辑链），输出结构化的问题报告。

        Args:
            all_chapters_text: 论文所有章节的完整文本。
            global_state_summary: 全局状态摘要，包含符号定义表、
                建模决策记录等信息。

        Returns:
            JSON 格式的一致性检查报告字符串，包含 consistency_score、
            issues 列表和 summary。
        """
        logger.info("ConsistencyAgent: 开始全文一致性检查")

        system_prompt = get_consistency_check_prompt(
            all_chapters_text=all_chapters_text,
            global_state_summary=global_state_summary,
        )

        # 一致性检查 Agent 只需要一轮对话
        prompt = "请根据上述信息，对论文全文执行 7 项一致性检查，输出 JSON 格式的检查报告。"

        result = await super().run(
            prompt=prompt,
            system_prompt=system_prompt,
            sub_title="全文一致性检查",
        )

        logger.info("ConsistencyAgent: 一致性检查完成")
        return result
