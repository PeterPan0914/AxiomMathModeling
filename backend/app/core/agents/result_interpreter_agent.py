"""结果解读 Agent 模块，负责解读代码执行结果并提取关键发现。"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts.result_interpreter import RESULT_INTERPRETER_PROMPT

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger

from app.utils.log_util import logger


@dataclass
class SanityCheck:
    """合理性检查结果。"""
    is_reasonable: bool = True
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class KeyFindings:
    """关键发现。"""
    conclusion: str = ""
    key_numbers: dict[str, str] = field(default_factory=dict)
    methodological_notes: str = ""


@dataclass
class Writeability:
    """结论可写性判断。"""
    can_claim: list[str] = field(default_factory=list)
    cannot_claim: list[str] = field(default_factory=list)
    suggested_framing: str = ""


@dataclass
class InterpreterResult:
    """结果解读的完整输出。"""
    sanity_check: SanityCheck = field(default_factory=SanityCheck)
    key_findings: KeyFindings = field(default_factory=KeyFindings)
    writeability: Writeability = field(default_factory=Writeability)
    raw_response: str = ""


class ResultInterpreterAgent(Agent):
    """结果解读 Agent，解读代码执行结果并提取关键发现。

    职责：
    1. 量级合理性检查（预测值是否在合理范围内？）
    2. 模型诊断（R²>0.99 检查过拟合，RMSE/MAE 比值检查异常值）
    3. 关键发现提取（最重要的数值和结论）
    4. 结论可写性判断（能支撑什么论断？不能支撑什么？）
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
        self.system_prompt = RESULT_INTERPRETER_PROMPT

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        code_output: str,
        subtask_title: str,
        model_spec: str = "",
    ) -> InterpreterResult:
        """解读代码执行结果。

        Args:
            code_output: 代码执行的文本输出。
            subtask_title: 子任务标题（如 ques1, eda）。
            model_spec: 建模方案的描述（可选，帮助理解结果）。

        Returns:
            InterpreterResult 对象。
        """
        logger.info(f"ResultInterpreter: 开始解读 {subtask_title} 的结果")

        # 注入 system prompt（仅首次）
        if not self.chat_history:
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )

        # 构造解读提示
        prompt = f"请解读以下代码执行结果：\n\n【子任务】{subtask_title}\n"
        if model_spec:
            prompt += f"\n【建模方案】\n{model_spec}\n"
        prompt += f"\n【代码执行输出】\n{code_output}"

        # 截断过长的输出（避免 token 爆炸）
        if len(prompt) > 15000:
            prompt = prompt[:15000] + "\n\n... (输出过长，已截断)"

        await self.append_chat_history({"role": "user", "content": prompt})

        response = await self._chat(
            history=self.chat_history,
            agent_name=self.__class__.__name__,
            sub_title=subtask_title,
        )

        response_content = response.content or ""
        assistant_msg: dict = {"role": "assistant", "content": response_content}
        if response.reasoning_content:
            assistant_msg["reasoning_content"] = response.reasoning_content
        await self.append_chat_history(assistant_msg)

        # 记录诊断日志
        if self.diagnostic_logger:
            self.diagnostic_logger.log_interaction(
                agent_name=self.__class__.__name__,
                sub_title=f"结果解读-{subtask_title}",
                messages=self.chat_history[:-1],
                response_content=response_content,
                response_reasoning=response.reasoning_content,
            )

        # 解析结果
        result = self._parse_response(response_content)

        # 输出诊断信息
        if not result.sanity_check.is_reasonable:
            logger.warning(
                f"ResultInterpreter: {subtask_title} 结果可能不合理: "
                f"{result.sanity_check.issues}"
            )
        if result.sanity_check.warnings:
            for w in result.sanity_check.warnings:
                logger.warning(f"ResultInterpreter: {subtask_title} 警告: {w}")

        logger.info(
            f"ResultInterpreter: {subtask_title} 解读完成, "
            f"结论: {result.key_findings.conclusion[:80]}..."
        )

        return result

    def _parse_response(self, response: str) -> InterpreterResult:
        """解析 LLM 返回的 JSON 结果。"""
        # 尝试提取 JSON
        json_str = response.strip()
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        json_str = re.sub(r'\[thinking\].*?\[/thinking\]', '', json_str, flags=re.DOTALL)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试用正则提取
            logger.warning("ResultInterpreter: JSON 解析失败，使用默认值")
            return InterpreterResult(raw_response=response)

        # 构造结果
        sc_data = data.get("sanity_check", {})
        sanity = SanityCheck(
            is_reasonable=sc_data.get("is_reasonable", True),
            issues=sc_data.get("issues", []),
            warnings=sc_data.get("warnings", []),
        )

        kf_data = data.get("key_findings", {})
        findings = KeyFindings(
            conclusion=kf_data.get("conclusion", ""),
            key_numbers=kf_data.get("key_numbers", {}),
            methodological_notes=kf_data.get("methodological_notes", ""),
        )

        wb_data = data.get("writeability", {})
        writeability = Writeability(
            can_claim=wb_data.get("can_claim", []),
            cannot_claim=wb_data.get("cannot_claim", []),
            suggested_framing=wb_data.get("suggested_framing", ""),
        )

        return InterpreterResult(
            sanity_check=sanity,
            key_findings=findings,
            writeability=writeability,
            raw_response=response,
        )
