"""CriticAgent 模块，系统的"免疫系统"——常驻质疑者，不生产内容，只质疑内容。

CriticAgent 在以下时机被触发：
1. ModelerAgent 选定方法后 → 质疑方法选择（method_choice）
2. CoderAgent 输出结果后 → 质疑结果解读（result_interpretation）
3. WriterAgent 写完每一章后 → 质疑论证逻辑（chapter_logic）

每个质疑维度都包含严重程度评级（fatal/important/minor）和具体修复建议。
当存在 fatal 级别问题时，CriticAgent 会阻止流程继续并建议回退到指定 Agent。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts.critic import (
    get_chapter_critique_prompt,
    get_method_critique_prompt,
    get_result_critique_prompt,
)
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger


# 质疑类型 → 触发回退时应返回的 Agent 名称
_RETURN_AGENT_MAP = {
    "method_choice": "ModelerAgent",
    "result_interpretation": "CoderAgent",
    "chapter_logic": "WriterAgent",
}


@dataclass
class CritiqueResult:
    """CriticAgent 的质疑结果。

    Attributes:
        decision: 质疑结论，"approve" 表示通过，"revise" 表示需小修但可继续，
            "reject" 表示需要大改，阻止下一步执行。
        issues: 所有发现的问题列表，每个问题包含内容、严重程度、位置和修复建议。
        suggestions: 改进建议列表（从 issues 中提取的修复建议汇总）。
        fatal_issues: fatal 级别的问题子集。
        return_to_agent: 当 decision 为 "reject" 时，应回退到的 Agent 名称。
    """

    decision: str  # "approve" | "revise" | "reject"
    issues: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    fatal_issues: list[dict] = field(default_factory=list)
    return_to_agent: str | None = None


def _strip_thinking_tags(text: str) -> str:
    """剥离 thinking 模型返回的 [thinking] 标签。

    Args:
        text: 可能包含 thinking 标签的原始文本。

    Returns:
        清理后的文本。
    """
    return re.sub(r"\[thinking\].*?\[/thinking\]", "", text, flags=re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON 对象。

    尝试多种策略：直接解析、剥离代码块标记后解析、正则提取。

    Args:
        text: LLM 输出文本。

    Returns:
        解析后的字典，无法解析时返回 None。
    """
    # 剥离 thinking 标签和代码块标记
    cleaned = _strip_thinking_tags(text)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # 直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个 JSON 对象（匹配最外层花括号）
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
                        return json.loads(cleaned[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break

    # 正则兜底：提取顶层键值对
    try:
        pattern = r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.|"(?!,\s*\n)|"(?!\s*\n\s*}))*)"'
        matches = re.findall(pattern, cleaned, re.DOTALL)
        if matches:
            return {k: v.replace('\\"', '"') for k, v in matches}
    except re.error:
        pass

    return None


def _extract_issues_from_json(parsed: dict, critique_type: str) -> list[dict]:
    """从解析后的 JSON 中提取统一格式的问题列表。

    不同质疑类型的 JSON 结构不同，此函数将其归一化为统一格式：
    [{issue, severity, location, fix}]

    Args:
        parsed: 解析后的 JSON 字典。
        critique_type: 质疑类型，"method_choice" | "result_interpretation" | "chapter_logic"。

    Returns:
        归一化后的问题列表。
    """
    issues: list[dict] = []

    if critique_type == "method_choice":
        for critique in parsed.get("critiques", []):
            issues.append({
                "issue": critique.get("question", ""),
                "dimension": critique.get("dimension", ""),
                "severity": critique.get("severity", "minor"),
                "location": critique.get("current_status", ""),
                "fix": critique.get("fix", ""),
            })

    elif critique_type == "result_interpretation":
        for check in parsed.get("checks", []):
            issues.append({
                "issue": check.get("finding", ""),
                "dimension": check.get("dimension", ""),
                "severity": check.get("severity", "minor"),
                "location": check.get("location", ""),
                "fix": check.get("fix", ""),
            })

    elif critique_type == "chapter_logic":
        for check in parsed.get("logic_checks", []):
            location = check.get("location", {})
            location_str = ""
            if isinstance(location, dict):
                para = location.get("paragraph", "")
                sent = location.get("sentence", "")
                location_str = f"段落{para}: {sent}"
            else:
                location_str = str(location)

            issues.append({
                "issue": check.get("problem", ""),
                "dimension": check.get("check_type", ""),
                "severity": check.get("severity", "minor"),
                "location": location_str,
                "fix": check.get("fix", ""),
            })

    return issues


class CriticAgent(Agent):
    """系统的"免疫系统"——常驻质疑者，不生产内容，只质疑内容。

    在 ModelerAgent 选定方法、CoderAgent 输出结果、WriterAgent 写完章节后
    分别触发质疑，阻止低质量内容流向下一个 Agent。

    Attributes:
        max_json_retries: JSON 解析失败时的最大重试次数。
    """

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        max_json_retries: int = 3,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        """初始化 CriticAgent。

        Args:
            task_id: 任务 ID。
            model: LLM 模型实例。
            context_window: 上下文窗口大小。
            max_json_retries: JSON 解析失败时的最大重试次数。
            cancel_event: 取消信号事件。
            diagnostic_logger: 诊断日志记录器。
        """
        super().__init__(
            task_id,
            model,
            context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )
        self.max_json_retries = max_json_retries

    async def critique(
        self,
        target_output: str,
        critique_type: str,
        global_state_summary: str = "",
        chapter_num: int = 0,
    ) -> CritiqueResult:
        """对目标内容执行质疑并返回结构化结果。

        根据 critique_type 选择对应的质疑 prompt，调用 LLM 获取质疑结果，
        解析 JSON 后按严重程度分级：fatal → reject，important → revise，否则 → approve。

        Args:
            target_output: 被质疑的内容文本。
            critique_type: 质疑类型，必须是 "method_choice"、"result_interpretation"
                或 "chapter_logic" 之一。
            global_state_summary: 全局状态摘要，用于跨章节一致性检查。
                仅 chapter_logic 类型使用，但建议所有类型都传入以保持一致性。
            chapter_num: 章节编号，仅 chapter_logic 类型使用。

        Returns:
            CritiqueResult 对象，包含决策、问题列表、建议和回退目标。

        Raises:
            ValueError: critique_type 不在支持的范围内。
        """
        # 验证质疑类型
        valid_types = ("method_choice", "result_interpretation", "chapter_logic")
        if critique_type not in valid_types:
            raise ValueError(
                f"不支持的质疑类型: {critique_type}，"
                f"必须是 {valid_types} 之一"
            )

        logger.info(f"CriticAgent: 开始质疑 (类型: {critique_type})")

        # 根据类型选择 prompt
        system_prompt = self._get_system_prompt(critique_type)
        user_prompt = self._build_user_prompt(
            target_output, critique_type, global_state_summary, chapter_num
        )

        # 尝试获取并解析 JSON 结果（支持重试）
        parsed = None
        raw_response = ""

        for attempt in range(self.max_json_retries):
            # 每次重试都重新构建消息历史，避免污染
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if attempt > 0:
                messages.append({
                    "role": "user",
                    "content": "你返回的 JSON 格式有误，请严格按要求的 JSON 格式重新输出。"
                    "注意：字符串值内的双引号必须转义为 \\\"，不要包含未转义的特殊字符。",
                })

            response = await self._chat(
                history=messages,
                agent_name=self.__class__.__name__,
                sub_title=f"质疑_{critique_type}",
            )

            raw_response = response.content or ""

            # 记录诊断日志
            if self.diagnostic_logger:
                usage_dict = None
                if response.usage:
                    usage_dict = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                await self.diagnostic_logger.log_interaction(
                    agent_name=self.__class__.__name__,
                    sub_title=f"critique_{critique_type}",
                    messages=messages,
                    response_content=raw_response,
                    response_reasoning=response.reasoning_content,
                    usage=usage_dict,
                )

            parsed = _extract_json(raw_response)
            if parsed:
                break

            logger.warning(
                f"CriticAgent: JSON 解析失败 (第{attempt + 1}/{self.max_json_retries}次)"
            )

        if not parsed:
            # 所有重试都失败，返回一个安全的降级结果
            logger.error(
                f"CriticAgent: {self.max_json_retries} 次尝试后仍无法解析 JSON，"
                f"返回降级结果。原始输出: {raw_response[:200]}..."
            )
            return CritiqueResult(
                decision="revise",
                issues=[{
                    "issue": "CriticAgent 无法解析质疑结果，建议人工检查",
                    "severity": "important",
                    "location": "",
                    "fix": "请检查被质疑内容的质量",
                }],
                suggestions=["CriticAgent 质疑结果解析失败，建议人工审查"],
                fatal_issues=[],
                return_to_agent=None,
            )

        # 提取归一化的问题列表
        issues = _extract_issues_from_json(parsed, critique_type)

        # 分离 fatal 级别问题
        fatal_issues = [i for i in issues if i.get("severity") == "fatal"]
        important_issues = [i for i in issues if i.get("severity") == "important"]

        # 提取所有修复建议
        suggestions = [i.get("fix", "") for i in issues if i.get("fix")]

        # 根据严重程度决定决策
        if fatal_issues:
            decision = "reject"
            return_to = _RETURN_AGENT_MAP.get(critique_type)
            logger.warning(
                f"CriticAgent: 发现 {len(fatal_issues)} 个致命问题，"
                f"建议回退到 {return_to}"
            )
        elif important_issues:
            decision = "revise"
            return_to = None
            logger.info(
                f"CriticAgent: 发现 {len(important_issues)} 个重要问题，"
                f"建议修改但不阻止流程"
            )
        else:
            decision = "approve"
            return_to = None
            logger.info("CriticAgent: 未发现严重问题，通过质疑")

        result = CritiqueResult(
            decision=decision,
            issues=issues,
            suggestions=suggestions,
            fatal_issues=fatal_issues,
            return_to_agent=return_to,
        )

        # 记录结构化质疑结果到诊断日志
        if self.diagnostic_logger:
            await self.diagnostic_logger.log_tool_result(
                agent_name=self.__class__.__name__,
                tool_name=f"critique_{critique_type}",
                sub_title="critique_result",
                tool_input={
                    "critique_type": critique_type,
                    "target_output_length": len(target_output),
                    "chapter_num": chapter_num,
                },
                tool_output=json.dumps(
                    {
                        "decision": result.decision,
                        "total_issues": len(result.issues),
                        "fatal_count": len(result.fatal_issues),
                        "return_to_agent": result.return_to_agent,
                    },
                    ensure_ascii=False,
                ),
            )

        return result

    def _get_system_prompt(self, critique_type: str) -> str:
        """获取质疑角色的系统提示词。

        Args:
            critique_type: 质疑类型。

        Returns:
            系统提示词字符串。
        """
        prompts = {
            "method_choice": (
                "你是一位有 10 年数学建模竞赛评审经验的学术评审委员会成员，"
                "以挑剔著称。你的工作是找出方法选择中的漏洞和风险，"
                "确保论文的方法论证经得起评委的严格审查。"
                "请严格按 JSON 格式输出质疑结果。"
            ),
            "result_interpretation": (
                "你是一位统计学教授，专门审查学生提交的数据分析报告。"
                "你的工作是发现数据解读中的过度声称、遗漏的反面证据、"
                "数字矛盾和图表误读。"
                "请严格按 JSON 格式输出检查结果。"
            ),
            "chapter_logic": (
                "你是一位逻辑学家和学术写作编辑的结合体，专门挑论文的逻辑漏洞。"
                "你的工作是找出逻辑跳跃、循环论证、空洞断言、前后矛盾和悬空引用。"
                "请严格按 JSON 格式输出检查结果。"
            ),
        }
        return prompts.get(critique_type, "")

    def _build_user_prompt(
        self,
        target_output: str,
        critique_type: str,
        global_state_summary: str,
        chapter_num: int,
    ) -> str:
        """根据质疑类型构建用户提示词。

        Args:
            target_output: 被质疑的内容。
            critique_type: 质疑类型。
            global_state_summary: 全局状态摘要。
            chapter_num: 章节编号。

        Returns:
            用户提示词字符串。
        """
        if critique_type == "method_choice":
            return get_method_critique_prompt(target_output)
        elif critique_type == "result_interpretation":
            return get_result_critique_prompt(target_output)
        elif critique_type == "chapter_logic":
            return get_chapter_critique_prompt(
                chapter_num=chapter_num,
                chapter_content=target_output,
                global_state_summary=global_state_summary,
            )
        else:
            # 理论上不会到达（已在 critique() 中校验），防御性兜底
            return f"请质疑以下内容：\n\n{target_output}"
