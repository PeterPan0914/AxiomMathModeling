"""写作手 Agent 模块，负责基于建模结果撰写学术论文。"""

import asyncio
import os
import re
from typing import TYPE_CHECKING
from app.core.agents.agent import Agent
from app.core.llm.llm import LLM

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger
from app.core.prompts import get_writer_prompt
from app.schemas.enums import CompTemplate, FormatOutPut
from app.config.setting import ApiType
from app.tools.openalex_scholar import OpenAlexScholar
from app.utils.log_util import logger
from app.services.redis_manager import redis_manager
from app.schemas.response import SystemMessage, WriterMessage
import json
from app.core.functions import writer_tools, writer_tools_anthropic
from app.schemas.A2A import WriterResponse


# ---------------------------------------------------------------------------
# PaperSanitizer: 论文内容净化器
# ---------------------------------------------------------------------------

class PaperSanitizer:
    """论文内容净化器，移除所有非论文元素。

    处理：
    1. HTML 注释（<!-- -->）和新的 TRACKING 标记
    2. [thinking] 标签
    3. LLM 元叙述前缀（"以下是"、"作为AI"等）
    4. 代码块包裹（```markdown ... ```）
    5. 多余空行
    """

    _THINKING_PATTERN = re.compile(
        r'\[thinking\].*?\[/thinking\]', re.DOTALL
    )
    _CODE_FENCE_PATTERN = re.compile(
        r'^```(?:markdown)?\s*\n(.*?)\n```\s*$', re.DOTALL
    )
    _HTML_COMMENT_PATTERN = re.compile(r'<!--.*?-->', re.DOTALL)
    _TRACKING_PATTERN = re.compile(
        r'~~~TRACKING_START.*?~~~TRACKING_END\s*', re.DOTALL
    )
    _META_PREFIX_PATTERNS = [
        re.compile(r'^(?:Here\s+is|Below\s+is|The\s+following\s+is|I\s+will\s+now)\b.*?\n', re.IGNORECASE),
        re.compile(r'^(?:以下是|下面是|下面是论文|现在开始撰写|这是).*?\n'),
        re.compile(r'^(?:Sure|Certainly|Of\s+course|OK).*?[.。：:]\s*\n', re.IGNORECASE),
    ]

    @classmethod
    def sanitize(cls, content: str) -> str:
        """净化论文内容，移除所有非论文元素。"""
        if not content or not content.strip():
            return ""

        result = content

        # 1. 移除 TRACKING 标记（新格式）
        result = cls._TRACKING_PATTERN.sub('', result)

        # 2. 移除 HTML 注释（旧格式，防御性剥离）
        result = cls._HTML_COMMENT_PATTERN.sub('', result)

        # 3. 移除思考标签
        result = cls._THINKING_PATTERN.sub('', result)

        # 4. 移除代码块包裹
        result = cls._unwrap_code_fence(result)

        # 5. 移除 LLM 元叙述前缀
        result = cls._strip_meta_prefixes(result)

        # 6. 清理多余空行（最多保留两个连续换行）
        result = re.sub(r'\n{4,}', '\n\n\n', result)

        return result.strip() + "\n"

    @classmethod
    def _unwrap_code_fence(cls, content: str) -> str:
        match = cls._CODE_FENCE_PATTERN.match(content.strip())
        if match:
            return match.group(1)
        return content

    @classmethod
    def _strip_meta_prefixes(cls, content: str) -> str:
        result = content
        for pattern in cls._META_PREFIX_PATTERNS:
            result = pattern.sub('', result).lstrip()
        return result


# ---------------------------------------------------------------------------
# FigurePathResolver: 图表路径解析器
# ---------------------------------------------------------------------------

class FigurePathResolver:
    """图表路径解析器，确保论文中的图片引用指向正确的文件位置。"""

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir
        self._path_map: dict[str, str] = {}

    def register_images(self, images: list[str]) -> None:
        """注册可用图片列表。"""
        for img in images:
            short_name = os.path.basename(img)
            self._path_map[short_name] = img
            stem = os.path.splitext(short_name)[0]
            self._path_map[stem] = img

    def resolve_content(self, content: str) -> str:
        """扫描论文内容，修正所有图片引用路径。"""
        def _replace_fig(match: re.Match) -> str:
            alt_text = match.group(1)
            original_path = match.group(2).strip()

            if original_path in self._path_map:
                return f'![{alt_text}]({self._path_map[original_path]})'

            stem = os.path.splitext(original_path)[0]
            if stem in self._path_map:
                return f'![{alt_text}]({self._path_map[stem]})'

            basename = os.path.basename(original_path)
            if basename in self._path_map:
                return f'![{alt_text}]({self._path_map[basename]})'

            logger.warning(f"[FigurePathResolver] 未找到图片 '{original_path}'")
            return match.group(0)

        pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        return pattern.sub(_replace_fig, content)

    def validate_references(self, content: str) -> list[str]:
        """验证论文中引用的所有图片是否都存在。"""
        missing: list[str] = []
        refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', content)
        for ref in refs:
            full_path = os.path.join(self.work_dir, ref)
            if not os.path.exists(full_path):
                missing.append(ref)
        return missing


# ---------------------------------------------------------------------------
# EquationValidator: 公式编号校验器
# ---------------------------------------------------------------------------

class EquationValidator:
    """校验论文中公式编号的连续性和正确性。"""

    @staticmethod
    def validate(text: str) -> list[str]:
        """校验公式编号。

        Returns:
            问题列表（如编号不连续、重复等）。
        """
        tags = re.findall(r'\\tag\{(\d+)\}', text)
        if not tags:
            return []

        numbers = [int(t) for t in tags]
        issues = []

        if numbers and numbers[0] != 1:
            issues.append(f"公式编号应从 1 开始，实际从 {numbers[0]} 开始")

        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i-1] + 1:
                issues.append(
                    f"公式编号不连续: 第 {i} 个 tag 是 {numbers[i]}，"
                    f"前一个是 {numbers[i-1]}（预期 {numbers[i-1]+1}）"
                )

        seen = set()
        for n in numbers:
            if n in seen:
                issues.append(f"公式编号重复: {n}")
            seen.add(n)

        return issues


# TODO: 并行 parallel
# TODO: 获取当前文件下的文件
# TODO: 引用cites tool


class WriterAgent(Agent):
    """写作手 Agent，基于建模和代码执行结果撰写竞赛论文。"""
    def __init__(
        self,
        task_id: str,
        model: LLM,
        comp_template: CompTemplate = CompTemplate.CHINA,
        format_output: FormatOutPut = FormatOutPut.Markdown,
        scholar: OpenAlexScholar | None = None,
        context_window: int = 128000,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: 'DiagnosticLogger | None' = None,
    ) -> None:
        super().__init__(task_id, model, context_window, cancel_event=cancel_event, diagnostic_logger=diagnostic_logger)
        self.format_out_put = format_output
        self.comp_template = comp_template
        self.scholar = scholar
        self.is_first_run = True
        self.system_prompt = get_writer_prompt(format_output)
        self.available_images: list[str] = []

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str,
        available_images: list[str] | None = None,
        sub_title: str | None = None,
    ) -> WriterResponse:
        """执行写作任务。

        Args:
            prompt: 写作提示。
            available_images: 可用的图片相对路径列表。
            sub_title: 子任务标题。

        Returns:
            WriterResponse 对象，内容已经过 PaperSanitizer 净化。
        """
        logger.info(
            f"[WriterAgent] >>> 开始写作: sub_title={sub_title}, "
            f"prompt长度={len(prompt)}, 图片数={len(available_images or [])}"
        )

        # 根据 api_type 选择 tools 格式
        api_type = self.model.api_type
        tools = writer_tools_anthropic if api_type == ApiType.ANTHROPIC else writer_tools

        if self.is_first_run:
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )

        # 注册图片到 FigurePathResolver
        if available_images:
            self.available_images = available_images
            image_lines = "\n".join(
                [f"- ![{os.path.basename(img)}]({img})" for img in available_images]
            )
            image_prompt = (
                f"\n\n【必须插入的图片列表】\n"
                f"以下图片是代码手生成的，你必须在论文相关段落后用 Markdown 格式逐一插入：\n"
                f"{image_lines}\n"
                f"插入格式为独占一行的 ![描述](文件名)，每张图片后需配3行以上的分析解读。\n"
            )
            prompt = prompt + image_prompt

        await self.append_chat_history({"role": "user", "content": prompt})

        # 获取历史消息用于本次对话
        response = await self._chat(
            history=self.chat_history,
            tools=tools,
            tool_choice="auto",
            agent_name=self.__class__.__name__,
            sub_title=sub_title,
        )

        footnotes = []
        response_content: str = ""

        if response.tool_calls:
            logger.info("[WriterAgent] 检测到工具调用")
            tool_call = response.tool_calls[0]
            tool_id = tool_call.id
            if tool_call.name == "search_papers":
                logger.info("[WriterAgent] 调用工具: search_papers")
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"写作手调用{tool_call.name}工具"),
                )

                query = json.loads(tool_call.arguments)["query"]

                await redis_manager.publish_message(
                    self.task_id,
                    WriterMessage(content=query),
                )

                assistant_msg: dict = {"role": "assistant", "content": response.content}
                if response.reasoning_content:
                    assistant_msg["reasoning_content"] = response.reasoning_content
                if response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in response.tool_calls
                    ]
                await self.append_chat_history(assistant_msg)

                try:
                    assert self.scholar is not None, "scholar 未初始化"
                    papers = await self.scholar.search_papers(query)
                except Exception as e:
                    error_msg = f"搜索文献失败: {str(e)}"
                    logger.error(error_msg)
                    return WriterResponse(
                        response_content=error_msg, footnotes=footnotes
                    )
                assert self.scholar is not None, "scholar 未初始化"
                papers_str = self.scholar.papers_to_str(papers)
                logger.info(f"[WriterAgent] 搜索文献结果\n{papers_str}")

                if self.diagnostic_logger:
                    self.diagnostic_logger.log_tool_result(
                        agent_name=self.__class__.__name__,
                        tool_name="search_papers",
                        sub_title=sub_title or "unknown",
                        tool_input={"query": query},
                        tool_output=papers_str,
                    )

                await self.append_chat_history(
                    {
                        "role": "tool",
                        "content": papers_str,
                        "tool_call_id": tool_id,
                        "name": "search_papers",
                    }
                )
                next_response = await self._chat(
                    history=self.chat_history,
                    tools=tools,
                    tool_choice="auto",
                    agent_name=self.__class__.__name__,
                    sub_title=sub_title,
                )
                response_content = next_response.content or ""
                final_reasoning = next_response.reasoning_content
        else:
            response_content = response.content or ""
            final_reasoning = response.reasoning_content

        # ---- PaperSanitizer: 净化论文内容 ----
        response_content = PaperSanitizer.sanitize(response_content)

        # ---- FigurePathResolver: 修正图片路径 ----
        if self.available_images:
            resolver = FigurePathResolver(work_dir="")
            resolver.register_images(self.available_images)
            response_content = resolver.resolve_content(response_content)

        # ---- EquationValidator: 校验公式编号 ----
        eq_issues = EquationValidator.validate(response_content)
        if eq_issues:
            for issue in eq_issues:
                logger.warning(f"[WriterAgent-公式编号] {issue}")

        # ---- 断言：最终内容不得包含 HTML 注释 ----
        html_comments = re.findall(r'<!--.*?-->', response_content, re.DOTALL)
        if html_comments:
            logger.warning(
                f"[WriterAgent] 检测到 {len(html_comments)} 个 HTML 注释残留，强制剥离"
            )
            response_content = re.sub(r'<!--.*?-->', '', response_content, flags=re.DOTALL)

        assistant_msg: dict = {"role": "assistant", "content": response_content}
        if final_reasoning:
            assistant_msg["reasoning_content"] = final_reasoning
        await self.append_chat_history(assistant_msg)

        logger.info(
            f"[WriterAgent] <<< 写作完成: sub_title={sub_title}, "
            f"输出长度={len(response_content)}"
        )
        return WriterResponse(response_content=response_content, footnotes=footnotes)

    def update_system_prompt_for_chapter(
        self,
        chapter_type: str,
        global_state_summary: str = "",
    ) -> None:
        """根据章节类型更新系统提示词。

        Args:
            chapter_type: 章节类型常量（CHAPTER_*）。
            global_state_summary: 全局状态摘要。
        """
        from app.core.prompts.writer import get_writer_system_prompt
        new_prompt = get_writer_system_prompt(
            chapter_type=chapter_type,
            global_state_summary=global_state_summary,
            competition_type=self.comp_template.value if hasattr(self.comp_template, 'value') else str(self.comp_template),
            format_output=self.format_out_put,
        )
        # 更新系统提示（替换 history 中的第一条消息）
        if self.chat_history and self.chat_history[0].get("role") == "system":
            self.chat_history[0]["content"] = new_prompt
        else:
            self.chat_history.insert(0, {"role": "system", "content": new_prompt})
        self.system_prompt = new_prompt

    async def summarize(self) -> str:
        """总结对话内容，生成任务执行摘要。"""
        try:
            await self.append_chat_history(
                {"role": "user", "content": "请简单总结以上完成什么任务取得什么结果:"}
            )
            # 获取历史消息用于本次对话
            response = await self._chat(
                history=self.chat_history, agent_name=self.__class__.__name__
            )
            response_content = response.content or ""
            summary_msg: dict = {"role": "assistant", "content": response_content}
            if response.reasoning_content:
                summary_msg["reasoning_content"] = response.reasoning_content
            await self.append_chat_history(summary_msg)
            return response_content
        except Exception as e:
            logger.error(f"总结生成失败: {str(e)}")
            # 返回一个基础总结，避免完全失败
            return "由于网络原因无法生成详细总结，但已完成主要任务处理。"
