"""代码手 Agent 模块，负责生成和执行 Python 代码完成建模任务。"""

import asyncio
from typing import TYPE_CHECKING
from app.core.agents.agent import Agent
from app.config.setting import settings, ApiType
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger
from app.services.redis_manager import redis_manager
from app.schemas.response import SystemMessage, InterpreterMessage
from app.tools.base_interpreter import BaseCodeInterpreter
from app.core.llm.llm import LLM
from app.schemas.A2A import CoderToWriter
from app.core.prompts import CODER_PROMPT
from app.utils.common_utils import get_current_files
import json
from app.core.prompts import get_reflection_prompt
from app.core.functions import coder_tools, coder_tools_anthropic

# TODO: 时间等待过久，stop 进程
# TODO: 支持 cuda
# TODO: 引入创新方案：


class CoderAgent(Agent):
    """代码手 Agent，通过 LLM 生成代码并在解释器中执行，支持错误反思和重试。"""
    def __init__(
        self,
        task_id: str,
        model: LLM,
        work_dir: str,  # 工作目录
        max_chat_turns: int | None = settings.MAX_CHAT_TURNS,  # 最大聊天次数，None表示无限制
        max_retries: int | None = settings.MAX_RETRIES,  # 最大反思次数，None表示无限制
        code_interpreter: BaseCodeInterpreter | None = None,
        context_window: int = 128000,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: 'DiagnosticLogger | None' = None,
    ) -> None:
        super().__init__(task_id, model, context_window, cancel_event=cancel_event, diagnostic_logger=diagnostic_logger)
        self.work_dir = work_dir
        self.max_chat_turns = max_chat_turns
        self.current_chat_turns = 0
        self.max_retries = max_retries
        self.is_first_run = True
        self.system_prompt = CODER_PROMPT
        self.code_interpreter = code_interpreter

    async def run(self, prompt: str, subtask_title: str) -> CoderToWriter:  # type: ignore[reportIncompatibleMethodOverride]
        """执行代码手子任务，生成并运行代码。

        Args:
            prompt: 子任务描述。
            subtask_title: 子任务标题，用于分段输出。

        Returns:
            CoderToWriter 对象，包含代码执行结果和生成的图片列表。
        """
        logger.info(f"{self.__class__.__name__}:开始:执行子任务: {subtask_title}")
        assert self.code_interpreter is not None, "code_interpreter 未初始化"
        self.code_interpreter.add_section(subtask_title)

        # 根据 api_type 选择 tools 格式
        api_type = self.model.api_type
        tools = coder_tools_anthropic if api_type == ApiType.ANTHROPIC else coder_tools

        # 如果是第一次运行，则添加系统提示
        if self.is_first_run:
            logger.info("首次运行，添加系统提示和数据集文件信息")
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )
            # 当前数据集文件
            await self.append_chat_history(
                {
                    "role": "user",
                    "content": f"当前文件夹下的数据集文件{get_current_files(self.work_dir, 'data')}",
                }
            )

        # 添加 sub_task
        logger.info(f"添加子任务提示: {prompt}")
        await self.append_chat_history({"role": "user", "content": prompt})

        retry_count = 0
        last_error_message = ""

        while True:
            if self.max_retries is not None and retry_count >= self.max_retries:
                logger.error(f"超过最大尝试次数: {self.max_retries}")
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content="超过最大尝试次数", type="error"),
                )
                logger.warning(f"任务失败，超过最大尝试次数{self.max_retries}, 最后错误信息: {last_error_message}")
                return CoderToWriter(
                    code_response=f"任务失败，超过最大尝试次数{self.max_retries}, 最后错误信息: {last_error_message}",
                    created_images=[])


            # 仅当显式设置了 max_chat_turns 时才限制，否则不限制轮次
            if self.max_chat_turns is not None and self.current_chat_turns >= self.max_chat_turns:
                logger.error(f"超过最大聊天次数: {self.max_chat_turns}")
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content="超过最大聊天次数", type="error"),
                )
                return CoderToWriter(
                    code_response=f"任务因超过最大聊天轮次({self.max_chat_turns})而结束，已完成部分结果",
                    created_images=await self.code_interpreter.get_created_images(subtask_title) if self.code_interpreter else [],
                )

            self.current_chat_turns += 1
            logger.info(f"当前对话轮次: {self.current_chat_turns}")
            
            try:
                response = await self._chat(
                    history=self.chat_history,
                    tools=tools,
                    tool_choice="auto",
                    agent_name=self.__class__.__name__,
                )
                # 如果有工具调用（可能有多个）
                if response.tool_calls:
                    logger.info(f"检测到 {len(response.tool_calls)} 个工具调用")

                    # 校验所有 tool_call：仅支持 execute_code
                    unknown_tools = [tc for tc in response.tool_calls if tc.name != "execute_code"]
                    if unknown_tools:
                        # 至少存在一个未识别的工具名：让 LLM 重新生成（计入 retry 防止死循环）
                        unknown_names = ", ".join(tc.name for tc in unknown_tools)
                        logger.warning(f"未预期的工具名称: {unknown_names}")
                        await self.append_chat_history(
                            {"role": "assistant", "content": response.content or ""}
                        )
                        await self.append_chat_history(
                            {"role": "user", "content": f"你调用了不存在的工具 ({unknown_names})，请只使用 execute_code 工具。"}
                        )
                        retry_count += 1
                        last_error_message = f"未预期的工具名称: {unknown_names}"
                        continue

                    # 一次性把 assistant 消息（含全部 tool_calls）追加到 history
                    assistant_msg: dict = {"role": "assistant", "content": response.content}
                    if response.reasoning_content:
                        assistant_msg["reasoning_content"] = response.reasoning_content
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in response.tool_calls
                    ]
                    await self.append_chat_history(assistant_msg)

                    # 依次执行每个 tool_call，串行调用内核
                    for tool_call in response.tool_calls:
                        tool_id = tool_call.id
                        code = json.loads(tool_call.arguments)["code"]
                        await redis_manager.publish_message(
                            self.task_id,
                            SystemMessage(content=f"代码手调用{tool_call.name}工具"),
                        )
                        await redis_manager.publish_message(
                            self.task_id,
                            InterpreterMessage(input={"code": code}),
                        )
                        (text_to_gpt, error_occurred, error_message) = await self.code_interpreter.execute_code(code)

                        if self.diagnostic_logger:
                            await self.diagnostic_logger.log_tool_result(
                                agent_name=self.__class__.__name__,
                                tool_name="execute_code",
                                sub_title=subtask_title,
                                tool_input={"code": code},
                                tool_output=error_message if error_occurred else text_to_gpt,
                                is_error=error_occurred,
                            )

                        if error_occurred:
                            logger.warning(f"代码执行错误: {error_message}")
                            # 工具失败：追加 tool 响应，注入反思 prompt，计入 retry
                            await self.append_chat_history(
                                {"role": "tool", "tool_call_id": tool_id, "name": "execute_code", "content": error_message}
                            )
                            await redis_manager.publish_message(
                                self.task_id,
                                SystemMessage(content="代码手反思纠正错误", type="error"),
                            )
                            last_error_message = error_message
                            reflection_prompt = get_reflection_prompt(error_message, code)
                            await self.append_chat_history({"role": "user", "content": reflection_prompt})
                            retry_count += 1
                            # 任何一次失败都跳出当前 tool_call 列表，进入下一轮让 LLM 反思
                            break
                        else:
                            await self.append_chat_history(
                                {"role": "tool", "tool_call_id": tool_id, "name": "execute_code", "content": text_to_gpt}
                            )
                    # for-loop 结束：
                    #   - 若 break 触发，continue 进入下一轮反思
                    #   - 若全部成功，continue 进入下一轮让 LLM 决定后续
                    continue
                else:
                    # 没有工具调用，但可能 LLM 把代码作为文本返回了（部分模型不支持 tool_use）
                    # 尝试从 content 中提取代码块并执行
                    content = response.content or ""
                    import re
                    code_blocks = re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL)

                    if code_blocks:
                        # 提取到代码块，逐个执行
                        logger.info(f"LLM 未调用工具，但从文本中提取到 {len(code_blocks)} 个代码块，自动执行")
                        for i, code in enumerate(code_blocks):
                            code = code.strip()
                            if not code:
                                continue
                            logger.info(f"执行提取的代码块 {i+1}/{len(code_blocks)}")
                            text_to_gpt, error_occurred, error_message = await self.code_interpreter.execute_code(code)

                            if self.diagnostic_logger:
                                await self.diagnostic_logger.log_tool_result(
                                    agent_name=self.__class__.__name__,
                                    tool_name="execute_code (extracted)",
                                    sub_title=subtask_title,
                                    tool_input={"code": code},
                                    tool_output=error_message if error_occurred else text_to_gpt,
                                    is_error=error_occurred,
                                )

                            if error_occurred:
                                logger.warning(f"提取的代码块执行失败: {error_message}")
                                # 执行失败时，将错误反馈给 LLM 重试
                                await self.append_chat_history({"role": "assistant", "content": content})
                                reflection_prompt = get_reflection_prompt(error_message, code)
                                await self.append_chat_history({"role": "user", "content": reflection_prompt})
                                retry_count += 1
                                break  # 跳出代码块循环，进入下一轮对话
                        else:
                            # 所有代码块执行成功，继续对话让 LLM 决定是否还需要执行更多代码
                            await self.append_chat_history({"role": "assistant", "content": content})
                            await self.append_chat_history({
                                "role": "user",
                                "content": "代码已执行完成。如果还需要继续分析或执行更多代码，请继续；如果任务已完成，请说明结果。"
                            })
                            # 计一次 retry，防止 LLM 一直说“代码已完成”陷入无限循环
                            retry_count += 1
                        continue
                    else:
                        # 没有代码块，认为任务完成
                        logger.info("没有工具调用也没有代码块，任务完成")
                        return CoderToWriter(
                            code_response=response.content,
                            created_images=await self.code_interpreter.get_created_images(
                                subtask_title
                            ),
                        )
                    
            except Exception as e:
                logger.error(f"执行过程中发生异常: {str(e)}")
                retry_count += 1
                last_error_message = str(e)
                continue
            logger.info(f"{self.__class__.__name__}:完成:执行子任务: {subtask_title}")
