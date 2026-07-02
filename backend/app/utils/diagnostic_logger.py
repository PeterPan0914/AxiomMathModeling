"""诊断日志模块，记录 Agent 的完整交互过程用于质量分析。"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any
from app.utils.log_util import logger


class DiagnosticLogger:
    """诊断日志记录器，将 Agent 的完整 prompt、response、工具调用结果写入磁盘。

    所有数据写入 task 工作目录下的 diagnostic/ 子目录：
    - interactions.jsonl: 每次 LLM 调用的完整记录（prompt + response）
    - tool_results.jsonl: 工具执行结果（代码输出、搜索结果等）
    - quality.json: Reflexion 质量跟踪数据
    """

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir
        self.diag_dir = os.path.join(work_dir, "diagnostic")
        os.makedirs(self.diag_dir, exist_ok=True)
        self._interactions_path = os.path.join(self.diag_dir, "interactions.jsonl")
        self._tool_results_path = os.path.join(self.diag_dir, "tool_results.jsonl")

    async def _append_jsonl(self, filepath: str, record: dict) -> None:
        """追加一条 JSON 记录到文件（异步，不阻塞事件循环）。"""
        try:
            await asyncio.to_thread(self._write_jsonl_sync, filepath, record)
        except Exception as e:
            logger.error(f"诊断日志写入失败 ({filepath}): {e}")

    @staticmethod
    def _write_jsonl_sync(filepath: str, record: dict) -> None:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json_file(filepath: str, content: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    async def log_interaction(
        self,
        agent_name: str,
        sub_title: str,
        messages: list[dict],
        response_content: str | None,
        response_reasoning: str | None = None,
        tool_calls: list[dict] | None = None,
        usage: dict | None = None,
    ) -> None:
        """记录一次完整的 LLM 交互（prompt + response）。

        Args:
            agent_name: Agent 类名。
            sub_title: 子任务标题。
            messages: 发送给 LLM 的完整消息列表。
            response_content: LLM 返回的文本内容。
            response_reasoning: LLM 的推理内容（如有）。
            tool_calls: 工具调用列表（如有）。
            usage: token 用量（如有）。
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "sub_title": sub_title,
            "prompt_messages": messages,
            "response": {
                "content": response_content,
                "reasoning_content": response_reasoning,
                "tool_calls": tool_calls,
                "usage": usage,
            },
        }
        await self._append_jsonl(self._interactions_path, record)

    async def log_tool_result(
        self,
        agent_name: str,
        tool_name: str,
        sub_title: str,
        tool_input: dict,
        tool_output: str,
        is_error: bool = False,
    ) -> None:
        """记录工具执行结果。

        Args:
            agent_name: Agent 类名。
            tool_name: 工具名称（execute_code / search_papers）。
            sub_title: 子任务标题。
            tool_input: 工具输入参数。
            tool_output: 工具输出结果。
            is_error: 是否为错误结果。
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "tool_name": tool_name,
            "sub_title": sub_title,
            "input": tool_input,
            "output": tool_output,
            "is_error": is_error,
        }
        await self._append_jsonl(self._tool_results_path, record)

    async def save_quality_data(self, quality_data: dict) -> None:
        """保存质量跟踪数据到 JSON 文件。

        Args:
            quality_data: QualityTracker.get_summary() 的输出。
        """
        path = os.path.join(self.diag_dir, "quality.json")
        try:
            await asyncio.to_thread(
                self._write_json_file, path,
                json.dumps(quality_data, ensure_ascii=False, indent=2),
            )
        except Exception as e:
            logger.error(f"quality data write failed: {e}")

    async def save_workflow_config(self, config: dict) -> None:
        """保存工作流配置信息（模型名、Reflexion 配置等）。

        Args:
            config: 配置字典。
        """
        path = os.path.join(self.diag_dir, "config.json")
        try:
            await asyncio.to_thread(
                self._write_json_file, path,
                json.dumps(config, ensure_ascii=False, indent=2),
            )
        except Exception as e:
            logger.error(f"workflow config write failed: {e}")

    async def save_structure_report(self, report: dict) -> None:
        """保存结构控制报告到 JSON 文件。

        Args:
            report: StructureController.check_full_paper() 的序列化输出。
        """
        path = os.path.join(self.diag_dir, "structure_report.json")
        try:
            await asyncio.to_thread(
                self._write_json_file, path,
                json.dumps(report, ensure_ascii=False, indent=2),
            )
        except Exception as e:
            logger.error(f"structure report write failed: {e}")
