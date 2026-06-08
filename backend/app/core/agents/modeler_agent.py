"""建模手 Agent 模块，负责分析问题并制定建模方案。"""

import asyncio
from typing import TYPE_CHECKING
from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts import MODELER_PROMPT, get_modeler_system_prompt

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger
from app.schemas.A2A import CoordinatorToModeler, ModelerToCoder, ModelSpec
from app.utils.log_util import logger
import json
import re


def extract_model_specs(questions_solution: dict[str, str]) -> dict[str, ModelSpec]:
    """从建模方案文本中提取结构化的 model_spec。

    Args:
        questions_solution: 建模手输出的各问题方案文本。

    Returns:
        各问题的 ModelSpec 字典。
    """
    specs: dict[str, ModelSpec] = {}
    pattern = r'---MODEL_SPEC_START---\s*\n(.*?)\n\s*---MODEL_SPEC_END---'

    for key, text in questions_solution.items():
        if not key.startswith("ques"):
            continue
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue

        spec_text = match.group(1)
        spec = ModelSpec()

        for line in spec_text.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            # 只 split 第一个冒号（值中可能包含冒号）
            colon_idx = line.index(":")
            field_name = line[:colon_idx].strip().upper()
            field_value = line[colon_idx + 1:].strip()

            if field_name == "OBJECTIVE":
                spec.objective = field_value
            elif field_name == "CONSTRAINTS":
                spec.constraints = [c.strip() for c in field_value.split(";") if c.strip()]
            elif field_name == "ALGORITHM":
                spec.algorithm = field_value
            elif field_name == "KEY_PARAMS":
                # 解析 key=value 对
                for param in field_value.split(";"):
                    param = param.strip()
                    if "=" in param:
                        pk, pv = param.split("=", 1)
                        spec.key_params[pk.strip()] = pv.strip()
            elif field_name == "EXPECTED_OUTPUT":
                spec.expected_output = field_value
            elif field_name == "VALIDATION_METHOD":
                spec.validation_method = field_value
            elif field_name == "PSEUDOCODE":
                # 伪代码可能跨行（以 |- 开头的 YAML 多行字符串格式）
                spec.pseudocode = field_value

        specs[key] = spec
        logger.info(f"提取到 {key} 的 model_spec: objective={spec.objective[:50]}...")

    return specs


def repair_json(json_str: str) -> dict | None:
    """尝试修复 LLM 输出的格式错误的 JSON。

    Args:
        json_str: 可能包含格式错误的 JSON 字符串。

    Returns:
        修复后的字典，无法修复时返回 None。
    """
    # 剥离 thinking 块（thinking 模型如 mimo-v2.5-pro 会返回）
    json_str = re.sub(r"\[thinking\].*?\[/thinking\]", "", json_str, flags=re.DOTALL)
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    # Try direct parse first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Fix unescaped newlines and quotes inside string values
    try:
        fixed = re.sub(
            r'(?<=: ")(.*?)(?=",\s*\n\s*"|"\s*\n\s*})',
            lambda m: m.group(0).replace('"', '\\"'),
            json_str,
            flags=re.DOTALL,
        )
        return json.loads(fixed)
    except (json.JSONDecodeError, re.error):
        pass

    # Extract key-value pairs with regex as last resort
    try:
        pattern = r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.|"(?!,\s*\n)|"(?!\s*\n\s*}))*)"'
        matches = re.findall(pattern, json_str, re.DOTALL)
        if matches:
            return {k: v.replace('\\"', '"') for k, v in matches}
    except re.error:
        pass

    return None


class ModelerAgent(Agent):
    """建模手 Agent，分析问题类型并制定建模方案、求解方法和可视化策略。"""
    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        max_retries: int = 5,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: 'DiagnosticLogger | None' = None,
    ) -> None:
        super().__init__(task_id, model, context_window, cancel_event=cancel_event, diagnostic_logger=diagnostic_logger)
        self.system_prompt = MODELER_PROMPT
        self.max_retries = max_retries

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        coordinator_to_modeler: CoordinatorToModeler,
        problem_analysis: str = "",
        literature_review_text: str = "",
        reformulation_text: str = "",
    ) -> ModelerToCoder:
        """根据协调者拆解的问题生成建模方案。

        Args:
            coordinator_to_modeler: 协调者传递的结构化问题信息。
            problem_analysis: 题目深度分析结果（可选，来自 ProblemAnalystAgent）。
            literature_review_text: 文献调研结果（可选，来自 LiteratureAgent）。
            reformulation_text: 问题重述结果（可选，来自 ProblemReformulationAgent）。

        Returns:
            ModelerToCoder 对象，包含各问题的建模解决方案。

        Raises:
            ValueError: 超过最大重试次数仍无法解析。
        """
        # 根据是否有题目分析动态生成系统提示词
        system_prompt = get_modeler_system_prompt(
            problem_analysis=problem_analysis,
            literature_review_text=literature_review_text,
            reformulation_text=reformulation_text,
        )
        await self.append_chat_history(
            {"role": "system", "content": system_prompt}
        )

        # 构造用户消息，包含问题信息
        user_msg = json.dumps(coordinator_to_modeler.questions, ensure_ascii=False)
        await self.append_chat_history(
            {"role": "user", "content": user_msg}
        )

        attempt = 0
        while attempt < self.max_retries:
            response = await self._chat(
                history=self.chat_history,
                agent_name=self.__class__.__name__,
            )

            json_str = response.content

            # 记录诊断日志
            if self.diagnostic_logger:
                self.diagnostic_logger.log_interaction(
                    agent_name=self.__class__.__name__,
                    sub_title="建模方案",
                    messages=self.chat_history,
                    response_content=json_str,
                    response_reasoning=response.reasoning_content,
                )
            if not json_str:
                raise ValueError("返回的 JSON 字符串为空，请检查输入内容。")

            questions_solution = repair_json(json_str)
            if questions_solution:
                # 提取结构化的 model_spec（给 CoderAgent 的精确接口）
                model_specs = extract_model_specs(questions_solution)
                logger.info(f"建模方案解析成功: {list(questions_solution.keys())}, "
                           f"提取到 {len(model_specs)} 个 model_spec")
                return ModelerToCoder(
                    questions_solution=questions_solution,
                    model_specs=model_specs,
                )

            attempt += 1
            logger.warning(
                f"JSON 解析失败 (第{attempt}/{self.max_retries}次)，请求模型重新生成"
            )

            if attempt >= self.max_retries:
                raise ValueError(
                    f"ModelerAgent 在 {self.max_retries} 次尝试后仍无法生成有效的 JSON。"
                    f"最后的输出: {json_str[:200]}..."
                )

            retry_msg: dict = {"role": "assistant", "content": json_str}
            if response.reasoning_content:
                retry_msg["reasoning_content"] = response.reasoning_content
            await self.append_chat_history(retry_msg)
            await self.append_chat_history(
                {
                    "role": "user",
                    "content": "你返回的JSON格式有误，请严格按照JSON格式重新输出，注意字符串值内的双引号必须转义为\\\"，不要包含未转义的特殊字符。",
                }
            )
