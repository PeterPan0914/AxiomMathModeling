"""评审 Agent 模块，负责对论文进行多维度质量评审。"""

import asyncio
from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts import get_reviewer_prompt
from app.schemas.enums import FormatOutPut
from app.utils.log_util import logger
from app.schemas.A2A import ReviewResponse


class ReviewAgent(Agent):
    """评审 Agent，对论文进行多维度质量评审并提供改进建议。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(task_id, model, context_window, cancel_event=cancel_event)
        self.system_prompt = get_reviewer_prompt()
        self.is_first_run = True

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        paper_content: str,
        section_name: str | None = None,
        sub_title: str | None = None,
    ) -> ReviewResponse:
        """评审论文内容并返回评分和改进建议。

        Args:
            paper_content: 待评审的论文内容。
            section_name: 章节名称（可选）。
            sub_title: 子任务标题。

        Returns:
            ReviewResponse 对象，包含评分、反馈和改进建议。
        """
        logger.info(f"ReviewAgent: 开始评审 {section_name or '完整论文'}")

        if self.is_first_run:
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )

        # 构造评审提示
        review_prompt = self._build_review_prompt(paper_content, section_name)
        await self.append_chat_history({"role": "user", "content": review_prompt})

        # 获取评审结果
        response = await self._chat(
            history=self.chat_history,
            agent_name=self.__class__.__name__,
            sub_title=sub_title or "论文评审",
        )

        response_content = response.content or ""
        await self.append_chat_history({"role": "assistant", "content": response_content})

        # 解析评审结果
        review_result = self._parse_review_response(response_content)

        logger.info(
            f"ReviewAgent: 评审完成 - 总分: {review_result['overall_score']}/100"
        )

        return ReviewResponse(
            overall_score=review_result["overall_score"],
            math_score=review_result["math_score"],
            logic_score=review_result["logic_score"],
            language_score=review_result["language_score"],
            format_score=review_result["format_score"],
            feedback=response_content,
            improvements=review_result["improvements"],
            strengths=review_result["strengths"],
        )

    def _build_review_prompt(self, paper_content: str, section_name: str | None) -> str:
        """构建评审提示。

        Args:
            paper_content: 论文内容。
            section_name: 章节名称。

        Returns:
            格式化的评审提示。
        """
        section_info = f"章节: {section_name}" if section_name else "完整论文"

        return f"""请对以下论文内容进行严格的质量评审。

{section_info}

【评审要求】
1. 从以下四个维度进行评分（每项 0-25 分，总分 100 分）：
   - 数学正确性 (25分): 公式准确性、推导完整性、计算正确性
   - 逻辑连贯性 (25分): 论证有效性、过渡流畅性、结构清晰性
   - 语言质量 (25分): 学术性、准确性、流畅性
   - 格式规范 (25分): 引用格式、图表整合、排版一致性

2. 列出论文的优点（至少 2 个）

3. 列出需要改进的地方（至少 3 个），每个改进点要具体说明：
   - 问题在哪里
   - 为什么是问题
   - 如何改进

4. 给出总体评价和是否达到质量标准的判断（总分 >= 80 分为达标）

【论文内容】
{paper_content}

【输出格式】
请严格按照以下格式输出：

## 评分
- 数学正确性: X/25
- 逻辑连贯性: X/25
- 语言质量: X/25
- 格式规范: X/25
- 总分: X/100

## 优点
1. [优点1]
2. [优点2]

## 改进建议
1. [问题]: [改进方法]
2. [问题]: [改进方法]
3. [问题]: [改进方法]

## 总体评价
[是否达标] - [简要说明]
"""

    def _parse_review_response(self, response: str) -> dict:
        """解析评审响应，提取评分和改进建议。

        Args:
            response: 评审响应文本。

        Returns:
            包含评分和建议的字典。
        """
        import re

        # 默认值
        result = {
            "math_score": 15,
            "logic_score": 15,
            "language_score": 15,
            "format_score": 15,
            "overall_score": 60,
            "improvements": [],
            "strengths": [],
        }

        # 提取数学正确性分数
        math_match = re.search(r"数学正确性[：:]\s*(\d+)", response)
        if math_match:
            result["math_score"] = int(math_match.group(1))

        # 提取逻辑连贯性分数
        logic_match = re.search(r"逻辑连贯性[：:]\s*(\d+)", response)
        if logic_match:
            result["logic_score"] = int(logic_match.group(1))

        # 提取语言质量分数
        language_match = re.search(r"语言质量[：:]\s*(\d+)", response)
        if language_match:
            result["language_score"] = int(language_match.group(1))

        # 提取格式规范分数
        format_match = re.search(r"格式规范[：:]\s*(\d+)", response)
        if format_match:
            result["format_score"] = int(format_match.group(1))

        # 提取总分
        total_match = re.search(r"总分[：:]\s*(\d+)", response)
        if total_match:
            result["overall_score"] = int(total_match.group(1))
        else:
            # 计算总分
            result["overall_score"] = (
                result["math_score"]
                + result["logic_score"]
                + result["language_score"]
                + result["format_score"]
            )

        # 提取改进建议
        improvements_section = re.search(
            r"## 改进建议\s*\n(.*?)(?=##|$)", response, re.DOTALL
        )
        if improvements_section:
            improvements_text = improvements_section.group(1)
            improvements = re.findall(r"\d+\.\s*(.+)", improvements_text)
            result["improvements"] = [imp.strip() for imp in improvements]

        # 提取优点
        strengths_section = re.search(
            r"## 优点\s*\n(.*?)(?=##|$)", response, re.DOTALL
        )
        if strengths_section:
            strengths_text = strengths_section.group(1)
            strengths = re.findall(r"\d+\.\s*(.+)", strengths_text)
            result["strengths"] = [s.strip() for s in strengths]

        return result
