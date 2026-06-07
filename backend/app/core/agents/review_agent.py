"""评审 Agent 模块，支持三审制（方法论/写作/格式）和段落级定位。"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger

from app.core.prompts import get_reviewer_prompt
from app.core.prompts.reviewer import (
    get_method_reviewer_prompt,
    get_writing_reviewer_prompt,
    get_format_reviewer_prompt,
)
from app.schemas.enums import FormatOutPut
from app.utils.log_util import logger
from app.schemas.A2A import ReviewResponse, ParagraphIssue


class ReviewAgent(Agent):
    """评审 Agent，对论文进行多维度质量评审并提供改进建议。

    支持两种模式：
    1. 综合评审（默认）：使用完整的四维度评审提示词
    2. 专项评审（role="method"/"writing"/"format"）：使用三审制的专项提示词
    """

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        super().__init__(task_id, model, context_window, cancel_event=cancel_event, diagnostic_logger=diagnostic_logger)
        self.system_prompt = get_reviewer_prompt()
        self.is_first_run = True
        # 三审制的专用 prompt
        self._role_prompts = {
            "method": get_method_reviewer_prompt(),
            "writing": get_writing_reviewer_prompt(),
            "format": get_format_reviewer_prompt(),
        }

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        paper_content: str,
        section_name: str | None = None,
        sub_title: str | None = None,
        role: str | None = None,
    ) -> ReviewResponse:
        """评审论文内容并返回评分和改进建议。

        Args:
            paper_content: 待评审的论文内容。
            section_name: 章节名称（可选）。
            sub_title: 子任务标题。
            role: 评审角色，None 表示综合评审，
                  "method"/"writing"/"format" 表示专项评审。

        Returns:
            ReviewResponse 对象，包含评分、反馈和改进建议。
        """
        logger.info(f"ReviewAgent: 开始评审 {section_name or '完整论文'} (角色: {role or '综合'})")

        # 选择对应的 prompt
        if role and role in self._role_prompts:
            # 专项评审使用独立的 Agent 历史（避免污染综合评审上下文）
            prompt = self._role_prompts[role]
        else:
            prompt = self.system_prompt

        if self.is_first_run:
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": prompt}
            )

        # 构造评审提示
        review_prompt = self._build_review_prompt(paper_content, section_name, role)
        await self.append_chat_history({"role": "user", "content": review_prompt})

        # 获取评审结果
        response = await self._chat(
            history=self.chat_history,
            agent_name=self.__class__.__name__,
            sub_title=sub_title or "论文评审",
        )

        response_content = response.content or ""
        assistant_msg: dict = {"role": "assistant", "content": response_content}
        if response.reasoning_content:
            assistant_msg["reasoning_content"] = response.reasoning_content
        await self.append_chat_history(assistant_msg)

        # 解析评审结果
        review_result = self._parse_review_response(response_content, role)

        logger.info(
            f"ReviewAgent: 评审完成 - 总分: {review_result['overall_score']}/100"
        )

        # 记录诊断日志：结构化评审数据
        if self.diagnostic_logger:
            self.diagnostic_logger.log_tool_result(
                agent_name=self.__class__.__name__,
                tool_name=f"review_{role or 'full'}",
                sub_title=section_name or "unknown",
                tool_input={"paper_content_length": len(paper_content), "role": role},
                tool_output=json.dumps(review_result, ensure_ascii=False),
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
            paragraph_issues=review_result.get("paragraph_issues", []),
        )

    def _build_review_prompt(
        self, paper_content: str, section_name: str | None, role: str | None = None
    ) -> str:
        """构建评审提示。"""
        section_info = f"章节: {section_name}" if section_name else "完整论文"

        # 专项评审使用精简的 prompt
        if role == "method":
            return f"""请对以下论文内容进行**方法论专项评审**。

{section_info}

【审查重点】
1. 模型选择论证是否充分（至少2种不同方法族的候选方法？对比表格？选择理由具体到数据特征？）
2. 假设是否有数据支撑（正态性检验？独立性检验？）
3. 数学公式是否正确，推导是否完整
4. 结果是否合理（R²是否过高？是否有模型诊断？）
5. 鲁棒性分析是否采用六维度框架

【论文内容】
{paper_content}

请严格按输出格式要求评审，改进建议必须定位到具体段落（"【第X段】"格式）。
"""
        elif role == "writing":
            return f"""请对以下论文内容进行**写作质量专项评审**。

{section_info}

【审查重点】
1. 逻辑连贯性：有没有逻辑跳跃？过渡是否自然？
2. 段落结构：每段是否有主题句→论据→结论？
3. 图表论证：是否有"观察→解读→关联"三步论证？
4. 语言质量：是否使用学术中文？有没有口语化表达？
5. 前后呼应：摘要承诺是否在正文兑现？
6. 禁止模板句式：是否使用了"针对问题X，本文采用了..."等模板句式？

【论文内容】
{paper_content}

请严格按输出格式要求评审，改进建议必须定位到具体段落（"【第X段】"格式）。
"""
        elif role == "format":
            return f"""请对以下论文内容进行**格式规范专项评审**。

{section_info}

【审查重点】
1. 图表格式：标题、标签、插入格式（![描述](文件名.png)）
2. 公式格式：行内$...$、块级$$...$$、每个公式后有"其中"说明
3. 引用格式：一致性、无重复
4. 章节结构：层次清晰、编号连续
5. 篇幅控制：各章节字数是否在目标范围内

【论文内容】
{paper_content}

请严格按输出格式要求评审，改进建议必须定位到具体段落（"【第X段】"格式）。
"""
        else:
            # 综合评审（原有逻辑）
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

    def _parse_review_response(self, response: str, role: str | None = None) -> dict:
        """解析评审响应，提取评分和改进建议。

        Args:
            response: 评审响应文本。
            role: 评审角色。

        Returns:
            包含评分和建议的字典。
        """
        # 默认值
        result = {
            "math_score": 15,
            "logic_score": 15,
            "language_score": 15,
            "format_score": 15,
            "overall_score": 60,
            "improvements": [],
            "strengths": [],
            "paragraph_issues": [],
        }

        # 专项评审：解析单维度总分
        if role in ("method", "writing", "format"):
            score_match = re.search(r'(?:方法论|写作|格式)总分[：:]\s*(\d+)', response)
            if score_match:
                score = int(score_match.group(1))
                result["overall_score"] = score
                # 映射到对应维度
                if role == "method":
                    result["math_score"] = score
                elif role == "writing":
                    result["logic_score"] = score
                elif role == "format":
                    result["format_score"] = score
        else:
            # 综合评审：解析四维度
            math_match = re.search(r"数学正确性[：:]\s*(\d+)", response)
            if math_match:
                result["math_score"] = int(math_match.group(1))

            logic_match = re.search(r"逻辑连贯性[：:]\s*(\d+)", response)
            if logic_match:
                result["logic_score"] = int(logic_match.group(1))

            language_match = re.search(r"语言质量[：:]\s*(\d+)", response)
            if language_match:
                result["language_score"] = int(language_match.group(1))

            format_match = re.search(r"格式规范[：:]\s*(\d+)", response)
            if format_match:
                result["format_score"] = int(format_match.group(1))

            total_match = re.search(r"总分[：:]\s*(\d+)", response)
            if total_match:
                result["overall_score"] = int(total_match.group(1))
            else:
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

            # 提取段落级定位
            paragraph_pattern = r'【第(\d+)段】(.+?)(?=【第\d+段】|$)'
            para_matches = re.findall(paragraph_pattern, improvements_text, re.DOTALL)
            for para_idx, para_content in para_matches:
                # 解析 "问题: 建议" 格式
                parts = para_content.split(":", 1)
                issue = parts[0].strip() if parts else para_content.strip()
                fix = parts[1].strip() if len(parts) > 1 else ""
                result["paragraph_issues"].append({
                    "paragraph_index": int(para_idx),
                    "issue": issue,
                    "fix": fix,
                    "severity": "MAJOR",
                })

        # 提取优点
        strengths_section = re.search(
            r"## 优点\s*\n(.*?)(?=##|$)", response, re.DOTALL
        )
        if strengths_section:
            strengths_text = strengths_section.group(1)
            strengths = re.findall(r"\d+\.\s*(.+)", strengths_text)
            result["strengths"] = [s.strip() for s in strengths]

        return result


class MultiReviewer:
    """三审制协调器，并行运行三个专项审稿人，合并结果。

    使用方式：
        multi_reviewer = MultiReviewer(task_id, model)
        merged_response = await multi_reviewer.run(paper_content, section_name)
    """

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        self.task_id = task_id
        self.model = model
        self.context_window = context_window
        self.cancel_event = cancel_event
        self.diagnostic_logger = diagnostic_logger

    async def run(
        self,
        paper_content: str,
        section_name: str | None = None,
    ) -> ReviewResponse:
        """并行运行三个专项审稿人，合并结果。

        Args:
            paper_content: 待评审的论文内容。
            section_name: 章节名称。

        Returns:
            合并后的 ReviewResponse。
        """
        roles = ["method", "writing", "format"]

        # 并行运行三个审稿人
        tasks = []
        for role in roles:
            reviewer = ReviewAgent(
                task_id=self.task_id,
                model=self.model,
                context_window=self.context_window,
                cancel_event=self.cancel_event,
                diagnostic_logger=self.diagnostic_logger,
            )
            tasks.append(reviewer.run(
                paper_content=paper_content,
                section_name=section_name,
                sub_title=f"{section_name}_{role}",
                role=role,
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        merged = ReviewResponse()
        all_strengths = []
        all_improvements = []
        all_para_issues = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"MultiReviewer: {roles[i]} 评审失败: {result}")
                continue

            all_strengths.extend(result.strengths)
            all_improvements.extend(result.improvements)
            all_para_issues.extend(result.paragraph_issues)

            # 合并分数（加权平均）
            if roles[i] == "method":
                merged.math_score = result.overall_score
            elif roles[i] == "writing":
                merged.logic_score = result.overall_score
                merged.language_score = result.overall_score
            elif roles[i] == "format":
                merged.format_score = result.overall_score

        # 计算总分（四维度平均）
        scores = [merged.math_score, merged.logic_score, merged.language_score, merged.format_score]
        valid_scores = [s for s in scores if s > 0]
        merged.overall_score = sum(valid_scores) // len(valid_scores) if valid_scores else 60

        merged.strengths = list(set(all_strengths))[:5]  # 去重，最多5条
        merged.improvements = list(set(all_improvements))[:8]  # 去重，最多8条
        merged.paragraph_issues = all_para_issues
        merged.feedback = "\n\n".join([
            f"### {r.feedback}" for r in results if not isinstance(r, Exception)
        ])

        return merged
