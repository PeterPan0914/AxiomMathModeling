"""评审 Agent 模块，支持三审制（方法论/写作/格式）和段落级定位。"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

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


class ReviewSynthesizer:
    """合并三份评审报告，去重、排序并生成改进路线图。

    使用方式：
        synthesizer = ReviewSynthesizer()
        report = synthesizer.synthesize(method_review, writing_review, format_review)
    """

    # 严重程度对应的权重基数
    _SEVERITY_WEIGHT: dict[str, float] = {
        "CRITICAL": 1.0,
        "MAJOR": 0.6,
        "MINOR": 0.3,
    }

    # 影响范围权重：方法论和逻辑问题影响最大
    _IMPACT_SCOPE: dict[str, float] = {
        "method": 1.0,
        "logic": 0.9,
        "writing": 0.6,
        "format": 0.4,
    }

    # 修改难度权重（越大越容易修改，优先级越高）
    _MODIFICATION_EASE: dict[str, float] = {
        "easy": 1.0,       # 格式、错别字等
        "medium": 0.6,     # 段落重写、补充论证
        "hard": 0.3,       # 重新建模、重新实验
    }

    def synthesize(
        self,
        method_review: ReviewResponse,
        writing_review: ReviewResponse,
        format_review: ReviewResponse,
        global_state_summary: str = "",
    ) -> dict[str, Any]:
        """合并三份评审报告，生成改进路线图。

        Args:
            method_review: 方法论审稿人的评审结果。
            writing_review: 写作审稿人的评审结果。
            format_review: 格式审稿人的评审结果。
            global_state_summary: 全局状态摘要（可选，用于更精确的去重判断）。

        Returns:
            合成报告字典，包含 overall_score, roadmap, score_breakdown,
            red_issues, green_issues 等字段。
        """
        # 1. 收集所有问题并标注来源
        all_issues = self._collect_all_issues(method_review, writing_review, format_review)

        # 2. 去重
        deduped_issues = self._deduplicate_issues(all_issues)

        # 3. 计算优先级分数并排序
        scored_issues = self._score_and_rank(deduped_issues)

        # 4. 标记红色（不改会扣分）和绿色（改了加分）
        red_issues, green_issues = self._classify_issues(scored_issues)

        # 5. 生成改进路线图
        roadmap = self._generate_roadmap(scored_issues)

        # 6. 汇总分数
        score_breakdown = {
            "method": method_review.overall_score,
            "writing": writing_review.overall_score,
            "format": format_review.overall_score,
        }
        overall_score = self._compute_overall_score(
            method_review, writing_review, format_review
        )

        # 7. 合并优点和反馈
        all_strengths = self._merge_strengths(method_review, writing_review, format_review)
        combined_feedback = self._combine_feedback(
            method_review, writing_review, format_review
        )

        return {
            "overall_score": overall_score,
            "roadmap": roadmap,
            "score_breakdown": score_breakdown,
            "red_issues": red_issues,
            "green_issues": green_issues,
            "strengths": all_strengths,
            "combined_feedback": combined_feedback,
            "total_issues_count": len(scored_issues),
        }

    # ---- 内部方法 ----

    def _collect_all_issues(
        self,
        method_review: ReviewResponse,
        writing_review: ReviewResponse,
        format_review: ReviewResponse,
    ) -> list[dict[str, Any]]:
        """从三份评审中收集所有问题，标注来源维度。"""
        issues: list[dict[str, Any]] = []

        for review, dimension in [
            (method_review, "method"),
            (writing_review, "writing"),
            (format_review, "format"),
        ]:
            for issue in review.paragraph_issues:
                issues.append({
                    "paragraph_index": issue.paragraph_index,
                    "sentence": issue.sentence,
                    "issue": issue.issue,
                    "severity": issue.severity or "MINOR",
                    "fix": issue.fix,
                    "issue_type": issue.issue_type,
                    "source_dimension": dimension,
                    "source_score": review.overall_score,
                })

            # 也将 improvements 文本作为轻量级 issue 纳入
            for imp_text in review.improvements:
                issues.append({
                    "paragraph_index": 0,
                    "sentence": "",
                    "issue": imp_text,
                    "severity": "MINOR",
                    "fix": "",
                    "issue_type": "improvement_text",
                    "source_dimension": dimension,
                    "source_score": review.overall_score,
                })

        return issues

    def _deduplicate_issues(self, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """去除重复问题。

        多个审稿人可能从不同角度指出同一个段落的同一个问题。
        以 (paragraph_index, 语义相似度) 为依据去重，保留严重程度更高的。
        """
        if not issues:
            return []

        # 按段落索引分组
        by_paragraph: dict[int, list[dict[str, Any]]] = {}
        no_para: list[dict[str, Any]] = []
        for issue in issues:
            para_idx = issue.get("paragraph_index", 0)
            if para_idx > 0:
                by_paragraph.setdefault(para_idx, []).append(issue)
            else:
                no_para.append(issue)

        deduped: list[dict[str, Any]] = []

        for _para_idx, para_issues in by_paragraph.items():
            # 对同段落的问题做简单文本相似度去重
            seen_clusters: list[dict[str, Any]] = []
            for issue in para_issues:
                merged_into = False
                for cluster in seen_clusters:
                    if self._issues_are_similar(cluster["issue"], issue["issue"]):
                        # 保留严重程度更高的
                        sev_order = {"CRITICAL": 3, "MAJOR": 2, "MINOR": 1}
                        if sev_order.get(issue["severity"], 0) > sev_order.get(cluster["severity"], 0):
                            cluster.update(issue)
                        # 合并来源
                        if issue["source_dimension"] not in cluster.get("source_dimensions", set()):
                            cluster.setdefault("source_dimensions", set()).add(issue["source_dimension"])
                        merged_into = True
                        break
                if not merged_into:
                    issue_copy = dict(issue)
                    issue_copy["source_dimensions"] = {issue["source_dimension"]}
                    seen_clusters.append(issue_copy)

            deduped.extend(seen_clusters)

        # 无段落定位的问题做文本去重
        seen_texts: set[str] = set()
        for issue in no_para:
            normalized = issue["issue"].strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                issue_copy = dict(issue)
                issue_copy["source_dimensions"] = {issue["source_dimension"]}
                deduped.append(issue_copy)

        return deduped

    def _issues_are_similar(self, text_a: str, text_b: str) -> bool:
        """简单判断两个问题文本是否描述同一问题。

        使用关键词重叠率：如果两个文本共享超过 60% 的关键词，视为相似。
        """
        # 去除标点和空白，提取关键词
        def _extract_keywords(text: str) -> set[str]:
            cleaned = re.sub(r'[^\w\s]', '', text)
            words = set(cleaned.split())
            # 过滤停用词
            stopwords = {"的", "了", "是", "在", "和", "与", "对", "将", "被", "从", "到", "有", "这", "那", "不", "也", "都", "就", "把", "让", "用", "以", "为"}
            return words - stopwords

        kw_a = _extract_keywords(text_a)
        kw_b = _extract_keywords(text_b)

        if not kw_a or not kw_b:
            return False

        overlap = len(kw_a & kw_b)
        min_len = min(len(kw_a), len(kw_b))
        return (overlap / min_len) >= 0.6 if min_len > 0 else False

    def _score_and_rank(self, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """计算每个问题的优先级分数并排序。

        优先级 = severity_weight * impact_scope * modification_ease
        """
        for issue in issues:
            sev = self._SEVERITY_WEIGHT.get(issue.get("severity", "MINOR"), 0.3)
            impact = self._IMPACT_SCOPE.get(issue.get("source_dimension", "format"), 0.5)
            # 根据 issue_type 和 severity 推断修改难度
            mod_ease = self._estimate_modification_ease(issue)
            issue["priority_score"] = round(sev * impact * mod_ease, 3)

        # 按优先级降序排列
        issues.sort(key=lambda x: x["priority_score"], reverse=True)
        return issues

    def _estimate_modification_ease(self, issue: dict[str, Any]) -> float:
        """根据问题类型和严重程度推断修改难度。"""
        severity = issue.get("severity", "MINOR")
        issue_type = issue.get("issue_type", "")
        dimension = issue.get("source_dimension", "")

        # 格式问题最容易修改
        if dimension == "format":
            return self._MODIFICATION_EASE["easy"]

        # 语言和写作问题一般是中等难度
        if dimension == "writing":
            if severity == "CRITICAL":
                return self._MODIFICATION_EASE["medium"]
            return self._MODIFICATION_EASE["easy"]

        # 方法论问题：假设不成立 → 困难，公式错误 → 中等
        if dimension == "method":
            hard_types = {"assumption_unjustified", "model_selection", "data_leakage"}
            if issue_type in hard_types or severity == "CRITICAL":
                return self._MODIFICATION_EASE["hard"]
            return self._MODIFICATION_EASE["medium"]

        return self._MODIFICATION_EASE["medium"]

    def _classify_issues(
        self, scored_issues: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将问题分为红色（不改会扣分）和绿色（改了加分）。

        红色：severity 为 CRITICAL 或 MAJOR，且优先级分数高于阈值。
        绿色：severity 为 MINOR，改了可以提升论文品质。
        """
        red_issues: list[dict[str, Any]] = []
        green_issues: list[dict[str, Any]] = []

        # 优先级阈值：大于 0.3 视为红色（需要强制修改）
        RED_THRESHOLD = 0.3

        for issue in scored_issues:
            # 清理不可序列化的字段
            clean_issue = {k: v for k, v in issue.items() if k != "source_dimensions"}
            if "source_dimensions" in issue:
                clean_issue["source_dimensions"] = list(issue["source_dimensions"])

            if issue["priority_score"] >= RED_THRESHOLD:
                red_issues.append(clean_issue)
            else:
                green_issues.append(clean_issue)

        return red_issues, green_issues

    def _generate_roadmap(self, scored_issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """生成改进路线图：按优先级排列，标注预估收益和具体修改指令。"""
        roadmap: list[dict[str, Any]] = []

        for i, issue in enumerate(scored_issues[:15]):  # 最多处理 15 个问题
            # 预估修复后的分数提升
            score_gain = self._estimate_score_gain(issue)

            roadmap.append({
                "order": i + 1,
                "issue": issue.get("issue", ""),
                "severity": issue.get("severity", "MINOR"),
                "dimension": issue.get("source_dimension", ""),
                "estimated_score_gain": score_gain,
                "specific_instruction": issue.get("fix", "请根据问题描述自行修改"),
                "verification_criterion": self._generate_verification_criterion(issue),
                "priority_score": issue.get("priority_score", 0),
                "is_red": issue["priority_score"] >= 0.3,
            })

        return roadmap

    def _estimate_score_gain(self, issue: dict[str, Any]) -> int:
        """预估修复单个问题后的分数提升。"""
        severity = issue.get("severity", "MINOR")
        dimension = issue.get("source_dimension", "format")

        # 基础分值
        base_gain = {"CRITICAL": 5, "MAJOR": 3, "MINOR": 1}.get(severity, 1)

        # 方法论问题修复收益最高
        multiplier = {"method": 1.2, "logic": 1.1, "writing": 1.0, "format": 0.8}.get(dimension, 1.0)

        return max(1, round(base_gain * multiplier))

    def _generate_verification_criterion(self, issue: dict[str, Any]) -> str:
        """为每个问题生成验证标准，用于判断修改是否到位。"""
        dimension = issue.get("source_dimension", "")
        issue_type = issue.get("issue_type", "")

        # 根据维度和问题类型生成验证标准
        verification_map: dict[str, str] = {
            "assumption_unjustified": "检查是否补充了假设检验数据或论证",
            "logic_gap": "检查推理链是否完整，逻辑跳跃是否消除",
            "empty_assertion": "检查是否用具体数据或文献替换了空洞断言",
            "formula_error": "检查公式是否修正，变量定义是否完整",
            "improvement_text": "检查改进措施是否已落实到论文中",
        }

        if issue_type in verification_map:
            return verification_map[issue_type]

        if dimension == "method":
            return "检查方法论论证是否完整，假设是否有数据支撑"
        elif dimension == "writing":
            return "检查论证是否严密，图表解读是否包含三段式"
        elif dimension == "format":
            return "检查格式是否符合竞赛规范"
        return "检查问题是否已修复"

    def _compute_overall_score(
        self,
        method_review: ReviewResponse,
        writing_review: ReviewResponse,
        format_review: ReviewResponse,
    ) -> int:
        """计算综合分数。

        方法论权重最高（40%），写作 35%，格式 25%。
        """
        method_score = method_review.overall_score or 60
        writing_score = writing_review.overall_score or 60
        format_score = format_review.overall_score or 60

        weighted = method_score * 0.4 + writing_score * 0.35 + format_score * 0.25
        return round(weighted)

    def _merge_strengths(
        self,
        method_review: ReviewResponse,
        writing_review: ReviewResponse,
        format_review: ReviewResponse,
    ) -> list[str]:
        """合并三份评审的优点列表，去重后保留最多 5 条。"""
        all_strengths: list[str] = []
        for review in [method_review, writing_review, format_review]:
            all_strengths.extend(review.strengths)

        # 简单去重（保留首次出现的）
        seen: set[str] = set()
        unique: list[str] = []
        for s in all_strengths:
            normalized = s.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)

        return unique[:5]

    def _combine_feedback(
        self,
        method_review: ReviewResponse,
        writing_review: ReviewResponse,
        format_review: ReviewResponse,
    ) -> str:
        """合并三份评审的反馈文本。"""
        parts: list[str] = []
        for label, review in [
            ("方法论评审", method_review),
            ("写作评审", writing_review),
            ("格式评审", format_review),
        ]:
            if review.feedback:
                parts.append(f"### {label}\n{review.feedback}")
        return "\n\n".join(parts)


class MultiReviewer:
    """三审制协调器，并行运行三个专项审稿人，合并结果。

    使用 ReviewSynthesizer 进行去重、排序和路线图生成。

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
        self.synthesizer = ReviewSynthesizer()

    async def run(
        self,
        paper_content: str,
        section_name: str | None = None,
    ) -> ReviewResponse:
        """并行运行三个专项审稿人，使用 ReviewSynthesizer 合并结果。

        Args:
            paper_content: 待评审的论文内容。
            section_name: 章节名称。

        Returns:
            合并后的 ReviewResponse，包含路线图和红/绿标记。
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

        # 分离成功和失败的结果
        method_review = ReviewResponse()
        writing_review = ReviewResponse()
        format_review = ReviewResponse()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"MultiReviewer: {roles[i]} 评审失败: {result}")
                continue
            if roles[i] == "method":
                method_review = result
            elif roles[i] == "writing":
                writing_review = result
            elif roles[i] == "format":
                format_review = result

        # 使用 ReviewSynthesizer 合成报告
        synthesis = self.synthesizer.synthesize(
            method_review, writing_review, format_review
        )

        # 构建合并后的 ReviewResponse
        merged = ReviewResponse(
            overall_score=synthesis["overall_score"],
            math_score=synthesis["score_breakdown"]["method"],
            logic_score=synthesis["score_breakdown"]["writing"],
            language_score=synthesis["score_breakdown"]["writing"],
            format_score=synthesis["score_breakdown"]["format"],
            feedback=synthesis["combined_feedback"],
            improvements=[
                item["specific_instruction"]
                for item in synthesis["roadmap"]
                if item["specific_instruction"]
            ][:8],
            strengths=synthesis["strengths"],
            paragraph_issues=[
                ParagraphIssue(
                    paragraph_index=issue.get("paragraph_index", 0),
                    sentence=issue.get("sentence", ""),
                    issue=issue.get("issue", ""),
                    severity=issue.get("severity", "MINOR"),
                    fix=issue.get("fix", ""),
                    issue_type=issue.get("issue_type", ""),
                )
                for issue in synthesis["red_issues"] + synthesis["green_issues"]
            ],
        )

        # 记录诊断日志
        if self.diagnostic_logger:
            self.diagnostic_logger.log_tool_result(
                agent_name="MultiReviewer",
                tool_name="review_synthesis",
                sub_title=section_name or "unknown",
                tool_input={"paper_content_length": len(paper_content)},
                tool_output=json.dumps(
                    {
                        "overall_score": synthesis["overall_score"],
                        "score_breakdown": synthesis["score_breakdown"],
                        "total_issues": synthesis["total_issues_count"],
                        "red_issues_count": len(synthesis["red_issues"]),
                        "green_issues_count": len(synthesis["green_issues"]),
                        "roadmap_top3": synthesis["roadmap"][:3],
                    },
                    ensure_ascii=False,
                ),
            )

        return merged
