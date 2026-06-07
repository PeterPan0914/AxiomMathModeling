"""Agent 间通信数据模型定义。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# 建模规格 & Agent 间传递结构
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """单个问题的结构化模型规格，给 CoderAgent 的接口。

    同时承载论文叙事所需的方法对比、假设论证等元数据，
    供 WriterAgent 生成论文时使用。
    """

    # ---- 原有字段 ----
    objective: str = ""
    constraints: list[str] = []
    algorithm: str = ""
    key_params: dict[str, str] = {}
    expected_output: str = ""
    validation_method: str = ""
    pseudocode: str = ""

    # ---- 扩展：CoderAgent 执行规范 ----
    library: str = ""
    hyperparameters: dict[str, Any] = {}
    required_data_format: dict[str, Any] = {}
    required_outputs: list[str] = []
    validation_protocol: dict[str, Any] = {}
    plots_required: list[str] = []
    sanity_checks: list[str] = []


class CoordinatorToModeler(BaseModel):
    """协调者传递给建模手的数据结构。"""

    questions: dict
    ques_count: int


class ModelerToCoder(BaseModel):
    """建模手传递给代码手的数据结构。"""

    questions_solution: dict[str, str]
    model_specs: dict[str, ModelSpec] = {}


class CoderToWriter(BaseModel):
    """代码手传递给写作手的数据结构。"""

    code_response: str | None = None
    code_output: str | None = None
    created_images: list[str] | None = None


class WriterResponse(BaseModel):
    """写作手的响应数据结构。"""

    response_content: Any
    footnotes: list[tuple[str, str]] | None = None


# ---------------------------------------------------------------------------
# 评审相关
# ---------------------------------------------------------------------------


class ParagraphIssue(BaseModel):
    """段落级评审问题。"""

    chapter: str = ""
    paragraph_index: int = 0
    sentence: str = ""
    issue: str = ""
    severity: str = "MINOR"  # CRITICAL / MAJOR / MINOR
    fix: str = ""
    issue_type: str = ""  # assumption_unjustified, logic_gap, empty_assertion ...


class ReviewResponse(BaseModel):
    """评审 Agent 的响应数据结构。"""

    overall_score: int = 0
    math_score: int = 0
    logic_score: int = 0
    language_score: int = 0
    format_score: int = 0
    feedback: str = ""
    improvements: list[str] = []
    strengths: list[str] = []
    paragraph_issues: list[ParagraphIssue] = []

    # ---- 扩展：Reflexion 追踪 ----
    round: int = 0
    reviewer: str = ""
    diff_from_previous: str | None = None


# ---------------------------------------------------------------------------
# 质疑 & 合理性检查
# ---------------------------------------------------------------------------


class FigureNarrative(BaseModel):
    """单张图表的三段式解读（观察 -> 含义 -> 处置）。"""

    filename: str = ""
    description: str = ""
    observation: str = ""   # 客观观察 1-2 句
    meaning: str = ""        # 含义解读 2-3 句
    disposition: str = ""    # 处置论证 2-3 句


class CritiqueResult(BaseModel):
    """CriticAgent 质疑输出。"""

    decision: str = "approve"  # approve / revise / reject
    issues: list[dict[str, Any]] = []
    suggestions: list[str] = []
    fatal_issues: list[dict[str, Any]] = []
    return_to_agent: str | None = None


class SanityReport(BaseModel):
    """DataSanityAgent 合理性检查报告。"""

    status: str = "ok_with_caveats"  # ok_with_caveats / needs_remodeling
    issues: list[dict[str, Any]] = []
    figure_narratives: list[FigureNarrative] = []
    paper_talking_points: list[str] = []


# ---------------------------------------------------------------------------
# 题目分析 & 子问题规格
# ---------------------------------------------------------------------------


class SubProblemSpec(BaseModel):
    """单个子问题的结构化规格（ProblemAnalystAgent 输出）。"""

    id: str = ""
    description: str = ""
    core_challenge: str = ""
    dependency_on: list[str] = []
    provides_to: list[str] = []
    recommended_method_family: list[str] = []
    forbidden_methods: list[str] = []
    key_outputs_needed: list[str] = []
    examiner_red_flags: list[str] = []


# ---------------------------------------------------------------------------
# 建模决策记录
# ---------------------------------------------------------------------------


class CandidateEvaluation(BaseModel):
    """ModelerAgent 对单个候选方法的评估。"""

    method: str = ""
    pros: list[str] = []
    cons: list[str] = []
    applicability_check: dict[str, str] = {}
    verdict: str = ""  # 选用 / 排除 / 备选
    reason: str = ""


class KeyAssumption(BaseModel):
    """模型关键假设及其验证记录。"""

    assumption: str = ""
    justification: str = ""
    test_used: str = ""
    test_result: str = ""


# ---------------------------------------------------------------------------
# Agent 决策日志 & 章节承诺
# ---------------------------------------------------------------------------


class AgentDecision(BaseModel):
    """Agent 决策日志条目，可被 CriticAgent 质疑。"""

    timestamp: str = ""
    agent: str = ""
    decision: str = ""
    reasoning: str = ""
    can_be_challenged: bool = True
    challenged_by: str | None = None


class ChapterPromise(BaseModel):
    """论文章节做出的承诺及兑现状态。"""

    claim: str = ""
    delivered: bool = False
    evidence: str | None = None
    gap: str | None = None
