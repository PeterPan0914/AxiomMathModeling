"""Agent 间通信数据模型定义。"""

from pydantic import BaseModel
from typing import Any


class ModelSpec(BaseModel):
    """单个问题的结构化模型规格，给 CoderAgent 的接口。"""
    objective: str = ""                     # 目标函数描述
    constraints: list[str] = []             # 约束条件列表
    algorithm: str = ""                     # 求解算法
    key_params: dict[str, str] = {}         # 关键参数及来源
    expected_output: str = ""               # 预期输出格式
    validation_method: str = ""             # 验证方法
    pseudocode: str = ""                    # 伪代码（给 CoderAgent 参考）


class CoordinatorToModeler(BaseModel):
    """协调者传递给建模手的数据结构。"""
    questions: dict
    ques_count: int


class ModelerToCoder(BaseModel):
    """建模手传递给代码手的数据结构。"""
    questions_solution: dict[str, str]
    model_specs: dict[str, ModelSpec] = {}  # 结构化的模型规格


class CoderToWriter(BaseModel):
    """代码手传递给写作手的数据结构。"""
    code_response: str | None = None
    code_output: str | None = None
    created_images: list[str] | None = None


class WriterResponse(BaseModel):
    """写作手的响应数据结构。"""
    response_content: Any
    footnotes: list[tuple[str, str]] | None = None


class ParagraphIssue(BaseModel):
    """段落级评审问题。"""
    chapter: str = ""           # 章节名
    paragraph_index: int = 0    # 段落序号
    sentence: str = ""          # 问题句子
    issue: str = ""             # 问题描述
    severity: str = "MINOR"     # CRITICAL/MAJOR/MINOR
    fix: str = ""               # 具体修改建议


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
    paragraph_issues: list[ParagraphIssue] = []  # 段落级问题定位
