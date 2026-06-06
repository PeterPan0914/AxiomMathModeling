"""Agent 间通信数据模型定义。"""

from pydantic import BaseModel
from typing import Any


class CoordinatorToModeler(BaseModel):
    """协调者传递给建模手的数据结构。"""
    questions: dict
    ques_count: int


class ModelerToCoder(BaseModel):
    """建模手传递给代码手的数据结构。"""
    questions_solution: dict[str, str]


class CoderToWriter(BaseModel):
    """代码手传递给写作手的数据结构。"""
    code_response: str | None = None
    code_output: str | None = None
    created_images: list[str] | None = None


class WriterResponse(BaseModel):
    """写作手的响应数据结构。"""
    response_content: Any
    footnotes: list[tuple[str, str]] | None = None


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
