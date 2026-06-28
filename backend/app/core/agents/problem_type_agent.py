"""问题类型识别 Agent 模块。

使用诊断决策树识别每个子问题的统计问题类型：
生存分析、纵向回归、组合优化、多分类不平衡等。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger


@dataclass
class SparsityReport:
    """数据稀疏性分析报告。"""
    observations_per_subject: float = 0.0
    total_samples: int = 0
    num_features: int = 0
    sparsity_level: str = "moderate"  # dense / moderate / sparse / very_sparse
    has_censoring: bool = False
    censoring_type: str = ""  # right / left / interval
    censoring_rate: float = 0.0
    has_repeated_measures: bool = False
    recommendation: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations_per_subject": self.observations_per_subject,
            "total_samples": self.total_samples,
            "sparsity_level": self.sparsity_level,
            "has_censoring": self.has_censoring,
            "censoring_type": self.censoring_type,
            "censoring_rate": self.censoring_rate,
            "has_repeated_measures": self.has_repeated_measures,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


@dataclass
class ProblemTypeInfo:
    """单个子问题的问题类型信息。"""
    question_id: str = ""
    primary_type: str = ""
    # survival_analysis / count_data / longitudinal_regression /
    # classification / optimization / time_series / regression /
    # clustering / evaluation / mechanism_dynamics
    sub_types: list[str] = field(default_factory=list)
    key_indicators: list[str] = field(default_factory=list)
    censoring_detected: bool = False
    censoring_details: str = ""
    sparsity_report: SparsityReport = field(default_factory=SparsityReport)
    recommended_method_family: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "primary_type": self.primary_type,
            "sub_types": self.sub_types,
            "key_indicators": self.key_indicators,
            "censoring_detected": self.censoring_detected,
            "censoring_details": self.censoring_details,
            "sparsity_report": self.sparsity_report.to_dict(),
            "recommended_method_family": self.recommended_method_family,
            "anti_patterns": self.anti_patterns,
            "confidence": self.confidence,
        }


@dataclass
class ProblemTypeReport:
    """完整的问题类型识别报告。"""
    sub_problem_types: dict[str, ProblemTypeInfo] = field(default_factory=dict)
    overall_data_characteristics: str = ""
    cross_problem_dependencies: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_problem_types": {k: v.to_dict() for k, v in self.sub_problem_types.items()},
            "overall_data_characteristics": self.overall_data_characteristics,
            "cross_problem_dependencies": self.cross_problem_dependencies,
        }


PROBLEM_TYPE_PROMPT = """# Role
你是一名统计学和生物统计学领域的专家顾问，擅长识别问题的统计类型。
你的核心能力是：从题目描述中识别出**生存分析**、**计数数据**、**纵向回归**等容易被误判的统计问题类型。

# 诊断决策树

你必须对每个子问题按以下顺序回答诊断问题：

## 问题A：结局变量是什么类型？
- 结局是"事件发生的时间"（如达标时间、发病时间、存活时间） → **生存分析 (survival_analysis)**
- 结局是"计数"（如发病次数、缺陷数） → **计数数据 (count_data)**
- 结局是连续数值 → 继续问题B
- 结局是类别标签 → **分类 (classification)**

## 问题B：数据是否有时间/重复测量结构？
- 同一个体有多次纵向观测 → **纵向回归 (longitudinal_regression)**
- 数据是纯时间序列 → **时间序列 (time_series)**
- 横截面数据 → **回归 (regression)**

## 问题C：是否存在删失（censoring）？【关键！】
- 有些个体在研究结束时仍未达到终点事件 → **右删失 (right censoring)**
- **重要规则**：如果存在删失，问题类型必须升级为生存分析

## 问题D：数据稀疏性评估
- 每个个体平均观测次数：
  - 1-3次 → **very_sparse**，强烈推荐 GPR
  - 4-10次 → **sparse**，LMM 可用但需谨慎
  - 10次以上 → **moderate/dense**

# 高频误判模式（必须警惕！）

1. **"纵向观测数据预测事件发生时间"** → 这是**生存分析**，不是普通回归！
   - 结局变量是事件发生时间（time-to-event）
   - 部分个体在观测期内未达到终点 → 右删失
   - 每个个体仅1-3次观测 → 非常稀疏
   - **错误做法**：用 LMM 直接外推
   - **正确做法**：GPR 量化不确定性 + Cox/DeepHit 生存分析

# 输出格式

严格按以下 JSON 格式输出：

```json
{
  "sub_problem_types": {
    "ques1": {
      "question_id": "ques1",
      "primary_type": "survival_analysis",
      "sub_types": ["survival_analysis", "longitudinal_regression"],
      "key_indicators": ["结局变量是事件发生时间", "部分样本右删失", "每个体仅1-3次观测"],
      "censoring_detected": true,
      "censoring_details": "右删失，约30%的样本在观测期结束时仍未达到终点",
      "sparsity_report": {
        "observations_per_subject": 2.1,
        "total_samples": 500,
        "sparsity_level": "very_sparse",
        "has_censoring": true,
        "censoring_type": "right",
        "censoring_rate": 0.30,
        "has_repeated_measures": true,
        "recommendation": "使用 GPR 而非 LMM 直接外推",
        "rationale": "GPR 输出后验概率分布+置信区间，LMM 仅输出点估计"
      },
      "recommended_method_family": ["GPR", "Cox_PH", "DeepHit", "Kaplan_Meier"],
      "anti_patterns": ["LMM直接外推: 稀疏数据外推误差大", "OLS回归: 违反独立性假设"],
      "confidence": 0.95
    }
  },
  "overall_data_characteristics": "整体数据特征描述",
  "cross_problem_dependencies": "子问题间依赖关系"
}
```
"""


class ProblemTypeAgent(Agent):
    """问题类型识别 Agent。

    使用诊断决策树识别每个子问题的统计问题类型。
    """

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        super().__init__(
            task_id, model, context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )
        self.system_prompt = PROBLEM_TYPE_PROMPT

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        ques_all: str,
        questions: dict,
    ) -> ProblemTypeReport:
        """对每个子问题进行问题类型深度分类。"""
        logger.info("ProblemTypeAgent: 开始问题类型分类")

        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )

        prompt = f"""请对以下数学建模竞赛题目进行问题类型深度分类。

【原始题目】
{ques_all}

【已拆解的问题】
{json.dumps(questions, ensure_ascii=False, indent=2)}

请严格按照诊断决策树对每个子问题进行分类，输出 JSON 格式结果。"""

        await self.append_chat_history({"role": "user", "content": prompt})

        response = await self._chat(
            history=self.chat_history,
            agent_name=self.__class__.__name__,
            sub_title="问题类型分类",
        )

        response_content = response.content or ""
        assistant_msg: dict = {"role": "assistant", "content": response_content}
        if response.reasoning_content:
            assistant_msg["reasoning_content"] = response.reasoning_content
        await self.append_chat_history(assistant_msg)

        if self.diagnostic_logger:
            await self.diagnostic_logger.log_interaction(
                agent_name=self.__class__.__name__,
                sub_title="问题类型分类",
                messages=self.chat_history[:-1],
                response_content=response_content,
                response_reasoning=response.reasoning_content,
            )

        return self._parse_response(response_content)

    def _parse_response(self, response: str) -> ProblemTypeReport:
        """解析 LLM JSON 响应。"""
        json_str = response.strip()
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        json_str = re.sub(r'\[thinking\].*?\[/thinking\]', '', json_str, flags=re.DOTALL)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("ProblemTypeAgent: JSON 解析失败，返回空报告")
            return ProblemTypeReport()

        report = ProblemTypeReport(
            overall_data_characteristics=data.get("overall_data_characteristics", ""),
            cross_problem_dependencies=data.get("cross_problem_dependencies", ""),
        )

        for q_id, q_data in data.get("sub_problem_types", {}).items():
            sp_data = q_data.get("sparsity_report", {})
            sp = SparsityReport(
                observations_per_subject=sp_data.get("observations_per_subject", 0),
                total_samples=sp_data.get("total_samples", 0),
                sparsity_level=sp_data.get("sparsity_level", "moderate"),
                has_censoring=sp_data.get("has_censoring", False),
                censoring_type=sp_data.get("censoring_type", ""),
                censoring_rate=sp_data.get("censoring_rate", 0),
                has_repeated_measures=sp_data.get("has_repeated_measures", False),
                recommendation=sp_data.get("recommendation", ""),
                rationale=sp_data.get("rationale", ""),
            )
            report.sub_problem_types[q_id] = ProblemTypeInfo(
                question_id=q_id,
                primary_type=q_data.get("primary_type", ""),
                sub_types=q_data.get("sub_types", []),
                key_indicators=q_data.get("key_indicators", []),
                censoring_detected=q_data.get("censoring_detected", False),
                censoring_details=q_data.get("censoring_details", ""),
                sparsity_report=sp,
                recommended_method_family=q_data.get("recommended_method_family", []),
                anti_patterns=q_data.get("anti_patterns", []),
                confidence=q_data.get("confidence", 0.0),
            )

        return report


def problem_type_report_to_text(report: ProblemTypeReport) -> str:
    """将 ProblemTypeReport 转换为人类可读文本，用于 prompt 注入。"""
    lines = []
    for q_id, pti in report.sub_problem_types.items():
        lines.append(f"### {q_id}")
        lines.append(f"- 问题类型: {pti.primary_type}")
        if pti.sub_types:
            lines.append(f"- 子类型: {', '.join(pti.sub_types)}")
        lines.append(f"- 关键指标: {'; '.join(pti.key_indicators)}")
        if pti.censoring_detected:
            lines.append(f"- 删失检测: {pti.censoring_type}删失, {pti.censoring_details}")
        sp = pti.sparsity_report
        lines.append(f"- 数据稀疏性: {sp.sparsity_level} (每个体 {sp.observations_per_subject:.1f} 次观测)")
        if sp.recommendation:
            lines.append(f"- 推荐: {sp.recommendation}")
        lines.append(f"- 推荐方法族: {', '.join(pti.recommended_method_family)}")
        if pti.anti_patterns:
            lines.append(f"- 禁用方法:")
            for ap in pti.anti_patterns:
                lines.append(f"  - {ap}")
        lines.append(f"- 置信度: {pti.confidence:.0%}")
        lines.append("")
    return "\n".join(lines)
