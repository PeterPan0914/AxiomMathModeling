"""领域知识规则库：ICC、R²、AIC/BIC 的合理性检查规则。

获奖论文的标准：
- ICC > 0.5（说明个体差异被充分捕捉）
- ICC 目标值 > 0.7（优秀水平，如获奖论文的 0.743）
- R²_conditional > R²_marginal（随机效应确实贡献了额外解释力）
- AIC 差异 > 2 才有实质意义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from app.utils.log_util import logger


class Severity(str, Enum):
    """问题严重程度。"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class RuleViolation:
    """单条规则违反记录。"""
    rule_id: str = ""
    metric_name: str = ""
    actual_value: float = 0.0
    expected_range: str = ""
    severity: Severity = Severity.INFO
    message: str = ""
    recommendation: str = ""


@dataclass
class SanityCheckReport:
    """合理性检查报告。"""
    violations: list[RuleViolation] = field(default_factory=list)
    overall_status: str = "pass"  # pass / warning / fail
    score_penalty: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.WARNING)

    def to_text(self) -> str:
        if not self.violations:
            return "所有指标检查通过，无异常。"

        lines = [f"## 合理性检查报告 (状态: {self.overall_status}, 扣分: {self.score_penalty})"]
        for v in self.violations:
            icon = {"critical": "!!!", "warning": "!!", "info": "i"}[v.severity.value]
            lines.append(f"[{icon}] {v.rule_id}: {v.message}")
            lines.append(f"     指标: {v.metric_name} = {v.actual_value:.4f}, 期望: {v.expected_range}")
            lines.append(f"     建议: {v.recommendation}")
        return "\n".join(lines)


def check_icc(icc_value: float) -> list[RuleViolation]:
    """检查 ICC（组内相关系数）的合理性。"""
    violations = []

    if icc_value < 0.1:
        violations.append(RuleViolation(
            rule_id="ICC_LOW",
            metric_name="ICC",
            actual_value=icc_value,
            expected_range="> 0.1",
            severity=Severity.WARNING,
            message=f"ICC = {icc_value:.4f}，组间差异极小。混合效应模型可能不必要。",
            recommendation="检查数据是否有真正的层次结构。如果没有，改用 OLS。",
        ))
    elif icc_value < 0.3:
        violations.append(RuleViolation(
            rule_id="ICC_MODERATE",
            metric_name="ICC",
            actual_value=icc_value,
            expected_range="> 0.5（获奖论文标准）",
            severity=Severity.WARNING,
            message=f"ICC = {icc_value:.4f}，低于获奖论文标准（0.743）。模型可能未充分捕捉个体差异。",
            recommendation="尝试增加随机效应项（随机斜率）、或添加更多个体层面的预测变量。",
        ))
    elif icc_value < 0.5:
        violations.append(RuleViolation(
            rule_id="ICC_BELOW_TARGET",
            metric_name="ICC",
            actual_value=icc_value,
            expected_range="> 0.5（目标）",
            severity=Severity.INFO,
            message=f"ICC = {icc_value:.4f}，中等水平。AI 论文的典型 ICC（0.35）即在此区间。",
            recommendation="尝试添加个体层面的协变量或交互项以提高 ICC。",
        ))

    if icc_value > 0.95:
        violations.append(RuleViolation(
            rule_id="ICC_TOO_HIGH",
            metric_name="ICC",
            actual_value=icc_value,
            expected_range="< 0.95",
            severity=Severity.WARNING,
            message=f"ICC = {icc_value:.4f}，极高。可能是数据泄露或过度拟合。",
            recommendation="检查是否存在数据泄露。",
        ))

    return violations


def check_r_squared(
    r2_marginal: float,
    r2_conditional: float,
) -> list[RuleViolation]:
    """检查 R² 的合理性。"""
    violations = []

    if r2_marginal < 0:
        violations.append(RuleViolation(
            rule_id="R2_NEGATIVE",
            metric_name="R²_marginal",
            actual_value=r2_marginal,
            expected_range="[0, 1]",
            severity=Severity.CRITICAL,
            message="R²_marginal 为负数，模型拟合极差。",
            recommendation="检查数据预处理、模型公式是否正确。",
        ))

    if r2_conditional < r2_marginal:
        violations.append(RuleViolation(
            rule_id="R2_INCONSISTENT",
            metric_name="R²_conditional < R²_marginal",
            actual_value=r2_conditional,
            expected_range=f">= R²_marginal ({r2_marginal:.4f})",
            severity=Severity.CRITICAL,
            message="R²_conditional < R²_marginal，不一致。",
            recommendation="检查 R² 的计算公式。",
        ))

    if r2_conditional > 0.99:
        violations.append(RuleViolation(
            rule_id="R2_TOO_HIGH",
            metric_name="R²_conditional",
            actual_value=r2_conditional,
            expected_range="< 0.99",
            severity=Severity.WARNING,
            message=f"R²_conditional = {r2_conditional:.4f}，极高。可能存在过拟合。",
            recommendation="用交叉验证确认泛化性能。",
        ))

    return violations


def run_full_sanity_check(
    icc: float = 0.0,
    r2_marginal: float = 0.0,
    r2_conditional: float = 0.0,
) -> SanityCheckReport:
    """执行完整的合理性检查。"""
    all_violations: list[RuleViolation] = []
    all_violations.extend(check_icc(icc))
    all_violations.extend(check_r_squared(r2_marginal, r2_conditional))

    has_critical = any(v.severity == Severity.CRITICAL for v in all_violations)
    has_warning = any(v.severity == Severity.WARNING for v in all_violations)

    if has_critical:
        overall_status = "fail"
        score_penalty = 15
    elif has_warning:
        overall_status = "warning"
        score_penalty = 5
    else:
        overall_status = "pass"
        score_penalty = 0

    return SanityCheckReport(
        violations=all_violations,
        overall_status=overall_status,
        score_penalty=score_penalty,
    )
