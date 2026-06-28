"""质量评估模块，提供论文质量评估和跟踪功能。

评分体系统一为三维度 0-100 分制：
- method_score (0-40): 方法论质量（模型选择、假设检验、推导完整性、验证充分性）
- writing_score (0-30): 写作质量（逻辑连贯、语言质量、论证深度）
- format_score (0-30): 格式规范（公式编号、图表规范、参考文献）

总分 = method_score + writing_score + format_score (0-100)

新增 MandatoryMinimums 机制：方法分低于阈值时强制重写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from app.utils.log_util import logger


# ---------------------------------------------------------------------------
# 评分维度权重常量
# ---------------------------------------------------------------------------

SCORE_WEIGHT_METHOD: float = 0.40
SCORE_WEIGHT_WRITING: float = 0.30
SCORE_WEIGHT_FORMAT: float = 0.30

# MandatoryMinimums: 低于此阈值必须重写
MANDATORY_METHOD_FLOOR: int = 24      # 40 * 0.6
MANDATORY_WRITING_FLOOR: int = 18     # 30 * 0.6
MANDATORY_FORMAT_FLOOR: int = 18      # 30 * 0.6
MANDATORY_COMPOSITE_FLOOR: int = 60   # 总分 60 分


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def clamp_score(value: int, dimension: str, max_score: int = 100) -> int:
    """将分数限制在 [0, max_score] 范围内。

    Args:
        value: 原始分数。
        dimension: 维度名称（用于日志）。
        max_score: 满分值。

    Returns:
        限制后的分数。
    """
    if value < 0:
        logger.warning(f"[评分] {dimension}={value} < 0，修正为 0")
        return 0
    if value > max_score:
        logger.warning(f"[评分] {dimension}={value} > {max_score}，修正为 {max_score}")
        return max_score
    return value


@dataclass
class PassFailResult:
    """MandatoryMinimums 检查结果。"""
    is_passing: bool = True
    reason: str = ""
    action: str = "PASS"  # PASS / TARGETED_IMPROVEMENT / REWRITE_METHOD_SECTIONS / REJECT_AND_REWRITE_ALL

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_passing": self.is_passing,
            "reason": self.reason,
            "action": self.action,
        }


class MandatoryMinimums:
    """强制最低分检查。"""

    @classmethod
    def check_passing(
        cls,
        method_score: int,
        writing_score: int,
        format_score: int,
    ) -> PassFailResult:
        """检查是否通过强制最低分。

        Args:
            method_score: 方法论分数 (0-40)。
            writing_score: 写作分数 (0-30)。
            format_score: 格式分数 (0-30)。

        Returns:
            PassFailResult 对象。
        """
        composite = method_score + writing_score + format_score

        # 检查总分
        if composite < MANDATORY_COMPOSITE_FLOOR:
            return PassFailResult(
                is_passing=False,
                reason=f"总分 {composite} < {MANDATORY_COMPOSITE_FLOOR}",
                action="REWRITE_METHOD_SECTIONS",
            )

        # 检查方法论分数
        if method_score < MANDATORY_METHOD_FLOOR:
            return PassFailResult(
                is_passing=False,
                reason=f"方法论分 {method_score} < {MANDATORY_METHOD_FLOOR}",
                action="REWRITE_METHOD_SECTIONS",
            )

        return PassFailResult(is_passing=True, action="PASS")


# ---------------------------------------------------------------------------
# 改进追踪
# ---------------------------------------------------------------------------

@dataclass
class ImprovementRecord:
    """改进记录，追踪每轮改进的详情。"""
    iteration: int = 0
    section_key: str = ""
    target_issues: list[str] = field(default_factory=list)
    changes_made: list[str] = field(default_factory=list)
    score_before: dict[str, int] = field(default_factory=dict)
    score_after: dict[str, int] = field(default_factory=dict)
    content_length_before: int = 0
    content_length_after: int = 0
    is_substantive_change: bool = True
    is_score_improved: bool = False
    regressions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "section_key": self.section_key,
            "target_issues": self.target_issues,
            "changes_made": self.changes_made,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "is_substantive_change": self.is_substantive_change,
            "is_score_improved": self.is_score_improved,
            "regressions": self.regressions,
        }


@dataclass
class RegressionAlert:
    """回归告警，当某维度分数下降时触发。"""
    dimension: str = ""
    score_before: int = 0
    score_after: int = 0
    drop_amount: int = 0
    severity: str = "warning"  # warning / critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "drop_amount": self.drop_amount,
            "severity": self.severity,
        }


def detect_regressions(
    scores_before: dict[str, int],
    scores_after: dict[str, int],
    threshold: int = 5,
) -> list[RegressionAlert]:
    """检测分数回归。

    Args:
        scores_before: 改进前的分数。
        scores_after: 改进后的分数。
        threshold: 下降阈值（默认 5 分）。

    Returns:
        回归告警列表。
    """
    alerts = []
    for dim in scores_before:
        before = scores_before.get(dim, 0)
        after = scores_after.get(dim, 0)
        drop = before - after
        if drop >= threshold:
            severity = "critical" if drop >= 10 else "warning"
            alerts.append(RegressionAlert(
                dimension=dim,
                score_before=before,
                score_after=after,
                drop_amount=drop,
                severity=severity,
            ))
    return alerts


def verify_substantive_change(
    content_before: str,
    content_after: str,
) -> tuple[bool, str]:
    """验证改进是否为实质性变化。

    Args:
        content_before: 改进前的内容。
        content_after: 改进后的内容。

    Returns:
        (is_substantive, reason) 元组。
    """
    if not content_before or not content_after:
        return True, "内容为空，无法比较"

    # 检查长度变化
    len_before = len(content_before)
    len_after = len(content_after)
    ratio = len_after / len_before if len_before > 0 else 1.0

    if ratio < 0.3:
        return False, f"内容大幅缩短（{ratio:.0%}），可能是删除而非改进"

    if ratio > 3.0:
        return False, f"内容大幅膨胀（{ratio:.0%}），可能是堆砌而非改进"

    # 检查内容相似度（比较行差异）
    lines_before = set(content_before.split('\n'))
    lines_after = set(content_after.split('\n'))
    if lines_before and lines_after:
        union = lines_before | lines_after
        common = lines_before & lines_after
        if len(union) > 0:
            similarity = len(common) / len(union)
            if similarity > 0.95:
                return False, f"内容相似度 {similarity:.1%}，属于表面修改"

    return True, "实质性变化"


# ---------------------------------------------------------------------------
# QualityScore: 统一三维度 0-100 分制
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """质量评分数据结构（统一三维度 0-100 分制）。

    - method_score (0-40): 方法论质量
    - writing_score (0-30): 写作质量
    - format_score (0-30): 格式规范
    - overall_score (0-100): 总分（自动计算）
    """
    method_score: int = 0    # 方法论 (0-40)
    writing_score: int = 0   # 写作 (0-30)
    format_score: int = 0    # 格式 (0-30)
    overall_score: int = 0   # 总分 (0-100)

    # 兼容旧字段（映射到新维度）
    math_score: int = 0      # 已废弃，映射到 method_score
    logic_score: int = 0     # 已废弃，映射到 writing_score
    language_score: int = 0  # 已废弃，映射到 writing_score

    def __post_init__(self):
        """自动计算总分，兼容旧字段映射。"""
        # 兼容旧字段映射
        if self.method_score == 0 and self.math_score > 0:
            self.method_score = self.math_score
        if self.writing_score == 0 and (self.logic_score > 0 or self.language_score > 0):
            self.writing_score = self.logic_score + self.language_score

        # 分数限制
        self.method_score = clamp_score(self.method_score, "method_score", 40)
        self.writing_score = clamp_score(self.writing_score, "writing_score", 30)
        self.format_score = clamp_score(self.format_score, "format_score", 30)

        # 自动计算总分
        if self.overall_score == 0:
            self.overall_score = self.method_score + self.writing_score + self.format_score

    @property
    def is_passing(self) -> bool:
        """是否达到质量标准。"""
        result = MandatoryMinimums.check_passing(
            self.method_score, self.writing_score, self.format_score
        )
        return result.is_passing

    @property
    def weakest_dimension(self) -> tuple[str, int]:
        """返回最弱的维度和分数。"""
        dimensions = {
            '方法论': self.method_score,
            '写作': self.writing_score,
            '格式': self.format_score,
        }
        return min(dimensions.items(), key=lambda x: x[1])

    @property
    def strongest_dimension(self) -> tuple[str, int]:
        """返回最强的维度和分数。"""
        dimensions = {
            '方法论': self.method_score,
            '写作': self.writing_score,
            '格式': self.format_score,
        }
        return max(dimensions.items(), key=lambda x: x[1])

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            'method_score': self.method_score,
            'writing_score': self.writing_score,
            'format_score': self.format_score,
            'overall_score': self.overall_score,
            'is_passing': self.is_passing,
            'weakest_dimension': self.weakest_dimension[0],
        }


@dataclass
class QualityReport:
    """质量报告数据结构。"""
    section_name: str
    score: QualityScore
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    feedback: str = ""
    iteration: int = 1

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            'section_name': self.section_name,
            'score': self.score.to_dict(),
            'strengths': self.strengths,
            'improvements': self.improvements,
            'feedback': self.feedback,
            'iteration': self.iteration,
        }


class QualityTracker:
    """质量跟踪器，记录和分析质量变化。"""

    def __init__(self):
        self.reports: dict[str, list[QualityReport]] = {}
        self.improvement_log: list[ImprovementRecord] = []

    def add_report(self, report: QualityReport):
        """添加质量报告。"""
        if report.section_name not in self.reports:
            self.reports[report.section_name] = []
        self.reports[report.section_name].append(report)

        logger.info(
            f"QualityTracker: {report.section_name} 第 {report.iteration} 轮 "
            f"得分 {report.score.overall_score}/100 "
            f"(方法{report.score.method_score} 写作{report.score.writing_score} 格式{report.score.format_score})"
        )

    def add_improvement(self, record: ImprovementRecord):
        """添加改进记录。"""
        self.improvement_log.append(record)

    def get_latest_score(self, section_name: str) -> QualityScore | None:
        """获取指定章节的最新评分。"""
        if section_name not in self.reports or not self.reports[section_name]:
            return None
        return self.reports[section_name][-1].score

    def get_improvement_trend(self, section_name: str) -> list[int]:
        """获取指定章节的评分变化趋势。"""
        if section_name not in self.reports:
            return []
        return [report.score.overall_score for report in self.reports[section_name]]

    def get_total_iterations(self, section_name: str) -> int:
        """获取指定章节的总迭代次数。"""
        if section_name not in self.reports:
            return 0
        return len(self.reports[section_name])

    def get_summary(self) -> dict:
        """获取质量跟踪摘要。"""
        summary = {}
        for section_name, reports in self.reports.items():
            scores = [r.score.overall_score for r in reports]
            summary[section_name] = {
                'iterations': len(reports),
                'initial_score': scores[0] if scores else 0,
                'final_score': scores[-1] if scores else 0,
                'improvement': scores[-1] - scores[0] if len(scores) > 1 else 0,
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
            }
        return summary

    def print_summary(self):
        """打印质量跟踪摘要。"""
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("质量跟踪摘要")
        logger.info("=" * 60)
        for section_name, stats in summary.items():
            logger.info(f"\n{section_name}:")
            logger.info(f"  迭代次数: {stats['iterations']}")
            logger.info(f"  初始分数: {stats['initial_score']}/100")
            logger.info(f"  最终分数: {stats['final_score']}/100")
            logger.info(f"  提升幅度: {stats['improvement']} 分")
            logger.info(f"  最高分数: {stats['max_score']}/100")
            logger.info(f"  最低分数: {stats['min_score']}/100")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Checklist 评分系统（10 维布尔值检查）
# ---------------------------------------------------------------------------

@dataclass
class ChecklistItem:
    """单个检查项。"""
    id: str = ""
    category: str = ""  # method / writing / format
    description: str = ""
    passed: bool = False
    evidence: str = ""
    severity: str = "MAJOR"  # CRITICAL / MAJOR / MINOR


@dataclass
class ChecklistScore:
    """10 维布尔值 Checklist 评分。"""
    items: list[ChecklistItem] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.items:
            return 0.0
        return sum(1 for item in self.items if item.passed) / len(self.items)

    @property
    def critical_failures(self) -> list[ChecklistItem]:
        return [item for item in self.items
                if not item.passed and item.severity == "CRITICAL"]

    @property
    def has_fatal(self) -> bool:
        return len(self.critical_failures) > 0

    def to_legacy_score(self) -> int:
        """转换为百分制（兼容现有系统）。"""
        base = self.pass_rate * 100
        critical_penalty = len(self.critical_failures) * 15
        return max(0, min(100, int(base - critical_penalty)))


# 标准 Checklist 定义
DEFAULT_CHECKLIST = [
    ChecklistItem(id="M1", category="method", description="每个模型假设是否有统计检验支撑", severity="CRITICAL"),
    ChecklistItem(id="M2", category="method", description="是否有至少2种不同方法族的对比表格", severity="CRITICAL"),
    ChecklistItem(id="M3", category="method", description="公式是否正确，推导是否完整", severity="CRITICAL"),
    ChecklistItem(id="M4", category="method", description="数值是否在合理范围（R², 残差等）", severity="MAJOR"),
    ChecklistItem(id="M5", category="method", description="是否有六维度鲁棒性分析", severity="MAJOR"),
    ChecklistItem(id="W1", category="writing", description="每张图表是否有三段式论证（观察→含义→处置）", severity="MAJOR"),
    ChecklistItem(id="W2", category="writing", description="论证链是否完整，无逻辑跳跃", severity="MAJOR"),
    ChecklistItem(id="W3", category="writing", description="是否使用学术中文，无口语化/AI味", severity="MINOR"),
    ChecklistItem(id="F1", category="format", description="公式编号是否连续、无重复", severity="MAJOR"),
    ChecklistItem(id="F2", category="format", description="参考文献是否存在且格式规范", severity="MINOR"),
]




