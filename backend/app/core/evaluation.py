"""质量评估模块，提供论文质量评估和跟踪功能。"""

from dataclasses import dataclass, field
from typing import Optional
from app.utils.log_util import logger


@dataclass
class QualityScore:
    """质量评分数据结构。"""
    math_score: int = 0  # 数学正确性 (0-25)
    logic_score: int = 0  # 逻辑连贯性 (0-25)
    language_score: int = 0  # 语言质量 (0-25)
    format_score: int = 0  # 格式规范 (0-25)
    overall_score: int = 0  # 总分 (0-100)

    def __post_init__(self):
        """自动计算总分。"""
        if self.overall_score == 0:
            self.overall_score = (
                self.math_score
                + self.logic_score
                + self.language_score
                + self.format_score
            )

    @property
    def is_passing(self) -> bool:
        """是否达到质量标准（>= 80 分）。"""
        return self.overall_score >= 80

    @property
    def weakest_dimension(self) -> tuple[str, int]:
        """返回最弱的维度和分数。"""
        dimensions = {
            '数学正确性': self.math_score,
            '逻辑连贯性': self.logic_score,
            '语言质量': self.language_score,
            '格式规范': self.format_score,
        }
        return min(dimensions.items(), key=lambda x: x[1])

    @property
    def strongest_dimension(self) -> tuple[str, int]:
        """返回最强的维度和分数。"""
        dimensions = {
            '数学正确性': self.math_score,
            '逻辑连贯性': self.logic_score,
            '语言质量': self.language_score,
            '格式规范': self.format_score,
        }
        return max(dimensions.items(), key=lambda x: x[1])

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            'math_score': self.math_score,
            'logic_score': self.logic_score,
            'language_score': self.language_score,
            'format_score': self.format_score,
            'overall_score': self.overall_score,
            'is_passing': self.is_passing,
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

    def add_report(self, report: QualityReport):
        """添加质量报告。

        Args:
            report: 质量报告。
        """
        if report.section_name not in self.reports:
            self.reports[report.section_name] = []
        self.reports[report.section_name].append(report)

        logger.info(
            f"QualityTracker: {report.section_name} 第 {report.iteration} 轮 "
            f"得分 {report.score.overall_score}/100"
        )

    def get_latest_score(self, section_name: str) -> Optional[QualityScore]:
        """获取指定章节的最新评分。

        Args:
            section_name: 章节名称。

        Returns:
            最新评分，如果不存在则返回 None。
        """
        if section_name not in self.reports or not self.reports[section_name]:
            return None
        return self.reports[section_name][-1].score

    def get_improvement_trend(self, section_name: str) -> list[int]:
        """获取指定章节的评分变化趋势。

        Args:
            section_name: 章节名称。

        Returns:
            评分列表。
        """
        if section_name not in self.reports:
            return []
        return [report.score.overall_score for report in self.reports[section_name]]

    def get_total_iterations(self, section_name: str) -> int:
        """获取指定章节的总迭代次数。

        Args:
            section_name: 章节名称。

        Returns:
            迭代次数。
        """
        if section_name not in self.reports:
            return 0
        return len(self.reports[section_name])

    def get_summary(self) -> dict:
        """获取质量跟踪摘要。

        Returns:
            摘要字典。
        """
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


# 全局质量跟踪器实例
quality_tracker = QualityTracker()
