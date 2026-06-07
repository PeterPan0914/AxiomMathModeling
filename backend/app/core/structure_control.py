"""论文结构与篇幅控制模块。

提供动态章节长度检测、去重校验和结构化反馈生成功能，
确保 AI 生成的论文篇幅接近优秀论文标准（~25000字）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.utils.log_util import logger


class SectionStatus(str, Enum):
    """章节篇幅状态。"""
    TOO_SHORT = "过短"
    APPROPRIATE = "适中"
    TOO_LONG = "过长"


@dataclass
class SectionTarget:
    """章节篇幅目标配置。

    Attributes:
        name: 章节名称（用于日志和提示）。
        key: 对应 flows 中的键名。
        min_pct: 最小占比（百分比，如 2.0 表示 2%）。
        max_pct: 最大占比。
        quality_requirements: 该章节的质量要求列表。
        anti_patterns: 该章节的反模式列表（禁止出现的写法）。
    """
    name: str
    key: str
    min_pct: float
    max_pct: float
    quality_requirements: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)


# ── 默认目标（4 题论文，目标 25000 字） ──────────────────────────

DEFAULT_TARGET_LENGTH = 25000

SECTION_TARGETS_4PROBLEM: list[SectionTarget] = [
    SectionTarget(
        name="摘要",
        key="firstPage",
        min_pct=2.0, max_pct=3.0,
        quality_requirements=[
            "不含公式，以洞察驱动",
            "每个问题单独成段，含具体数值结果",
            "总字数 500-750 字",
        ],
        anti_patterns=[
            "不得出现空泛描述（'取得了较好效果'）",
            "不得在摘要中列公式",
        ],
    ),
    SectionTarget(
        name="问题重述",
        key="RepeatQues",
        min_pct=3.0, max_pct=4.0,
        quality_requirements=[
            "问题背景需有文献支撑，体现研究意义",
            "不得逐字复制原题，需用自己的语言概括",
        ],
        anti_patterns=[
            "不得大段复制题目原文",
        ],
    ),
    SectionTarget(
        name="问题分析",
        key="analysisQues",
        min_pct=8.0, max_pct=10.0,
        quality_requirements=[
            "以挑战为导向而非以方案为导向",
            "先分析难点和约束，再引出解题思路",
            "每个问题 400-600 字",
        ],
        anti_patterns=[
            "不得重复问题重述的内容",
            "不得直接给出模型名称而不分析问题本质",
        ],
    ),
    SectionTarget(
        name="模型假设",
        key="modelAssumption",
        min_pct=2.0, max_pct=3.0,
        quality_requirements=[
            "假设需可检验、非泛泛而谈",
            "每条假设说明合理性依据",
        ],
        anti_patterns=[
            "不得出现'假设模型合理'等无意义假设",
        ],
    ),
    SectionTarget(
        name="符号说明",
        key="symbol",
        min_pct=2.0, max_pct=3.0,
        quality_requirements=[
            "符号表完整，覆盖论文所有重要变量",
            "分层组织：全局符号 → 问题专用符号",
            "含符号、含义、单位三列",
        ],
        anti_patterns=[
            "不得遗漏关键变量",
        ],
    ),
    SectionTarget(
        name="数据预处理与探索性分析",
        key="eda",
        min_pct=8.0, max_pct=10.0,
        quality_requirements=[
            "多维度分析：分布、趋势、相关性、异常值",
            "插入 5-8 张图表",
            "含描述性统计表格",
            "字数 2000-2500 字",
        ],
        anti_patterns=[
            "不得只放图表不解读",
            "不得只有数据清洗没有可视化分析",
        ],
    ),
    SectionTarget(
        name="问题一模型",
        key="ques1",
        min_pct=15.0, max_pct=18.0,
        quality_requirements=[
            "含完整推导过程，不能跳步",
            "含模型诊断（残差分析、拟合优度等）",
            "含机制解释（为什么模型有效）",
            "字数 3750-4500 字",
        ],
        anti_patterns=[
            "不得只有公式没有推导",
            "不得只有结果没有分析",
        ],
    ),
    SectionTarget(
        name="问题二模型",
        key="ques2",
        min_pct=15.0, max_pct=18.0,
        quality_requirements=[
            "含优化问题的标准形式（目标函数 + 约束条件）",
            "含求解算法说明（如遗传算法、模拟退火等）",
            "含多场景对比分析",
            "字数 3750-4500 字",
        ],
        anti_patterns=[
            "不得省略约束条件",
            "不得只给最优解而不分析解的性质",
        ],
    ),
    SectionTarget(
        name="问题三模型",
        key="ques3",
        min_pct=15.0, max_pct=18.0,
        quality_requirements=[
            "使用进阶方法（如集成学习、深度学习等）",
            "含特征重要性分析",
            "含模型对比实验",
            "字数 3750-4500 字",
        ],
        anti_patterns=[
            "不得使用过于简单的方法而不解释原因",
        ],
    ),
    SectionTarget(
        name="问题四模型",
        key="ques4",
        min_pct=10.0, max_pct=12.0,
        quality_requirements=[
            "含集成/融合策略说明",
            "含逐类别指标（precision、recall、F1）",
            "含混淆矩阵分析",
            "字数 2500-3000 字",
        ],
        anti_patterns=[
            "不得只报告总体准确率",
        ],
    ),
    SectionTarget(
        name="灵敏度分析",
        key="sensitivity_analysis",
        min_pct=5.0, max_pct=6.0,
        quality_requirements=[
            "多场景分析，非仅单因素（OAT）",
            "含龙卷风图或蜘蛛图",
            "定量说明参数变化对结果的影响幅度",
            "字数 1250-1500 字",
        ],
        anti_patterns=[
            "不得只做定性描述不做定量分析",
        ],
    ),
    SectionTarget(
        name="模型评价",
        key="judge",
        min_pct=3.0, max_pct=4.0,
        quality_requirements=[
            "诚实评价，优缺点并重",
            "优缺点需具体，非泛泛而谈",
            "改进方案需有针对性",
            "字数 750-1000 字",
        ],
        anti_patterns=[
            "不得只写优点不写缺点",
        ],
    ),
]


def _get_target_map() -> dict[str, SectionTarget]:
    """构建 key -> SectionTarget 的映射。"""
    return {t.key: t for t in SECTION_TARGETS_4PROBLEM}


def _count_chinese_chars(text: str) -> int:
    """统计中文字符数（含中文标点）。

    同时计算英文单词数（每词按 1.5 字符折算），
    公式块（$$...$$）按 30 字符/个折算，
    使字符数估计更贴近排版后的实际篇幅。
    """
    # 中文字符 + 中文标点
    chinese = len(re.findall(r'[一-鿿　-〿＀-￯]', text))
    # 英文单词（不含公式内容）
    # 先去掉公式块
    text_no_formula = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text_no_formula = re.sub(r'\$[^$]+\$', '', text_no_formula)
    english_words = len(re.findall(r'[a-zA-Z]+', text_no_formula))
    english_chars = int(english_words * 1.5)
    # 公式块数
    formula_blocks = len(re.findall(r'\$\$.*?\$\$', text, flags=re.DOTALL))
    inline_formulas = len(re.findall(r'\$[^$]+\$', text))
    formula_chars = formula_blocks * 30 + inline_formulas * 10
    # 图表标签
    figures = len(re.findall(r'!\[.*?\]\(.*?\)', text))
    figure_chars = figures * 50

    return chinese + english_chars + formula_chars + figure_chars


def _extract_tables(text: str) -> list[str]:
    """提取 Markdown 表格内容。"""
    tables = re.findall(r'(\|.+\|(?:\n\|.+\|)+)', text)
    return tables


def _extract_figures(text: str) -> list[str]:
    """提取图片引用。"""
    return re.findall(r'!\[.*?\]\((.*?)\)', text)


# ── 公开接口 ────────────────────────────────────────────────────────


@dataclass
class SectionReport:
    """单个章节的篇幅检测报告。

    Attributes:
        section_name: 章节名称。
        key: 章节键名。
        char_count: 实际字符数。
        target_min: 目标最小字符数。
        target_max: 目标最大字符数。
        pct_of_total: 占总目标的百分比。
        status: 篇幅状态。
        feedback: 自动生成的反馈提示。
        quality_issues: 质量问题列表。
        redundancy_issues: 去重问题列表。
    """
    section_name: str
    key: str
    char_count: int
    target_min: int
    target_max: int
    pct_of_total: float
    status: SectionStatus
    feedback: str
    quality_issues: list[str] = field(default_factory=list)
    redundancy_issues: list[str] = field(default_factory=list)


@dataclass
class PaperStructureReport:
    """整篇论文的结构检测报告。

    Attributes:
        total_chars: 总字符数。
        target_length: 目标总字符数。
        section_reports: 各章节报告列表。
        global_issues: 全局问题列表（如重复图表）。
        overall_feedback: 整体反馈。
    """
    total_chars: int
    target_length: int
    section_reports: list[SectionReport]
    global_issues: list[str] = field(default_factory=list)
    overall_feedback: str = ""


class StructureController:
    """论文结构与篇幅控制器。

    提供以下功能：
    1. 单章节篇幅检测与反馈
    2. 全文结构报告
    3. 图表去重检测
    4. 结果分析质量检查（禁止复述表格数据）

    Args:
        target_length: 论文目标总字数（默认 25000）。
        section_targets: 章节目标配置列表（默认使用 4 题论文配置）。
    """

    def __init__(
        self,
        target_length: int = DEFAULT_TARGET_LENGTH,
        section_targets: list[SectionTarget] | None = None,
    ):
        self.target_length = target_length
        self.section_targets = section_targets or SECTION_TARGETS_4PROBLEM
        self._target_map = _get_target_map()
        # 跟踪已出现的图表，用于去重
        self._seen_tables: dict[str, list[str]] = {}  # key -> [table_content, ...]
        self._seen_figures: dict[str, list[str]] = {}  # key -> [figure_ref, ...]

    def check_section(
        self,
        key: str,
        content: str,
        context: str = "",
    ) -> SectionReport:
        """检测单个章节的篇幅和质量。

        Args:
            key: 章节键名（如 "eda", "ques1"）。
            content: 章节文本内容。
            context: 额外上下文（如之前的反馈）。

        Returns:
            SectionReport 对象，包含篇幅状态和反馈。
        """
        target = self._target_map.get(key)
        if target is None:
            # 未知章节，不做限制
            return SectionReport(
                section_name=key,
                key=key,
                char_count=_count_chinese_chars(content),
                target_min=0,
                target_max=999999,
                pct_of_total=0.0,
                status=SectionStatus.APPROPRIATE,
                feedback="",
            )

        char_count = _count_chinese_chars(content)
        target_min = int(self.target_length * target.min_pct / 100)
        target_max = int(self.target_length * target.max_pct / 100)
        pct_of_total = char_count / self.target_length * 100

        # 判断状态
        if char_count < target_min * 0.7:
            status = SectionStatus.TOO_SHORT
        elif char_count > target_max * 1.3:
            status = SectionStatus.TOO_LONG
        else:
            status = SectionStatus.APPROPRIATE

        # 生成反馈
        feedback = self._generate_section_feedback(
            target, char_count, target_min, target_max, status
        )

        # 质量问题检查
        quality_issues = self._check_quality(target, content)

        # 去重检查
        redundancy_issues = self._check_redundancy(key, content)

        report = SectionReport(
            section_name=target.name,
            key=key,
            char_count=char_count,
            target_min=target_min,
            target_max=target_max,
            pct_of_total=pct_of_total,
            status=status,
            feedback=feedback,
            quality_issues=quality_issues,
            redundancy_issues=redundancy_issues,
        )

        logger.info(
            f"[结构控制] {target.name}: {char_count} 字 "
            f"(目标 {target_min}-{target_max}), "
            f"占比 {pct_of_total:.1f}%, 状态={status.value}"
        )

        return report

    def check_full_paper(
        self, sections: dict[str, str]
    ) -> PaperStructureReport:
        """对完整论文进行结构检测。

        Args:
            sections: 章节字典，键为章节名，值为内容。

        Returns:
            PaperStructureReport 对象。
        """
        # 重置去重跟踪
        self._seen_tables.clear()
        self._seen_figures.clear()

        section_reports: list[SectionReport] = []
        total_chars = 0

        for key, content in sections.items():
            report = self.check_section(key, content)
            section_reports.append(report)
            total_chars += report.char_count

        # 全局去重检查
        global_issues = self._check_global_redundancy()

        # 整体反馈
        overall_feedback = self._generate_overall_feedback(
            total_chars, section_reports, global_issues
        )

        paper_report = PaperStructureReport(
            total_chars=total_chars,
            target_length=self.target_length,
            section_reports=section_reports,
            global_issues=global_issues,
            overall_feedback=overall_feedback,
        )

        logger.info(
            f"[结构控制] 全文总字数: {total_chars} / {self.target_length} "
            f"({total_chars / self.target_length * 100:.1f}%)"
        )

        return paper_report

    def get_section_length_hint(self, key: str) -> str:
        """获取章节篇幅提示字符串，用于嵌入写作 prompt。

        Args:
            key: 章节键名。

        Returns:
            格式化的篇幅提示文本。
        """
        target = self._target_map.get(key)
        if target is None:
            return ""
        target_min = int(self.target_length * target.min_pct / 100)
        target_max = int(self.target_length * target.max_pct / 100)
        return (
            f"【篇幅要求】目标 {target_min}-{target_max} 字"
            f"（占全文 {target.min_pct}-{target.max_pct}%）。"
        )

    def get_anti_redundancy_hint(self, key: str) -> str:
        """获取去重提示字符串。

        Args:
            key: 章节键名。

        Returns:
            格式化的去重提示文本。
        """
        target = self._target_map.get(key)
        if target is None:
            return ""
        lines = ["【反冗余要求】"]
        if target.anti_patterns:
            for ap in target.anti_patterns:
                lines.append(f"- {ap}")
        lines.append("- 结果分析不得复述表格数据，必须进行解读和因果分析")
        lines.append("- 引用图表后至少配 3 行独立分析，不得用'同上'省略")
        return "\n".join(lines)

    # ── 内部方法 ────────────────────────────────────────────────────

    def _generate_section_feedback(
        self,
        target: SectionTarget,
        char_count: int,
        target_min: int,
        target_max: int,
        status: SectionStatus,
    ) -> str:
        """根据篇幅状态生成反馈提示。"""
        if status == SectionStatus.TOO_SHORT:
            shortage = target_min - char_count
            return (
                f"此章节「{target.name}」过短（当前 {char_count} 字，"
                f"目标至少 {target_min} 字，缺少 {shortage} 字）。"
                f"需要补充：\n"
                + "\n".join(f"- {req}" for req in target.quality_requirements)
            )
        elif status == SectionStatus.TOO_LONG:
            excess = char_count - target_max
            return (
                f"此章节「{target.name}」过长（当前 {char_count} 字，"
                f"目标最多 {target_max} 字，超出 {excess} 字）。"
                f"需要精简：\n"
                "- 删除重复内容\n"
                "- 合并相似段落\n"
                "- 精简已充分展开的论证\n"
                + "\n".join(f"- {ap}" for ap in target.anti_patterns)
            )
        else:
            return ""

    def _check_quality(
        self, target: SectionTarget, content: str
    ) -> list[str]:
        """检查章节内容质量问题。"""
        issues: list[str] = []

        # 检查结果分析是否只是复述表格
        if target.key.startswith("ques"):
            # 检查是否有表格后跟空洞描述
            table_lines = re.findall(r'\|.+\|', content)
            if table_lines:
                # 检查表格之后是否有实质分析（不含"如表所示""由表可知"之后紧跟句号）
                post_table = re.findall(
                    r'(?:由表|如表|表\d).*?[。.](?:\s*\n|\s*$)', content
                )
                for pt in post_table:
                    if len(pt) < 30:
                        issues.append(
                            f"表格引用后分析过短（'{pt.strip()}'），"
                            "需展开至少3行独立解读"
                        )

        # 检查问题分析是否重复问题重述
        if target.key == "analysisQues":
            # 简单启发：如果出现"题目要求""题目指出"等，可能在复述
            restatement_markers = re.findall(
                r'题目(?:要求|指出|给出|描述|中)', content
            )
            if len(restatement_markers) > 2:
                issues.append(
                    "问题分析章节出现过多题目复述语句，"
                    "应以'挑战/难点'为导向分析问题本质"
                )

        # 检查图片是否有解读
        figure_refs = re.findall(r'!\[.*?\]\(.*?\)\s*\n([\s\S]*?)(?=!\[|$|\n#)', content)
        for ref_text in figure_refs:
            clean = re.sub(r'\s+', '', ref_text)
            if len(clean) < 20:
                issues.append("图片后缺少足够的分析解读（至少3行）")

        return issues

    def _check_redundancy(self, key: str, content: str) -> list[str]:
        """检查章节内部的图表重复。"""
        issues: list[str] = []

        # 提取图片
        figures = _extract_figures(content)
        seen_figs: set[str] = set()
        for fig in figures:
            if fig in seen_figs:
                issues.append(f"图片 {fig} 在本章节中重复引用")
            seen_figs.add(fig)

        # 提取表格（简单去重：比较表头行）
        tables = _extract_tables(content)
        table_headers: set[str] = set()
        for table in tables:
            header = table.split('\n')[0] if '\n' in table else table
            if header in table_headers:
                issues.append(f"表格 '{header[:30]}...' 在本章节中重复")
            table_headers.add(header)

        # 记录到全局跟踪
        self._seen_figures.setdefault(key, []).extend(figures)
        self._seen_tables.setdefault(key, []).extend(
            [t.split('\n')[0] if '\n' in t else t for t in tables]
        )

        return issues

    def _check_global_redundancy(self) -> list[str]:
        """跨章节检查图表重复。"""
        issues: list[str] = []

        # 跨章节图片去重
        all_figures: dict[str, list[str]] = {}
        for key, figs in self._seen_figures.items():
            for fig in figs:
                all_figures.setdefault(fig, []).append(key)

        for fig, keys in all_figures.items():
            if len(keys) > 1:
                issues.append(
                    f"图片 {fig} 在多个章节中重复出现: {', '.join(keys)}"
                )

        # 跨章节表格去重（按表头）
        all_table_headers: dict[str, list[str]] = {}
        for key, headers in self._seen_tables.items():
            for header in headers:
                all_table_headers.setdefault(header, []).append(key)

        for header, keys in all_table_headers.items():
            if len(keys) > 1:
                issues.append(
                    f"表格 '{header[:30]}...' 在多个章节中重复出现: {', '.join(keys)}"
                )

        return issues

    def _generate_overall_feedback(
        self,
        total_chars: int,
        section_reports: list[SectionReport],
        global_issues: list[str],
    ) -> str:
        """生成整篇论文的结构反馈。"""
        lines: list[str] = []

        # 总字数评估
        ratio = total_chars / self.target_length
        if ratio < 0.7:
            lines.append(
                f"全文总字数 {total_chars}，严重不足（目标 {self.target_length}，"
                f"仅达 {ratio * 100:.0f}%）。需要大幅扩充各章节内容。"
            )
        elif ratio < 0.9:
            lines.append(
                f"全文总字数 {total_chars}，偏少（目标 {self.target_length}，"
                f"达 {ratio * 100:.0f}%）。请参考各章节反馈补充内容。"
            )
        elif ratio > 1.3:
            lines.append(
                f"全文总字数 {total_chars}，超出目标较多（目标 {self.target_length}，"
                f"达 {ratio * 100:.0f}%）。请精简冗余内容。"
            )
        else:
            lines.append(
                f"全文总字数 {total_chars}，基本达标（目标 {self.target_length}，"
                f"达 {ratio * 100:.0f}%）。"
            )

        # 列出问题章节
        problem_sections = [
            r for r in section_reports
            if r.status != SectionStatus.APPROPRIATE
        ]
        if problem_sections:
            lines.append("\n### 需要调整的章节：")
            for r in problem_sections:
                lines.append(
                    f"- **{r.section_name}**: {r.status.value} "
                    f"({r.char_count}字, 目标{r.target_min}-{r.target_max}字)"
                )

        # 全局去重问题
        if global_issues:
            lines.append("\n### 全局去重问题：")
            for issue in global_issues:
                lines.append(f"- {issue}")

        return "\n".join(lines)


def get_structure_control_prompt_section() -> str:
    """返回用于嵌入 WriterAgent 系统提示词的结构控制段落。"""
    return """
---

# 论文篇幅与结构控制（强制执行！）

## 目标总篇幅
论文目标总篇幅为 **25000 字**（中文字符），这是优秀论文的典型篇幅。
当前系统生成的论文往往只有 9000 字左右，严重不足。

## 各章节篇幅目标

| 章节 | 占比 | 目标字数 | 核心质量要求 |
|------|------|---------|------------|
| 摘要 | 2-3% | 500-750 | 不含公式，洞察驱动，含具体数值 |
| 问题重述 | 3-4% | 750-1000 | 有文献支撑，非逐字复制 |
| 问题分析 | 8-10% | 2000-2500 | 挑战导向，先分析难点再引出思路 |
| 模型假设 | 2-3% | 500-750 | 可检验假设，说明合理性 |
| 符号说明 | 2-3% | 500-750 | 完整分层，含单位 |
| 数据预处理与EDA | 8-10% | 2000-2500 | 多维度分析，5-8张图表 |
| 问题一模型 | 15-18% | 3750-4500 | 完整推导+诊断+机制 |
| 问题二模型 | 15-18% | 3750-4500 | 优化形式+算法+多场景 |
| 问题三模型 | 15-18% | 3750-4500 | 进阶方法+特征重要性 |
| 问题四模型 | 10-12% | 2500-3000 | 集成策略+逐类指标 |
| 灵敏度分析 | 5-6% | 1250-1500 | 多场景+龙卷风图 |
| 模型评价 | 3-4% | 750-1000 | 诚实具体，优缺点并重 |
| 参考文献 | 1-2% | 250-500 | 8-12篇引用 |

## 写作时的篇幅控制规则

### 规则一：每个章节必须达到最低字数
- 在撰写每个章节前，先确认该章节的目标字数范围
- 撰写完成后，估算字数是否达标
- 如果不达标，必须补充内容后再输出

### 规则二：充实内容的具体方法（针对过短章节）

**模型章节扩充策略（目标 3750-4500 字）：**
1. **推导过程**：从基本原理出发，逐步推导，不跳步（约 800-1000 字）
2. **公式解读**：每个重要公式后用"其中"详细说明每个变量（约 300-500 字）
3. **模型机制**：解释为什么该模型适合此问题，物理/数学直觉是什么（约 500-800 字）
4. **求解步骤**：详细描述算法流程或数值方法（约 500-800 字）
5. **结果分析**：深入解读计算结果，含趋势、对比、异常值讨论（约 500-800 字）
6. **图表解读**：每张图表配至少3行分析（约 300-500 字）

**EDA 章节扩充策略（目标 2000-2500 字）：**
1. 数据来源和背景介绍（200-300 字）
2. 数据质量评估：缺失值、异常值、数据类型（300-500 字）
3. 描述性统计：关键变量的均值、中位数、标准差、分布特征（300-500 字）
4. 单变量分析：每个关键变量的分布图解读（400-600 字）
5. 多变量分析：相关性、散点图、分组对比（400-600 字）
6. 数据预处理决策及理由（200-300 字）

**问题分析扩充策略（目标 2000-2500 字）：**
1. 每个问题 500-600 字
2. 第一段：问题类型、数学本质、核心难点（150-200 字）
3. 第二段：约束条件、边界条件、特殊情形（150-200 字）
4. 第三段：解题思路、模型选择理由、与备选方案对比（150-200 字）

### 规则三：反冗余要求（严格遵守！）

**禁止重复图表：**
- 同一张图片不得在多个章节中重复插入
- 同一个表格不得在多个章节中重复出现
- 如需引用其他章节的图表，使用"如第X章图Y所示"文字引用，不重复插入

**禁止复述表格数据：**
- 错误示例："由表3可知，模型准确率为95.2%，召回率为92.1%，F1值为93.6%。"
- 正确示例："由表3可知，模型在各类别上均表现出较高的识别精度。其中类别A的F1值最高（93.6%），表明该类别的特征最为显著；而类别C的召回率相对较低（88.3%），可能源于该类样本量不足导致的欠拟合。"

**禁止问题分析重复问题重述：**
- 问题重述：概括题目要求
- 问题分析：分析解题难点和思路，不重复题目内容

### 规则四：自查清单
在输出每个章节前，对照检查：
- [ ] 字数是否达到目标范围？
- [ ] 公式后是否有"其中"说明？
- [ ] 图表引用后是否有3行以上分析？
- [ ] 是否有与已写章节重复的图表？
- [ ] 结果分析是否解读而非复述？
"""


def get_reviewer_structure_check_prompt() -> str:
    """返回用于嵌入 ReviewAgent 提示词的结构检查段落。"""
    return """
---

# 结构与篇幅检查（评审必查项）

## 篇幅检查
检查论文章节是否达到以下字数要求：

| 章节 | 最低字数 | 最高字数 |
|------|---------|---------|
| 摘要 | 500 | 750 |
| 问题重述 | 750 | 1000 |
| 问题分析 | 2000 | 2500 |
| 模型假设 | 500 | 750 |
| 符号说明 | 500 | 750 |
| 数据预处理 | 2000 | 2500 |
| 每个问题模型 | 3750 | 4500 |
| 灵敏度分析 | 1250 | 1500 |
| 模型评价 | 750 | 1000 |

若章节明显过短，扣分并注明"此章节需扩充：[具体建议]"。

## 去重检查
1. **重复图表**：检查是否有相同的图片或表格在多个章节出现
2. **复述数据**：检查结果分析是否只是复述表格数值而缺乏解读
3. **重复内容**：检查问题分析是否重复了问题重述的内容

发现重复时，在改进建议中具体指出：
- "图X在第Y章和第Z章重复出现，请删除一处"
- "第N节结果分析仅复述表格数据，缺乏因果分析"

## 结构完整性检查
- 每个公式后是否有"其中"变量说明？
- 每张图表后是否有至少3行分析？
- 参考文献是否为 8-12 篇？
- 各章节之间是否有合理过渡？
"""
