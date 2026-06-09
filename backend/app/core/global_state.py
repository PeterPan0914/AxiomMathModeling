"""全局状态模块，替代 PaperContext 作为整个任务的共享大脑。

所有 Agent 读取和更新此状态，实现真正的共享上下文。
每个 Agent 调用时获得的不是"上一个 Agent 的输出文本"，
而是整个任务到目前为止所有决策的结构化记录。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ============================================================
# 一、TaskMeta：任务元信息
# ============================================================


@dataclass
class TaskMeta:
    """任务元信息，贯穿整个任务生命周期。"""

    task_id: str = ""
    competition_type: str = ""          # 国赛 / 美赛 / 其他
    problem_year: str = ""              # 题目年份，如 "2024"
    problem_type: str = ""              # 连续型 / 离散型 / 优化型 等
    deadline_hours: float = 72.0        # 截止时间（小时）
    current_phase: str = "init"         # 当前阶段标识
    created_at: str = ""                # ISO 格式时间戳

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "competition_type": self.competition_type,
            "problem_year": self.problem_year,
            "problem_type": self.problem_type,
            "deadline_hours": self.deadline_hours,
            "current_phase": self.current_phase,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskMeta:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# 二、ProblemUnderstanding：题目理解
# ============================================================


@dataclass
class DataInventory:
    """数据资产清单，描述附件数据的基本特征。"""

    files: list[str] = field(default_factory=list)
    sample_size: dict[str, int] = field(default_factory=dict)     # {文件名: 样本量}
    variable_types: dict[str, int] = field(default_factory=dict)  # {类型: 数量}
    known_issues: list[str] = field(default_factory=list)         # 已知数据问题
    data_range_checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "sample_size": self.sample_size,
            "variable_types": self.variable_types,
            "known_issues": self.known_issues,
            "data_range_checks": self.data_range_checks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataInventory:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SubProblem:
    """子问题定义，包含 DAG 依赖关系。"""

    id: str = ""                          # 如 "Q1"
    description: str = ""                 # 问题描述
    core_challenge: str = ""              # 核心难点
    dependency_on: list[str] = field(default_factory=list)        # 依赖哪些子问题
    provides_to: list[str] = field(default_factory=list)          # 提供给哪些子问题
    recommended_method_family: list[str] = field(default_factory=list)
    forbidden_methods: list[str] = field(default_factory=list)    # 禁用方法及原因
    key_outputs_needed: list[str] = field(default_factory=list)
    examiner_red_flags: list[str] = field(default_factory=list)   # 评委关注的雷区

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "core_challenge": self.core_challenge,
            "dependency_on": self.dependency_on,
            "provides_to": self.provides_to,
            "recommended_method_family": self.recommended_method_family,
            "forbidden_methods": self.forbidden_methods,
            "key_outputs_needed": self.key_outputs_needed,
            "examiner_red_flags": self.examiner_red_flags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubProblem:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LiteratureContext:
    """文献调研上下文。"""

    existing_approaches: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    innovation_space: str = ""            # 可以在哪里创新

    def to_dict(self) -> dict[str, Any]:
        return {
            "existing_approaches": self.existing_approaches,
            "known_limitations": self.known_limitations,
            "innovation_space": self.innovation_space,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LiteratureContext:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProblemUnderstanding:
    """题目理解，由 ProblemAnalystAgent 和 CoordinatorAgent 共同构建。"""

    raw_text: str = ""                              # 原始题目文本
    core_question: str = ""                         # 题目本质问题（一句话）
    examiner_intent: str = ""                       # 出题人想考察什么能力
    scoring_priorities: list[str] = field(default_factory=list)
    data_inventory: DataInventory = field(default_factory=DataInventory)
    sub_problems: list[SubProblem] = field(default_factory=list)
    inter_problem_dependencies: str = ""            # 子问题间的依赖描述
    literature_context: LiteratureContext = field(default_factory=LiteratureContext)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "core_question": self.core_question,
            "examiner_intent": self.examiner_intent,
            "scoring_priorities": self.scoring_priorities,
            "data_inventory": self.data_inventory.to_dict(),
            "sub_problems": [sp.to_dict() for sp in self.sub_problems],
            "inter_problem_dependencies": self.inter_problem_dependencies,
            "literature_context": self.literature_context.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProblemUnderstanding:
        return cls(
            raw_text=d.get("raw_text", ""),
            core_question=d.get("core_question", ""),
            examiner_intent=d.get("examiner_intent", ""),
            scoring_priorities=d.get("scoring_priorities", []),
            data_inventory=DataInventory.from_dict(d.get("data_inventory", {})),
            sub_problems=[SubProblem.from_dict(sp) for sp in d.get("sub_problems", [])],
            inter_problem_dependencies=d.get("inter_problem_dependencies", ""),
            literature_context=LiteratureContext.from_dict(d.get("literature_context", {})),
        )


# ============================================================
# 三、ModelingDecision：建模决策
# ============================================================


@dataclass
class KeyAssumption:
    """模型的关键假设及其验证情况。"""

    assumption: str = ""                  # 假设内容
    justification: str = ""               # 为什么做出这个假设
    test_used: str = ""                   # 用了什么检验方法
    test_result: str = ""                 # 检验结果

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumption": self.assumption,
            "justification": self.justification,
            "test_used": self.test_used,
            "test_result": self.test_result,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KeyAssumption:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CandidateEvaluation:
    """候选方法的评估记录。"""

    method: str = ""
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    applicability_check: dict[str, str] = field(default_factory=dict)  # {检验项: 结果}
    verdict: str = ""                     # "选用" / "排除" / "备选"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "pros": self.pros,
            "cons": self.cons,
            "applicability_check": self.applicability_check,
            "verdict": self.verdict,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateEvaluation:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelingDecision:
    """某个子问题的完整建模决策。"""

    candidates_evaluated: list[CandidateEvaluation] = field(default_factory=list)
    final_choice: str = ""                # 最终选择的方法名
    key_assumptions: list[KeyAssumption] = field(default_factory=list)
    mathematical_formulation: str = ""    # LaTeX 公式
    model_spec_for_coder: dict[str, Any] = field(default_factory=dict)  # 传给 CoderAgent 的规格

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates_evaluated": [c.to_dict() for c in self.candidates_evaluated],
            "final_choice": self.final_choice,
            "key_assumptions": [a.to_dict() for a in self.key_assumptions],
            "mathematical_formulation": self.mathematical_formulation,
            "model_spec_for_coder": self.model_spec_for_coder,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelingDecision:
        return cls(
            candidates_evaluated=[
                CandidateEvaluation.from_dict(c) for c in d.get("candidates_evaluated", [])
            ],
            final_choice=d.get("final_choice", ""),
            key_assumptions=[
                KeyAssumption.from_dict(a) for a in d.get("key_assumptions", [])
            ],
            mathematical_formulation=d.get("mathematical_formulation", ""),
            model_spec_for_coder=d.get("model_spec_for_coder", {}),
        )


# ============================================================
# 四、CodeResult：代码执行结果
# ============================================================


@dataclass
class SanityCheckResult:
    """数据合理性检查结果。"""

    value_range_ok: bool = True
    no_data_leakage: bool = True
    residuals_normal: bool = True
    residuals_issue: str = ""
    action_required: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_range_ok": self.value_range_ok,
            "no_data_leakage": self.no_data_leakage,
            "residuals_normal": self.residuals_normal,
            "residuals_issue": self.residuals_issue,
            "action_required": self.action_required,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SanityCheckResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FigureRecord:
    """生成的图表记录，含三段式解读。"""

    filename: str = ""
    description: str = ""
    key_observation: str = ""             # 观察：图中客观发生了什么
    paper_narrative: str = ""             # 完整的三段式叙述（供 WriterAgent 使用）

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "description": self.description,
            "key_observation": self.key_observation,
            "paper_narrative": self.paper_narrative,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FigureRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CodeResult:
    """某个子问题的代码执行结果。"""

    execution_status: str = "pending"     # pending / success / failed
    raw_outputs: dict[str, Any] = field(default_factory=dict)
    key_metrics: dict[str, float] = field(default_factory=dict)    # {"MAE": 12.3, ...}
    sanity_check_results: SanityCheckResult = field(default_factory=SanityCheckResult)
    generated_figures: list[FigureRecord] = field(default_factory=list)
    conclusions_supported: list[str] = field(default_factory=list)       # 数据能支持的结论
    conclusions_not_supported: list[str] = field(default_factory=list)   # 数据不能支持的结论
    links_to_other_q: str = ""            # 与其他子问题的关联

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_status": self.execution_status,
            "raw_outputs": self.raw_outputs,
            "key_metrics": self.key_metrics,
            "sanity_check_results": self.sanity_check_results.to_dict(),
            "generated_figures": [f.to_dict() for f in self.generated_figures],
            "conclusions_supported": self.conclusions_supported,
            "conclusions_not_supported": self.conclusions_not_supported,
            "links_to_other_q": self.links_to_other_q,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodeResult:
        return cls(
            execution_status=d.get("execution_status", "pending"),
            raw_outputs=d.get("raw_outputs", {}),
            key_metrics=d.get("key_metrics", {}),
            sanity_check_results=SanityCheckResult.from_dict(d.get("sanity_check_results", {})),
            generated_figures=[FigureRecord.from_dict(f) for f in d.get("generated_figures", [])],
            conclusions_supported=d.get("conclusions_supported", []),
            conclusions_not_supported=d.get("conclusions_not_supported", []),
            links_to_other_q=d.get("links_to_other_q", ""),
        )


# ============================================================
# 五、PaperState：论文状态
# ============================================================


@dataclass
class CrossReference:
    """图表/公式的交叉引用记录。"""

    chapter: int = 0
    description: str = ""
    cited_in: list[int] = field(default_factory=list)   # 被引用的章节列表

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter": self.chapter,
            "description": self.description,
            "cited_in": self.cited_in,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrossReference:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ChapterPromise:
    """章节做出的承诺及其兑现情况。"""

    claims: list[str] = field(default_factory=list)      # 本章做出的声明
    deliveries: dict[str, dict[str, Any]] = field(default_factory=dict)
    # deliveries 格式: {"声明内容": {"delivered": bool, "evidence": str, "gap": str}}

    def to_dict(self) -> dict[str, Any]:
        return {"claims": self.claims, "deliveries": self.deliveries}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChapterPromise:
        return cls(
            claims=d.get("claims", []),
            deliveries=d.get("deliveries", {}),
        )


@dataclass
class PaperState:
    """论文撰写状态，追踪写作进度和一致性。"""

    core_thesis: str = ""                               # 本文核心论点
    chapters_completed: list[str] = field(default_factory=list)
    chapters_in_progress: list[str] = field(default_factory=list)
    global_notation_table: dict[str, str] = field(default_factory=dict)   # {符号: 含义}
    cross_references: dict[str, CrossReference] = field(default_factory=dict)
    chapter_promises: dict[str, ChapterPromise] = field(default_factory=dict)
    tone_consistency: dict[str, Any] = field(default_factory=dict)
    # tone_consistency 内容示例:
    #   "problem_domain_terms": {"预测目标": "销售量"},
    #   "forbidden_phrases_used": [...],
    #   "style_notes": "技术性但保持可读性"

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_thesis": self.core_thesis,
            "chapters_completed": self.chapters_completed,
            "chapters_in_progress": self.chapters_in_progress,
            "global_notation_table": self.global_notation_table,
            "cross_references": {k: v.to_dict() for k, v in self.cross_references.items()},
            "chapter_promises": {k: v.to_dict() for k, v in self.chapter_promises.items()},
            "tone_consistency": self.tone_consistency,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PaperState:
        return cls(
            core_thesis=d.get("core_thesis", ""),
            chapters_completed=d.get("chapters_completed", []),
            chapters_in_progress=d.get("chapters_in_progress", []),
            global_notation_table=d.get("global_notation_table", {}),
            cross_references={
                k: CrossReference.from_dict(v)
                for k, v in d.get("cross_references", {}).items()
            },
            chapter_promises={
                k: ChapterPromise.from_dict(v)
                for k, v in d.get("chapter_promises", {}).items()
            },
            tone_consistency=d.get("tone_consistency", {}),
        )


# ============================================================
# 六、ReviewHistory：评审历史
# ============================================================


@dataclass
class ReviewIssue:
    """评审发现的单个问题。"""

    location: dict[str, Any] = field(default_factory=dict)
    # location 示例: {"chapter": 6, "paragraph": 3, "sentence": "假设数据独立同分布"}
    issue_type: str = ""                  # assumption_unjustified / logic_gap / ...
    severity: str = "medium"              # critical / high / medium / low
    description: str = ""
    fix_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "fix_instruction": self.fix_instruction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReviewIssue:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReviewRecord:
    """单轮评审记录。"""

    round: int = 0
    reviewer: str = ""                    # 评审者标识，如 "MethodAgent"
    score: int = 0                        # 百分制评分
    issues: list[ReviewIssue] = field(default_factory=list)
    diff_from_previous: str | None = None  # 与上一轮的差异描述

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "reviewer": self.reviewer,
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "diff_from_previous": self.diff_from_previous,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReviewRecord:
        return cls(
            round=d.get("round", 0),
            reviewer=d.get("reviewer", ""),
            score=d.get("score", 0),
            issues=[ReviewIssue.from_dict(i) for i in d.get("issues", [])],
            diff_from_previous=d.get("diff_from_previous"),
        )


# ============================================================
# 七、AgentDecision：Agent 决策日志
# ============================================================


@dataclass
class AgentDecision:
    """Agent 的一个可追溯决策。"""

    timestamp: str = ""
    agent: str = ""                       # Agent 名称
    decision: str = ""                    # 做了什么决策
    reasoning: str = ""                   # 决策推理过程
    can_be_challenged: bool = True        # 是否可被质疑
    challenged_by: str | None = None      # 被谁质疑了

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "agent": self.agent,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "can_be_challenged": self.can_be_challenged,
            "challenged_by": self.challenged_by,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentDecision:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# GlobalState：全局状态主类
# ============================================================


class GlobalState:
    """全局状态，整个任务的共享大脑。

    所有 Agent 读取此状态获取上下文，写入此状态留下决策记录。
    替代原有的 PaperContext，提供更完整的结构化共享信息。

    典型用法::

        state = GlobalState(task_id="abc-123")
        state.problem_understanding.core_question = "预测未来销量"
        state.update_modeling_decision("Q1", ModelingDecision(...))
        state.log_decision("ModelerAgent", "选择 Prophet", "数据有季节性")
        summary = state.inject_summary("WriterAgent")
    """

    def __init__(self, task_id: str = "", competition_type: str = "") -> None:
        self.task_meta = TaskMeta(
            task_id=task_id,
            competition_type=competition_type,
            created_at=datetime.now().isoformat(),
        )
        self.problem_understanding = ProblemUnderstanding()
        # 键为子问题 ID（如 "Q1"、"Q2"）
        self.modeling_decisions: dict[str, ModelingDecision] = {}
        self.code_results: dict[str, CodeResult] = {}
        self.paper_state = PaperState()
        self.review_history: list[ReviewRecord] = []
        self.agent_decisions_log: list[AgentDecision] = []

    # ----------------------------------------------------------
    # 决策记录
    # ----------------------------------------------------------

    def log_decision(
        self,
        agent: str,
        decision: str,
        reasoning: str,
        can_be_challenged: bool = True,
    ) -> None:
        """记录 Agent 决策。

        Args:
            agent: 做出决策的 Agent 名称。
            decision: 决策内容描述。
            reasoning: 决策推理过程。
            can_be_challenged: 是否可被其他 Agent 质疑。
        """
        self.agent_decisions_log.append(AgentDecision(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            decision=decision,
            reasoning=reasoning,
            can_be_challenged=can_be_challenged,
        ))

    def challenge_decision(self, index: int, challenger: str) -> None:
        """标记某个决策已被质疑。

        Args:
            index: 决策在 agent_decisions_log 中的索引。
            challenger: 质疑者的 Agent 名称。

        Raises:
            IndexError: 索引超出范围。
            ValueError: 决策不可被质疑。
        """
        if index < 0 or index >= len(self.agent_decisions_log):
            raise IndexError(f"决策索引 {index} 超出范围（共 {len(self.agent_decisions_log)} 条）")
        decision = self.agent_decisions_log[index]
        if not decision.can_be_challenged:
            raise ValueError(f"决策 '{decision.decision}' 不可被质疑")
        decision.challenged_by = challenger

    def get_unchallenged_decisions(self) -> list[AgentDecision]:
        """获取所有可被质疑但尚未被质疑的决策。

        Returns:
            未被质疑的决策列表。
        """
        return [
            d for d in self.agent_decisions_log
            if d.can_be_challenged and d.challenged_by is None
        ]

    # ----------------------------------------------------------
    # 建模决策更新
    # ----------------------------------------------------------

    def update_modeling_decision(self, q_id: str, decision: ModelingDecision) -> None:
        """更新某个子问题的建模决策。

        Args:
            q_id: 子问题标识，如 "Q1"。
            decision: 建模决策对象。
        """
        self.modeling_decisions[q_id] = decision

    # ----------------------------------------------------------
    # 代码结果更新
    # ----------------------------------------------------------

    def update_code_result(self, q_id: str, result: CodeResult) -> None:
        """更新某个子问题的代码执行结果。

        Args:
            q_id: 子问题标识，如 "Q1"。
            result: 代码结果对象。
        """
        self.code_results[q_id] = result

    # ----------------------------------------------------------
    # 论文状态更新
    # ----------------------------------------------------------

    def update_paper_state(self, **kwargs: Any) -> None:
        """更新论文状态的指定字段。

        支持的字段名：core_thesis, chapters_completed, chapters_in_progress,
        global_notation_table, cross_references, chapter_promises, tone_consistency。

        Args:
            **kwargs: 要更新的字段名和值。

        Raises:
            ValueError: 字段名不属于 PaperState。
        """
        valid_fields = set(PaperState.__dataclass_fields__)
        for key, value in kwargs.items():
            if key not in valid_fields:
                raise ValueError(f"PaperState 没有字段 '{key}'，有效字段: {valid_fields}")
            setattr(self.paper_state, key, value)

    # ----------------------------------------------------------
    # 评审记录
    # ----------------------------------------------------------

    def add_review_record(self, record: ReviewRecord) -> None:
        """添加评审记录。

        Args:
            record: 评审记录对象。
        """
        self.review_history.append(record)

    # ----------------------------------------------------------
    # 符号表管理（兼容 PaperContext 功能）
    # ----------------------------------------------------------

    def add_symbols(self, symbols_dict: dict[str, str]) -> None:
        """向全局符号表添加符号定义。

        Args:
            symbols_dict: {符号: 含义} 字典。
        """
        self.paper_state.global_notation_table.update(symbols_dict)

    # ----------------------------------------------------------
    # Writer 输出提取（兼容 PaperContext 功能）
    # ----------------------------------------------------------

    def extract_from_writer_response(self, content: str) -> None:
        """从 WriterAgent 的输出中提取符号、关键数值和章节信息。

        提取内容：
        1. 符号定义（表格中的 $符号$ 格式）
        2. 关键数值（R², RMSE, MAE, Accuracy, AUC, p 值）
        3. 章节摘要（前 200 字）

        Args:
            content: WriterAgent 输出的论文文本。
        """
        # 提取符号定义
        self._extract_symbols(content)

        # 提取关键数值（写入最新代码结果或创建临时记录）
        numbers = self._extract_key_numbers(content)
        if numbers:
            # 如果有正在处理的代码结果，追加到那里
            for _q_id, cr in self.code_results.items():
                cr.key_metrics.update(numbers)
                break

    def _extract_symbols(self, content: str) -> None:
        """从论文文本中提取符号定义（表格格式）。"""
        symbol_pattern = r'\$([^$]+)\$'
        table_rows = re.findall(r'\|([^|]+)\|([^|]+)\|([^|]*)\|', content)
        for row in table_rows:
            sym_match = re.search(symbol_pattern, row[0])
            if sym_match:
                sym = sym_match.group(1).strip()
                meaning = row[1].strip()
                if sym and meaning:
                    self.paper_state.global_notation_table[sym] = meaning

    def _extract_key_numbers(self, content: str) -> dict[str, str]:
        """从文本中提取关键数值指标。"""
        numbers: dict[str, str] = {}

        patterns = [
            (r'R\^?2\s*[=≈:]\s*([\d.]+)', "R²"),
            (r'RMSE\s*[=≈:]\s*([\d.]+)', "RMSE"),
            (r'MAE\s*[=≈:]\s*([\d.]+)', "MAE"),
            (r'(?:准确率|Accuracy)\s*[=≈:]\s*([\d.]+%?)', "准确率"),
            (r'AUC\s*[=≈:]\s*([\d.]+)', "AUC"),
            (r'p\s*[=≈<]\s*([\d.]+)', "p值"),
        ]
        for pattern, name in patterns:
            match = re.search(pattern, content)
            if match:
                numbers[name] = match.group(1)

        return numbers

    # ----------------------------------------------------------
    # 注入摘要：为特定 Agent 生成精简上下文
    # ----------------------------------------------------------

    def inject_summary(self, for_agent: str) -> str:
        """为指定 Agent 生成精简的全局状态摘要，用于注入 prompt。

        不同 Agent 获得不同侧重的摘要，避免信息过载。

        Args:
            for_agent: Agent 名称，如 "WriterAgent"、"CoderAgent"、"ModelerAgent"。

        Returns:
            格式化的上下文文本。若无可用信息则返回空字符串。
        """
        parts: list[str] = []

        # 所有 Agent 都需要：核心问题
        if self.problem_understanding.core_question:
            parts.append(f"【核心问题】\n{self.problem_understanding.core_question}")

        # 所有 Agent 都需要：子问题列表
        if self.problem_understanding.sub_problems:
            sp_lines = []
            for sp in self.problem_understanding.sub_problems:
                dep = f"（依赖: {', '.join(sp.dependency_on)}）" if sp.dependency_on else ""
                sp_lines.append(f"- {sp.id}: {sp.description}{dep}")
            parts.append("【子问题】\n" + "\n".join(sp_lines))

        # 建模决策摘要（CoderAgent、WriterAgent 需要）
        if self.modeling_decisions and for_agent in ("CoderAgent", "WriterAgent", "ReviewAgent"):
            md_lines = []
            for q_id, md in self.modeling_decisions.items():
                assumptions = "; ".join(
                    a.assumption for a in md.key_assumptions[:3]
                ) if md.key_assumptions else "无"
                md_lines.append(
                    f"- {q_id}: {md.final_choice} | 假设: {assumptions}"
                )
            parts.append("【建模决策】\n" + "\n".join(md_lines))

        # 数据探查摘要（CoderAgent、ModelerAgent 最需要，直接影响模型选择）
        if for_agent in ("CoderAgent", "ModelerAgent"):
            inv = self.problem_understanding.data_inventory
            data_parts = []
            # DataProfiler 完整摘要
            profiler_summary = inv.data_range_checks.get("_profiler_summary", "")
            if profiler_summary:
                data_parts.append(profiler_summary[:2000])
            # 已知数据问题（稀疏性/不平衡/重复测量等信号）
            if inv.known_issues:
                issues_str = "\n".join(f"  - {issue}" for issue in inv.known_issues[:10])
                data_parts.append(f"【数据质量信号（建模时必须考虑！）】\n{issues_str}")
            if data_parts:
                parts.append("【DataProfiler 数据探查结果】\n" + "\n".join(data_parts))

        # 代码结果摘要（WriterAgent 需要）
        if self.code_results and for_agent in ("WriterAgent", "ReviewAgent"):
            cr_lines = []
            for q_id, cr in self.code_results.items():
                metrics = ", ".join(f"{k}={v}" for k, v in cr.key_metrics.items())
                status_icon = "OK" if cr.execution_status == "success" else cr.execution_status
                supported = "; ".join(cr.conclusions_supported[:2]) if cr.conclusions_supported else ""
                cr_lines.append(f"- {q_id} [{status_icon}]: {metrics}")
                if supported:
                    cr_lines.append(f"  结论: {supported}")
            parts.append("【代码结果】\n" + "\n".join(cr_lines))

        # 图表叙事（WriterAgent 需要）
        if self.code_results and for_agent == "WriterAgent":
            fig_lines = []
            for q_id, cr in self.code_results.items():
                for fig in cr.generated_figures:
                    if fig.paper_narrative:
                        fig_lines.append(f"- {fig.filename}: {fig.paper_narrative[:150]}...")
            if fig_lines:
                parts.append("【图表叙述（可直接引用）】\n" + "\n".join(fig_lines))

        # 符号表（WriterAgent 需要，防止前后不一致）
        if self.paper_state.global_notation_table and for_agent == "WriterAgent":
            sym_lines = [
                f"- {sym}: {meaning}"
                for sym, meaning in list(self.paper_state.global_notation_table.items())[:30]
            ]
            if len(self.paper_state.global_notation_table) > 30:
                sym_lines.append(f"... 共 {len(self.paper_state.global_notation_table)} 个符号")
            parts.append("【已定义符号（请保持一致）】\n" + "\n".join(sym_lines))

        # 论文状态（WriterAgent、ReviewAgent 需要）
        if for_agent in ("WriterAgent", "ReviewAgent"):
            ps_lines = []
            if self.paper_state.core_thesis:
                ps_lines.append(f"核心论点: {self.paper_state.core_thesis}")
            if self.paper_state.chapters_completed:
                ps_lines.append(f"已完成章节: {', '.join(self.paper_state.chapters_completed)}")
            if self.paper_state.chapters_in_progress:
                ps_lines.append(f"进行中章节: {', '.join(self.paper_state.chapters_in_progress)}")
            if ps_lines:
                parts.append("【论文状态】\n" + "\n".join(ps_lines))

        # 章节承诺（WriterAgent 需要）
        if self.paper_state.chapter_promises and for_agent == "WriterAgent":
            promise_lines = []
            for ch, promise in self.paper_state.chapter_promises.items():
                for claim in promise.claims:
                    delivered = promise.deliveries.get(claim, {}).get("delivered", False)
                    icon = "v" if delivered else "x"
                    promise_lines.append(f"- [{icon}] {ch}: {claim}")
            if promise_lines:
                parts.append("【章节承诺追踪】\n" + "\n".join(promise_lines))

        # 评审历史（WriterAgent 需要，知道上次评审了什么）
        if self.review_history and for_agent == "WriterAgent":
            last = self.review_history[-1]
            issue_summary = "; ".join(
                iss.description for iss in last.issues[:5]
            ) if last.issues else "无"
            parts.append(
                f"【最近评审（第 {last.round} 轮，{last.reviewer}，得分 {last.score}）】\n"
                f"主要问题: {issue_summary}"
            )

        # 未被质疑的决策（CriticAgent 需要）
        if for_agent == "CriticAgent":
            unchallenged = self.get_unchallenged_decisions()
            if unchallenged:
                dc_lines = [
                    f"- [{d.agent}] {d.decision} | 理由: {d.reasoning[:80]}"
                    for d in unchallenged[-10:]  # 最近 10 条
                ]
                parts.append("【待质疑决策】\n" + "\n".join(dc_lines))

        # 数据清单（CoderAgent、ModelerAgent 需要）
        if for_agent in ("CoderAgent", "ModelerAgent", "DataSanityAgent"):
            inv = self.problem_understanding.data_inventory
            if inv.files:
                inv_lines = [f"- 文件: {', '.join(inv.files)}"]
                if inv.sample_size:
                    inv_lines.append(f"- 样本量: {inv.sample_size}")
                if inv.known_issues:
                    inv_lines.append(f"- 已知问题: {'; '.join(inv.known_issues)}")
                parts.append("【数据清单】\n" + "\n".join(inv_lines))

        if not parts:
            return ""

        header = f"=== 全局状态（供 {for_agent} 参考）==="
        return header + "\n\n" + "\n\n".join(parts)

    # ----------------------------------------------------------
    # 完整摘要
    # ----------------------------------------------------------

    def to_summary(self) -> str:
        """生成人类可读的全局状态摘要。

        Returns:
            格式化的全文摘要。
        """
        lines: list[str] = []
        lines.append(f"任务 {self.task_meta.task_id} | {self.task_meta.competition_type}")
        lines.append(f"当前阶段: {self.task_meta.current_phase}")
        lines.append(f"创建时间: {self.task_meta.created_at}")

        # 题目理解
        pu = self.problem_understanding
        if pu.core_question:
            lines.append(f"\n核心问题: {pu.core_question}")
        if pu.sub_problems:
            lines.append(f"子问题数: {len(pu.sub_problems)}")
            for sp in pu.sub_problems:
                lines.append(f"  - {sp.id}: {sp.description[:60]}")

        # 建模决策
        if self.modeling_decisions:
            lines.append(f"\n建模决策 ({len(self.modeling_decisions)} 个):")
            for q_id, md in self.modeling_decisions.items():
                lines.append(f"  - {q_id}: {md.final_choice}")

        # 代码结果
        if self.code_results:
            lines.append(f"\n代码结果 ({len(self.code_results)} 个):")
            for q_id, cr in self.code_results.items():
                metrics = ", ".join(f"{k}={v}" for k, v in cr.key_metrics.items())
                lines.append(f"  - {q_id} [{cr.execution_status}]: {metrics}")

        # 论文状态
        ps = self.paper_state
        if ps.chapters_completed:
            lines.append(f"\n已完成章节: {', '.join(ps.chapters_completed)}")
        if ps.chapters_in_progress:
            lines.append(f"进行中章节: {', '.join(ps.chapters_in_progress)}")
        if ps.global_notation_table:
            lines.append(f"符号表: {len(ps.global_notation_table)} 个符号")

        # 评审历史
        if self.review_history:
            last = self.review_history[-1]
            lines.append(f"\n最近评审: 第 {last.round} 轮，得分 {last.score}")

        # 决策日志
        if self.agent_decisions_log:
            unchallenged = len(self.get_unchallenged_decisions())
            lines.append(f"\n决策日志: {len(self.agent_decisions_log)} 条（{unchallenged} 条未质疑）")

        return "\n".join(lines)

    # ----------------------------------------------------------
    # JSON 序列化
    # ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "task_meta": self.task_meta.to_dict(),
            "problem_understanding": self.problem_understanding.to_dict(),
            "modeling_decisions": {k: v.to_dict() for k, v in self.modeling_decisions.items()},
            "code_results": {k: v.to_dict() for k, v in self.code_results.items()},
            "paper_state": self.paper_state.to_dict(),
            "review_history": [r.to_dict() for r in self.review_history],
            "agent_decisions_log": [d.to_dict() for d in self.agent_decisions_log],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GlobalState:
        """从字典反序列化。

        Args:
            d: 由 to_dict() 生成的字典。

        Returns:
            反序列化后的 GlobalState 实例。
        """
        state = cls.__new__(cls)
        state.task_meta = TaskMeta.from_dict(d.get("task_meta", {}))
        state.problem_understanding = ProblemUnderstanding.from_dict(
            d.get("problem_understanding", {})
        )
        state.modeling_decisions = {
            k: ModelingDecision.from_dict(v)
            for k, v in d.get("modeling_decisions", {}).items()
        }
        state.code_results = {
            k: CodeResult.from_dict(v)
            for k, v in d.get("code_results", {}).items()
        }
        state.paper_state = PaperState.from_dict(d.get("paper_state", {}))
        state.review_history = [
            ReviewRecord.from_dict(r) for r in d.get("review_history", [])
        ]
        state.agent_decisions_log = [
            AgentDecision.from_dict(d_item) for d_item in d.get("agent_decisions_log", [])
        ]
        return state

    def save(self, path: str) -> None:
        """保存到 JSON 文件。

        Args:
            path: 文件路径（绝对或相对）。
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> GlobalState:
        """从 JSON 文件加载。

        Args:
            path: 文件路径。

        Returns:
            加载的 GlobalState 实例。文件不存在时返回空状态。
        """
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
