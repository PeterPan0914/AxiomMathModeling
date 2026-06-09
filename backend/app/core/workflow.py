"""工作流模块，编排多 Agent 协作完成数学建模任务。

执行顺序：
Phase 0: 初始化（诊断日志、LLM、质量跟踪器）
Phase 1: ProblemAnalystAgent（题目深度分析，前置）
Phase 2: LiteratureAgent（文献调研）
Phase 3: CoordinatorAgent（战略规划，接收 problem_analysis + literature_review）
Phase 4: ModelerAgent（建模，注入 problem_analysis 上下文）
Phase 5: 子任务循环：CoderAgent → ResultInterpreterAgent → WriterAgent
Phase 6: OutlineAgent（生成论文大纲，指导后续写作）
Phase 7: 写作剩余章节（问题分析、结果分析等）
Phase 8: ConsistencyAgent（全文一致性检查）
Phase 9: MultiReviewer（三审制） + ReviewSynthesizer
Phase 10: Reflexion 循环（低于阈值则改进并重新评审）
Phase 11: 保存 GlobalState + 诊断数据
"""

from __future__ import annotations

import asyncio
import json as json_module
from typing import TYPE_CHECKING

from app.core.agents import (
    WriterAgent,
    CoderAgent,
    CoordinatorAgent,
    ModelerAgent,
    ReviewAgent,
    MultiReviewer,
    ResultInterpreterAgent,
    CriticAgent,
    LiteratureAgent,
    OutlineAgent,
    ConsistencyAgent,
    ProblemAnalystAgent,
    # Phase 2 新增
    DependencyAgent,
    ProblemTypeAgent,
    ProblemReformulationAgent,
    ModelSearchAgent,
    ReviewerAgent,
    AwardJudgeAgent,
)
from app.core.dependency_graph import QuestionDependencyGraph, QuestionNode, DependencyEdge
from app.core.evaluation import QualityScore, QualityReport, QualityTracker
from app.core.flows import Flows
from app.core.global_state import GlobalState
from app.core.llm.llm_factory import LLMFactory
from app.core.paper_context import PaperContext
from app.core.structure_control import StructureController, SectionStatus
from app.models.user_output import UserOutput
from app.schemas.request import Problem
from app.schemas.response import SystemMessage
from app.services.redis_manager import redis_manager
from app.tools.interpreter_factory import create_interpreter
from app.tools.notebook_serializer import NotebookSerializer
from app.tools.openalex_scholar import OpenAlexScholar
from app.utils.common_utils import create_work_dir, get_config_template
from app.utils.diagnostic_logger import DiagnosticLogger
from app.utils.log_util import logger
from app.config.setting import settings

if TYPE_CHECKING:
    from app.schemas.A2A import WriterResponse


# Reflexion 配置（从 settings 读取）
def _get_reflexion_config():
    """获取 Reflexion 配置。"""
    return {
        'max_iterations': settings.REFLEXION_MAX_ITERATIONS,
        'quality_threshold': settings.REFLEXION_QUALITY_THRESHOLD,
        'enabled': settings.REFLEXION_ENABLED,
    }


class WorkFlow:
    """工作流基类。"""

    def __init__(self):
        pass

    def execute(self) -> None:
        """执行工作流。"""
        pass


class MathModelWorkFlow(WorkFlow):
    """数学建模工作流，协调协调者、建模手、代码手和写作手完成完整建模任务。"""

    task_id: str
    work_dir: str
    ques_count: int = 0
    questions: dict[str, str | int] = {}
    cancel_event: asyncio.Event | None = None

    async def _check_cancelled(self) -> None:
        """检查是否收到取消信号，若已取消则发布通知并抛出 CancelledError。"""
        if self.cancel_event and self.cancel_event.is_set():
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content="任务已停止", type="warning"),
            )
            raise asyncio.CancelledError("任务被用户停止")

    async def execute(self, problem: Problem):  # type: ignore[reportIncompatibleMethodOverride]
        """执行数学建模工作流。

        Args:
            problem: 包含题目信息、模板配置等的 Problem 对象。
        """
        self.task_id = problem.task_id
        self.work_dir = create_work_dir(self.task_id)

        # ================================================================
        # Phase 0: 初始化
        # ================================================================
        diag = DiagnosticLogger(self.work_dir)
        quality_tracker = QualityTracker()
        structure_controller = StructureController()
        diag.save_workflow_config({
            "task_id": self.task_id,
            "comp_template": problem.comp_template.value if hasattr(problem.comp_template, 'value') else str(problem.comp_template),
            "format_output": problem.format_output.value if hasattr(problem.format_output, 'value') else str(problem.format_output),
            "reflexion_enabled": settings.REFLEXION_ENABLED,
            "reflexion_max_iterations": settings.REFLEXION_MAX_ITERATIONS,
            "reflexion_quality_threshold": settings.REFLEXION_QUALITY_THRESHOLD,
            "models": {
                "coordinator": settings.COORDINATOR_MODEL,
                "modeler": settings.MODELER_MODEL,
                "coder": settings.CODER_MODEL,
                "writer": settings.WRITER_MODEL,
            },
        })

        llm_factory = LLMFactory(self.task_id)
        coordinator_llm, modeler_llm, coder_llm, writer_llm = llm_factory.get_all_llms()

        # 初始化全局状态（替代 PaperContext，作为系统的"共享大脑"）
        global_state = GlobalState(task_id=self.task_id)
        global_state.task_meta.current_phase = "init"

        # 同时保留 PaperContext 以兼容 Flows 等现有模块
        paper_context = PaperContext()

        user_output = UserOutput(work_dir=self.work_dir, ques_count=0)

        # ================================================================
        # Phase 0.5: DataProfiler（数据形态探查，纯 Python，不消耗 LLM）
        # ================================================================
        data_profiler_text = ""
        try:
            from app.core.data_profiler import DataProfiler, format_profiles_for_prompt
            data_profiler = DataProfiler(work_dir=self.work_dir)
            data_profiles = data_profiler.profile_all()
            data_profiler_text = format_profiles_for_prompt(data_profiles)
            if data_profiles:
                logger.info(f"[DataProfiler] 探查完成: {len(data_profiles)} 个文件")
                for p in data_profiles:
                    if p.detected_signals:
                        logger.info(f"[DataProfiler] {p.file_name} 信号: {p.detected_signals}")
                    # 更新 GlobalState 数据资产清单，供 CoderAgent/ModelerAgent 通过 inject_summary 访问
                    global_state.problem_understanding.data_inventory.files.append(p.file_name)
                    if p.detected_signals:
                        global_state.problem_understanding.data_inventory.known_issues.extend(
                            [f"{p.file_name}: {s}" for s in p.detected_signals]
                        )
            # 将完整 DataProfiler 摘要存入 GlobalState，CoderAgent/ModelerAgent 可通过 inject_summary 获取
            if data_profiler_text:
                global_state.problem_understanding.data_inventory.data_range_checks["_profiler_summary"] = data_profiler_text
        except Exception as e:
            logger.warning(f"Phase 0.5: DataProfiler 失败（不影响流程）: {e}")


        # ================================================================
        # Phase 1: ProblemAnalystAgent（前置，第一个运行）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 1: 题目深度分析..."),
        )
        await self._check_cancelled()

        problem_analyst = ProblemAnalystAgent(
            self.task_id, coordinator_llm,
            context_window=settings.COORDINATOR_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )
        problem_analysis = None
        problem_analysis_text = ""
        try:
            # 注入 DataProfiler 结果到 ProblemAnalystAgent
            ques_all_with_profiler = problem.ques_all
            if data_profiler_text:
                ques_all_with_profiler = problem.ques_all + "\n\n" + data_profiler_text

            problem_analysis = await problem_analyst.run(ques_all_with_profiler, {})
            logger.info(f"[ProblemAnalyst] 陷阱: {problem_analysis.pitfalls}")
            logger.info(f"[ProblemAnalyst] 评分重点: {problem_analysis.scoring_focus}")

            # 将分析结果写入 GlobalState
            global_state.problem_understanding.raw_text = problem.ques_all
            global_state.task_meta.current_phase = "problem_analyzed"
            global_state.log_decision(
                agent="ProblemAnalystAgent",
                decision="完成题目深度分析",
                reasoning=f"陷阱: {problem_analysis.pitfalls[:3]}...",
            )

            # 注入 PaperContext（兼容）
            paper_context.set_core_argument(problem_analysis.data_characteristics)

            problem_analysis_text = (
                f"陷阱识别: {'; '.join(problem_analysis.pitfalls)}\n"
                f"评分重点: {'; '.join(problem_analysis.scoring_focus)}\n"
                f"禁用方法: {json_module.dumps(problem_analysis.forbidden_methods, ensure_ascii=False)}"
            )

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content="Phase 1: 题目深度分析完成", type="success"),
            )
        except Exception as e:
            logger.warning(f"Phase 1: 题目深度分析失败（不影响流程）: {e}")
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"题目深度分析失败: {e}，继续流程", type="warning"),
            )

        # ================================================================
        # Phase 2: LiteratureAgent（文献调研）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 2: 文献调研..."),
        )
        await self._check_cancelled()

        literature_review_text = ""
        try:
            # 初始化 OpenAlex 学术搜索客户端
            openalex_scholar = None
            if settings.OPENALEX_EMAIL:
                openalex_scholar = OpenAlexScholar(
                    task_id=self.task_id,
                    email=settings.OPENALEX_EMAIL,
                    api_key=settings.OPENALEX_API_KEY,
                )

            literature_agent = LiteratureAgent(
                self.task_id, coordinator_llm,
                context_window=settings.COORDINATOR_CONTEXT_WINDOW,
                cancel_event=self.cancel_event,
                diagnostic_logger=diag,
                openalex_scholar=openalex_scholar,
            )
            literature_review_text = await literature_agent.run(
                problem_description=problem.ques_all,
                competition_type=problem.comp_template.value if hasattr(problem.comp_template, 'value') else str(problem.comp_template),
            )

            # 将文献调研结果写入 GlobalState
            global_state.problem_understanding.literature_context.innovation_space = literature_review_text[:500]
            global_state.log_decision(
                agent="LiteratureAgent",
                decision="完成文献调研",
                reasoning=f"文献调研结果长度: {len(literature_review_text)} 字符",
            )

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content="Phase 2: 文献调研完成", type="success"),
            )
        except Exception as e:
            logger.warning(f"Phase 2: 文献调研失败（不影响流程）: {e}")
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"文献调研失败: {e}，继续流程", type="warning"),
            )

        # ================================================================
        # Phase 3: CoordinatorAgent（战略规划）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 3: 战略规划与问题拆解..."),
        )
        await self._check_cancelled()

        coordinator_agent = CoordinatorAgent(
            self.task_id, coordinator_llm,
            context_window=settings.COORDINATOR_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        try:
            coordinator_response = await coordinator_agent.run(
                problem.ques_all,
                problem_analysis=problem_analysis_text,
                literature_review=literature_review_text,
            )
            self.questions = coordinator_response.questions
            self.ques_count = coordinator_response.ques_count

            # 更新 GlobalState
            global_state.task_meta.current_phase = "coordinated"
            global_state.problem_understanding.core_question = str(
                self.questions.get("core_question", "")
            )
            global_state.log_decision(
                agent="CoordinatorAgent",
                decision=f"拆解为 {self.ques_count} 个子问题",
                reasoning=f"子问题: {list(self.questions.keys())}",
            )
        except Exception as e:
            logger.error(f"Phase 3: CoordinatorAgent 执行失败: {e}")
            raise e

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 3: 战略规划完成", type="success"),
        )

        # 更新 user_output 的 ques_count
        user_output = UserOutput(work_dir=self.work_dir, ques_count=self.ques_count)

        # ================================================================
        # Phase 3.1: ProblemTypeAgent（问题类型识别）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 3.1: 问题类型识别与数据稀疏性分析..."),
        )
        await self._check_cancelled()

        problem_type_report = None
        problem_type_analysis_text = ""
        try:
            problem_type_agent = ProblemTypeAgent(
                self.task_id, coordinator_llm,
                context_window=settings.COORDINATOR_CONTEXT_WINDOW,
                cancel_event=self.cancel_event,
                diagnostic_logger=diag,
            )
            problem_type_report = await problem_type_agent.run(
                problem.ques_all, self.questions
            )

            global_state.task_meta.current_phase = "problem_type_classified"
            for q_id, pti in problem_type_report.sub_problem_types.items():
                logger.info(
                    f"[ProblemType] {q_id}: type={pti.primary_type}, "
                    f"sparsity={pti.sparsity_report.sparsity_level}, "
                    f"censoring={pti.censoring_detected}, confidence={pti.confidence:.2f}"
                )

            # 将问题类型分析注入 problem_analysis_text
            from app.core.agents.problem_type_agent import problem_type_report_to_text
            problem_type_analysis_text = problem_type_report_to_text(problem_type_report)
            problem_analysis_text += f"\n\n问题类型识别:\n{problem_type_analysis_text}"

            global_state.log_decision(
                agent="ProblemTypeAgent",
                decision=f"识别出 {len(problem_type_report.sub_problem_types)} 个子问题类型",
                reasoning=str({k: v.primary_type for k, v in problem_type_report.sub_problem_types.items()}),
            )

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content="Phase 3.1: 问题类型识别完成", type="success"),
            )
        except Exception as e:
            logger.warning(f"Phase 3.1: 问题类型识别失败（不影响流程）: {e}")

        # ================================================================
        # Phase 3.2: DependencyAgent（子问题依赖分析）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 3.2: 分析问题依赖关系..."),
        )
        await self._check_cancelled()

        dep_graph = QuestionDependencyGraph()
        try:
            dependency_agent = DependencyAgent(
                self.task_id, coordinator_llm,
                context_window=settings.COORDINATOR_CONTEXT_WINDOW,
                cancel_event=self.cancel_event,
                diagnostic_logger=diag,
            )
            dep_graph = await dependency_agent.run(
                ques_all=problem.ques_all,
                questions=self.questions,
            )

            global_state.task_meta.current_phase = "dependency_analyzed"
            global_state.problem_understanding.inter_problem_dependencies = (
                f"执行顺序: {dep_graph.execution_order}, 依赖边数: {len(dep_graph.edges)}"
            )
            global_state.log_decision(
                agent="DependencyAgent",
                decision=f"构建依赖图: {dep_graph.execution_order}",
                reasoning=f"依赖边: {[(e.source, e.target) for e in dep_graph.edges]}",
            )

            # 保存依赖图到诊断目录
            dep_graph.save(f"{self.work_dir}/diagnostic/dependency_graph.json")

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"Phase 3.2: 依赖关系分析完成，执行顺序: {dep_graph.execution_order}",
                    type="success",
                ),
            )
        except Exception as e:
            logger.warning(f"Phase 3.2: 依赖分析失败（退化为平铺执行）: {e}")
            # 退化：构建无依赖的平铺图
            ques_keys = [k for k in self.questions.keys() if k.startswith("ques") and k != "ques_count"]
            for idx, qk in enumerate(ques_keys):
                dep_graph.add_node(QuestionNode(id=qk, description=str(self.questions.get(qk, ""))[:200]))
            dep_graph.execution_order = ques_keys

        # ================================================================
        # Phase 3.3: ProblemReformulationAgent（问题重述）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 3.3: 问题重述（将题目翻译为标准数学问题类型）..."),
        )
        await self._check_cancelled()

        reformulation_result = None
        try:
            reformulation_agent = ProblemReformulationAgent(
                self.task_id, modeler_llm,
                context_window=settings.MODELER_CONTEXT_WINDOW,
                cancel_event=self.cancel_event,
                diagnostic_logger=diag,
            )
            reformulation_result = await reformulation_agent.run(
                problem_description=problem.ques_all,
                coordinator_questions=self.questions,
                problem_analysis_text=problem_analysis_text,
            )

            global_state.task_meta.current_phase = "reformulated"
            for q_key, sp in reformulation_result.sub_problems.items():
                logger.info(f"[Reformulation] {q_key}: {sp.standard_problem_type} ({sp.problem_type_cn})")

            global_state.log_decision(
                agent="ProblemReformulationAgent",
                decision=f"完成问题重述，{len(reformulation_result.sub_problems)} 个子问题",
                reasoning="; ".join(
                    f"{k}: {sp.standard_problem_type}" for k, sp in reformulation_result.sub_problems.items()
                ),
            )

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"Phase 3.3: 问题重述完成: " +
                            "; ".join(f"{k}→{sp.problem_type_cn}" for k, sp in reformulation_result.sub_problems.items()),
                    type="success",
                ),
            )
        except Exception as e:
            logger.warning(f"Phase 3.3: 问题重述失败（退化为无重述模式）: {e}")

        # ================================================================
        # Phase 4: ModelerAgent（建模，注入问题重述结果）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 4: 建模手开始建模..."),
        )
        await self._check_cancelled()

        modeler_agent = ModelerAgent(
            self.task_id, modeler_llm,
            context_window=settings.MODELER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        # 构建依赖上下文摘要给 ModelerAgent
        dep_summary = ""
        if dep_graph.edges:
            dep_lines = []
            for edge in dep_graph.edges:
                dep_lines.append(f"- {edge.target} 依赖 {edge.source}: 需要使用 {edge.what_to_use}")
            dep_summary = (
                "\n\n【问题依赖关系（建模时必须考虑！后续问题的方案需利用前序结论）】\n"
                + "\n".join(dep_lines)
                + "\n\n执行顺序: " + " → ".join(dep_graph.execution_order)
            )

        # 构建问题重述文本（供 ModelerAgent 约束选型）
        reformulation_text = ""
        if reformulation_result:
            parts = []
            for q_key, sp in reformulation_result.sub_problems.items():
                parts.append(f"### {q_key}: {sp.problem_type_cn} ({sp.standard_problem_type})")
                parts.append(f"- 重述: {sp.reformulated_statement}")
                parts.append(f"- 推荐模型: {', '.join(sp.recommended_model_families)}")
                if sp.forbidden_model_families:
                    parts.append(f"- 禁止模型: {', '.join(sp.forbidden_model_families)}")
                parts.append(f"- 创新方向: {sp.innovation_direction}")
                parts.append("")
            if reformulation_result.innovation_packaging:
                parts.append(f"## 创新包装建议\n{reformulation_result.innovation_packaging}")
            reformulation_text = "\n".join(parts)

        modeler_response = await modeler_agent.run(
            coordinator_response,
            problem_analysis=problem_analysis_text + dep_summary,
            literature_review_text=literature_review_text[:3000],
            reformulation_text=reformulation_text,
        )

        # 更新 GlobalState：记录建模决策
        global_state.task_meta.current_phase = "modeled"
        if hasattr(modeler_response, 'questions_solution'):
            for q_key, solution_text in modeler_response.questions_solution.items():
                from app.core.global_state import ModelingDecision
                global_state.update_modeling_decision(
                    q_key,
                    ModelingDecision(
                        final_choice=solution_text[:200],
                        mathematical_formulation=solution_text,
                    ),
                )
        global_state.log_decision(
            agent="ModelerAgent",
            decision="完成建模方案制定",
            reasoning=f"子问题方案数: {len(getattr(modeler_response, 'questions_solution', {}))}",
        )

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 4: 建模完成", type="success"),
        )

        # ================================================================
        # 初始化代码解释器
        # 必须在 Phase 4.5 之前创建：ModelSearchAgent 需要执行代码进行变量筛选
        # Phase 5 的 CoderAgent 也复用此实例
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="创建代码沙盒环境"),
        )
        notebook_serializer = NotebookSerializer(work_dir=self.work_dir)
        code_interpreter = await create_interpreter(
            kind="local",
            task_id=self.task_id,
            work_dir=self.work_dir,
            notebook_serializer=notebook_serializer,
            timeout=3000,
        )
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="创建完成"),
        )

        # ================================================================
        # Phase 4.5: ModelSearchAgent（变量筛选，仅对包含 model_search_protocol 的子问题）
        # ================================================================
        model_search_results: dict[str, 'ModelSearchResult'] = {}

        ques_needing_search = {}
        if hasattr(modeler_response, 'model_specs') and modeler_response.model_specs:
            for q_key, spec in modeler_response.model_specs.items():
                if spec.model_search_protocol:
                    ques_needing_search[q_key] = spec

        if ques_needing_search:
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"Phase 4.5: 发现 {len(ques_needing_search)} 个子问题需要变量筛选..."),
            )
            await self._check_cancelled()

            try:
                model_search_agent = ModelSearchAgent(
                    task_id=problem.task_id,
                    model=coder_llm,
                    work_dir=self.work_dir,
                    code_interpreter=code_interpreter,
                    context_window=settings.CODER_CONTEXT_WINDOW,
                    cancel_event=self.cancel_event,
                    diagnostic_logger=diag,
                )

                for q_key, spec in ques_needing_search.items():
                    await self._check_cancelled()
                    data_desc = ""
                    if global_state.problem_understanding.data_inventory.files:
                        data_desc = f"数据文件: {', '.join(global_state.problem_understanding.data_inventory.files)}"

                    search_result = await model_search_agent.run(
                        ques_key=q_key, model_spec=spec, data_description=data_desc,
                    )
                    model_search_results[q_key] = search_result

                    if search_result.success:
                        logger.info(
                            f"[ModelSearchAgent] {q_key}: 最优={search_result.best_model_id}, "
                            f"AIC={search_result.best_aic:.2f}, ICC={search_result.icc:.4f}"
                        )
                        global_state.log_decision(
                            agent="ModelSearchAgent",
                            decision=f"{q_key}: 选择 {search_result.best_model_id}",
                            reasoning=f"AIC={search_result.best_aic:.2f}, ICC={search_result.icc:.4f}",
                        )
                    else:
                        logger.warning(f"[ModelSearchAgent] {q_key}: {search_result.error_message}")
            except Exception as e:
                logger.warning(f"Phase 4.5: ModelSearchAgent 失败（不影响流程）: {e}")

        # ================================================================
        # Phase 5: 子任务循环（Coder → ResultInterpreter → Writer）
        # 注意：code_interpreter 已在 Phase 4 完成后提前初始化，此处直接使用
        # ================================================================

        # 初始化 Agent 实例
        coder_agent = CoderAgent(
            task_id=problem.task_id,
            model=coder_llm,
            work_dir=self.work_dir,
            max_chat_turns=settings.MAX_CHAT_TURNS,
            max_retries=settings.MAX_RETRIES,
            code_interpreter=code_interpreter,
            context_window=settings.CODER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        writer_agent = WriterAgent(
            task_id=problem.task_id,
            model=writer_llm,
            comp_template=problem.comp_template,
            format_output=problem.format_output,
            scholar=None,  # Scholar 在 LiteratureAgent 中已使用
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        result_interpreter = ResultInterpreterAgent(
            task_id=problem.task_id,
            model=writer_llm,
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        # 初始化 ReviewerAgent（正确性守卫，替代 CriticAgent）
        reviewer_agent = ReviewerAgent(
            task_id=problem.task_id,
            model=writer_llm,
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        # 初始化 AwardJudgeAgent（国奖评审官）
        award_judge_agent = AwardJudgeAgent(
            task_id=problem.task_id,
            model=writer_llm,
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        flows = Flows(self.questions)
        solution_flows = flows.get_solution_flows(self.questions, modeler_response)
        config_template = get_config_template(problem.comp_template)

        # 按依赖图的拓扑序排列子任务
        ques_keys_ordered = [k for k in dep_graph.execution_order if k in solution_flows]
        non_ques_keys = [k for k in solution_flows if k not in ques_keys_ordered]
        eda_keys = [k for k in non_ques_keys if k == "eda"]
        sa_keys = [k for k in non_ques_keys if k == "sensitivity_analysis"]
        other_keys = [k for k in non_ques_keys if k not in ("eda", "sensitivity_analysis")]
        ordered_keys = eda_keys + ques_keys_ordered + other_keys + sa_keys

        total_subtasks = len(ordered_keys)
        completed_subtasks = 0
        failed_subtasks = []

        for key in ordered_keys:
            value = solution_flows[key]
            await self._check_cancelled()
            completed_subtasks += 1

            try:
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"Phase 5: 代码手开始求解{key} ({completed_subtasks}/{total_subtasks})"),
                )

                # 构建依赖上下文（注入前序问题的结论）
                dependency_context = dep_graph.build_dependency_context(key)

                # 构建增强版 coder_prompt，注入依赖上下文 + 变量筛选结果
                enhanced_coder_prompt = value["coder_prompt"]
                if dependency_context:
                    enhanced_coder_prompt = f"{dependency_context}\n\n【本题任务】\n{value['coder_prompt']}"

                # 注入 ModelSearchAgent 的变量筛选结果
                if key in model_search_results and model_search_results[key].success:
                    search_summary = model_search_results[key].to_summary_text()
                    enhanced_coder_prompt += f"""

【模型搜索结果（来自 ModelSearchAgent，必须使用此最优模型）】
{search_summary}

【重要】
- 以上模型搜索已通过 AIC/BIC 比较和似然比检验
- 请直接使用上述最优模型进行拟合和预测
- 不需要重新进行变量筛选
- 但仍需报告模型的诊断指标（残差图、Q-Q 图等）
"""

                # 5a: CoderAgent 执行代码
                coder_response = await coder_agent.run(
                    prompt=enhanced_coder_prompt, subtask_title=key
                )

                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"Phase 5: 代码手求解成功{key}", type="success"),
                )

                # 5b: ResultInterpreterAgent 解读结果
                code_output = code_interpreter.get_code_output(key)
                model_spec_text = modeler_response.questions_solution.get(key, "") if hasattr(modeler_response, 'questions_solution') else ""

                # 从 GlobalState 中获取模型类型（如果有的话）
                model_type = "通用"
                if problem_analysis and hasattr(problem_analysis, 'method_families'):
                    for family_key, methods in problem_analysis.method_families.items():
                        if key in family_key or family_key in key:
                            model_type = family_key
                            break

                # 提前初始化为 None，确保无论 try 是否成功，下方引用都安全
                interpreter_result = None
                try:
                    interpreter_result = await result_interpreter.run(
                        code_output=code_output or "",
                        subtask_title=key,
                        model_spec=model_spec_text[:2000],
                        model_type=model_type,
                    )

                    # 将解读结果写入 GlobalState
                    from app.core.global_state import CodeResult, SanityCheckResult, FigureRecord
                    code_result = CodeResult(
                        execution_status="success",
                        key_metrics=interpreter_result.key_findings.key_numbers,
                        sanity_check_results=SanityCheckResult(
                            value_range_ok=interpreter_result.sanity_check.is_reasonable,
                            residuals_issue="; ".join(interpreter_result.sanity_check.issues) if interpreter_result.sanity_check.issues else "",
                            action_required="; ".join(interpreter_result.sanity_check.warnings) if interpreter_result.sanity_check.warnings else "",
                        ),
                        generated_figures=[
                            FigureRecord(
                                filename=fig.filename,
                                description=fig.description,
                                key_observation=fig.observation,
                                paper_narrative=f"{fig.observation} {fig.meaning} {fig.disposition}",
                            )
                            for fig in interpreter_result.figure_narratives
                        ],
                        conclusions_supported=interpreter_result.writeability.can_claim,
                        conclusions_not_supported=interpreter_result.writeability.cannot_claim,
                    )
                    global_state.update_code_result(key, code_result)

                    # 注入 PaperContext（兼容）
                    if interpreter_result.key_findings.conclusion:
                        paper_context.update_key_result(
                            section_key=key,
                            conclusion=interpreter_result.key_findings.conclusion,
                            key_numbers=interpreter_result.key_findings.key_numbers,
                        )

                    if not interpreter_result.sanity_check.is_reasonable:
                        warn_msg = f"{key} 结果可能不合理: {', '.join(interpreter_result.sanity_check.issues)}"
                        logger.warning(f"[ResultInterpreter] {warn_msg}")
                        await redis_manager.publish_message(
                            self.task_id,
                            SystemMessage(content=warn_msg, type="warning"),
                        )
                except Exception as e:
                    logger.warning(f"Phase 5b: 结果解读失败（不影响流程）: {e}")

                # 更新 PaperContext：从代码执行结果中提取关键数据
                if coder_response.code_response:
                    paper_context.update_key_result(
                        section_key=key,
                        conclusion=coder_response.code_response[:300] if coder_response.code_response else "",
                        figures=coder_response.created_images or [],
                    )
                code_output = code_interpreter.get_code_output(key)
                if code_output:
                    paper_context._extract_key_numbers(key, code_output)
                if hasattr(modeler_response, 'questions_solution') and key in modeler_response.questions_solution:
                    model_text = modeler_response.questions_solution[key]
                    first_line = model_text.split("\n")[0][:100] if model_text else ""
                    paper_context.model_choices[key] = first_line

                # 5c: WriterAgent 写模型章节
                writer_prompt = flows.get_writer_prompt(
                    key, coder_response.code_response or "", code_interpreter, config_template,
                    paper_context=paper_context,
                )

                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"Phase 5: 论文手开始写{key}部分"),
                )

                writer_response = await writer_agent.run(
                    writer_prompt,
                    available_images=coder_response.created_images,
                    sub_title=key,
                )

                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"Phase 5: 论文手完成{key}部分", type="success"),
                )

                user_output.set_res(key, writer_response)

                # 更新 GlobalState 论文状态
                global_state.paper_state.chapters_completed.append(key)
                global_state.extract_from_writer_response(writer_response.response_content)

                # Phase 5d-1: ReviewerAgent 正确性审查
                review_verdict = None
                try:
                    review_verdict = await reviewer_agent.review(
                        target_output=writer_response.response_content,
                        review_type="chapter_logic",
                        global_state_summary=global_state.inject_summary("ReviewerAgent"),
                    )

                    if review_verdict.issues:
                        logger.info(
                            f"[ReviewerAgent] {key}: 正确性分 {review_verdict.correctness_score}/100, "
                            f"发现 {len(review_verdict.issues)} 个问题 (决策: {review_verdict.decision})"
                        )

                    # reject 时注入反馈并重写
                    if review_verdict.decision == "reject":
                        logger.warning(f"[ReviewerAgent] {key}: 致命问题，启动重写")
                        await redis_manager.publish_message(
                            self.task_id,
                            SystemMessage(
                                content=f"ReviewerAgent 对 {key} 发现致命问题，正在重写...",
                                type="warning",
                            ),
                        )
                        issues_text = "\n".join(
                            f"- [{it.get('severity')}] {it.get('issue', '')}"
                            for it in review_verdict.issues
                        )
                        rewrite_prompt = f"""请根据以下正确性审查反馈重新撰写本章节。

【原始任务】
{writer_prompt}

【正确性审查反馈（致命问题）】
{issues_text}

【修改要求】
1. 重点解决审查中指出的所有致命问题
2. 确保数学正确性、逻辑自洽性、数据一致性
3. 保持论文的整体结构和格式

请输出完整的修改后内容。"""
                        writer_response = await writer_agent.run(
                            rewrite_prompt,
                            available_images=coder_response.created_images,
                            sub_title=f"{key} (正确性重写)",
                        )
                        user_output.set_res(key, writer_response)
                        global_state.extract_from_writer_response(writer_response.response_content)
                except Exception as e:
                    logger.warning(f"Phase 5d-1: ReviewerAgent 审查失败（不影响流程）: {e}")

                # Phase 5d-2: AwardJudgeAgent 国奖潜力评估
                # 注意：不过滤 reject 决策——ReviewerAgent 重写后的内容同样需要国奖评审
                # 原逻辑 decision != "reject" 会导致重写后的章节绕过国奖评审
                if review_verdict is not None:
                    try:
                        award_verdict = await award_judge_agent.evaluate(
                            paper_content=writer_response.response_content,
                            global_state_summary=global_state.inject_summary("AwardJudgeAgent"),
                            competition_type=problem.comp_template.value if hasattr(problem.comp_template, 'value') else str(problem.comp_template),
                        )

                        logger.info(
                            f"[AwardJudgeAgent] {key}: 国奖分 {award_verdict.total_score}/100 "
                            f"(创新性 {award_verdict.innovation_score}/30), 决策: {award_verdict.decision}"
                        )

                        # Innovation < 20 或总分 < 60：注入反馈并重写
                        if award_verdict.decision in ("innovation_reject", "rewrite"):
                            is_innovation = award_verdict.decision == "innovation_reject"
                            reason = (
                                f"创新性不足 ({award_verdict.innovation_score}/30)"
                                if is_innovation
                                else f"国奖总分不足 ({award_verdict.total_score}/100)"
                            )
                            logger.warning(f"[AwardJudgeAgent] {key}: {reason}，启动重写")
                            await redis_manager.publish_message(
                                self.task_id,
                                SystemMessage(content=f"AwardJudgeAgent 对 {key}: {reason}，正在重写...", type="warning"),
                            )

                            award_feedback_parts = []
                            if is_innovation:
                                award_feedback_parts.append("【创新性不足 —— 这是国奖硬门槛，必须突破】")
                            for s in award_verdict.improvement_suggestions[:5]:
                                award_feedback_parts.append(
                                    f"- [{s.get('dimension')}] {s.get('item', '')} (扣{s.get('points_lost', 0)}分): {s.get('fix', '')}"
                                )
                            award_feedback_text = "\n".join(award_feedback_parts)

                            award_rewrite_prompt = f"""请根据以下国奖评审反馈修改本章节。
当前国奖评分: {award_verdict.total_score}/100，创新性: {award_verdict.innovation_score}/30

【原始任务】
{writer_prompt}

【国奖评审反馈】
{award_feedback_text}

【修改要求】
{"1. 首要任务：提升创新性。考虑引入跨学科方法、独特的视角变换。" if is_innovation else "1. 按优先级解决国奖评审中的扣分项。"}
2. 增强解释性：图表论证使用三段式（观察→含义→处置）。
3. 保持数学正确性。

请输出完整的修改后内容。"""

                            writer_response = await writer_agent.run(
                                award_rewrite_prompt,
                                available_images=coder_response.created_images,
                                sub_title=f"{key} (国奖重写)",
                            )
                            user_output.set_res(key, writer_response)
                            global_state.extract_from_writer_response(writer_response.response_content)
                    except Exception as e:
                        logger.warning(f"Phase 5d-2: AwardJudgeAgent 评估失败（不影响流程）: {e}")

                # 记录结论到依赖图（供后续子问题使用）
                core_conclusion = ""
                key_outputs = {}
                # 用 is not None 判断（比 dir() 更可靠，dir() 在异步上下文中不含局部变量）
                if interpreter_result is not None:
                    if hasattr(interpreter_result, 'key_findings') and interpreter_result.key_findings.conclusion:
                        core_conclusion = interpreter_result.key_findings.conclusion[:500]
                    if hasattr(interpreter_result, 'key_findings'):
                        key_outputs = getattr(interpreter_result.key_findings, 'key_numbers', {}) or {}
                if not core_conclusion and coder_response.code_response:
                    core_conclusion = coder_response.code_response[:500]

                dep_graph.record_conclusion(
                    node_id=key,
                    core_conclusion=core_conclusion,
                    key_outputs=key_outputs,
                    conclusion_source="ResultInterpreterAgent",
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                error_msg = f"子任务 {key} 执行失败: {str(e)}"
                logger.error(error_msg)
                failed_subtasks.append(key)
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=error_msg, type="error"),
                )
                continue

        if failed_subtasks:
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"警告: {len(failed_subtasks)} 个子任务失败: {', '.join(failed_subtasks)}",
                    type="warning"
                ),
            )

        # 关闭沙盒
        await code_interpreter.cleanup()
        logger.info(user_output.get_res())

        # ================================================================
        # Phase 6: OutlineAgent（生成论文大纲）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 6: 生成论文大纲..."),
        )
        await self._check_cancelled()

        outline_text = ""
        try:
            outline_agent = OutlineAgent(
                task_id=problem.task_id,
                model=writer_llm,
                context_window=settings.WRITER_CONTEXT_WINDOW,
                cancel_event=self.cancel_event,
                diagnostic_logger=diag,
            )
            outline_text = await outline_agent.run(
                global_state_summary=global_state.to_summary(),
                competition_type=problem.comp_template.value if hasattr(problem.comp_template, 'value') else str(problem.comp_template),
            )
            logger.info(f"[OutlineAgent] 大纲生成完成，长度: {len(outline_text)}")

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content="Phase 6: 论文大纲生成完成", type="success"),
            )
        except Exception as e:
            logger.warning(f"Phase 6: 论文大纲生成失败（不影响流程）: {e}")
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"论文大纲生成失败: {e}，继续流程", type="warning"),
            )

        # ================================================================
        # Phase 7: 写作剩余章节（问题分析、结果分析等）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 7: 撰写剩余章节..."),
        )

        review_agent = ReviewAgent(
            task_id=problem.task_id,
            model=writer_llm,
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        multi_reviewer = MultiReviewer(
            task_id=problem.task_id,
            model=writer_llm,
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        write_flows = flows.get_write_flows(
            user_output, config_template, problem.ques_all,
            paper_context=paper_context,
        )
        failed_write_sections = []

        for key, value in write_flows.items():
            await self._check_cancelled()

            try:
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"Phase 7: 论文手开始写{key}部分"),
                )

                # 如果有大纲，将大纲注入写作 prompt
                enhanced_prompt = value
                if outline_text:
                    enhanced_prompt = (
                        f"{value}\n\n"
                        f"【论文大纲参考】\n{outline_text[:3000]}\n\n"
                        f"【全局状态摘要】\n{global_state.inject_summary('WriterAgent')}"
                    )

                writer_response = await self._writing_with_reflexion(
                    writer_agent=writer_agent,
                    review_agent=review_agent,
                    prompt=enhanced_prompt,
                    sub_title=key,
                    quality_tracker=quality_tracker,
                    multi_reviewer=multi_reviewer,
                )

                user_output.set_res(key, writer_response)

                # 更新 PaperContext 和 GlobalState
                paper_context.extract_from_writer_response(key, writer_response.response_content)
                global_state.extract_from_writer_response(writer_response.response_content)
                if key not in global_state.paper_state.chapters_completed:
                    global_state.paper_state.chapters_completed.append(key)

                # 结构控制：检查章节篇幅
                section_report = structure_controller.check_section(
                    key, writer_response.response_content
                )
                if section_report.status == SectionStatus.TOO_SHORT:
                    warn_msg = (
                        f"章节「{section_report.section_name}」篇幅不足 "
                        f"({section_report.char_count}字, "
                        f"目标{section_report.target_min}-{section_report.target_max}字)。"
                        f"{section_report.feedback}"
                    )
                    logger.warning(f"[结构控制] {warn_msg}")
                    await redis_manager.publish_message(
                        self.task_id,
                        SystemMessage(content=warn_msg, type="warning"),
                    )
                elif section_report.status == SectionStatus.TOO_LONG:
                    warn_msg = (
                        f"章节「{section_report.section_name}」篇幅超出 "
                        f"({section_report.char_count}字, "
                        f"目标{section_report.target_min}-{section_report.target_max}字)。"
                    )
                    logger.warning(f"[结构控制] {warn_msg}")

                for issue in section_report.redundancy_issues:
                    logger.warning(f"[结构控制-去重] {section_report.section_name}: {issue}")
                for issue in section_report.quality_issues:
                    logger.warning(f"[结构控制-质量] {section_report.section_name}: {issue}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                error_msg = f"写作子任务 {key} 执行失败: {str(e)}"
                logger.error(error_msg)
                failed_write_sections.append(key)
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=error_msg, type="error"),
                )
                continue

        if failed_write_sections:
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"警告: {len(failed_write_sections)} 个写作任务失败: {', '.join(failed_write_sections)}",
                    type="warning"
                ),
            )

        logger.info(user_output.get_res())

        # ================================================================
        # Phase 8: ConsistencyAgent（全文一致性检查）
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 8: 全文一致性检查..."),
        )
        await self._check_cancelled()

        # 拼接所有章节文本（在 try 之前定义，确保后续可用）
        all_sections = {
            key: val.get("response_content", "")
            for key, val in user_output.get_res().items()
        }

        try:
            consistency_agent = ConsistencyAgent(
                task_id=problem.task_id,
                model=writer_llm,
                context_window=settings.WRITER_CONTEXT_WINDOW,
                cancel_event=self.cancel_event,
                diagnostic_logger=diag,
            )

            # 拼接所有章节文本
            all_chapters_text = "\n\n".join(
                f"## {k}\n{v}" for k, v in all_sections.items() if v
            )

            consistency_report = await consistency_agent.run(
                all_chapters_text=all_chapters_text,
                global_state_summary=global_state.to_summary(),
            )
            logger.info(f"[ConsistencyAgent] 一致性检查完成，报告长度: {len(consistency_report)}")

            # 将一致性检查结果记录到诊断日志
            if diag:
                diag.save_structure_report({
                    "consistency_report_length": len(consistency_report),
                    "consistency_report_preview": consistency_report[:2000],
                })

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content="Phase 8: 全文一致性检查完成", type="success"),
            )
        except Exception as e:
            logger.warning(f"Phase 8: 全文一致性检查失败（不影响流程）: {e}")
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"一致性检查失败: {e}，继续流程", type="warning"),
            )

        # 全文结构控制检查
        paper_report = structure_controller.check_full_paper(all_sections)

        if paper_report.overall_feedback:
            logger.info(f"[结构控制] 全文报告:\n{paper_report.overall_feedback}")
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"论文结构检测完成:\n{paper_report.overall_feedback}",
                    type="info",
                ),
            )

        for issue in paper_report.global_issues:
            logger.warning(f"[结构控制-全局去重] {issue}")

        diag.save_structure_report({
            "total_chars": paper_report.total_chars,
            "target_length": paper_report.target_length,
            "ratio": paper_report.total_chars / paper_report.target_length,
            "sections": [
                {
                    "name": r.section_name,
                    "key": r.key,
                    "char_count": r.char_count,
                    "target_min": r.target_min,
                    "target_max": r.target_max,
                    "status": r.status.value,
                    "quality_issues": r.quality_issues,
                    "redundancy_issues": r.redundancy_issues,
                }
                for r in paper_report.section_reports
            ],
            "global_issues": paper_report.global_issues,
        })

        # ================================================================
        # Phase 9: MultiReviewer（三审制） + ReviewSynthesizer
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 9: 三审制评审..."),
        )
        await self._check_cancelled()

        # 拼接完整论文用于全局评审
        full_paper_text = "\n\n".join(
            f"## {k}\n{v.get('response_content', '')}"
            for k, v in user_output.get_res().items()
            if v.get("response_content")
        )

        try:
            # 三审制并行评审
            final_review = await multi_reviewer.run(
                paper_content=full_paper_text,
                section_name="完整论文",
            )

            logger.info(
                f"[MultiReviewer] 三审制完成 - 综合得分: {final_review.overall_score}/100"
            )

            # 将评审结果写入 GlobalState
            from app.core.global_state import ReviewRecord
            review_record = ReviewRecord(
                round=1,
                reviewer="MultiReviewer",
                score=final_review.overall_score,
            )
            global_state.add_review_record(review_record)

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"Phase 9: 三审制评审完成，综合得分: {final_review.overall_score}/100",
                    type="success" if final_review.overall_score >= settings.REFLEXION_QUALITY_THRESHOLD else "warning",
                ),
            )
        except Exception as e:
            logger.warning(f"Phase 9: 三审制评审失败（不影响流程）: {e}")
            final_review = None
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"三审制评审失败: {e}，继续流程", type="warning"),
            )

        # ================================================================
        # Phase 10: Reflexion 循环（如果分数低于阈值）
        # ================================================================
        if final_review and final_review.overall_score < settings.REFLEXION_QUALITY_THRESHOLD:
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"Phase 10: 综合得分 {final_review.overall_score}/100 低于阈值 "
                            f"{settings.REFLEXION_QUALITY_THRESHOLD}，启动 Reflexion 改进循环...",
                ),
            )

            # 对得分最低的章节进行改进
            reflexion_config = _get_reflexion_config()
            max_iterations = reflexion_config['max_iterations']

            for iteration in range(max_iterations):
                await self._check_cancelled()

                # 找出需要改进的章节（基于评审反馈）
                sections_to_improve = self._identify_sections_to_improve(
                    final_review, user_output
                )

                if not sections_to_improve:
                    logger.info(f"[Reflexion] 第 {iteration + 1} 轮：没有需要改进的章节")
                    break

                for section_key in sections_to_improve:
                    await self._check_cancelled()

                    section_content = user_output.get_res().get(section_key, {}).get("response_content", "")
                    if not section_content:
                        continue

                    improvement_prompt = f"""请根据以下评审反馈修改论文内容。

【评审反馈】
{final_review.feedback[:3000]}

【改进建议】
{chr(10).join(f'- {imp}' for imp in final_review.improvements[:5])}

【原始章节内容】
{section_content[:5000]}

【修改要求】
1. 重点解决评审中指出的问题
2. 保持论文的整体结构
3. 确保修改后的内容更加准确和完善
4. 注入全局状态信息以保持一致性

{global_state.inject_summary('WriterAgent')}

请输出完整的修改后内容。"""

                    try:
                        revised_response = await writer_agent.run(
                            prompt=improvement_prompt,
                            sub_title=f"{section_key} (Reflexion {iteration + 1})",
                        )
                        user_output.set_res(section_key, revised_response)

                        # 更新 GlobalState
                        global_state.extract_from_writer_response(revised_response.response_content)
                    except Exception as e:
                        logger.warning(f"[Reflexion] {section_key} 改进失败: {e}")

                # 重新评审
                full_paper_text = "\n\n".join(
                    f"## {k}\n{v.get('response_content', '')}"
                    for k, v in user_output.get_res().items()
                    if v.get("response_content")
                )

                try:
                    final_review = await multi_reviewer.run(
                        paper_content=full_paper_text,
                        section_name="完整论文",
                    )

                    # 记录评审历史
                    review_record = ReviewRecord(
                        round=iteration + 2,
                        reviewer="MultiReviewer",
                        score=final_review.overall_score,
                    )
                    global_state.add_review_record(review_record)

                    logger.info(
                        f"[Reflexion] 第 {iteration + 1} 轮改进后得分: {final_review.overall_score}/100"
                    )

                    await redis_manager.publish_message(
                        self.task_id,
                        SystemMessage(
                            content=f"Reflexion 第 {iteration + 1} 轮: 得分 {final_review.overall_score}/100",
                            type="success" if final_review.overall_score >= settings.REFLEXION_QUALITY_THRESHOLD else "warning",
                        ),
                    )

                    if final_review.overall_score >= settings.REFLEXION_QUALITY_THRESHOLD:
                        logger.info("[Reflexion] 质量达标，停止改进循环")
                        break
                except Exception as e:
                    logger.warning(f"[Reflexion] 第 {iteration + 1} 轮重新评审失败: {e}")
                    break

        # ================================================================
        # Phase 11: 保存 GlobalState + 诊断数据
        # ================================================================
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="Phase 11: 保存数据..."),
        )

        # 打印质量跟踪摘要
        quality_tracker.print_summary()

        # 保存诊断数据
        diag.save_quality_data(quality_tracker.get_summary())

        # 保存 PaperContext（兼容）
        paper_context.save(self.work_dir)
        logger.info(
            f"[PaperContext] 已保存论文上下文，共 {len(paper_context.key_results)} 个关键结果，"
            f"{len(paper_context.defined_symbols)} 个符号定义"
        )

        # 保存 GlobalState
        global_state.task_meta.current_phase = "completed"
        global_state_path = f"{self.work_dir}/diagnostic/global_state.json"
        global_state.save(global_state_path)
        logger.info(
            f"[GlobalState] 已保存全局状态: {len(global_state.agent_decisions_log)} 条决策记录, "
            f"{len(global_state.modeling_decisions)} 个建模决策, "
            f"{len(global_state.code_results)} 个代码结果"
        )

        user_output.save_result()

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="所有阶段完成!", type="success"),
        )

    def _identify_sections_to_improve(
        self,
        review_result: 'ReviewResponse',
        user_output: UserOutput,
    ) -> list[str]:
        """根据评审结果识别需要改进的章节。

        Args:
            review_result: 评审结果。
            user_output: 用户输出对象。

        Returns:
            需要改进的章节 key 列表。
        """
        # 基于段落级定位找到需要改进的章节
        sections_to_improve: list[str] = []

        if review_result.paragraph_issues:
            # 从 paragraph_issues 中提取章节信息
            for issue in review_result.paragraph_issues:
                # 尝试从 issue 中提取章节 key
                if hasattr(issue, 'section_key') and issue.section_key:
                    if issue.section_key not in sections_to_improve:
                        sections_to_improve.append(issue.section_key)

        # 如果没有精确定位，改进得分较低的章节
        if not sections_to_improve:
            # 默认改进所有章节
            all_keys = list(user_output.get_res().keys())
            # 优先改进摘要和结论（通常评审重点）
            priority_keys = ["firstPage", "judge", "sensitivity_analysis"]
            for key in priority_keys:
                if key in all_keys and key not in sections_to_improve:
                    sections_to_improve.append(key)
            # 补充其他章节
            for key in all_keys:
                if key not in sections_to_improve:
                    sections_to_improve.append(key)

        return sections_to_improve[:3]  # 每轮最多改进 3 个章节

    async def _writing_with_reflexion(
        self,
        writer_agent: WriterAgent,
        review_agent: ReviewAgent,
        prompt: str,
        sub_title: str,
        quality_tracker: QualityTracker | None = None,
        multi_reviewer: MultiReviewer | None = None,
    ) -> WriterResponse:
        """带 Reflexion 循环的写作流程。

        实现：生成 -> 评审 -> 反馈 -> 改进 的迭代循环。
        使用三审制（方法论/写作/格式）进行并行评审。

        Args:
            writer_agent: 写作 Agent。
            review_agent: 评审 Agent（综合评审，用于 Reflexion 改进轮次）。
            prompt: 写作提示。
            sub_title: 子任务标题。
            quality_tracker: 质量跟踪器。
            multi_reviewer: 三审制协调器（首轮使用，提供更全面的反馈）。

        Returns:
            WriterResponse 对象。
        """
        from app.schemas.A2A import WriterResponse

        reflexion_config = _get_reflexion_config()

        # 如果 Reflexion 未启用，直接生成
        if not reflexion_config['enabled']:
            return await writer_agent.run(prompt=prompt, sub_title=sub_title)

        max_iterations = reflexion_config['max_iterations']
        quality_threshold = reflexion_config['quality_threshold']

        current_response = None
        feedback = ""

        for iteration in range(max_iterations):
            await self._check_cancelled()

            # 1. 生成论文
            if iteration == 0:
                current_response = await writer_agent.run(
                    prompt=prompt,
                    sub_title=sub_title,
                )
            else:
                refinement_prompt = f"""请根据以下评审反馈修改论文内容。

【原始任务】
{prompt}

【评审反馈】
{feedback}

【修改要求】
1. 重点解决评审中指出的问题
2. 保持论文的整体结构
3. 确保修改后的内容更加准确和完善

请输出完整的修改后内容。"""

                current_response = await writer_agent.run(
                    prompt=refinement_prompt,
                    sub_title=f"{sub_title} (修订 {iteration + 1})",
                )

            # 2. 评审（首轮使用三审制，后续轮次使用综合评审以节省 token）
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"正在评审{sub_title} (第 {iteration + 1} 轮)"),
            )

            if iteration == 0 and multi_reviewer:
                review_result = await multi_reviewer.run(
                    paper_content=current_response.response_content,
                    section_name=sub_title,
                )
            else:
                review_result = await review_agent.run(
                    paper_content=current_response.response_content,
                    section_name=sub_title,
                    sub_title=f"评审 {sub_title}",
                )

            # 3. 记录质量报告
            quality_score = QualityScore(
                math_score=review_result.math_score,
                logic_score=review_result.logic_score,
                language_score=review_result.language_score,
                format_score=review_result.format_score,
            )
            quality_report = QualityReport(
                section_name=sub_title,
                score=quality_score,
                strengths=review_result.strengths,
                improvements=review_result.improvements,
                feedback=review_result.feedback,
                iteration=iteration + 1,
            )
            if quality_tracker:
                quality_tracker.add_report(quality_report)

            # 4. 检查质量是否达标
            if review_result.overall_score >= quality_threshold:
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(
                        content=f"{sub_title} 质量达标 (得分: {review_result.overall_score}/100)",
                        type="success"
                    ),
                )
                break

            # 5. 生成反馈
            feedback = self._generate_reflexion_feedback(review_result)

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"{sub_title} 需要改进 (得分: {review_result.overall_score}/100, 第 {iteration + 1} 轮修订)",
                    type="warning"
                ),
            )

            logger.info(
                f"Reflexion: {sub_title} 第 {iteration + 1} 轮评审得分 "
                f"{review_result.overall_score}/100, 继续改进..."
            )

        return current_response

    def _generate_reflexion_feedback(self, review_result: 'ReviewResponse') -> str:
        """根据评审结果生成反馈。

        Args:
            review_result: 评审结果。

        Returns:
            格式化的反馈字符串。
        """
        feedback_parts = [
            f"## 评审反馈 (总分: {review_result.overall_score}/100)",
            "",
            "### 评分明细",
            f"- 数学正确性: {review_result.math_score}/25",
            f"- 逻辑连贯性: {review_result.logic_score}/25",
            f"- 语言质量: {review_result.language_score}/25",
            f"- 格式规范: {review_result.format_score}/25",
            "",
        ]

        if review_result.strengths:
            feedback_parts.append("### 优点")
            for strength in review_result.strengths:
                feedback_parts.append(f"- {strength}")
            feedback_parts.append("")

        if review_result.improvements:
            feedback_parts.append("### 需要改进的地方")
            for i, improvement in enumerate(review_result.improvements, 1):
                feedback_parts.append(f"{i}. {improvement}")
            feedback_parts.append("")

        return "\n".join(feedback_parts)
