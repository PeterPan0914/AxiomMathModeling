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
)
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
            problem_analysis = await problem_analyst.run(problem.ques_all, {})
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
        # Phase 4: ModelerAgent（建模）
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

        modeler_response = await modeler_agent.run(
            coordinator_response, problem_analysis=problem_analysis_text
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
        # Phase 5: 子任务循环（Coder → ResultInterpreter → Writer）
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

        # 初始化 CriticAgent（质疑者，Phase 5 中使用）
        critic_agent = CriticAgent(
            task_id=problem.task_id,
            model=writer_llm,
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
            diagnostic_logger=diag,
        )

        flows = Flows(self.questions)
        solution_flows = flows.get_solution_flows(self.questions, modeler_response)
        config_template = get_config_template(problem.comp_template)

        total_subtasks = len(solution_flows)
        completed_subtasks = 0
        failed_subtasks = []

        for key, value in solution_flows.items():
            await self._check_cancelled()
            completed_subtasks += 1

            try:
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"Phase 5: 代码手开始求解{key} ({completed_subtasks}/{total_subtasks})"),
                )

                # 5a: CoderAgent 执行代码
                coder_response = await coder_agent.run(
                    prompt=value["coder_prompt"], subtask_title=key
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

                # Phase 5d: CriticAgent 质疑写作结果
                try:
                    critique_result = await critic_agent.critique(
                        target_output=writer_response.response_content,
                        critique_type="chapter_logic",
                        global_state_summary=global_state.inject_summary("CriticAgent"),
                    )
                    if critique_result.issues:
                        logger.info(
                            f"[CriticAgent] {key}: 发现 {len(critique_result.issues)} 个问题 "
                            f"(决策: {critique_result.decision})"
                        )
                        for issue in critique_result.issues:
                            logger.warning(f"  - [{issue.get('severity')}] {issue.get('issue', '')[:100]}")

                        # reject 时注入反馈并重新写作（最多重试 1 次）
                        if critique_result.decision == "reject":
                            logger.warning(f"[CriticAgent] {key}: 致命问题，启动重写")
                            await redis_manager.publish_message(
                                self.task_id,
                                SystemMessage(
                                    content=f"CriticAgent 对 {key} 提出严重质疑，正在重新写作...",
                                    type="warning",
                                ),
                            )

                            # 构造带反馈的重写提示
                            issues_text = "\n".join(
                                f"- [{it.get('severity')}] {it.get('issue', '')}"
                                for it in critique_result.issues
                            )
                            rewrite_prompt = f"""请根据以下评审反馈重新撰写本章节。

【原始任务】
{writer_prompt}

【评审反馈（致命问题）】
{issues_text}

【修改要求】
1. 重点解决评审中指出的所有问题
2. 保持论文的整体结构和格式
3. 确保内容准确、逻辑自洽

请输出完整的修改后内容。"""

                            writer_response = await writer_agent.run(
                                rewrite_prompt,
                                available_images=coder_response.created_images,
                                sub_title=f"{key} (重写)",
                            )
                            user_output.set_res(key, writer_response)
                            global_state.extract_from_writer_response(writer_response.response_content)
                            logger.info(f"[CriticAgent] {key}: 重写完成")
                except Exception as e:
                    logger.warning(f"Phase 5d: CriticAgent 质疑失败（不影响流程）: {e}")

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
