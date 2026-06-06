"""工作流模块，编排多 Agent 协作完成数学建模任务。"""

import asyncio
from app.core.agents import WriterAgent, CoderAgent, CoordinatorAgent, ModelerAgent, ReviewAgent
from app.schemas.request import Problem
from app.schemas.response import SystemMessage
from app.tools.openalex_scholar import OpenAlexScholar
from app.utils.log_util import logger
from app.utils.common_utils import create_work_dir, get_config_template
from app.models.user_output import UserOutput
from app.config.setting import settings
from app.tools.interpreter_factory import create_interpreter
from app.services.redis_manager import redis_manager
from app.tools.notebook_serializer import NotebookSerializer
from app.core.flows import Flows
from app.core.llm.llm_factory import LLMFactory

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
        # RichPrinter.workflow_start()
        # RichPrinter.workflow_end()
        pass


class MathModelWorkFlow(WorkFlow):
    """数学建模工作流，协调协调者、建模手、代码手和写作手完成完整建模任务。"""
    task_id: str  #
    work_dir: str  # worklow work dir
    ques_count: int = 0  # 问题数量
    questions: dict[str, str | int] = {}  # 问题
    cancel_event: asyncio.Event | None = None  # 取消信号

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

        llm_factory = LLMFactory(self.task_id)
        coordinator_llm, modeler_llm, coder_llm, writer_llm = llm_factory.get_all_llms()

        coordinator_agent = CoordinatorAgent(
            self.task_id, coordinator_llm,
            context_window=settings.COORDINATOR_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
        )

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="识别用户意图和拆解问题ing..."),
        )

        await self._check_cancelled()

        try:
            coordinator_response = await coordinator_agent.run(problem.ques_all)
            self.questions = coordinator_response.questions
            self.ques_count = coordinator_response.ques_count
        except Exception as e:
            #  非数学建模问题
            logger.error(f"CoordinatorAgent 执行失败: {e}")
            raise e

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="识别用户意图和拆解问题完成,任务转交给建模手"),
        )

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="建模手开始建模ing..."),
        )

        await self._check_cancelled()

        modeler_agent = ModelerAgent(
            self.task_id, modeler_llm,
            context_window=settings.MODELER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
        )

        modeler_response = await modeler_agent.run(coordinator_response)

        user_output = UserOutput(work_dir=self.work_dir, ques_count=self.ques_count)

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="正在创建代码沙盒环境"),
        )

        notebook_serializer = NotebookSerializer(work_dir=self.work_dir)
        code_interpreter = await create_interpreter(
            kind="local",
            task_id=self.task_id,
            work_dir=self.work_dir,
            notebook_serializer=notebook_serializer,
            timeout=3000,
        )
        
        assert settings.OPENALEX_EMAIL is not None, "OPENALEX_EMAIL 未配置"
        scholar = OpenAlexScholar(
            task_id=self.task_id,
            email=settings.OPENALEX_EMAIL,
            api_key=settings.OPENALEX_API_KEY,
        )

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="创建完成"),
        )

        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="初始化代码手"),
        )

        # modeler_agent
        coder_agent = CoderAgent(
            task_id=problem.task_id,
            model=coder_llm,
            work_dir=self.work_dir,
            max_chat_turns=settings.MAX_CHAT_TURNS,
            max_retries=settings.MAX_RETRIES,
            code_interpreter=code_interpreter,
            context_window=settings.CODER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
        )

        writer_agent = WriterAgent(
            task_id=problem.task_id,
            model=writer_llm,
            comp_template=problem.comp_template,
            format_output=problem.format_output,
            scholar=scholar,
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
        )

        # 初始化评审 Agent（用于 Reflexion 循环）
        review_agent = ReviewAgent(
            task_id=problem.task_id,
            model=writer_llm,  # 使用同一个 LLM
            context_window=settings.WRITER_CONTEXT_WINDOW,
            cancel_event=self.cancel_event,
        )

        flows = Flows(self.questions)

        ################################################ solution steps
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
                    SystemMessage(content=f"代码手开始求解{key} ({completed_subtasks}/{total_subtasks})"),
                )

                coder_response = await coder_agent.run(
                    prompt=value["coder_prompt"], subtask_title=key
                )

                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"代码手求解成功{key}", type="success"),
                )

                writer_prompt = flows.get_writer_prompt(
                    key, coder_response.code_response or "", code_interpreter, config_template
                )

                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"论文手开始写{key}部分"),
                )

                writer_response = await writer_agent.run(
                    writer_prompt,
                    available_images=coder_response.created_images,
                    sub_title=key,
                )

                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content=f"论文手完成{key}部分", type="success"),
                )

                user_output.set_res(key, writer_response)

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
                # 继续执行下一个子任务，而不是终止整个工作流
                continue

        if failed_subtasks:
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(
                    content=f"警告: {len(failed_subtasks)} 个子任务失败: {', '.join(failed_subtasks)}",
                    type="warning"
                ),
            )

    async def _writing_with_reflexion(
        self,
        writer_agent: WriterAgent,
        review_agent: ReviewAgent,
        prompt: str,
        sub_title: str,
    ) -> 'WriterResponse':
        """带 Reflexion 循环的写作流程。

        实现：生成 -> 评审 -> 反馈 -> 改进 的迭代循环。

        Args:
            writer_agent: 写作 Agent。
            review_agent: 评审 Agent。
            prompt: 写作提示。
            sub_title: 子任务标题。

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
                # 使用反馈重新生成
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

            # 2. 评审
            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"正在评审{sub_title} (第 {iteration + 1} 轮)"),
            )

            review_result = await review_agent.run(
                paper_content=current_response.response_content,
                section_name=sub_title,
                sub_title=f"评审 {sub_title}",
            )

            # 3. 检查质量是否达标
            if review_result.overall_score >= quality_threshold:
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(
                        content=f"{sub_title} 质量达标 (得分: {review_result.overall_score}/100)",
                        type="success"
                    ),
                )
                break

            # 4. 生成反馈
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

        # 关闭沙盒

        await code_interpreter.cleanup()
        logger.info(user_output.get_res())

        ################################################ write steps with Reflexion

        write_flows = flows.get_write_flows(
            user_output, config_template, problem.ques_all
        )
        for key, value in write_flows.items():
            await self._check_cancelled()

            await redis_manager.publish_message(
                self.task_id,
                SystemMessage(content=f"论文手开始写{key}部分"),
            )

            # Reflexion 循环：生成 -> 评审 -> 改进
            writer_response = await self._writing_with_reflexion(
                writer_agent=writer_agent,
                review_agent=review_agent,
                prompt=value,
                sub_title=key,
            )

            user_output.set_res(key, writer_response)

        logger.info(user_output.get_res())

        user_output.save_result()
