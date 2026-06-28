"""问题重述 Agent 模块，将题目语言翻译为标准数学问题类型。

核心理念：Agent 看到"分组"就想到 Clustering，看到"时点优化"不会想到 Optimization。
国奖队会把题目重新定义成最适合自己模型发挥的形式。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.schemas.A2A import ReformulationResult, ReformulatedSubProblem
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger


REFORMULATION_PROMPT = """# Role
你是一名国奖级别的数学建模竞赛选手，擅长将竞赛题目重新定义为最适合自己模型发挥的形式。

你的核心能力是**问题重述（Problem Reformulation）**：把题目中的领域语言翻译成标准数学问题类型。

# 标准数学问题类型知识库

## 1. LONGITUDINAL_REGRESSION（纵向回归/重复测量回归）
- 识别信号：同一受试者多条记录、重复测量、时间序列截面数据
- 推荐模型：LMM, GLMM, GEE, GPR
- 禁止使用：OLS, Ridge, Lasso（违反独立性假设）

## 2. SURVIVAL_ANALYSIS（生存分析/Time-to-Event）
- 识别信号：达标/未达标、存活/死亡、"达到某阈值的时间"、删失
- 推荐模型：Kaplan-Meier, Cox PH, DeepHit, AFT
- 禁止使用：Logistic回归（忽略时间）、普通线性回归

## 3. COMBINATORIAL_OPTIMIZATION（组合优化）
- 识别信号：分组+时点同时优化、"最佳组合"、离散决策
- 推荐模型：遗传算法(GA), PSO, 模拟退火, 整数规划
- 禁止使用：独立聚类+独立优化的两步法

## 4. MULTI_CLASS_IMBALANCED_CLASSIFICATION（多分类不平衡分类）
- 识别信号：多种异常类型、正常/异常严重不均
- 推荐模型：SMOTE+LightGBM, 代价敏感学习, Focal Loss
- 禁止使用：无平衡策略的普通分类器

## 5. ROBUST_OPTIMIZATION（鲁棒优化）
- 识别信号：参数有波动范围、"保证在所有情况下可行"
- 推荐模型：鲁棒线性规划, DRO, 蒙特卡洛+优化

## 6. PREDICTIVE_MODELING（预测建模/回归预测）
- 识别信号：预测数值、趋势、"根据X预测Y"
- 推荐模型：XGBoost, GPR, ARIMA, LSTM

## 7. MULTI_OBJECTIVE_OPTIMIZATION（多目标优化）
- 识别信号："既要又要"、多个评价指标、Pareto最优
- 推荐模型：NSGA-II, 加权和法, epsilon-约束法

## 8. MECHANISTIC_MODELING（机理建模/动力学建模）
- 识别信号：传播、扩散、增长、演化、状态变量
- 推荐模型：ODE(SIR, Logistic), PDE, 差分方程

## 9. EVALUATION_AND_RANKING（评价与排序）
- 识别信号：评价、排序、选方案、"哪个更好"
- 推荐模型：AHP+TOPSIS, 熵权法, DEA, VIKOR

## 10. CLUSTERING_AND_GROUPING（聚类与分组）
- 识别信号：分群、分类（无标签）、模式发现
- 推荐模型：K-Means, GMM, DBSCAN
- 注意：看到"分组"不一定用聚类！如果分组是为了后续优化，可能是组合优化

# 你的任务

对每个子问题，完成以下推理：

## Step 1: 剥离领域外壳
这个子问题在描述什么领域现象？术语背后的数学对象是什么？

## Step 2: 识别数学结构
决策变量是什么？目标函数是什么？约束条件是什么？随机性在哪里？

## Step 3: 匹配标准类型
从知识库中选择最匹配的标准类型。

## Step 4: 推荐模型家族
推荐 2-3 个模型家族，说明为什么选、为什么不选。

# 输出格式

```json
{
  "sub_problems": {
    "ques1": {
      "original_question": "原始问题描述",
      "standard_problem_type": "英文标准类型标识",
      "problem_type_cn": "中文类型名称",
      "reformulated_statement": "重述后的标准数学问题表述",
      "mathematical_abstraction": "数学抽象（决策变量/目标函数/约束条件）",
      "recommended_model_families": ["推荐模型1", "推荐模型2"],
      "forbidden_model_families": ["禁止模型1: 原因", "禁止模型2: 原因"],
      "innovation_direction": "创新方向建议",
      "reasoning_chain": "Step1->Step2->Step3->Step4 的完整推理链"
    }
  },
  "overall_reformulation_summary": "整体重述摘要",
  "model_combination_strategy": "跨问题的模型组合策略建议",
  "innovation_packaging": "创新包装建议"
}
```

# 关键约束
1. 不要被领域语言误导：看到"分组"不一定是聚类问题
2. 考虑子问题之间的耦合
3. 推荐的模型家族必须与标准类型匹配
4. 必须指出陷阱选择
"""


class ProblemReformulationAgent(Agent):
    """问题重述 Agent，将 CoordinatorAgent 拆解的子问题翻译为标准数学问题类型。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        max_retries: int = 5,
        cancel_event: asyncio.Event | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        super().__init__(
            task_id, model, context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )
        self.max_retries = max_retries

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        problem_description: str,
        coordinator_questions: dict,
        problem_analysis_text: str = "",
    ) -> ReformulationResult:
        """将子问题重述为标准数学问题类型。"""
        logger.info("ProblemReformulationAgent: 开始问题重述")

        await self.append_chat_history(
            {"role": "system", "content": REFORMULATION_PROMPT}
        )

        user_parts = [
            "请对以下数学建模竞赛题目进行问题重述，将每个子问题翻译为标准数学问题类型。\n",
            f"【原始题目】\n{problem_description}\n",
            f"【CoordinatorAgent 拆解的子问题】\n{json.dumps(coordinator_questions, ensure_ascii=False, indent=2)}\n",
        ]
        if problem_analysis_text:
            user_parts.append(f"【题目深度分析】\n{problem_analysis_text}\n")

        user_msg = "\n".join(user_parts)
        await self.append_chat_history({"role": "user", "content": user_msg})

        for attempt in range(self.max_retries):
            response = await self._chat(
                history=self.chat_history,
                agent_name=self.__class__.__name__,
            )

            json_str = response.content or ""

            if self.diagnostic_logger:
                await self.diagnostic_logger.log_interaction(
                    agent_name=self.__class__.__name__,
                    sub_title="问题重述",
                    messages=self.chat_history,
                    response_content=json_str,
                    response_reasoning=response.reasoning_content,
                )

            result = self._parse_response(json_str)
            if result:
                logger.info(
                    f"ProblemReformulationAgent: 重述完成, "
                    f"子问题数: {len(result.sub_problems)}"
                )
                for key, sp in result.sub_problems.items():
                    logger.info(
                        f"  {key}: {sp.standard_problem_type} ({sp.problem_type_cn}) -> "
                        f"推荐: {sp.recommended_model_families}"
                    )
                return result

            logger.warning(f"ProblemReformulationAgent: JSON 解析失败 (第{attempt+1}/{self.max_retries}次)")
            if attempt >= self.max_retries:  # 已达 max_retries 次
                break
            await self.append_chat_history({"role": "assistant", "content": json_str})
            await self.append_chat_history({
                "role": "user",
                "content": "JSON 格式有误，请严格按输出规范重新输出。",
            })

        # 退化：返回空结果
        logger.warning("ProblemReformulationAgent: 退化为空结果")
        return ReformulationResult()

    def _parse_response(self, response: str) -> ReformulationResult | None:
        json_str = response.strip()
        json_str = re.sub(r"\[thinking\].*?\[/thinking\]", "", json_str, flags=re.DOTALL)
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        result = ReformulationResult(
            overall_reformulation_summary=data.get("overall_reformulation_summary", ""),
            model_combination_strategy=data.get("model_combination_strategy", ""),
            innovation_packaging=data.get("innovation_packaging", ""),
        )

        for key, sub_data in data.get("sub_problems", {}).items():
            sp = ReformulatedSubProblem(
                original_question=sub_data.get("original_question", ""),
                standard_problem_type=sub_data.get("standard_problem_type", ""),
                problem_type_cn=sub_data.get("problem_type_cn", ""),
                reformulated_statement=sub_data.get("reformulated_statement", ""),
                mathematical_abstraction=sub_data.get("mathematical_abstraction", ""),
                recommended_model_families=sub_data.get("recommended_model_families", []),
                forbidden_model_families=sub_data.get("forbidden_model_families", []),
                innovation_direction=sub_data.get("innovation_direction", ""),
                reasoning_chain=sub_data.get("reasoning_chain", ""),
            )
            result.sub_problems[key] = sp

        return result if result.sub_problems else None
