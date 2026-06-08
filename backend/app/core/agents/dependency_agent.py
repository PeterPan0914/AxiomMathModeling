"""问题依赖分析 Agent，识别子问题间的 DAG 依赖关系并构建执行图。"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.dependency_graph import (
    DependencyEdge,
    QuestionDependencyGraph,
    QuestionNode,
)
from app.core.llm.llm import LLM
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger


DEPENDENCY_AGENT_SYSTEM_PROMPT = """# Role
你是一名数学建模竞赛的战略分析师，专门负责分析子问题之间的依赖关系。

# 核心任务
国赛题目通常是 Q1 → Q2 → Q3 → Q4 层层递进的树结构，不是四个独立任务。
你的任务是精确识别这些子问题之间的依赖关系，构建问题依赖图（DAG）。

# 依赖关系的三种类型

## 1. 直接结论依赖（最常见）
Q2 需要使用 Q1 的模型结论/参数/预测结果作为输入。

## 2. 方法继承依赖
Q3 需要在 Q1/Q2 的方法基础上扩展（增加变量、改变约束等）。

## 3. 独立但共享前提
Q4 可能与 Q1 共享相同的数学基础，但不直接使用 Q1 的结论。

# 如何识别依赖关系

## 强信号（几乎必然是依赖）
- "利用/基于/参考/根据 第X问 的结果/模型/结论"
- "在第X问的基础上"
- "考虑到第X问发现的规律"

## 中等信号（很可能是依赖）
- 后续问题提到与前序问题相同的数学对象
- 后续问题的约束条件包含了前序问题的结果
- 问题之间有递进关系（简单→复杂，单一→多因素）

# 输出规范

严格按以下 JSON 格式输出：

```json
{
  "nodes": [
    {
      "id": "Q1",
      "description": "问题描述",
      "core_challenge": "核心难点",
      "priority": "高/中/低",
      "expected_output_type": "模型/数值/判定/排名"
    }
  ],
  "edges": [
    {
      "source": "Q1",
      "target": "Q2",
      "what_to_use": "Q1建立的模型及其拟合参数",
      "how_to_use": "作为Q2优化问题的目标函数"
    }
  ],
  "execution_order": ["Q1", "Q2", "Q3", "Q4"],
  "reasoning": "详细的依赖分析推理过程"
}
```

# 规则
1. 每个子问题必须在 nodes 中出现
2. edges 中的 source 和 target 必须引用 nodes 中已定义的 id
3. execution_order 必须与 edges 的依赖关系一致（拓扑序）
4. 禁止出现环依赖
5. what_to_use 必须具体描述需要使用什么结论
6. how_to_use 必须说明使用方式（作为输入/作为约束/作为基准/扩展其方法）
"""


class DependencyAgent(Agent):
    """问题依赖分析 Agent，构建子问题间的 DAG 依赖图。

    在 CoordinatorAgent 完成问题拆解后运行。
    """

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
        self.system_prompt = DEPENDENCY_AGENT_SYSTEM_PROMPT
        self.max_retries = max_retries

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        ques_all: str,
        questions: dict,
    ) -> QuestionDependencyGraph:
        """分析子问题间的依赖关系，构建依赖图。"""
        logger.info("DependencyAgent: 开始分析子问题依赖关系")

        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )

        user_parts = [
            "【完整题目】", ques_all, "",
            "【已拆解的子问题】",
            json.dumps(questions, ensure_ascii=False, indent=2),
        ]
        user_msg = "\n".join(user_parts)
        await self.append_chat_history({"role": "user", "content": user_msg})

        for attempt in range(self.max_retries):
            try:
                response = await self._chat(
                    history=self.chat_history,
                    agent_name=self.__class__.__name__,
                )
                json_str = response.content or ""

                if self.diagnostic_logger:
                    self.diagnostic_logger.log_interaction(
                        agent_name=self.__class__.__name__,
                        sub_title="依赖关系分析",
                        messages=self.chat_history,
                        response_content=json_str,
                        response_reasoning=response.reasoning_content,
                    )

                json_str = re.sub(r"\[thinking\].*?\[/thinking\]", "", json_str, flags=re.DOTALL)
                json_str = json_str.replace("```json", "").replace("```", "").strip()
                json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)

                if not json_str:
                    raise ValueError("返回的 JSON 字符串为空")

                data = json.loads(json_str)

                graph = QuestionDependencyGraph()

                for node_data in data.get("nodes", []):
                    graph.add_node(QuestionNode(
                        id=node_data["id"],
                        description=node_data.get("description", ""),
                        core_challenge=node_data.get("core_challenge", ""),
                        priority=node_data.get("priority", "中"),
                        expected_output_type=node_data.get("expected_output_type", ""),
                    ))

                for edge_data in data.get("edges", []):
                    graph.add_edge(DependencyEdge(
                        source=edge_data["source"],
                        target=edge_data["target"],
                        what_to_use=edge_data.get("what_to_use", ""),
                        how_to_use=edge_data.get("how_to_use", ""),
                    ))

                order = graph.topological_sort()
                logger.info(f"DependencyAgent: 依赖图构建完成, 执行顺序: {order}, 边数: {len(graph.edges)}")

                # 补充缺失的节点
                ques_keys = {k for k in questions.keys() if k.startswith("ques") and k != "ques_count"}
                missing = ques_keys - set(graph.nodes.keys())
                if missing:
                    logger.warning(f"DependencyAgent: 以下问题未在依赖图中出现: {missing}")
                    for mk in missing:
                        graph.add_node(QuestionNode(
                            id=mk, description=str(questions.get(mk, ""))[:200],
                        ))
                    graph.topological_sort()

                return graph

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"DependencyAgent: 解析失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt >= self.max_retries - 1:
                    break
                await self.append_chat_history({
                    "role": "system",
                    "content": self.system_prompt + f"\n上次响应格式错误: {e}。请严格输出JSON格式",
                })

        # 退化策略：构建无依赖的平铺图
        logger.warning("DependencyAgent: 退化为无依赖平铺图")
        graph = QuestionDependencyGraph()
        ques_keys = [k for k in questions.keys() if k.startswith("ques") and k != "ques_count"]
        for idx, qk in enumerate(ques_keys):
            graph.add_node(QuestionNode(id=qk, description=str(questions.get(qk, ""))[:200]))
        graph.execution_order = ques_keys
        return graph
