"""问题依赖图模块，管理子问题间的 DAG 依赖关系和执行上下文。"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from app.utils.log_util import logger


@dataclass
class DependencyEdge:
    """依赖边，描述一个子问题对另一个子问题的依赖。"""

    source: str = ""            # 被依赖的问题 ID
    target: str = ""            # 依赖方问题 ID
    what_to_use: str = ""       # 需要使用上一问的什么结论
    how_to_use: str = ""        # 如何使用

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target,
                "what_to_use": self.what_to_use, "how_to_use": self.how_to_use}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DependencyEdge:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class QuestionNode:
    """依赖图中的节点，代表一个子问题。"""

    id: str = ""
    description: str = ""
    core_challenge: str = ""
    priority: str = "中"
    expected_output_type: str = ""

    # 运行时填充
    execution_order: int = -1
    status: str = "pending"  # pending / running / completed / failed
    core_conclusion: str = ""
    key_outputs: dict[str, str] = field(default_factory=dict)
    conclusion_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "description": self.description,
            "core_challenge": self.core_challenge, "priority": self.priority,
            "expected_output_type": self.expected_output_type,
            "execution_order": self.execution_order, "status": self.status,
            "core_conclusion": self.core_conclusion,
            "key_outputs": self.key_outputs,
            "conclusion_source": self.conclusion_source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuestionNode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class QuestionDependencyGraph:
    """问题依赖图，管理子问题间的 DAG 依赖关系。

    核心职责：
    1. 存储节点（子问题）和边（依赖关系）
    2. 拓扑排序，确定执行顺序
    3. 检测环依赖
    4. 为每个子问题生成"强制上下文"（依赖问题的结论注入）
    5. 追踪每个子问题的执行状态和核心结论
    """

    def __init__(self) -> None:
        self.nodes: dict[str, QuestionNode] = {}
        self.edges: list[DependencyEdge] = []
        self.adjacency: dict[str, list[str]] = {}       # target -> [sources]
        self.reverse_adj: dict[str, list[str]] = {}     # source -> [targets]
        self.execution_order: list[str] = []

    def add_node(self, node: QuestionNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
        if node.id not in self.reverse_adj:
            self.reverse_adj[node.id] = []

    def add_edge(self, edge: DependencyEdge) -> None:
        self.edges.append(edge)
        self.adjacency.setdefault(edge.target, []).append(edge.source)
        self.reverse_adj.setdefault(edge.source, []).append(edge.target)

    def topological_sort(self) -> list[str]:
        """Kahn 算法拓扑排序，同时检测环依赖。"""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        queue: deque[str] = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                queue.append(nid)

        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for dependent in self.reverse_adj.get(nid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self.nodes):
            remaining = set(self.nodes.keys()) - set(order)
            raise ValueError(
                f"检测到环依赖！无法排序的节点: {remaining}。"
                f"依赖关系: {[(e.source, e.target) for e in self.edges]}"
            )

        self.execution_order = order
        for idx, nid in enumerate(order):
            self.nodes[nid].execution_order = idx
        return order

    def get_dependencies(self, node_id: str) -> list[str]:
        return self.adjacency.get(node_id, [])

    def can_execute(self, node_id: str) -> bool:
        for dep_id in self.get_dependencies(node_id):
            if self.nodes[dep_id].status != "completed":
                return False
        return True

    def record_conclusion(
        self,
        node_id: str,
        core_conclusion: str,
        key_outputs: dict[str, str] | None = None,
        conclusion_source: str = "",
    ) -> None:
        """记录某个子问题的核心结论。"""
        node = self.nodes[node_id]
        node.core_conclusion = core_conclusion
        if key_outputs:
            node.key_outputs.update(key_outputs)
        node.conclusion_source = conclusion_source
        node.status = "completed"

    def build_dependency_context(self, node_id: str) -> str:
        """为某个子问题构建"强制上下文"，注入其所有依赖问题的结论。

        这是 DependencyAgent 的核心功能：确保每个子问题在执行时
        能看到它所依赖的所有前序问题的核心结论。
        """
        dep_edges = [e for e in self.edges if e.target == node_id]
        if not dep_edges:
            return ""

        parts: list[str] = []
        parts.append("=== 前序问题结论（你必须基于这些结论继续工作，不可忽略） ===\n")

        for edge in dep_edges:
            source_node = self.nodes.get(edge.source)
            if not source_node or source_node.status != "completed":
                parts.append(f"【{edge.source}】尚未完成，但你的问题依赖它的结论。\n")
                continue

            what_block = f"需要使用: {edge.what_to_use}" if edge.what_to_use else ""
            how_block = f"使用方式: {edge.how_to_use}" if edge.how_to_use else ""

            conclusion_block = source_node.core_conclusion or "(无结论)"
            key_outputs_block = ""
            if source_node.key_outputs:
                key_outputs_block = "\n关键数值: " + ", ".join(
                    f"{k}={v}" for k, v in source_node.key_outputs.items()
                )

            parts.append(
                f"【{edge.source}: {source_node.description[:80]}】\n"
                f"  {what_block}\n"
                f"  {how_block}\n"
                f"  核心结论: {conclusion_block}\n"
                f"  {key_outputs_block}\n"
            )

        parts.append(
            "=== 重要提醒 ===\n"
            "你必须在方案中明确引用上述前序问题的结论，说明你如何利用了它们。\n"
        )
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "execution_order": self.execution_order,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> QuestionDependencyGraph:
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuestionDependencyGraph:
        graph = cls()
        for nid, nd in d.get("nodes", {}).items():
            graph.add_node(QuestionNode.from_dict(nd))
        for ed in d.get("edges", []):
            graph.add_edge(DependencyEdge.from_dict(ed))
        graph.execution_order = d.get("execution_order", [])
        for edge in graph.edges:
            graph.adjacency.setdefault(edge.target, []).append(edge.source)
            graph.reverse_adj.setdefault(edge.source, []).append(edge.target)
        return graph
