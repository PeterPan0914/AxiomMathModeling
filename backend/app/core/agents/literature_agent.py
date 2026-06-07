"""文献调研 Agent 模块，为建模提供文献支撑和方法论参考。"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.llm.llm import LLM
from app.core.prompts.literature import get_literature_research_prompt
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.tools.openalex_scholar import OpenAlexScholar
    from app.utils.diagnostic_logger import DiagnosticLogger

# 内置常见题型方法知识库，当 OpenAlex 不可用时作为 fallback
BUILTIN_KNOWLEDGE: dict[str, dict] = {
    "时间序列预测": {
        "主流方法": [
            "ARIMA（自回归积分滑动平均）",
            "Prophet（Facebook 时间序列分解）",
            "LSTM（长短期记忆网络）",
            "Transformer 时序模型",
        ],
        "获奖偏好": [
            "Prophet + 贝叶斯区间估计",
            "混合模型（经典统计 + 机器学习）",
            "多模型对比 + 灵敏度分析",
        ],
        "常见陷阱": [
            "数据非平稳未做差分或变换",
            "忽略季节性和周期性成分",
            "训练集/测试集划分存在时间泄露",
            "过拟合：参数过多而样本量不足",
            "预测区间过窄，未考虑模型不确定性",
        ],
    },
    "优化问题": {
        "主流方法": [
            "线性规划（LP）",
            "整数规划 / 混合整数规划（MIP）",
            "遗传算法（GA）",
            "粒子群优化（PSO）",
            "模拟退火（SA）",
        ],
        "获奖偏好": [
            "多目标优化 + Pareto 前沿分析",
            "精确算法与启发式算法对比",
            "灵敏度分析 + 参数扰动实验",
        ],
        "常见陷阱": [
            "约束条件遗漏（尤其是隐含约束）",
            "陷入局部最优未做全局搜索",
            "无灵敏度分析，结论鲁棒性存疑",
            "目标函数建模不合理，与实际需求脱节",
        ],
    },
    "分类/回归": {
        "主流方法": [
            "随机森林（Random Forest）",
            "XGBoost / LightGBM",
            "神经网络（MLP / CNN）",
            "支持向量机（SVM）",
        ],
        "获奖偏好": [
            "集成方法 + 精细特征工程",
            "多模型对比 + 可解释性分析（SHAP）",
            "交叉验证 + 完整的评估指标体系",
        ],
        "常见陷阱": [
            "数据泄露：特征中包含未来信息",
            "类别不平衡未做处理",
            "无交叉验证，单次划分结果不可靠",
            "只报告准确率，忽略精确率/召回率/F1",
        ],
    },
    "图论/网络": {
        "主流方法": [
            "最短路径算法（Dijkstra / Floyd）",
            "网络流（最大流 / 最小费用流）",
            "社区检测（Louvain / GN）",
            "PageRank 及其变体",
        ],
        "获奖偏好": [
            "图神经网络 + 传统图算法对比",
            "复杂网络特征分析（度分布、聚类系数）",
            "多层网络建模",
        ],
        "常见陷阱": [
            "图建模不合理（有向/无向、加权/无权选择错误）",
            "忽略网络的动态演化特性",
            "社区划分结果缺乏稳定性验证",
        ],
    },
    "评价/决策": {
        "主流方法": [
            "层次分析法（AHP）",
            "TOPSIS（逼近理想解排序）",
            "灰色关联分析",
            "熵权法",
            "模糊综合评价",
        ],
        "获奖偏好": [
            "组合赋权（主观 + 客观）",
            "多方法对比验证一致性",
            "灵敏度分析检验权重扰动影响",
        ],
        "常见陷阱": [
            "AHP 一致性检验未通过仍使用结果",
            "指标体系构建缺乏依据",
            "主观赋权缺乏专家数据支撑",
            "评价结果对权重过于敏感但未做分析",
        ],
    },
    "微分方程/动力学": {
        "主流方法": [
            "常微分方程（ODE）数值求解",
            "偏微分方程（PDE）有限差分/有限元",
            "传染病动力学模型（SIR/SEIR）",
            "系统动力学（Vensim）",
        ],
        "获奖偏好": [
            "参数标定 + 实际数据拟合",
            "模型简化与复杂模型对比",
            "稳定性分析 + 相图",
        ],
        "常见陷阱": [
            "参数无实际数据标定，凭空取值",
            "忽略初始条件敏感性",
            "模型过于复杂导致无法解析求解",
            "未做模型验证（与实际数据对比）",
        ],
    },
}


class LiteratureAgent(Agent):
    """文献调研 Agent——在 ModelerAgent 之前运行，为建模提供文献支撑。

    通过 OpenAlex API 搜索相关学术论文，或使用内置知识库作为 fallback，
    综合分析后输出结构化的文献调研报告，指导后续方法选型。
    """

    def __init__(
        self,
        task_id: str,
        model: LLM,
        context_window: int = 128000,
        cancel_event=None,
        diagnostic_logger: DiagnosticLogger | None = None,
        openalex_scholar: OpenAlexScholar | None = None,
    ) -> None:
        """初始化文献调研 Agent。

        Args:
            task_id: 任务 ID。
            model: LLM 模型实例。
            context_window: 模型上下文窗口大小。
            cancel_event: 取消信号事件。
            diagnostic_logger: 诊断日志记录器。
            openalex_scholar: 可选的 OpenAlex 学术搜索客户端。
        """
        super().__init__(
            task_id,
            model,
            context_window,
            cancel_event=cancel_event,
            diagnostic_logger=diagnostic_logger,
        )
        self.openalex_scholar = openalex_scholar

    def _match_builtin_knowledge(self, problem_description: str) -> str:
        """根据题目描述匹配内置知识库，生成参考文本。

        通过关键词匹配识别题目所属的问题类型，
        返回对应知识库条目的格式化文本。

        Args:
            problem_description: 题目描述文本。

        Returns:
            内置知识库匹配结果的格式化文本，未匹配到则返回空字符串。
        """
        # 关键词到问题类型的映射
        keyword_map: dict[str, list[str]] = {
            "时间序列预测": ["时间序列", "预测", "时序", "趋势", "季节", "ARIMA", "Prophet"],
            "优化问题": ["优化", "规划", "调度", "分配", "最短路径", "最小成本", "最大收益"],
            "分类/回归": ["分类", "回归", "识别", "判别", "预测模型", "机器学习"],
            "图论/网络": ["图论", "网络", "最短路", "连通", "社区", "拓扑"],
            "评价/决策": ["评价", "评估", "决策", "排序", "选优", "指标", "权重"],
            "微分方程/动力学": ["微分方程", "动力学", "传染病", "增长模型", "扩散", "SIR"],
        }

        # 统计各类型的匹配关键词数
        scores: dict[str, int] = {}
        desc_lower = problem_description.lower()
        for problem_type, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw.lower() in desc_lower)
            if score > 0:
                scores[problem_type] = score

        if not scores:
            return ""

        # 取匹配度最高的问题类型（最多取前 2 个）
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        matched_types = [t for t, _ in sorted_types]

        # 格式化输出
        sections = []
        for problem_type in matched_types:
            knowledge = BUILTIN_KNOWLEDGE[problem_type]
            section = f"### 问题类型：{problem_type}\n"
            for key, values in knowledge.items():
                section += f"- {key}：\n"
                for v in values:
                    section += f"  - {v}\n"
            sections.append(section)

        return "\n".join(sections)

    async def _search_openalex(self, problem_description: str, limit: int = 8) -> str:
        """使用 OpenAlex 搜索相关论文并格式化为文本。

        Args:
            problem_description: 题目描述，用作搜索查询。
            limit: 最大返回论文数。

        Returns:
            格式化的论文摘要文本，搜索失败时返回空字符串。
        """
        if not self.openalex_scholar:
            return ""

        try:
            # 构造搜索关键词：提取题目中的核心术语
            papers = await self.openalex_scholar.search_papers(
                query=problem_description, limit=limit
            )
            if not papers:
                logger.info("LiteratureAgent: OpenAlex 未检索到相关论文")
                return ""

            # 格式化论文列表为可读文本
            paper_str = self.openalex_scholar.papers_to_str(papers)
            logger.info(f"LiteratureAgent: OpenAlex 检索到 {len(papers)} 篇论文")
            return paper_str

        except Exception as e:
            logger.warning(f"LiteratureAgent: OpenAlex 搜索失败，将使用内置知识库: {e}")
            return ""

    def _extract_json_from_response(self, response_text: str) -> str | None:
        """从 LLM 响应中提取 JSON 字符串。

        处理 LLM 可能在 JSON 前后添加的说明文字、thinking 标签、代码块标记等。

        Args:
            response_text: LLM 的原始响应文本。

        Returns:
            提取到的 JSON 字符串，无法提取时返回 None。
        """
        # 剥离 thinking 块
        text = re.sub(r"\[thinking\].*?\[/thinking\]", "", response_text, flags=re.DOTALL)
        text = text.replace("```json", "").replace("```", "").strip()

        # 尝试直接解析
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # 尝试提取第一个 { 到最后一个 } 之间的内容
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        return None

    async def run(
        self,
        problem_description: str,
        competition_type: str = "国赛",
    ) -> str:
        """执行文献调研并返回结构化结果。

        工作流程：
        1. 如果有 openalex_scholar，用它搜索相关论文
        2. 如果没有，使用内置的"常见题型方法知识库"作为 fallback
        3. 将搜索结果 + 题目描述发给 LLM 综合分析
        4. 返回结构化的文献调研结果 JSON 字符串

        Args:
            problem_description: 题目描述文本。
            competition_type: 竞赛类型（国赛/美赛/其他）。

        Returns:
            结构化的文献调研结果 JSON 字符串。
        """
        logger.info(f"LiteratureAgent: 开始文献调研，竞赛类型={competition_type}")

        # 第一步：获取文献资料（优先 OpenAlex，fallback 到内置知识库）
        similar_papers = await self._search_openalex(problem_description)
        data_source = "OpenAlex"

        if not similar_papers:
            similar_papers = self._match_builtin_knowledge(problem_description)
            data_source = "内置知识库"
            if similar_papers:
                logger.info("LiteratureAgent: 使用内置知识库作为文献参考")
            else:
                logger.info("LiteratureAgent: 未匹配到内置知识库，LLM 将基于自身知识分析")

        # 第二步：构造 prompt 并调用 LLM
        system_prompt = get_literature_research_prompt(
            problem_description=problem_description,
            competition_type=competition_type,
            similar_papers=similar_papers,
        )

        user_prompt = f"""请基于以上信息，对该题目进行系统性的文献调研分析。

数据来源：{data_source}
竞赛类型：{competition_type}

请严格按照输出规则，以 JSON 格式输出完整的文献调研报告。"""

        # 调用 LLM（复用基类的 run 方法，传入 system_prompt 和 user_prompt）
        response = await super().run(
            prompt=user_prompt,
            system_prompt=system_prompt,
            sub_title="文献调研",
        )

        # 第三步：提取并验证 JSON 输出
        json_str = self._extract_json_from_response(response)
        if json_str:
            logger.info("LiteratureAgent: 文献调研完成，已提取有效 JSON 结果")
            return json_str

        # JSON 提取失败，返回包含原始响应的兜底结构
        logger.warning("LiteratureAgent: 无法从 LLM 响应中提取 JSON，返回兜底结果")
        fallback = json.dumps(
            {
                "mainstream_methods": [],
                "award_winning_methods": [],
                "known_limitations": [],
                "innovation_opportunities": [],
                "recommended_approach": {
                    "method": "待定",
                    "reason": "文献调研未能生成有效结果，需由 ModelerAgent 自行判断",
                    "differentiation": "",
                    "risk_assessment": "",
                },
                "methods_to_avoid": [],
                "_raw_response": response[:2000] if response else "",
                "_note": "此为兜底结果，原始 LLM 响应未能解析为有效 JSON",
            },
            ensure_ascii=False,
        )
        return fallback
