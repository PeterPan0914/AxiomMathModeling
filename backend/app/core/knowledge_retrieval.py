"""知识检索模块，提供 RAG (Retrieval-Augmented Generation) 功能。"""

from typing import Optional
from app.utils.log_util import logger


class KnowledgeRetrieval:
    """知识检索器，用于检索相关知识以增强生成质量。"""

    def __init__(self, enabled: bool = False):
        """初始化知识检索器。

        Args:
            enabled: 是否启用知识检索。
        """
        self.enabled = enabled
        self.knowledge_base: dict[str, list[dict]] = {
            'model_templates': [],
            'best_practices': [],
            'common_errors': [],
            'academic_papers': [],
        }

    def add_knowledge(self, category: str, content: dict):
        """添加知识到知识库。

        Args:
            category: 知识类别。
            content: 知识内容。
        """
        if category in self.knowledge_base:
            self.knowledge_base[category].append(content)
            logger.debug(f"KnowledgeRetrieval: 添加知识到 {category}")

    def retrieve(self, query: str, category: Optional[str] = None, top_k: int = 5) -> list[dict]:
        """检索相关知识。

        Args:
            query: 查询内容。
            category: 知识类别（可选）。
            top_k: 返回的最大结果数。

        Returns:
            相关知识列表。
        """
        if not self.enabled:
            return []

        results = []

        # 简单的关键词匹配检索
        categories = [category] if category else self.knowledge_base.keys()

        for cat in categories:
            if cat not in self.knowledge_base:
                continue

            for item in self.knowledge_base[cat]:
                # 计算简单的相关性分数
                score = self._calculate_relevance(query, item)
                if score > 0:
                    results.append({
                        'category': cat,
                        'content': item,
                        'score': score,
                    })

        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def _calculate_relevance(self, query: str, item: dict) -> float:
        """计算查询和知识项的相关性分数。

        Args:
            query: 查询内容。
            item: 知识项。

        Returns:
            相关性分数（0-1）。
        """
        # 简单的关键词匹配
        query_lower = query.lower()
        item_text = str(item).lower()

        # 计算匹配的关键词数量
        query_words = set(query_lower.split())
        item_words = set(item_text.split())

        if not query_words:
            return 0.0

        matches = query_words.intersection(item_words)
        score = len(matches) / len(query_words)

        return score

    def get_model_template(self, model_type: str) -> Optional[dict]:
        """获取模型模板。

        Args:
            model_type: 模型类型。

        Returns:
            模型模板，如果不存在则返回 None。
        """
        templates = {
            'linear_programming': {
                'name': '线性规划',
                'formulation': 'min c^T x\ns.t. Ax <= b\nx >= 0',
                'solver': ['PuLP', 'scipy.optimize.linprog'],
                'use_case': '资源分配、生产计划、运输问题',
            },
            'integer_programming': {
                'name': '整数规划',
                'formulation': 'min c^T x\ns.t. Ax <= b\nx ∈ Z^n',
                'solver': ['PuLP', 'OR-Tools'],
                'use_case': '选址问题、排班问题、组合优化',
            },
            'nonlinear_programming': {
                'name': '非线性规划',
                'formulation': 'min f(x)\ns.t. g(x) <= 0\nh(x) = 0',
                'solver': ['scipy.optimize.minimize', 'CVXPY'],
                'use_case': '参数估计、曲线拟合、工程优化',
            },
            'ode_system': {
                'name': '常微分方程组',
                'formulation': 'dx/dt = f(x, t)',
                'solver': ['scipy.integrate.solve_ivp'],
                'use_case': '动态系统、传染病模型、化学反应',
            },
            'monte_carlo': {
                'name': '蒙特卡洛模拟',
                'formulation': 'E[f(X)] ≈ (1/N) Σ f(x_i)',
                'solver': ['numpy.random', 'scipy.stats'],
                'use_case': '风险评估、不确定性分析、积分计算',
            },
            'regression': {
                'name': '回归分析',
                'formulation': 'y = Xβ + ε',
                'solver': ['scikit-learn', 'statsmodels'],
                'use_case': '预测、关系分析、趋势拟合',
            },
            'time_series': {
                'name': '时间序列分析',
                'formulation': 'ARIMA(p,d,q), SARIMA, Prophet',
                'solver': ['statsmodels', 'Prophet'],
                'use_case': '预测、趋势分析、季节性分析',
            },
        }

        return templates.get(model_type)

    def get_best_practice(self, topic: str) -> list[dict]:
        """获取最佳实践。

        Args:
            topic: 主题。

        Returns:
            最佳实践列表。
        """
        practices = {
            'sensitivity_analysis': [
                {'title': '使用 Sobol 方法', 'description': '全局敏感性分析，考虑参数交互'},
                {'title': '使用 Morris 方法', 'description': '筛选重要参数，计算效率高'},
                {'title': '生成龙卷风图', 'description': '直观展示参数重要性'},
            ],
            'model_validation': [
                {'title': '交叉验证', 'description': 'K-fold 交叉验证评估模型泛化能力'},
                {'title': '外部验证', 'description': '使用独立数据集验证模型'},
                {'title': '残差分析', 'description': '检查残差的正态性和独立性'},
            ],
            'paper_writing': [
                {'title': '执行摘要是关键', 'description': '评审首先阅读执行摘要，要能独立成文'},
                {'title': '假设要合理', 'description': '每个假设都要有合理性和依据'},
                {'title': '敏感性分析必不可少', 'description': '展示模型的稳健性'},
            ],
        }

        return practices.get(topic, [])


# 全局知识检索器实例
knowledge_retrieval = KnowledgeRetrieval(enabled=False)
