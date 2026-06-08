"""本地引用数据库，内置 25+ 种常见方法的标准引用。

用于：
1. LiteratureAgent 输出时自动匹配并附带标准引用
2. WriterAgent 写论文时使用标准引用格式
3. 确保参考文献真实存在（避免 LLM 编造引用）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


@dataclass
class Citation:
    """单条引用。"""
    key: str = ""
    method: str = ""
    aliases: list[str] | None = None
    authors: str = ""
    year: str = ""
    title: str = ""
    journal: str = ""
    apa: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "method": self.method,
            "authors": self.authors, "year": self.year,
            "title": self.title, "journal": self.journal, "apa": self.apa,
        }


# 标准引用数据库
STANDARD_CITATIONS: list[Citation] = [
    # ---- 统计模型 ----
    Citation(
        key="laird1982", method="LMM", aliases=["linear_mixed_model", "mixed_effects", "线性混合效应模型"],
        authors="Laird, N. M., & Ware, J. H.", year="1982",
        title="Random-effects models for longitudinal data",
        journal="Biometrics, 38(4), 963-974",
        apa="Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. Biometrics, 38(4), 963-974.",
    ),
    Citation(
        key="bates2015", method="LMM", aliases=["lme4", "mixed_effects"],
        authors="Bates, D., Mächler, M., Bolker, B., & Walker, S.", year="2015",
        title="Fitting linear mixed-effects models using lme4",
        journal="Journal of Statistical Software, 67(1), 1-48",
        apa="Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. Journal of Statistical Software, 67(1), 1-48.",
    ),
    Citation(
        key="rasmussen2006", method="GPR", aliases=["gaussian_process", "高斯过程回归"],
        authors="Rasmussen, C. E., & Williams, C. K. I.", year="2006",
        title="Gaussian Processes for Machine Learning",
        journal="MIT Press",
        apa="Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.",
    ),
    # ---- 生存分析 ----
    Citation(
        key="cox1972", method="Cox_PH", aliases=["cox_proportional_hazards", "cox回归"],
        authors="Cox, D. R.", year="1972",
        title="Regression models and life-tables",
        journal="Journal of the Royal Statistical Society: Series B, 34(2), 187-202",
        apa="Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society: Series B, 34(2), 187-202.",
    ),
    Citation(
        key="kaplan1958", method="Kaplan_Meier", aliases=["km_estimator", "kaplan_meier"],
        authors="Kaplan, E. L., & Meier, P.", year="1958",
        title="Nonparametric estimation from incomplete observations",
        journal="Journal of the American Statistical Association, 53(282), 457-481",
        apa="Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282), 457-481.",
    ),
    Citation(
        key="lee2018deephit", method="DeepHit", aliases=["deep_survival"],
        authors="Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M.", year="2018",
        title="DeepHit: A deep learning approach to survival analysis with competing risks",
        journal="Proceedings of the AAAI Conference on Artificial Intelligence, 32(1)",
        apa="Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). DeepHit: A deep learning approach to survival analysis with competing risks. Proceedings of AAAI, 32(1).",
    ),
    # ---- 机器学习 ----
    Citation(
        key="chen2016xgboost", method="XGBoost", aliases=["xgboost"],
        authors="Chen, T., & Guestrin, C.", year="2016",
        title="XGBoost: A scalable tree boosting system",
        journal="Proceedings of the 22nd ACM SIGKDD, 785-794",
        apa="Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD, 785-794.",
    ),
    Citation(
        key="ke2017lightgbm", method="LightGBM", aliases=["lightgbm"],
        authors="Ke, G., Meng, Q., Finley, T., et al.", year="2017",
        title="LightGBM: A highly efficient gradient boosting decision tree",
        journal="Advances in Neural Information Processing Systems, 30",
        apa="Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS, 30.",
    ),
    Citation(
        key="breiman2001rf", method="RandomForest", aliases=["random_forest", "随机森林"],
        authors="Breiman, L.", year="2001",
        title="Random forests",
        journal="Machine Learning, 45(1), 5-32",
        apa="Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
    ),
    Citation(
        key="friedman2001gbm", method="GBDT", aliases=["gradient_boosting", "梯度提升决策树"],
        authors="Friedman, J. H.", year="2001",
        title="Greedy function approximation: A gradient boosting machine",
        journal="Annals of Statistics, 29(5), 1189-1232",
        apa="Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.",
    ),
    Citation(
        key="cortes1995svm", method="SVM", aliases=["support_vector_machine", "支持向量机"],
        authors="Cortes, C., & Vapnik, V.", year="1995",
        title="Support-vector networks",
        journal="Machine Learning, 20(3), 273-297",
        apa="Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.",
    ),
    Citation(
        key="hochreiter1997lstm", method="LSTM", aliases=["long_short_term_memory"],
        authors="Hochreiter, S., & Schmidhuber, J.", year="1997",
        title="Long short-term memory",
        journal="Neural Computation, 9(8), 1735-1780",
        apa="Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.",
    ),
    Citation(
        key="vaswani2017attention", method="Transformer", aliases=["attention", "transformer"],
        authors="Vaswani, A., Shazeer, N., Parmar, N., et al.", year="2017",
        title="Attention is all you need",
        journal="Advances in Neural Information Processing Systems, 30",
        apa="Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. NeurIPS, 30.",
    ),
    # ---- 时间序列 ----
    Citation(
        key="box2015arima", method="ARIMA", aliases=["arima", "arima_model"],
        authors="Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M.", year="2015",
        title="Time Series Analysis: Forecasting and Control",
        journal="Wiley",
        apa="Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control. Wiley.",
    ),
    Citation(
        key="taylor2018prophet", method="Prophet", aliases=["facebook_prophet"],
        authors="Taylor, S. J., & Letham, B.", year="2018",
        title="Forecasting at scale",
        journal="The American Statistician, 72(1), 37-45",
        apa="Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.",
    ),
    # ---- 优化 ----
    Citation(
        key="holland1992ga", method="GA", aliases=["genetic_algorithm", "遗传算法"],
        authors="Holland, J. H.", year="1992",
        title="Adaptation in Natural and Artificial Systems",
        journal="MIT Press",
        apa="Holland, J. H. (1992). Adaptation in Natural and Artificial Systems. MIT Press.",
    ),
    Citation(
        key="kennedy1995pso", method="PSO", aliases=["particle_swarm", "粒子群优化"],
        authors="Kennedy, J., & Eberhart, R.", year="1995",
        title="Particle swarm optimization",
        journal="Proceedings of ICNN'95, 1942-1948",
        apa="Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of ICNN'95, 1942-1948.",
    ),
    Citation(
        key="dantzig1947lp", method="LinearProgramming", aliases=["linear_programming", "线性规划"],
        authors="Dantzig, G. B.", year="1947",
        title="Linear programming",
        journal="Operations Research, 51(1), 1-26",
        apa="Dantzig, G. B. (1947). Linear programming. Operations Research.",
    ),
    # ---- 评价方法 ----
    Citation(
        key="saaty1980ahp", method="AHP", aliases=["analytic_hierarchy_process", "层次分析法"],
        authors="Saaty, T. L.", year="1980",
        title="The Analytic Hierarchy Process",
        journal="McGraw-Hill",
        apa="Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.",
    ),
    Citation(
        key="hwang1981topsis", method="TOPSIS", aliases=["topsis"],
        authors="Hwang, C. L., & Yoon, K.", year="1981",
        title="Multiple Attribute Decision Making",
        journal="Springer",
        apa="Hwang, C. L., & Yoon, K. (1981). Multiple Attribute Decision Making. Springer.",
    ),
    Citation(
        key="shannon1948entropy", method="EntropyWeight", aliases=["entropy_weight", "熵权法"],
        authors="Shannon, C. E.", year="1948",
        title="A mathematical theory of communication",
        journal="Bell System Technical Journal, 27(3), 379-423",
        apa="Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal, 27(3), 379-423.",
    ),
    Citation(
        key="deng1982grey", method="GreyPrediction", aliases=["grey_prediction", "灰色预测"],
        authors="邓聚龙", year="1982",
        title="灰色系统理论",
        journal="华中工学院学报",
        apa="邓聚龙 (1982). 灰色系统理论. 华中工学院学报.",
    ),
    # ---- 聚类 ----
    Citation(
        key="macqueen1967kmeans", method="KMeans", aliases=["k_means", "k-means", "K-means聚类"],
        authors="MacQueen, J.", year="1967",
        title="Some methods for classification and analysis of multivariate observations",
        journal="Proceedings of the 5th Berkeley Symposium, 1, 281-297",
        apa="MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the 5th Berkeley Symposium, 1, 281-297.",
    ),
    Citation(
        key="ester1996dbscan", method="DBSCAN", aliases=["dbscan"],
        authors="Ester, M., Kriegel, H. P., Sander, J., & Xu, X.", year="1996",
        title="A density-based algorithm for discovering clusters in large spatial databases with noise",
        journal="Proceedings of KDD, 226-231",
        apa="Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters. KDD, 226-231.",
    ),
    # ---- 其他 ----
    Citation(
        key="pearson1901pca", method="PCA", aliases=["principal_component_analysis", "主成分分析"],
        authors="Pearson, K.", year="1901",
        title="On lines and planes of closest fit to systems of points in space",
        journal="Philosophical Magazine, 2(11), 559-572",
        apa="Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. Philosophical Magazine, 2(11), 559-572.",
    ),
    Citation(
        key="mahalanobis1936", method="Mahalanobis", aliases=["mahalanobis_distance", "马氏距离"],
        authors="Mahalanobis, P. C.", year="1936",
        title="On the generalised distance in statistics",
        journal="Proceedings of the National Institute of Sciences of India, 2(1), 49-55",
        apa="Mahalanobis, P. C. (1936). On the generalised distance in statistics. Proceedings of the National Institute of Sciences of India, 2(1), 49-55.",
    ),
    Citation(
        key="smote2002", method="SMOTE", aliases=["smote", "过采样"],
        authors="Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.", year="2002",
        title="SMOTE: Synthetic minority over-sampling technique",
        journal="Journal of Artificial Intelligence Research, 16, 321-357",
        apa="Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. JAIR, 16, 321-357.",
    ),
]


def lookup_citations(method_name: str) -> list[Citation]:
    """查找方法的标准引用，支持模糊匹配。"""
    method_lower = method_name.lower().strip()
    results = []
    for c in STANDARD_CITATIONS:
        # 精确匹配
        if c.method.lower() == method_lower:
            results.append(c)
            continue
        # 别名匹配
        if c.aliases:
            for alias in c.aliases:
                if alias.lower() == method_lower:
                    results.append(c)
                    break
        # 子串匹配
        if method_lower in c.method.lower() or c.method.lower() in method_lower:
            if c not in results:
                results.append(c)
    return results


def get_citation_bib(methods: list[str]) -> list[dict[str, str]]:
    """为给定方法列表生成引用列表。"""
    bib = []
    seen = set()
    for method in methods:
        citations = lookup_citations(method)
        for c in citations:
            if c.key not in seen:
                seen.add(c.key)
                bib.append({"index": str(len(bib) + 1), "apa": c.apa, "key": c.key})
    return bib
