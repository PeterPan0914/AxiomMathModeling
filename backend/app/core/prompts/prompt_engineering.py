"""Prompt Engineering 模块，提供高级提示词工程技术。"""


def get_chain_of_thought_prompt(task: str, context: str = "") -> str:
    """生成 Chain-of-Thought 提示词。

    Args:
        task: 任务描述。
        context: 上下文信息。

    Returns:
        CoT 提示词。
    """
    return f"""请一步一步思考以下问题。

{f"【上下文】{context}" if context else ""}

【任务】
{task}

【思考过程】
请按照以下步骤进行思考：

1. **理解问题**: 首先，确保你完全理解问题的要求
2. **分析背景**: 分析问题的背景和相关知识
3. **制定方案**: 制定解决问题的方案
4. **逐步推理**: 逐步推导，展示完整的思考过程
5. **验证结果**: 验证你的结论是否正确

请开始你的思考：
"""


def get_self_consistency_prompt(task: str, num_approaches: int = 3) -> str:
    """生成 Self-Consistency 提示词。

    Args:
        task: 任务描述。
        num_approaches: 尝试的方法数量。

    Returns:
        Self-Consistency 提示词。
    """
    return f"""请使用 {num_approaches} 种不同的方法来解决以下问题。

【任务】
{task}

【要求】
对于每种方法：
1. 使用不同的起始策略
2. 展示完整的推理过程
3. 给出最终答案

【方法 1】
请使用第一种方法：

【方法 2】
请使用第二种方法：

【方法 3】
请使用第三种方法：

【综合分析】
比较所有方法的结果：
- 如果所有方法得出相同答案，报告该答案并给出高置信度
- 如果方法之间有分歧，分析分歧原因，重新审视最可疑的推理过程
- 给出最终答案和置信度评估
"""


def get_tree_of_thought_prompt(problem: str, num_branches: int = 3) -> str:
    """生成 Tree-of-Thought 提示词。

    Args:
        problem: 问题描述。
        num_branches: 分支数量。

    Returns:
        ToT 提示词。
    """
    return f"""请使用 Tree-of-Thought 方法来解决以下问题。

【问题】
{problem}

【步骤 1】
生成 {num_branches} 种不同的初始方法来解决这个问题。
对于每种方法，请简要描述其核心思路。

【步骤 2】
评估每种方法的可行性（1-10 分），并简要说明理由。

【步骤 3】
选择最有前途的 2 种方法，将每种方法再推进一步。

【步骤 4】
评估扩展后的方法。如果其中一种明显能得出解决方案，请继续推进。
如果没有，请回溯并尝试第三种方法。

【步骤 5】
继续推进，直到得出完整的解决方案或穷尽所有分支。

在每一步，请明确说明你的评估和推理。
"""


def get_reflexion_prompt(
    original_task: str,
    previous_solution: str,
    review_feedback: str,
) -> str:
    """生成 Reflexion 改进提示词。

    Args:
        original_task: 原始任务。
        previous_solution: 之前的解决方案。
        review_feedback: 评审反馈。

    Returns:
        Reflexion 提示词。
    """
    return f"""请根据评审反馈改进你的解决方案。

【原始任务】
{original_task}

【之前的解决方案】
{previous_solution}

【评审反馈】
{review_feedback}

【改进要求】
1. 仔细阅读评审反馈，理解每个问题
2. 针对每个问题提出具体的改进方案
3. 实施改进，生成新的解决方案
4. 确保改进后的内容更加准确和完善

【反思】
在改进之前，请先反思：
- 之前为什么会出现这些问题？
- 如何避免类似问题？
- 改进的优先级是什么？

请输出完整的改进后内容。
"""


def get_academic_writing_prompt(section: str, context: str = "") -> str:
    """生成学术写作提示词。

    Args:
        section: 章节名称。
        context: 上下文信息。

    Returns:
        学术写作提示词。
    """
    return f"""请按照学术写作规范撰写以下章节。

【章节】
{section}

{f"【上下文】{context}" if context else ""}

【写作要求】

1. **语言风格**
   - 使用正式学术语言
   - 避免口语化表达
   - 使用被动语态为主
   - 句子长度适中（15-25 词）

2. **段落结构**
   - 每段一个主题
   - 主题句在开头
   - 支持性证据在中间
   - 过渡句在结尾

3. **数学公式**
   - 公式前有文字介绍
   - 公式后有解释
   - 所有符号都有定义
   - 重要公式要编号

4. **图表引用**
   - 图表有标题和标签
   - 正文中引用图表
   - 图表后有分析解读

5. **引用规范**
   - 使用 IEEE 引用格式
   - 每个引用在正文中出现
   - 引用要准确相关

请开始撰写：
"""


def get_sensitivity_analysis_prompt(model: str, parameters: list[str]) -> str:
    """生成敏感性分析提示词。

    Args:
        model: 模型描述。
        parameters: 参数列表。

    Returns:
        敏感性分析提示词。
    """
    params_str = "\n".join([f"- {p}" for p in parameters])

    return f"""请对以下模型进行敏感性分析。

【模型描述】
{model}

【待分析参数】
{params_str}

【分析要求】

1. **参数选择**
   - 确定每个参数的基准值
   - 确定每个参数的变化范围（±10%, ±20%, ±50%）

2. **分析方法**
   - 使用 One-At-a-Time (OAT) 方法
   - 或使用 Morris 方法进行筛选
   - 或使用 Sobol 方法进行全局分析

3. **结果展示**
   - 生成龙卷风图（Tornado Diagram）
   - 生成蜘蛛图（Spider Plot）
   - 生成参数重要性排序表

4. **结果解释**
   - 识别最敏感的参数
   - 分析参数交互效应
   - 讨论结果的实际意义

5. **稳健性评估**
   - 评估模型对参数不确定性的稳健性
   - 提出参数估计的改进建议

请输出完整的敏感性分析报告。
"""


def get_model_validation_prompt(model: str, data: str) -> str:
    """生成模型验证提示词。

    Args:
        model: 模型描述。
        data: 数据描述。

    Returns:
        模型验证提示词。
    """
    return f"""请对以下模型进行验证。

【模型描述】
{model}

【数据描述】
{data}

【验证要求】

1. **内部验证**
   - 交叉验证（K-fold）
   - 留一法验证（LOOCV）
   - 自助法验证（Bootstrap）

2. **外部验证**
   - 独立测试集验证
   - 时间序列验证（如果适用）
   - 外部数据集验证（如果可用）

3. **验证指标**
   - 回归问题：R², RMSE, MAE, MAPE
   - 分类问题：Accuracy, Precision, Recall, F1, AUC
   - 概率问题：Log-loss, Brier Score

4. **诊断分析**
   - 残差分析
   - 预测 vs 实际图
   - 残差 Q-Q 图
   - 影响点分析

5. **稳健性检查**
   - 参数敏感性分析
   - 假设检验
   - 异常值影响分析

请输出完整的模型验证报告。
"""
