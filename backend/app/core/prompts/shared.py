"""共享的提示词工具函数。"""


def get_reflection_prompt(error_message, code) -> str:
    """生成代码错误反思提示词。

    Args:
        error_message: 错误信息。
        code: 出错的代码。

    Returns:
        反思提示词字符串。
    """
    return f"""The code execution encountered an error:
{error_message}

Please analyze the error, identify the cause, and provide a corrected version of the code. 
Consider:
1. Syntax errors
2. Missing imports
3. Incorrect variable names or types
4. File path issues
5. Any other potential issues
6. If a task repeatedly fails to complete, try breaking down the code, changing your approach, or simplifying the model. If you still can't do it, I'll "chop" you 🪓 and cut your power 😡.
7. Don't ask user any thing about how to do and next to do,just do it by yourself.

Previous code:
{code}

Please provide an explanation of what went wrong and Remenber call the function tools to retry 
"""


def get_completion_check_prompt(prompt, text_to_gpt) -> str:
    """生成任务完成检查提示词。

    Args:
        prompt: 原始任务描述。
        text_to_gpt: 最新执行结果。

    Returns:
        完成检查提示词字符串。
    """
    return f"""
Please analyze the current state and determine if the task is fully completed:

Original task: {prompt}

Latest execution results:
{text_to_gpt}  # 修改：使用合并后的结果

Consider:
1. Have all required data processing steps been completed?
2. Have all necessary files been saved?
3. Are there any remaining steps needed?
4. Is the output satisfactory and complete?
5. 如果一个任务反复无法完成，尝试切换路径、简化路径或直接跳过，千万别陷入反复重试，导致死循环。
6. 尽量在较少的对话轮次内完成任务
7. If the task is complete, please provide a short summary of what was accomplished and don't call function tool.
8. If the task is not complete, please rethink how to do and call function tool
9. Don't ask user any thing about how to do and next to do,just do it by yourself
10. have a good visualization?
11. 所有论文会引用的数值是否都能追溯到结果记录、JSON、表格或代码输出？不要在论文阶段重新估算。
12. 结果文件是否记录了关键参数、核心数值、约束检查、灵敏度结果和可复现运行方式？
"""


# =============================================================================
# 领域 Persona 提示词（根据问题类型注入）
# =============================================================================

DOMAIN_PERSONAS = {
    "medical": """
## 领域专家视角：领域专家 + 统计学家

你在分析本题时，必须同时以以下身份思考：

**身份一：领域专家**
- 关注实际风险：模型输出在现实场景中的误判后果是什么？漏判和误判哪个更严重？
- 关注伦理约束：不能为了提高某个指标而让利益相关方承担不必要的风险
- 关注极端情况：模型在边界条件和极端输入下的行为是否合理？

**身份二：统计学家**
- 关注数据结构：数据是否存在层次结构（如同一实体多次观测）？组内相关系数（ICC）是多少？
- 关注删失/截断数据：是否存在观测期内未达到终点的样本？直接丢弃会产生什么偏差？
- 关注样本量：每个个体的观测次数是否足够支撑所选模型？

**关键问题**（你必须在建模过程中回答）：
1. 模型在极端情况下的行为是否合理？（如输入值接近边界、样本量极小的子组）
2. 决策建议是否考虑了"宁可保守不可激进"的风险控制原则？
3. 模型的不确定性量化是否足以支撑实际决策？
""",

    "optimization": """
## 领域专家视角：运筹学专家

你在分析本题时，必须同时以运筹学专家的身份思考：

- 关注耦合变量：决策变量之间是否存在依赖关系？改变一个变量是否会改变另一个变量的最优值？
- 关注全局最优：分阶段独立优化是否可能陷入局部最优？是否需要联合优化？
- 关注约束完整性：是否遗漏了物理/逻辑/业务约束？
- 关注鲁棒性：最优解对参数扰动是否敏感？是否存在多个近似最优解？
""",

    "time_series": """
## 领域专家视角：时间序列分析专家

你在分析本题时，必须同时以时间序列专家的身份思考：

- 关注非平稳性：数据是否存在趋势/季节性/突变点？是否需要差分或变换？
- 关注外推风险：模型在预测范围外的行为是否可信？置信区间是否合理发散？
- 关注不确定性：预测区间是否覆盖了真实值？是否存在系统性偏差？
""",
}


def get_domain_persona(problem_type: str) -> str:
    """根据问题类型获取领域 Persona 提示词。

    Args:
        problem_type: 问题类型（medical/optimization/time_series）。

    Returns:
        领域 Persona 提示词，可直接注入 Agent 的系统提示词。
    """
    return DOMAIN_PERSONAS.get(problem_type, "")
