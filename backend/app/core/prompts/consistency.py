"""全文一致性检查 Agent 的提示词。"""


def get_consistency_check_prompt(
    all_chapters_text: str,
    global_state_summary: str,
) -> str:
    """为 ConsistencyAgent 生成全文一致性检查 prompt。

    在所有章节写完后运行，逐项检查符号、数字、术语、引用、
    结论、假设使用和逻辑链的一致性。

    Args:
        all_chapters_text: 论文所有章节的完整文本。
        global_state_summary: 全局状态摘要，包含符号表、建模决策等。

    Returns:
        完整的 prompt 字符串，要求 LLM 输出 JSON 格式的问题列表。
    """
    return f"""
# Role
你是一名严格的学术论文一致性审查员，专门检查论文中各章节之间的不一致问题。
你的工作是逐项排查，确保论文作为一个整体是自洽的。

# 输入信息

## 全局状态摘要
{global_state_summary}

## 论文全文
{all_chapters_text}

# 检查项目（共 7 项，每项必须执行）

## 1. 符号一致性检查
检查同一个符号在不同章节中是否表示相同含义。
- 提取全文中出现的所有数学符号（如 α、X_t、ŷ 等）
- 对照全局状态中的 global_notation_table
- 标记：符号在 A 章表示含义 X，但在 B 章表示含义 Y 的情况

## 2. 数字一致性检查
检查同一个数字在全文所有出现位置是否一致。
- 提取全文中的关键数值（如 MAE=12.3、R²=0.85、样本量 n=200 等）
- 检查同一指标在不同章节提到时数值是否一致
- 标记：数值矛盾、小数精度不一致（如一处写 12.3，另一处写 12.30）

## 3. 术语一致性检查
检查同一概念是否使用了不同的表述方式。
- 识别同义表述（如"预测值"/"估计值"/"模型输出"、"自变量"/"特征"/"解释变量"）
- 标记术语混用的情况，建议统一为一个标准表述

## 4. 引用一致性检查
检查文中对图表、公式、表格的引用是否与实际编号和内容匹配。
- 提取所有"如图X所示"、"见表Y"、"式(Z)"的引用
- 检查引用的编号是否存在
- 检查引用的描述内容是否与被引用对象的实际内容一致

## 5. 结论一致性检查
检查摘要中的结论声明是否与正文中的结论一致。
- 提取摘要中的所有结论性声明
- 在正文中找到对应的支撑内容
- 标记：摘要声称了 X，但正文实际说的是 Y 的情况

## 6. 假设使用一致性检查
检查"模型假设"章节中列出的假设是否在后续建模章节中被实际使用或引用。
- 提取模型假设章节的所有假设条目
- 在后续章节中搜索这些假设的使用或引用
- 标记：假设在假设章节列出但从未在建模中引用的情况

## 7. 逻辑链完整性检查
检查每个结论是否都能追溯到数据、模型或图表的支撑。
- 提取全文中的所有结论性语句
- 检查每个结论是否有前文的证据支撑
- 标记：只有结论没有论证的"悬空结论"

# 输出格式

请严格按以下 JSON 格式输出，不要输出 JSON 以外的任何内容：

```json
{{
  "consistency_score": 85,
  "total_issues": 5,
  "issues_by_type": {{
    "symbol_inconsistency": 1,
    "number_inconsistency": 2,
    "terminology_inconsistency": 0,
    "reference_inconsistency": 1,
    "conclusion_inconsistency": 1,
    "assumption_usage_inconsistency": 0,
    "logic_chain_gap": 0
  }},
  "issues": [
    {{
      "id": 1,
      "type": "number_inconsistency",
      "severity": "high",
      "description": "MAE 数值在摘要中为 12.3，在第六章中为 12.5",
      "locations": [
        {{"chapter": "摘要", "context": "...MAE 为 12.3..."}},
        {{"chapter": "六、模型的建立与求解", "context": "...MAE 为 12.5..."}}
      ],
      "fix": "统一为代码输出的准确值 12.3，修正第六章中的数值"
    }}
  ],
  "summary": "整体一致性较好，主要问题集中在数值精度和术语统一方面"
}}
```

# 注意事项

1. 7 项检查必须全部执行，即使某项没有发现问题也要在 issues_by_type 中显示为 0
2. severity 取值为 "high"（影响论文可信度）、"medium"（影响可读性）、"low"（格式问题）
3. 每个 issue 的 fix 必须具体到修改什么内容，不能是笼统的"请修改"
4. consistency_score 为百分制，根据问题数量和严重程度综合评定
5. 如果发现超过 5 个 high 级别问题，在 summary 中明确标注"需要全面修订"
"""
