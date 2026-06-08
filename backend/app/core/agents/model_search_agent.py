"""模型搜索 Agent 模块，专门负责变量筛选和模型比较。

解决的问题：AI 论文直接选定最终模型，不做变量筛选。
该 Agent 接收 ModelerAgent 的搜索协议，在代码沙箱中执行
系统性的模型比较，输出 AIC/BIC 对比表。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from app.core.agents.agent import Agent
from app.core.domain_rules import run_full_sanity_check, SanityCheckReport
from app.core.llm.llm import LLM
from app.schemas.A2A import ModelSpec, ModelSearchProtocol
from app.utils.log_util import logger

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger
from app.tools.base_interpreter import BaseCodeInterpreter


class ModelSearchResult:
    """模型搜索结果。"""

    def __init__(self):
        self.comparison_table: str = ""
        self.best_model_id: str = ""
        self.best_model_formula: str = ""
        self.best_aic: float = float('inf')
        self.best_bic: float = float('inf')
        self.icc: float = 0.0
        self.r2_marginal: float = 0.0
        self.r2_conditional: float = 0.0
        self.elimination_log: list[dict] = []
        self.likelihood_ratio_tests: list[dict] = []
        self.final_coefficients: str = ""
        self.selection_justification: str = ""
        self.all_model_results: list[dict] = []
        self.sanity_report: SanityCheckReport | None = None
        self.success: bool = False
        self.error_message: str = ""

    def to_summary_text(self) -> str:
        """生成给 WriterAgent 的摘要文本。"""
        if not self.success:
            return f"模型搜索失败: {self.error_message}"

        lines = [
            "## 模型变量筛选结果",
            "",
            "### 候选模型 AIC/BIC 对比",
            self.comparison_table,
            "",
            f"### 最优模型: {self.best_model_id}",
            f"- 公式: {self.best_model_formula}",
            f"- AIC: {self.best_aic:.2f}",
            f"- BIC: {self.best_bic:.2f}",
            f"- ICC: {self.icc:.4f} ({self.icc*100:.1f}% 的变异源于个体差异)",
            f"- R²_marginal: {self.r2_marginal:.4f}",
            f"- R²_conditional: {self.r2_conditional:.4f}",
            "",
            "### 逐步筛选过程",
        ]
        for step in self.elimination_log:
            lines.append(f"- {step.get('step', '?')}: {step.get('action', '?')}")

        if self.sanity_report and self.sanity_report.violations:
            lines.append("")
            lines.append("### 合理性检查")
            lines.append(self.sanity_report.to_text())

        lines.append("")
        lines.append("### 选择论证")
        lines.append(self.selection_justification or "（无）")

        return "\n".join(lines)


_SEARCH_PROMPT_TEMPLATE = """你是模型搜索专家。你的任务是执行系统性变量筛选，找到最优的模型规格。

## 任务说明

{search_protocol_text}

## 数据文件信息

{data_description}

## 强制执行流程

### Step 1: 数据准备
1. 加载数据
2. 检查数据结构和缺失值
3. 计算初始 ICC（空模型）
4. 报告数据基本信息

### Step 2: 拟合全模型
1. 按照 FULL_MODEL 定义拟合全模型
2. 报告全模型的 AIC、BIC、对数似然
3. 报告全模型各系数的 p 值

### Step 3: 逐步筛选
按照搜索策略逐步测试每种候选模型：
1. 对每个候选模型，拟合并记录 AIC/BIC/对数似然
2. 对嵌套模型做似然比检验
3. 记录每步的删除理由

### Step 4: 选择最优模型
1. 按 AIC 从低到高排序所有模型
2. 用似然比检验确认简化模型不显著差于全模型
3. 选择 AIC 最低且通过似然比检验的模型

### Step 5: 最终模型报告
1. 最优模型的完整系数表（含 p 值和置信区间）
2. ICC、R²_marginal、R²_conditional
3. 残差诊断图

## 输出格式（必须严格遵循）

用 print 输出以下结构化信息（每行一个标签）：

```
===MODEL_SEARCH_RESULT_START===
BEST_MODEL_ID: <模型ID>
BEST_MODEL_FORMULA: <公式>
BEST_AIC: <数值>
BEST_BIC: <数值>
ICC: <数值>
R2_MARGINAL: <数值>
R2_CONDITIONAL: <数值>
===MODEL_SEARCH_RESULT_END===

===COMPARISON_TABLE_START===
<Markdown 格式的 AIC/BIC 对比表>
===COMPARISON_TABLE_END===

===ELIMINATION_LOG_START===
Step 1: 删除 XXX，理由: p=0.XX > 0.05，AIC 变化 +X.XX
Step 2: 删除 XXX，理由: ...
===ELIMINATION_LOG_END===

===COEFFICIENTS_START===
<最终模型的系数表，Markdown 格式>
===COEFFICIENTS_END===

===JUSTIFICATION_START===
<选择论证文本，3-5句话>
===JUSTIFICATION_END===
```

## 注意事项
1. 如果使用 mixedlm，注意 groups 参数必须正确指定分组变量
2. 如果模型不收敛，尝试简化随机效应结构
3. 所有随机操作设置随机种子 np.random.seed(42)
4. ICC 计算公式: Var(intercept) / (Var(intercept) + Var(residual))
"""


class ModelSearchAgent(Agent):
    """模型搜索 Agent，专门负责系统性变量筛选和模型比较。

    在 Phase 4.5 运行，对包含 model_search_protocol 的子问题执行搜索。
    """

    def __init__(
        self,
        task_id: str,
        model: LLM,
        work_dir: str,
        code_interpreter: BaseCodeInterpreter,
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
        self.work_dir = work_dir
        self.code_interpreter = code_interpreter
        self.max_retries = max_retries

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        ques_key: str,
        model_spec: ModelSpec,
        data_description: str = "",
    ) -> ModelSearchResult:
        """执行模型搜索。

        Args:
            ques_key: 问题标识，如 "ques1"。
            model_spec: 包含 model_search_protocol 的模型规格。
            data_description: 数据文件描述。

        Returns:
            ModelSearchResult 对象。
        """
        result = ModelSearchResult()

        protocol = model_spec.model_search_protocol
        if not protocol:
            result.error_message = "model_spec 中没有 model_search_protocol"
            return result

        prompt = _SEARCH_PROMPT_TEMPLATE.format(
            search_protocol_text=protocol.to_prompt_text(),
            data_description=data_description or "请先用 pd.read_csv() 加载数据并用 .info() 查看结构",
        )

        logger.info(f"[ModelSearchAgent] 开始模型搜索: {ques_key}, 策略={protocol.search_strategy}")

        await self.append_chat_history(
            {"role": "system", "content": "你是模型搜索专家，负责执行系统性变量筛选。"}
        )
        await self.append_chat_history({"role": "user", "content": prompt})

        for retry in range(self.max_retries):
            try:
                response = await self._chat(
                    history=self.chat_history,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "execute_code",
                            "description": "执行 Python 代码",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string", "description": "要执行的 Python 代码"}
                                },
                                "required": ["code"]
                            }
                        }
                    }],
                    tool_choice="auto",
                    agent_name=self.__class__.__name__,
                )

                if response.tool_calls:
                    tool_call = response.tool_calls[0]
                    if tool_call.name == "execute_code":
                        code = json.loads(tool_call.arguments)["code"]

                        assistant_msg = {"role": "assistant", "content": response.content}
                        if response.tool_calls:
                            assistant_msg["tool_calls"] = [{
                                "id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments}
                            } for tc in response.tool_calls]
                        await self.append_chat_history(assistant_msg)

                        text_to_gpt, error_occurred, error_message = \
                            await self.code_interpreter.execute_code(code)

                        if error_occurred:
                            await self.append_chat_history({
                                "role": "tool", "tool_call_id": tool_call.id,
                                "name": "execute_code", "content": error_message,
                            })
                            continue

                        await self.append_chat_history({
                            "role": "tool", "tool_call_id": tool_call.id,
                            "name": "execute_code", "content": text_to_gpt,
                        })

                        if "===MODEL_SEARCH_RESULT_START===" in text_to_gpt:
                            result = self._parse_search_output(text_to_gpt)
                            if result.success:
                                # 运行合理性检查
                                result.sanity_report = run_full_sanity_check(
                                    icc=result.icc,
                                    r2_marginal=result.r2_marginal,
                                    r2_conditional=result.r2_conditional,
                                )
                                logger.info(
                                    f"[ModelSearchAgent] 搜索完成: "
                                    f"最优={result.best_model_id}, "
                                    f"AIC={result.best_aic:.2f}, ICC={result.icc:.4f}"
                                )
                                return result
                else:
                    content = response.content or ""
                    if "===MODEL_SEARCH_RESULT_START===" in content:
                        result = self._parse_search_output(content)
                        if result.success:
                            return result
                    await self.append_chat_history({"role": "assistant", "content": content})
                    await self.append_chat_history({"role": "user", "content": "请继续执行模型搜索代码。"})

            except Exception as e:
                logger.error(f"[ModelSearchAgent] 执行异常: {e}")

        result.error_message = f"模型搜索在 {self.max_retries} 次重试后失败"
        return result

    def _parse_search_output(self, output: str) -> ModelSearchResult:
        """解析模型搜索输出。"""
        result = ModelSearchResult()

        result_pattern = r'===MODEL_SEARCH_RESULT_START===\s*\n(.*?)\n\s*===MODEL_SEARCH_RESULT_END==='
        match = re.search(result_pattern, output, re.DOTALL)
        if not match:
            result.error_message = "输出中未找到 MODEL_SEARCH_RESULT 标记"
            return result

        for line in match.group(1).strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            colon_idx = line.index(":")
            key = line[:colon_idx].strip()
            value = line[colon_idx + 1:].strip()
            try:
                if key == "BEST_MODEL_ID":
                    result.best_model_id = value
                elif key == "BEST_MODEL_FORMULA":
                    result.best_model_formula = value
                elif key == "BEST_AIC":
                    result.best_aic = float(value)
                elif key == "BEST_BIC":
                    result.best_bic = float(value)
                elif key == "ICC":
                    result.icc = float(value)
                elif key == "R2_MARGINAL":
                    result.r2_marginal = float(value)
                elif key == "R2_CONDITIONAL":
                    result.r2_conditional = float(value)
            except (ValueError, TypeError):
                pass

        table_pattern = r'===COMPARISON_TABLE_START===\s*\n(.*?)\n\s*===COMPARISON_TABLE_END==='
        match = re.search(table_pattern, output, re.DOTALL)
        if match:
            result.comparison_table = match.group(1).strip()

        just_pattern = r'===JUSTIFICATION_START===\s*\n(.*?)\n\s*===JUSTIFICATION_END==='
        match = re.search(just_pattern, output, re.DOTALL)
        if match:
            result.selection_justification = match.group(1).strip()

        result.success = bool(result.best_model_id and result.comparison_table)
        return result
