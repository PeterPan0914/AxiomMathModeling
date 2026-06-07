"""论文全局上下文模块，在整个写作过程中维护共享状态。

解决的核心问题：每个 Agent 都是"第一次见"，没有共享上下文。
PaperContext 在 solution 阶段和 writing 阶段持续维护，每章写完后更新，
下一章的 prompt 注入当前上下文。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class KeyResult:
    """某个问题的关键结果。"""
    conclusion: str = ""                    # 主要结论
    key_numbers: dict[str, str] = field(default_factory=dict)  # 关键数值 {"R²": "0.94"}
    figures: list[str] = field(default_factory=list)           # 对应图表文件名
    method_used: str = ""                   # 使用的方法
    method_reason: str = ""                 # 选择该方法的理由

    def to_dict(self) -> dict:
        return {
            "conclusion": self.conclusion,
            "key_numbers": self.key_numbers,
            "figures": self.figures,
            "method_used": self.method_used,
            "method_reason": self.method_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KeyResult:
        return cls(
            conclusion=d.get("conclusion", ""),
            key_numbers=d.get("key_numbers", {}),
            figures=d.get("figures", []),
            method_used=d.get("method_used", ""),
            method_reason=d.get("method_reason", ""),
        )


class PaperContext:
    """论文全局上下文，在整个任务执行过程中持续维护。

    职责：
    1. 追踪各问题的关键结果（数值、结论、图表）
    2. 维护已定义的符号表（防止前后不一致）
    3. 记录各章的论证钩子（确保前后呼应）
    4. 记录已使用的连接词和模板句（避免重复）
    5. 提供上下文注入文本（拼入 Writer prompt）
    """

    def __init__(self) -> None:
        self.core_argument: str = ""                            # 本文核心论点
        self.key_results: dict[str, KeyResult] = {}             # 各问题的关键结果
        self.defined_symbols: dict[str, str] = {}               # 已定义符号 {符号: 含义}
        self.used_connectors: list[str] = []                    # 已用过的连接词
        self.chapter_hooks: dict[str, str] = {}                 # 各章承诺的钩子
        self.model_choices: dict[str, str] = {}                 # 各问题选择的模型及理由
        self.section_summaries: dict[str, str] = {}             # 各章节的简短摘要

    def update_key_result(
        self,
        section_key: str,
        conclusion: str = "",
        key_numbers: dict[str, str] | None = None,
        figures: list[str] | None = None,
        method_used: str = "",
        method_reason: str = "",
    ) -> None:
        """更新某个问题的关键结果。"""
        if section_key not in self.key_results:
            self.key_results[section_key] = KeyResult()
        kr = self.key_results[section_key]
        if conclusion:
            kr.conclusion = conclusion
        if key_numbers:
            kr.key_numbers.update(key_numbers)
        if figures:
            kr.figures.extend(figures)
        if method_used:
            kr.method_used = method_used
        if method_reason:
            kr.method_reason = method_reason

    def add_symbols(self, symbols: dict[str, str]) -> None:
        """添加新定义的符号。"""
        self.defined_symbols.update(symbols)

    def add_section_summary(self, section_key: str, summary: str) -> None:
        """添加章节摘要。"""
        self.section_summaries[section_key] = summary

    def set_core_argument(self, argument: str) -> None:
        """设置论文核心论点。"""
        self.core_argument = argument

    def inject_into_prompt(self, section_key: str) -> str:
        """生成上下文注入文本，用于拼入 Writer prompt。

        Args:
            section_key: 当前要写的章节标识。

        Returns:
            格式化的上下文文本。如果没有任何上下文，返回空字符串。
        """
        parts: list[str] = []

        # 核心论点
        if self.core_argument:
            parts.append(f"【论文核心论点】\n{self.core_argument}")

        # 已完成章节的关键结果
        completed_results = {
            k: v for k, v in self.key_results.items()
            if v.conclusion and k != section_key
        }
        if completed_results:
            result_lines = []
            for key, kr in completed_results.items():
                display_name = key.replace("ques", "问题").replace("eda", "EDA分析")
                nums = ", ".join(f"{k}={v}" for k, v in kr.key_numbers.items()) if kr.key_numbers else ""
                line = f"- {display_name}: {kr.conclusion}"
                if nums:
                    line += f" (关键数据: {nums})"
                if kr.method_used:
                    line += f" [方法: {kr.method_used}]"
                result_lines.append(line)
            parts.append("【已完成分析的关键结果】\n" + "\n".join(result_lines))

        # 已定义的符号表（防止前后不一致）
        if self.defined_symbols and section_key not in ("symbol", "firstPage"):
            symbol_lines = [
                f"- {sym}: {meaning}"
                for sym, meaning in list(self.defined_symbols.items())[:30]
            ]
            if len(self.defined_symbols) > 30:
                symbol_lines.append(f"... 共 {len(self.defined_symbols)} 个符号")
            parts.append("【已定义的符号（请保持一致）】\n" + "\n".join(symbol_lines))

        # 各章选择的模型（保持术语一致）
        if self.model_choices and section_key.startswith("ques"):
            model_lines = [
                f"- {k.replace('ques', '问题')}: {v}"
                for k, v in self.model_choices.items()
                if k != section_key
            ]
            if model_lines:
                parts.append("【其他问题选择的模型（保持术语一致）】\n" + "\n".join(model_lines))

        # 已完成章节的摘要（保持前后呼应）
        if self.section_summaries:
            summary_lines = [
                f"- {k}: {v}"
                for k, v in self.section_summaries.items()
                if k != section_key
            ]
            if summary_lines:
                parts.append("【已完成章节摘要（用于前后呼应）】\n" + "\n".join(summary_lines[-5:]))

        # 当前章节在整体中的位置提示
        if section_key in self.chapter_hooks:
            parts.append(f"【本章的论证承诺】\n{self.chapter_hooks[section_key]}")

        if not parts:
            return ""

        return "\n\n" + "\n\n".join(parts)

    def extract_from_writer_response(self, section_key: str, content: str) -> None:
        """从 WriterAgent 的输出中提取信息更新上下文。

        提取：符号定义、数值结果、章节摘要。

        Args:
            section_key: 章节标识。
            content: WriterAgent 输出的论文文本。
        """
        # 提取符号定义（从符号说明章节）
        if section_key == "symbol":
            self._extract_symbols(content)

        # 提取数值结果（R², RMSE, MAE 等）
        self._extract_key_numbers(section_key, content)

        # 生成章节摘要（取前200字）
        clean = re.sub(r'[#*`\[\]()]', '', content)
        clean = re.sub(r'\s+', ' ', clean).strip()
        self.section_summaries[section_key] = clean[:200] + "..." if len(clean) > 200 else clean

    def _extract_symbols(self, content: str) -> None:
        """从符号说明章节提取符号定义。"""
        # 匹配 $符号$ 格式
        symbol_pattern = r'\$([^$]+)\$'
        # 匹配表格行中的符号
        table_rows = re.findall(r'\|([^|]+)\|([^|]+)\|([^|]*)\|', content)
        for row in table_rows:
            sym_match = re.search(symbol_pattern, row[0])
            if sym_match:
                sym = sym_match.group(1).strip()
                meaning = row[1].strip()
                if sym and meaning:
                    self.defined_symbols[sym] = meaning

    def _extract_key_numbers(self, section_key: str, content: str) -> None:
        """从结果分析中提取关键数值。"""
        numbers: dict[str, str] = {}

        # R² / R-squared
        r2_match = re.search(r'R\^?2\s*[=≈:]\s*([\d.]+)', content)
        if r2_match:
            numbers["R²"] = r2_match.group(1)

        # RMSE
        rmse_match = re.search(r'RMSE\s*[=≈:]\s*([\d.]+)', content)
        if rmse_match:
            numbers["RMSE"] = rmse_match.group(1)

        # MAE
        mae_match = re.search(r'MAE\s*[=≈:]\s*([\d.]+)', content)
        if mae_match:
            numbers["MAE"] = mae_match.group(1)

        # 准确率 / Accuracy
        acc_match = re.search(r'(?:准确率|Accuracy)\s*[=≈:]\s*([\d.]+%?)', content)
        if acc_match:
            numbers["准确率"] = acc_match.group(1)

        # AUC
        auc_match = re.search(r'AUC\s*[=≈:]\s*([\d.]+)', content)
        if auc_match:
            numbers["AUC"] = auc_match.group(1)

        # p-value
        pval_match = re.search(r'p\s*[=≈<]\s*([\d.]+)', content)
        if pval_match:
            numbers["p值"] = pval_match.group(1)

        if numbers:
            self.update_key_result(section_key, key_numbers=numbers)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "core_argument": self.core_argument,
            "key_results": {k: v.to_dict() for k, v in self.key_results.items()},
            "defined_symbols": self.defined_symbols,
            "used_connectors": self.used_connectors,
            "chapter_hooks": self.chapter_hooks,
            "model_choices": self.model_choices,
            "section_summaries": self.section_summaries,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PaperContext:
        """从字典反序列化。"""
        ctx = cls()
        ctx.core_argument = d.get("core_argument", "")
        ctx.key_results = {
            k: KeyResult.from_dict(v) for k, v in d.get("key_results", {}).items()
        }
        ctx.defined_symbols = d.get("defined_symbols", {})
        ctx.used_connectors = d.get("used_connectors", [])
        ctx.chapter_hooks = d.get("chapter_hooks", {})
        ctx.model_choices = d.get("model_choices", {})
        ctx.section_summaries = d.get("section_summaries", {})
        return ctx

    def save(self, work_dir: str) -> None:
        """保存到诊断目录。"""
        diag_dir = os.path.join(work_dir, "diagnostic")
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(diag_dir, "paper_context.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, work_dir: str) -> PaperContext:
        """从诊断目录加载。"""
        path = os.path.join(work_dir, "diagnostic", "paper_context.json")
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
