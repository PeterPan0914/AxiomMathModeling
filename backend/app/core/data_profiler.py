"""数据形态探针模块，对附件数据执行标准化探查并输出元特征字典。

纯 Python 实现，不依赖 LLM，不消耗调用次数。
在 workflow.py 的 Phase 1 中，ProblemAnalystAgent 之前或并行运行。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from app.utils.log_util import logger


@dataclass
class ColumnDetail:
    """单列的探查结果。"""
    name: str = ""
    dtype: str = ""
    missing_rate: float = 0.0
    n_unique: int = 0
    is_identifier: bool = False
    is_binary: bool = False
    value_range: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "dtype": self.dtype,
            "missing_rate": self.missing_rate, "n_unique": self.n_unique,
            "is_identifier": self.is_identifier, "is_binary": self.is_binary,
            "value_range": self.value_range,
        }


@dataclass
class DataProfile:
    """单个数据文件的探查结果。"""
    file_name: str = ""
    n_rows: int = 0
    n_cols: int = 0
    columns: list[ColumnDetail] = field(default_factory=list)
    has_time_column: bool = False
    has_censored_data: bool = False
    has_repeated_measures: bool = False
    class_imbalance_ratio: float | None = None
    imbalanced_columns: list[str] = field(default_factory=list)
    detected_signals: list[str] = field(default_factory=list)
    summary_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "n_rows": self.n_rows, "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
            "has_time_column": self.has_time_column,
            "has_censored_data": self.has_censored_data,
            "has_repeated_measures": self.has_repeated_measures,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "imbalanced_columns": self.imbalanced_columns,
            "detected_signals": self.detected_signals,
            "summary_stats": self.summary_stats,
        }


class DataProfiler:
    """数据形态探针，对附件数据执行标准化探查。"""

    TIME_KEYWORDS = {"date", "time", "年", "月", "日", "时间", "孕周", "week", "day"}
    CENSOR_KEYWORDS = {"status", "censored", "event", "删失", "截断", "是否达标", "是否异常"}

    def __init__(self, work_dir: str):
        self.work_dir = work_dir

    def profile_all(self) -> list[DataProfile]:
        """扫描工作目录下的所有数据文件并探查。"""
        profiles = []
        for fname in os.listdir(self.work_dir):
            fpath = os.path.join(self.work_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".xlsx", ".xls", ".csv", ".txt"):
                try:
                    profile = self._profile_file(fpath, fname)
                    profiles.append(profile)
                except Exception as e:
                    logger.warning(f"[DataProfiler] 无法探查 {fname}: {e}")
        return profiles

    def _profile_file(self, fpath: str, fname: str) -> DataProfile:
        """探查单个数据文件。"""
        ext = os.path.splitext(fname)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(fpath, nrows=5000)
        elif ext == ".csv":
            df = pd.read_csv(fpath, nrows=5000, encoding="utf-8", on_bad_lines="skip")
        else:
            df = pd.read_csv(fpath, nrows=5000, sep="\t", encoding="utf-8", on_bad_lines="skip")

        profile = DataProfile(file_name=fname, n_rows=len(df), n_cols=len(df.columns))

        # 逐列分析
        id_candidates = []
        for col in df.columns:
            detail = ColumnDetail(
                name=str(col),
                dtype=str(df[col].dtype),
                missing_rate=float(df[col].isna().mean()),
                n_unique=int(df[col].nunique()),
            )
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    detail.value_range = (float(non_null.min()), float(non_null.max()))
                if detail.n_unique <= 3:
                    detail.is_binary = True

            # 检测 ID 列（唯一值数 ≈ 行数）
            if detail.n_unique > len(df) * 0.9 and detail.n_unique > 10:
                detail.is_identifier = True
                id_candidates.append(str(col))

            profile.columns.append(detail)

        # 检测时间列
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in self.TIME_KEYWORDS):
                profile.has_time_column = True
                break

        # 检测删失/事件列
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in self.CENSOR_KEYWORDS):
                profile.has_censored_data = True
                break

        # 检测重复测量结构
        if id_candidates:
            id_col = id_candidates[0]
            counts = df.groupby(id_col).size()
            if counts.mean() > 1.5:
                profile.has_repeated_measures = True

        # 检测类别不平衡
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() <= 10:
                vc = df[col].value_counts()
                if len(vc) >= 2:
                    ratio = vc.iloc[0] / max(vc.iloc[-1], 1)
                    if ratio > 10:
                        profile.imbalanced_columns.append(str(col))
                        if profile.class_imbalance_ratio is None or ratio > profile.class_imbalance_ratio:
                            profile.class_imbalance_ratio = ratio

        # 汇总信号
        if profile.has_repeated_measures:
            profile.detected_signals.append("repeated_measures")
        if profile.has_censored_data:
            profile.detected_signals.append("censored_data")
        if profile.imbalanced_columns:
            profile.detected_signals.append(f"class_imbalance({','.join(profile.imbalanced_columns)})")

        # 描述性统计摘要
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profile.summary_stats = {
                "numeric_columns": len(numeric_cols),
                "mean_missing_rate": float(df[numeric_cols].isna().mean().mean()),
            }

        logger.info(
            f"[DataProfiler] {fname}: {profile.n_rows}行 x {profile.n_cols}列, "
            f"重复测量={profile.has_repeated_measures}, 删失={profile.has_censored_data}, "
            f"信号={profile.detected_signals}"
        )
        return profile


def format_profiles_for_prompt(profiles: list[DataProfile]) -> str:
    """将探查结果格式化为可注入 prompt 的文本。"""
    if not profiles:
        return ""

    lines = ["【数据探查结果（来自 DataProfiler）】\n"]
    for p in profiles:
        lines.append(f"### 文件: {p.file_name}")
        lines.append(f"- 行数: {p.n_rows}, 列数: {p.n_cols}")
        lines.append(f"- 时间列: {'有' if p.has_time_column else '无'}")
        lines.append(f"- 重复测量: {'有' if p.has_repeated_measures else '无'}")
        lines.append(f"- 删失数据: {'有' if p.has_censored_data else '无'}")
        if p.imbalanced_columns:
            lines.append(f"- 类别不平衡列: {', '.join(p.imbalanced_columns)} (最大/最小比={p.class_imbalance_ratio:.1f})")
        if p.detected_signals:
            lines.append(f"- 检测信号: {', '.join(p.detected_signals)}")

        # 列详情（仅显示有缺失或特殊的列）
        for c in p.columns:
            if c.missing_rate > 0.01 or c.is_identifier or c.is_binary:
                lines.append(f"  - {c.name}: dtype={c.dtype}, 缺失={c.missing_rate:.1%}, 唯一值={c.n_unique}")
        lines.append("")

    return "\n".join(lines)
