#!/usr/bin/env python3
"""
Hybrid Scholar — 并行搜索 + 交叉验证

同时调用 OpenAlex 和 AnySearch 学术搜索，对结果进行去重和交叉验证。
当同一篇论文同时被两个数据源收录时，标记为「交叉验证」，可信度更高。
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from openalex_scholar import OpenAlexScholar, Paper
from anysearch_academic import AnySearchAcademic


# ---------------------------------------------------------------------------
# 搜索结果联合数据类
# ---------------------------------------------------------------------------

@dataclass
class HybridPaper:
    """融合论文（带来源追踪）"""
    title: str
    authors: List[str]
    year: Optional[int]
    citations: int
    doi: Optional[str]
    abstract: Optional[str]
    sources: List[str] = field(default_factory=list)
    # sources = ["openalex"], ["anysearch"], 或 ["openalex", "anysearch"]

    @property
    def cross_validated(self) -> bool:
        return len(self.sources) >= 2

    @property
    def source_tag(self) -> str:
        if self.cross_validated:
            return "✓ 交叉验证"
        return self.sources[0] if self.sources else "?"

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "citations": self.citations,
            "doi": self.doi,
            "abstract": self.abstract,
            "sources": self.sources,
            "cross_validated": self.cross_validated,
        }


# ---------------------------------------------------------------------------
# Hybrid Scholar
# ---------------------------------------------------------------------------

class HybridScholar:
    """并行搜索 + 交叉验证器"""

    def __init__(self, email: Optional[str] = None,
                 anysearch_api_key: Optional[str] = None):
        self.openalex = OpenAlexScholar(email=email)
        self.anysearch = AnySearchAcademic(api_key=anysearch_api_key)

    def search_papers(
        self,
        query: str,
        limit: int = 8,
        sort: str = "relevance",
        min_citations: Optional[int] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        field_filter: Optional[str] = None,
        openalex_only: bool = False,
        anysearch_only: bool = False,
    ) -> Dict[str, Any]:
        """
        并行搜索 + 交叉验证。

        Args:
            query: 搜索关键词
            limit: 最终返回结果数量
            sort: 排序方式（仅 OpenAlex）
            min_citations: 最低引用量（仅 OpenAlex）
            year_from / year_to: 年份范围（仅 OpenAlex）
            field_filter: 领域过滤（仅 OpenAlex）
            openalex_only: 仅用 OpenAlex
            anysearch_only: 仅用 AnySearch

        Returns:
            {
                "cross_validated": [...],   # 同时被两个源收录
                "openalex_only": [...],     # 仅 OpenAlex
                "anysearch_only": [...],    # 仅 AnySearch
                "stats": {...},
            }
        """
        # 决定启用哪些源
        use_oa = not anysearch_only
        use_any = not openalex_only

        # 为了去重效果更好，每个源多取一些
        fetch_limit = max(limit * 2, 15) if (use_oa and use_any) else max(limit, 8)

        oa_papers: List[Paper] = []
        any_papers: List[Dict] = []

        # ---- 并行执行 ----
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = []

            if use_oa:
                futures.append(pool.submit(
                    self.openalex.search_papers,
                    query, limit=fetch_limit, sort=sort,
                    min_citations=min_citations,
                    year_from=year_from, year_to=year_to,
                    field_filter=field_filter,
                ))

            if use_any:
                futures.append(pool.submit(
                    self.anysearch.search_papers,
                    query, limit=fetch_limit,
                ))

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if not result:
                        continue
                    if isinstance(result[0], Paper):
                        oa_papers = result
                    elif isinstance(result[0], dict):
                        any_papers = result
                except Exception as e:
                    print(f"[杂交] 并行搜索异常: {e}", file=sys.stderr)

        # ---- 融合与去重 ----
        return self._fuse(oa_papers, any_papers, final_limit=limit)

    # ------------------------------------------------------------------
    # 融合 / 去重 / 交叉验证
    # ------------------------------------------------------------------

    def _fuse(self, oa_papers: List[Paper], any_papers: List[Dict],
              final_limit: int) -> Dict[str, Any]:
        """融合两个源的结果，去重并标记交叉验证。"""

        # Step 1 — 用 DOI 建立索引
        oa_by_doi: Dict[str, Paper] = {}
        oa_no_doi: List[Paper] = []
        for p in oa_papers:
            if p.doi:
                oa_by_doi[p.doi.lower()] = p
            else:
                oa_no_doi.append(p)

        any_by_doi: Dict[str, Dict] = {}
        any_no_doi: List[Dict] = []
        for p in any_papers:
            doi = (p.get("doi") or "").lower()
            if doi:
                any_by_doi[doi] = p
            else:
                any_no_doi.append(p)

        # Step 2 — 交叉验证（DOI 匹配）
        cross: List[HybridPaper] = []
        all_dois: Set[str] = set(oa_by_doi.keys()) | set(any_by_doi.keys())

        for doi in all_dois:
            oa_p = oa_by_doi.get(doi)
            any_p = any_by_doi.get(doi)
            if oa_p and any_p:
                # 优先使用 OpenAlex 的结构化数据
                cross.append(HybridPaper(
                    title=oa_p.title,
                    authors=oa_p.authors,
                    year=oa_p.publication_year,
                    citations=oa_p.cited_by_count,
                    doi=doi,
                    abstract=oa_p.abstract,
                    sources=["openalex", "anysearch"],
                ))

        # Step 3 — 非交叉部分（无 DOI 或单源）
        oa_only: List[HybridPaper] = []
        for p in oa_papers:
            key = (p.doi or "").lower()
            if key in any_by_doi:
                continue  # 已在 cross 中
            oa_only.append(HybridPaper(
                title=p.title,
                authors=p.authors,
                year=p.publication_year,
                citations=p.cited_by_count,
                doi=p.doi,
                abstract=p.abstract,
                sources=["openalex"],
            ))

        any_only: List[HybridPaper] = []
        for p in any_papers:
            key = (p.get("doi") or "").lower()
            if key in oa_by_doi:
                continue  # 已在 cross 中
            any_only.append(HybridPaper(
                title=p.get("title", "Unknown"),
                authors=p.get("authors", []),
                year=p.get("year"),
                citations=p.get("citations", 0),
                doi=p.get("doi"),
                abstract=p.get("abstract"),
                sources=["anysearch"],
            ))

        # Step 4 — 标题模糊去重（针对无 DOI 的论文）
        oa_only, any_only = self._fuzzy_dedup(oa_only, any_only)

        # Step 5 — 排序并截断
        cross.sort(key=lambda x: x.citations, reverse=True)
        oa_only.sort(key=lambda x: x.citations, reverse=True)
        any_only.sort(key=lambda x: x.citations, reverse=True)

        # 等比例分配名额
        total_needed = final_limit
        n_cross = min(len(cross), max(2, total_needed // 3))
        remaining = total_needed - n_cross
        n_oa = min(len(oa_only), remaining // 2)
        n_any = remaining - n_oa

        return {
            "query": self._current_query,
            "cross_validated": cross[:n_cross],
            "openalex_only": oa_only[:n_oa],
            "anysearch_only": any_only[:n_any],
            "stats": {
                "openalex_total": len(oa_papers),
                "anysearch_total": len(any_papers),
                "cross_validated": len(cross),
                "openalex_unique": len(oa_only),
                "anysearch_unique": len(any_only),
            },
        }

    def _fuzzy_dedup(self, oa_only: List[HybridPaper],
                     any_only: List[HybridPaper]) -> Tuple[List[HybridPaper], List[HybridPaper]]:
        """基于标题相似度的模糊去重（针对无 DOI 论文）。"""
        def normalize(title: str) -> str:
            t = title.lower().strip()
            t = re.sub(r'[^\w\s]', '', t)
            return ' '.join(t.split())

        def overlap(a: str, b: str) -> float:
            words_a = set(normalize(a).split())
            words_b = set(normalize(b).split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / max(len(words_a), len(words_b))

        # 检查 oa_only 中的标题与 any_only 中的标题
        kept_oa = list(oa_only)
        kept_any = list(any_only)

        for hp in oa_only:
            if hp.doi:
                continue
            for ap in any_only:
                if ap.doi:
                    continue
                if overlap(hp.title, ap.title) >= 0.7:
                    # 高相似度 → 合并到交叉验证，从单源列表中移除
                    if hp in kept_oa:
                        kept_oa.remove(hp)
                    if ap in kept_any:
                        kept_any.remove(ap)

        return kept_oa, kept_any

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------

    _current_query: str = ""

    def print_results(self, result: Dict[str, Any]):
        """打印可读的交叉验证结果。"""
        query = result.get("query", "")
        self._current_query = query  # stash for template
        stats = result["stats"]

        header = f"交叉验证搜索结果: {query}"
        print()
        print("=" * 60)
        print(f"  {header}")
        print("=" * 60)
        print(f"  数据源: OpenAlex + AnySearch")
        print(f"  统计: OpenAlex {stats['openalex_total']} 篇 | "
              f"AnySearch {stats['anysearch_total']} 篇 | "
              f"交叉验证 {stats['cross_validated']} 篇")
        print()

        # 交叉验证区域
        cross = result.get("cross_validated", [])
        if cross:
            self._print_section("交叉验证", "★", "OpenAlex + AnySearch 同时收录", cross, "verified")

        oa_only = result.get("openalex_only", [])
        if oa_only:
            self._print_section("OpenAlex 独有", "◆", "仅来自 OpenAlex", oa_only, "oa")

        any_only = result.get("anysearch_only", [])
        if any_only:
            self._print_section("AnySearch 独有", "◇", "仅来自 AnySearch", any_only, "any")

        if not cross and not oa_only and not any_only:
            print("  未找到相关论文。\n")

    _SECTION_COLORS = {
        "verified": "\033[33m",  # 金色
        "oa": "\033[36m",       # 青色
        "any": "\033[35m",      # 紫色
        "reset": "\033[0m",
    }
    # Windows 兼容：如果颜色不支持则静默降级
    _USE_COLOR = sys.platform != "win32" or os.environ.get("TERM", "").startswith("xterm")

    @classmethod
    def _c(cls, code: str) -> str:
        if cls._USE_COLOR:
            return cls._SECTION_COLORS.get(code, "")
        return ""

    def _print_section(self, title: str, icon: str, subtitle: str,
                       papers: List[HybridPaper], tag: str):
        c_tag = self._c(tag)
        c_reset = self._c("reset")

        print(f"  {c_tag}{icon} {title}{c_reset}")
        print(f"  {c_tag}  {subtitle}{c_reset}")
        print(f"  {c_tag}{'─' * 56}{c_reset}")

        for i, hp in enumerate(papers, 1):
            authors = ", ".join(hp.authors[:4])
            if len(hp.authors) > 4:
                authors += " et al."

            line = f"  [{i}] {hp.title}"
            print(f"  {c_tag}{line}{c_reset}")

            details = []
            if authors:
                details.append(f"作者: {authors}")
            if hp.year:
                details.append(f"年份: {hp.year}")
            if hp.citations:
                details.append(f"引用: {hp.citations}")
            if hp.doi:
                details.append(f"DOI: {hp.doi}")

            if details:
                print(f"     {' | '.join(details)}")
            if hp.abstract:
                preview = hp.abstract[:120].replace("\n", " ")
                print(f"     摘要: {preview}...")
            print()

        print()

    def results_to_json(self, result: Dict[str, Any]) -> str:
        """输出为 JSON。"""
        return json.dumps(result, ensure_ascii=False, indent=2,
                          default=self._json_default)

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, HybridPaper):
            return obj.to_dict()
        if isinstance(obj, Paper):
            return obj.to_dict()
        return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid Scholar — 并行搜索 + 交叉验证 (OpenAlex + AnySearch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 混合搜索（默认）
  python hybrid_scholar.py --query "grey prediction model"

  # 仅用 OpenAlex（传统模式）
  python hybrid_scholar.py --query "genetic algorithm" --openalex-only

  # 仅用 AnySearch
  python hybrid_scholar.py --query "reinforcement learning" --anysearch-only

  # 高级过滤 + 交叉验证
  python hybrid_scholar.py --query "TOPSIS" --min-citations 10 --year-from 2020 --field mathematics

  # JSON 输出
  python hybrid_scholar.py --query "LSTM" --json
        """,
    )
    parser.add_argument("--query", "-q", required=True, help="搜索关键词")
    parser.add_argument("--limit", "-n", type=int, default=8,
                        help="最终返回结果数量（默认 8）")
    parser.add_argument("--email", "-e", default="your@email.com",
                        help="OpenAlex 礼貌池邮箱")
    parser.add_argument("--anysearch-api-key",
                        help="AnySearch API Key（默认读取 ANYSEARCH_API_KEY 环境变量）")
    parser.add_argument("--sort", "-s",
                        choices=["relevance", "cited_by_count:desc",
                                 "cited_by_count:asc", "publication_year:desc",
                                 "publication_year:asc"],
                        default="relevance",
                        help="排序方式（仅 OpenAlex，默认相关性）")
    parser.add_argument("--min-citations", type=int,
                        help="最低引用量过滤（仅 OpenAlex）")
    parser.add_argument("--year-from", type=int,
                        help="起始年份（仅 OpenAlex）")
    parser.add_argument("--year-to", type=int,
                        help="结束年份（仅 OpenAlex）")
    parser.add_argument("--field",
                        choices=["mathematics", "computer_science", "engineering",
                                 "statistics", "operations_research", "physics", "economics"],
                        help="领域过滤（仅 OpenAlex）")
    parser.add_argument("--openalex-only", action="store_true",
                        help="仅使用 OpenAlex 搜索")
    parser.add_argument("--anysearch-only", action="store_true",
                        help="仅使用 AnySearch 搜索")
    parser.add_argument("--json", "-j", action="store_true",
                        help="以 JSON 格式输出")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    scholar = HybridScholar(
        email=args.email,
        anysearch_api_key=args.anysearch_api_key,
    )

    result = scholar.search_papers(
        query=args.query,
        limit=args.limit,
        sort=args.sort,
        min_citations=args.min_citations,
        year_from=args.year_from,
        year_to=args.year_to,
        field_filter=args.field,
        openalex_only=args.openalex_only,
        anysearch_only=args.anysearch_only,
    )

    if args.json:
        print(scholar.results_to_json(result))
    else:
        scholar.print_results(result)


if __name__ == "__main__":
    main()
