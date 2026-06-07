#!/usr/bin/env python3
"""
AnySearch Academic — 基于 AnySearch API 的学术论文搜索封装

直接通过 JSON-RPC 2.0 协议调用 AnySearch API，专注于 academic 垂直域。
不依赖 requests 库（使用标准库 urllib），与 openalex_scholar.py 保持依赖一致。
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

ENDPOINT = "https://api.anysearch.com/mcp"


class AnySearchAcademic:
    """AnySearch 学术搜索封装"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANYSEARCH_API_KEY", "")

    def search_papers(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """
        通过 AnySearch academic 垂直域搜索学术论文。

        Args:
            query: 搜索关键词
            limit: 返回结果数量上限

        Returns:
            统一格式的论文字典列表
        """
        arguments = {
            "query": query,
            "domain": "academic",
            "sub_domain": "academic.search",
            "content_types": ["academic"],
            "max_results": limit,
        }

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "search", "arguments": arguments},
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = urllib.request.Request(
                ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data: Dict = json.loads(resp.read().decode("utf-8"))
            return self._parse_response(data)
        except urllib.error.HTTPError as e:
            print(f"[AnySearch] HTTP 错误 ({e.code}): {e.reason}")
            return []
        except urllib.error.URLError as e:
            print(f"[AnySearch] 网络连接失败: {e.reason}")
            return []
        except json.JSONDecodeError:
            print("[AnySearch] API 返回数据格式异常")
            return []
        except Exception as e:
            print(f"[AnySearch] 搜索异常: {e}")
            return []

    def _parse_response(self, data: Dict) -> List[Dict[str, Any]]:
        """解析 AnySearch JSON-RPC 响应，提取论文信息。"""
        if "error" in data:
            msg = data["error"].get("message", str(data["error"]))
            print(f"[AnySearch] API 错误: {msg}")
            return []

        result = data.get("result", {})
        content = result.get("content", [])

        raw_items: List[Dict] = []
        for item in content:
            text = item.get("text", "")
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                raw_items.extend(parsed)
            else:
                raw_items.append(parsed)

        # 灵活解析不同响应格式
        papers = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            paper = self._normalize(raw)
            if paper.get("title"):
                papers.append(paper)

        return papers

    # ------------------------------------------------------------------
    # 字段映射：兼容 AnySearch 可能返回的不同字段名
    # ------------------------------------------------------------------
    _TITLE_KEYS = ["title", "display_name", "name", "paper_title"]
    _AUTHOR_KEYS = ["authors", "authorships", "authors_list"]
    _YEAR_KEYS = ["publication_year", "year", "pub_year", "date"]
    _CITATION_KEYS = ["cited_by_count", "citations", "citation_count", "times_cited"]
    _DOI_KEYS = ["doi", "doi_link", "identifier"]
    _ABSTRACT_KEYS = ["abstract", "abstract_inverted_index", "summary"]

    def _normalize(self, raw: Dict) -> Dict[str, Any]:
        """统一为项目通用字段格式。"""
        title = self._first_value(raw, self._TITLE_KEYS, "Unknown Title")

        # 作者：可能是字符串列表或对象列表
        authors_raw = self._first_value(raw, self._AUTHOR_KEYS, [])
        authors = self._normalize_authors(authors_raw)

        year = self._to_int(self._first_value(raw, self._YEAR_KEYS))
        citations = self._to_int(self._first_value(raw, self._CITATION_KEYS), 0)

        doi_raw = self._first_value(raw, self._DOI_KEYS, "")
        doi = self._normalize_doi(doi_raw)

        abstract_raw = self._first_value(raw, self._ABSTRACT_KEYS)
        abstract = self._normalize_abstract(abstract_raw)

        return {
            "title": title,
            "authors": authors,
            "year": year,
            "citations": citations,
            "doi": doi,
            "abstract": abstract,
            "source": "anysearch",
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _first_value(d: Dict, keys: List[str], default=None):
        for k in keys:
            v = d.get(k)
            if v is not None and v != "":
                return v
        return default

    @staticmethod
    def _normalize_authors(authors_raw) -> List[str]:
        if not authors_raw:
            return []
        if isinstance(authors_raw, list):
            result = []
            for a in authors_raw:
                if isinstance(a, str):
                    result.append(a)
                elif isinstance(a, dict):
                    name = a.get("display_name") or a.get("name") or a.get("author", {}).get("display_name", "")
                    if name:
                        result.append(name)
            return result
        if isinstance(authors_raw, str):
            return [s.strip() for s in authors_raw.split(";") if s.strip()]
        return []

    @staticmethod
    def _normalize_doi(doi_raw) -> Optional[str]:
        if not doi_raw:
            return None
        doi = str(doi_raw).strip()
        # 去除常见前缀
        for prefix in ["https://doi.org/", "http://doi.org/", "doi:" ]:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
        return doi or None

    @staticmethod
    def _normalize_abstract(raw) -> Optional[str]:
        if not raw:
            return None
        if isinstance(raw, str):
            return raw
        # 倒排索引：兼容 OpenAlex 格式
        if isinstance(raw, dict):
            try:
                max_pos = max(max(v) for v in raw.values())
                words = [""] * (max_pos + 1)
                for word, positions in raw.items():
                    for pos in positions:
                        words[pos] = word
                return " ".join(words).strip()
            except (ValueError, TypeError, KeyError):
                return None
        return str(raw)

    @staticmethod
    def _to_int(value, default=None):
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
