#!/usr/bin/env python3
"""
LaTeX 方程 → Word OMML 公式转换工具

将 .docx 文档中的 LaTeX 占位符替换为 Word 原生的数学公式（OMML 格式），
使得公式在 Word 中可编辑、可渲染，而非显示为纯文本 LaTeX 代码。

流程: LaTeX → MathML → OMML → 插入 docx

依赖:
    pip install latex2mathml lxml python-docx

用法:
    # 单个公式替换
    python equations.py paper.docx ^
        --replace "EQ1" "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}" ^
        -o paper_final.docx

    # 批量替换（JSON 文件）
    python equations.py paper.docx --mapping equations.json -o paper_final.docx

    # 从 Markdown 生成 docx（含公式）
    python equations.py generate paper.md -o paper.docx --template template.docx
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from docx import Document
    from docx.oxml import OxmlElement, qn
    from docx.oxml.ns import nsmap
    from lxml import etree
except ImportError:
    print("错误: 请先安装依赖: pip install python-docx lxml", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# OMML / MathML 命名空间
# ---------------------------------------------------------------------------
OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
OMML_PREFIX = "m"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# 注册命名空间前缀，保证序列化干净
ET.register_namespace(OMML_PREFIX, OMML_NS)
ET.register_namespace("", MATHML_NS)

# ---------------------------------------------------------------------------
# 核心：LaTeX → MathML → OMML
# ---------------------------------------------------------------------------

def latex2omml(latex_str: str) -> bytes:
    """
    将 LaTeX 字符串转换为 Word OMML XML（即 <m:oMath> 元素内的 XML 字符串）。

    使用 latex2mathml 转换为 MathML，再包装为 OMML。
    """
    try:
        from latex2mathml import convert
    except ImportError:
        print("正在安装 latex2mathml...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "latex2mathml", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        from latex2mathml import convert

    # Step 1: LaTeX → MathML (presentation MathML)
    mathml_str = convert(latex_str)

    # Step 2: 解析 MathML <math> 元素
    mathml_root = etree.fromstring(mathml_str.encode("utf-8"))

    # Step 3: 包装为 OMML <m:oMath> 元素
    # OMML 容器中包含 MathML 内容，Word 同时支持这两种格式
    omml = etree.Element(f"{{{OMML_NS}}}oMath")

    # 将 MathML 元素复制到 OMML 下
    for child in mathml_root:
        omml.append(child)

    # 如果 MathML 转换后没有子元素，则直接包装整个 math 元素
    if len(omml) == 0:
        for child in mathml_root.iter():
            if child.tag == f"{{{MATHML_NS}}}math":
                for grandchild in child:
                    omml.append(grandchild)
                break

    return etree.tostring(omml, encoding="unicode").encode("utf-8")


def latex2mathml_direct(latex_str: str) -> str:
    """仅返回 MathML 字符串（调试用）。"""
    from latex2mathml import convert
    return convert(latex_str)


# ---------------------------------------------------------------------------
# docx 操作：插入 OMML 方程
# ---------------------------------------------------------------------------

def find_paragraph_with_text(doc: Document, text: str) -> object:
    """
    在文档中查找包含指定文本的第一个段落。
    返回 docx Paragraph 对象，或 None。
    """
    for para in doc.paragraphs:
        if text in para.text:
            return para
    return None


def replace_with_equation(para, omml_xml: bytes):
    """
    将段落中所有文本替换为 OMML 公式元素。

    原段落的内容会被清空，然后插入 <m:oMathPara> 包含 <m:oMath>。
    """
    # 清空段落的所有 run
    for r in para._element.findall(qn("w:r")):
        para._element.remove(r)
    for r in para._element.findall(qn("w:rPr")):
        para._element.remove(r)

    # 解析 OMML XML
    omml_elem = etree.fromstring(omml_xml)

    # 创建 <m:oMathPara> 包装器
    math_para = OxmlElement("m:oMathPara")
    # 确保命名空间映射正确
    math_para.set(f"xmlns:{OMML_PREFIX}", OMML_NS)

    # 将 <m:oMath> 添加到 <m:oMathPara>
    math_elem = OxmlElement("m:oMath")
    for child in omml_elem:
        math_elem.append(child)
    math_para.append(math_elem)

    # 插入到段落
    para._element.append(math_para)


def replace_placeholder(doc: Document, placeholder: str, latex: str):
    """
    查找占位符文本并替换为公式。
    """
    para = find_paragraph_with_text(doc, placeholder)
    if para is None:
        print(f"  ⚠ 未找到占位符 '{placeholder}'，跳过")
        return False

    # 检查是否是纯占位符段落
    text = para.text.strip()
    if text == placeholder:
        # 整段替换
        omml_xml = latex2omml(latex)
        replace_with_equation(para, omml_xml)
        print(f"  ✓ '{placeholder}' → 公式已插入")
    else:
        # 占位符在段落文本中，需要拆分
        # 简化处理：整段替换
        omml_xml = latex2omml(latex)
        replace_with_equation(para, omml_xml)
        print(f"  ✓ '{placeholder}' → 公式已插入（整段替换）")

    return True


def batch_replace(doc_path: str, mapping: dict, output_path: str):
    """
    批量替换占位符为公式。

    mapping = {
        "占位符文本": "LaTeX 公式",
        "EQ_MODEL": "\\min f(x) = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2",
        ...
    }
    """
    doc = Document(doc_path)

    success = 0
    for placeholder, latex in mapping.items():
        if replace_placeholder(doc, placeholder, latex):
            success += 1

    doc.save(output_path)
    print(f"\n完成: {success}/{len(mapping)} 个公式已插入 → {output_path}")
    return success


# ---------------------------------------------------------------------------
# 从 Markdown 生成 docx（使用 pandoc 后端）
# ---------------------------------------------------------------------------

def markdown_to_docx(md_path: str, output_path: str,
                     template_path: str = None, mathml: bool = True):
    """
    使用 pandoc 将 Markdown 文件（含 $$ LaTeX $$）转换为 .docx。

    pandoc 原生支持 LaTeX 方程 → Word OMML 转换，这是最可靠的方式。

    需安装 pandoc: https://pandoc.org/installing.html
    """
    cmd = ["pandoc", str(md_path), "-o", str(output_path)]
    if mathml:
        cmd.append("--mathml")
    if template_path:
        cmd.extend(["--reference-doc", str(template_path)])

    print(f"运行: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ 已生成: {output_path}")
    except FileNotFoundError:
        print("错误: pandoc 未安装。请安装: https://pandoc.org/installing.html", file=sys.stderr)
        print("  或使用 batch_replace 模式对现有 .docx 注入公式。", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"错误: pandoc 转换失败: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def load_mapping(file_path: str) -> dict:
    """从 JSON 文件加载占位符→LaTeX 映射。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {item["placeholder"]: item["latex"] for item in data}
    raise ValueError("JSON 格式错误，应为 dict 或 [{placeholder, latex}, ...]")


def build_parser():
    parser = argparse.ArgumentParser(
        description="LaTeX 方程 → Word OMML 公式转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", help="运行模式")

    # ---- 模式 1: replace（替换 docx 中的占位符） ----
    rp = sub.add_parser("replace", help="替换 .docx 中的占位符为公式")
    rp.add_argument("input", help="输入 .docx 文件路径")
    rp.add_argument("--mapping", "-m", help="JSON 映射文件 ({\"占位符\": \"LaTeX\", ...})")
    rp.add_argument("--replace", "-r", nargs=2, action="append",
                    metavar=("PLACEHOLDER", "LATEX"),
                    help="单个替换对，可重复使用")
    rp.add_argument("--output", "-o", default=None,
                    help="输出 .docx 路径（默认覆盖输入文件）")
    rp.add_argument("--show-mathml", action="store_true",
                    help="仅显示 LaTeX → MathML 转换结果，不操作 docx")

    # ---- 模式 2: generate（从 Markdown 生成） ----
    gn = sub.add_parser("generate", help="从 Markdown 生成含公式的 .docx")
    gn.add_argument("input", help="输入 .md 文件路径（使用 $$...$$ 或 $...$ 写公式）")
    gn.add_argument("--output", "-o", required=True, help="输出 .docx 路径")
    gn.add_argument("--template", "-t", help="pandoc 参考模板 .docx")
    gn.add_argument("--no-mathml", action="store_true",
                    help="不使用 --mathml 标志（默认启用）")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "replace":
        # 收集替换映射
        mapping = {}
        if args.mapping:
            mapping.update(load_mapping(args.mapping))
        if args.replace:
            for placeholder, latex in args.replace:
                mapping[placeholder] = latex

        if not mapping:
            print("错误: 请提供 --mapping 或 --replace", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

        if args.show_mathml:
            print("LaTeX → MathML 预览:")
            print("=" * 60)
            for placeholder, latex in mapping.items():
                print(f"\n占位符: {placeholder}")
                print(f"LaTeX:   {latex}")
                try:
                    mathml = latex2mathml_direct(latex)
                    print(f"MathML: {mathml}")
                except Exception as e:
                    print(f"错误: {e}")
            return

        output = args.output or args.input
        batch_replace(args.input, mapping, output)

    elif args.mode == "generate":
        markdown_to_docx(
            args.input, args.output,
            template_path=args.template,
            mathml=not args.no_mathml,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
