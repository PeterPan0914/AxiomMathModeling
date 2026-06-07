"""工作流程定义模块，管理建模任务的求解和写作流程。"""

from __future__ import annotations
from typing import TYPE_CHECKING
from app.models.user_output import UserOutput
from app.tools.base_interpreter import BaseCodeInterpreter
from app.core.agents.modeler_agent import ModelerToCoder
from app.core.structure_control import StructureController

if TYPE_CHECKING:
    from app.core.paper_context import PaperContext


class Flows:
    """管理数学建模任务的求解流程和写作流程。"""
    def __init__(self, questions: dict[str, str | int]):
        self.flows: dict[str, dict] = {}
        self.questions: dict[str, str | int] = questions
        self.structure_controller = StructureController()

    def set_flows(self, ques_count: int):
        """根据问题数量设置流程节点。

        Args:
            ques_count: 问题数量。
        """
        ques_str = [f"ques{i}" for i in range(1, ques_count + 1)]
        seq = [
            "firstPage",
            "RepeatQues",
            "analysisQues",
            "modelAssumption",
            "symbol",
            "eda",
            *ques_str,
            "sensitivity_analysis",
            "judge",
        ]
        self.flows = {key: {} for key in seq}

    def get_solution_flows(
        self, questions: dict[str, str | int], modeler_response: ModelerToCoder
    ):
        """生成求解阶段的流程配置。

        Args:
            questions: 包含各问题描述的字典。
            modeler_response: 建模手的响应，包含各问题的解决方案。

        Returns:
            求解流程配置字典，键为任务名，值包含 coder_prompt 等信息。
        """
        questions_quesx = {
            key: value
            for key, value in questions.items()
            if key.startswith("ques") and key != "ques_count"
        }
        solutions = modeler_response.questions_solution
        model_specs = modeler_response.model_specs or {}

        def _build_coder_prompt(ques_key: str, ques_desc: str) -> str:
            """构建 CoderAgent 的 prompt，注入 model_spec（如有）。"""
            solution_text = solutions.get(ques_key, "")
            spec = model_specs.get(ques_key)
            spec_section = ""
            if spec:
                spec_section = f"""

【结构化模型规格（必须严格遵循）】
- 目标函数: {spec.objective}
- 约束条件: {'; '.join(spec.constraints) if spec.constraints else '无'}
- 求解算法: {spec.algorithm}
- 关键参数: {', '.join(f'{k}={v}' for k, v in spec.key_params.items()) if spec.key_params else '无'}
- 预期输出: {spec.expected_output}
- 验证方法: {spec.validation_method}
- 伪代码参考:
{spec.pseudocode}
"""
            return f"""
                        参考建模手给出的解决方案{solution_text}
                        完成如下问题{ques_desc}
                        {spec_section}
                    """

        ques_flow = {
            key: {
                "coder_prompt": _build_coder_prompt(key, value),
            }
            for key, value in questions_quesx.items()
        }
        flows = {
            "eda": {
                "coder_prompt": f"""
                        参考建模手给出的解决方案{solutions.get("eda", "对数据进行探索性分析")}
                        对当前目录下数据进行EDA分析(数据清洗,可视化),清洗后的数据保存当前目录下,**不需要复杂的模型**
                    """,
            },
            **ques_flow,
            "sensitivity_analysis": {
                "coder_prompt": f"""
                        参考建模手给出的方案{solutions.get("sensitivity_analysis", "对模型进行鲁棒性与灵敏度分析")}
                        执行六维度鲁棒性分析框架，包含以下组件：
                        1. 参数灵敏度分析：使用模型拟合的置信区间（非任意±10%），计算灵敏度指数，分析交互效应，绘制龙卷风图和交互热力图
                        2. 结构灵敏度分析：比较至少2种模型规格（不同算法或特征组合），用5折交叉验证对比
                        3. 数据灵敏度分析：5折交叉验证 + Bootstrap 1000次置信区间 + 留一子组分析
                        4. 场景分析：识别题目中的自由参数，在5+水平上扫描，识别临界阈值
                        5. 特征重要性分析：排列重要性（ML模型）或标准化系数（线性模型）
                        6. 稳定性验证（优化类问题）：10次独立运行，计算CV，评定稳定性等级
                        最后输出鲁棒性分析汇总。
                    """,
            },
        }
        return flows

    def get_write_flows(
        self, user_output: UserOutput, config_template: dict, bg_ques_all: str,
        paper_context: PaperContext | None = None,
    ):
        """生成写作阶段的流程配置。

        Args:
            user_output: 用户输出对象，包含已求解的结果。
            config_template: 论文模板配置。
            bg_ques_all: 问题背景和题目信息。

        Returns:
            写作流程配置字典，键为章节名，值为写作提示。
        """
        sc = self.structure_controller
        model_build_solve = user_output.get_model_build_solve()
        # PaperContext 上下文注入
        ctx_injection = paper_context.inject_into_prompt("firstPage") if paper_context else ""
        flows = {
            "firstPage": f"""【任务】撰写论文的标题、摘要和关键词。
{ctx_injection}

【问题背景】
{bg_ques_all}

【模型求解信息】
{model_build_solve}

{sc.get_section_length_hint("firstPage")}

{sc.get_anti_redundancy_hint("firstPage")}

【写作要求】
1. 标题：简洁明了，15-25字，体现核心方法和研究对象
2. 摘要：
   - 第一段：背景与整体方法（1-2句）
   - 中间段落：每个问题单独成段，必须包含具体方法和数值结果
   - 最后一段：敏感性分析与结论
   - 总字数500-750字
3. 关键词：4-5个，用空格分隔
4. 禁止在摘要中写公式

【模板参考】
{config_template["firstPage"]}
""",
            "RepeatQues": f"""【任务】撰写论文的问题重述章节。
{paper_context.inject_into_prompt("RepeatQues") if paper_context else ""}

【问题背景】
{bg_ques_all}

【模型求解信息】
{model_build_solve}

{sc.get_section_length_hint("RepeatQues")}

{sc.get_anti_redundancy_hint("RepeatQues")}

【写作要求】
1. 问题背景（1.1）：一段话，300-400字，说明研究背景和实际意义，可引用相关文献
2. 问题重述（1.2）：以"本文基于以上信息建立数学模型来解决以下问题"开头，逐条列出
3. 禁止逐字复制原题，需用自己的语言概括
4. 总字数750-1000字

【模板参考】
{config_template["RepeatQues"]}
""",
            "analysisQues": f"""【任务】撰写论文的问题分析章节。
{paper_context.inject_into_prompt("analysisQues") if paper_context else ""}

【问题背景】
{bg_ques_all}

【模型求解信息】
{model_build_solve}

{sc.get_section_length_hint("analysisQues")}

{sc.get_anti_redundancy_hint("analysisQues")}

【写作要求】
1. 每个问题用"## 2.X 问题X的分析"标题
2. 每个问题约500-600字，分三段：
   - 第一段：问题本质（类型、难点、约束，150-200字）
   - 第二段：数据特征与挑战（数据结构、统计假设、特殊情形，150-200字）
   - 第三段：解题思路（模型选择理由、与备选方案对比、预期结果，150-200字）
3. 必须以挑战/难点为导向，不得重复问题重述的内容
4. 必须说明模型选择理由，含与备选方案的对比

【模板参考】
{config_template["analysisQues"]}
""",
            "modelAssumption": f"""【任务】撰写论文的模型假设章节。
{paper_context.inject_into_prompt("modelAssumption") if paper_context else ""}

【问题背景】
{bg_ques_all}

【模型求解信息】
{model_build_solve}

{sc.get_section_length_hint("modelAssumption")}

{sc.get_anti_redundancy_hint("modelAssumption")}

【写作要求】
1. 写3-5条假设，总字数500-750字
2. 每条用编号格式：(1) 假设名称：假设内容。
3. 假设类型：数据有效性、环境稳定性、参数确定性、独立性等
4. 每条假设需说明合理性依据，不得出现'假设模型合理'等无意义假设

【模板参考】
{config_template["modelAssumption"]}
""",
            "symbol": f"""【任务】撰写论文的符号说明章节。
{paper_context.inject_into_prompt("symbol") if paper_context else ""}

【模型求解信息】
{model_build_solve}

{sc.get_section_length_hint("symbol")}

{sc.get_anti_redundancy_hint("symbol")}

【写作要求】
1. 以表格形式列出主要符号，总字数500-750字
2. 包含：符号（$...$ 格式）、含义、单位
3. 分层组织：全局符号在前，各问题专用符号在后
4. 符号应涵盖论文中所有重要变量，不得遗漏关键变量

【模板参考】
{config_template["symbol"]}
""",
            "judge": f"""【任务】撰写论文的模型评价章节。
{paper_context.inject_into_prompt("judge") if paper_context else ""}

【模型求解信息】
{model_build_solve}

{sc.get_section_length_hint("judge")}

{sc.get_anti_redundancy_hint("judge")}

【写作要求】
1. 模型优点（7.1）：3-5个，每个有具体依据（数值或对比支撑）
2. 模型缺点（7.2）：2-3个，每个有具体分析（非泛泛而谈）
3. 改进与推广（7.3）：针对每个缺点提出具体改进方案
4. 优点数量要多于缺点
5. 总字数750-1000字
6. 必须诚实评价，不得只写优点不写缺点

【模板参考】
{config_template["judge"]}
""",
        }
        return flows

    def get_writer_prompt(
        self,
        key: str,
        coder_response: str,
        code_interpreter: BaseCodeInterpreter,
        config_template: dict,
        paper_context: PaperContext | None = None,
    ) -> str:
        """根据不同的key生成对应的writer_prompt

        Args:
            key: 任务类型
            coder_response: 代码执行结果
            code_interpreter: 代码解释器实例

        Returns:
            str: 生成的writer_prompt
        """
        code_output = code_interpreter.get_code_output(key)

        questions_quesx_keys = self.get_questions_quesx_keys()
        bgc = self.questions["background"]
        ctx_injection = paper_context.inject_into_prompt(key) if paper_context else ""

        quesx_writer_prompt = {
            key: f"""【任务】撰写{key.replace("ques", "问题")}的模型建立与求解章节。
{ctx_injection}

【问题背景】
{bgc}

【代码手求解结果】
{coder_response}

【代码执行输出】
{code_output}

{sc.get_section_length_hint(key)}

{sc.get_anti_redundancy_hint(key)}

【写作要求】
1. 问题分析与模型选择：300-500字，说明问题类型、难点和模型选择论证
2. 模型的建立：800-1200字，包含完整数学公式（$$...$$ 格式），不跳步推导，每个公式后用"其中"说明变量
3. 模型的求解：800-1200字，详细描述算法流程、参数设定（必须有来源）、引用具体数值结果
4. 结果分析与机制解释：800-1000字，深入解读数值含义，解释背后的物理/数学机理
5. 图表分析：500-800字，每张图片后至少3行独立分析
6. 总字数约3750-4500字
7. 必须插入代码手生成的图片（![描述](文件名.png) 格式）
8. 禁止复述表格数据，必须进行解读和因果分析

【模板参考】
{config_template[key]}
"""
            for key in questions_quesx_keys
        }

        writer_prompt = {
            "eda": f"""【任务】撰写数据预处理与探索性分析章节。
{paper_context.inject_into_prompt("eda") if paper_context else ""}

【问题背景】
{bgc}

【代码手求解结果】
{coder_response}

【代码执行输出】
{code_output}

{sc.get_section_length_hint("eda")}

{sc.get_anti_redundancy_hint("eda")}

【写作要求】
1. 数据基本情况（200-300字）：来源、时间范围、变量个数、样本量、数据背景
2. 数据质量评估（300-500字）：缺失值检测与处理、异常值识别、数据类型检查
3. 描述性统计（300-500字）：关键变量的均值、中位数、标准差、最大最小值、分布特征
4. 单变量可视化分析（400-600字）：每个关键变量的分布图、箱线图解读
5. 多变量分析（400-600字）：相关性热力图、散点图、分组对比
6. 数据预处理决策及理由（200-300字）
7. 总字数约2000-2500字，插入5-8张图表
8. 使用"由表X可知""如图X所示"等句式引用图表
9. 必须插入代码手生成的图片，每张图后至少3行分析

【模板参考】
{config_template["eda"]}
""",
            **quesx_writer_prompt,
            "sensitivity_analysis": f"""【任务】撰写鲁棒性与灵敏度分析章节（六维度框架）。
{paper_context.inject_into_prompt("sensitivity_analysis") if paper_context else ""}

【问题背景】
{bgc}

【代码手求解结果】
{coder_response}

【代码执行输出】
{code_output}

{sc.get_section_length_hint("sensitivity_analysis")}

{sc.get_anti_redundancy_hint("sensitivity_analysis")}

【写作要求】
1. 参数灵敏度分析（200-300字）：使用模型拟合的置信区间（非任意±10%），报告灵敏度指数排名，分析交互效应，引用龙卷风图和热力图
2. 结构灵敏度分析（150-250字）：列出对比的模型规格（至少2种），用表格展示交叉验证指标（均值±标准差），说明最佳模型及优势
3. 数据灵敏度分析（200-300字）：报告5折CV结果、Bootstrap 95%置信区间、留一子组分析，引用Bootstrap分布图
4. 场景分析（200-300字）：识别关键自由参数，在5+水平上扫描，重点分析临界阈值及其实际意义
5. 特征重要性分析（100-150字）：列出特征重要性排序，识别关键特征
6. 稳定性验证（100-150字，优化类问题）：报告多次运行的CV和稳定性评级
7. 鲁棒性综合评估（100-150字）：综合六个维度给出整体评价
8. 总字数约1500-2000字
9. 必须插入代码手生成的所有鲁棒性分析图表
10. 每个结论必须有具体数据支撑（灵敏度指数、CV值、置信区间等），禁止空话

【模板参考】
{config_template["sensitivity_analysis"]}
""",
        }

        if key in writer_prompt:
            return writer_prompt[key]
        else:
            raise ValueError(f"未知的任务类型: {key}")

    def get_questions_quesx_keys(self) -> list[str]:
        """获取问题1,2...的键"""
        return list(self.get_questions_quesx().keys())

    def get_questions_quesx(self) -> dict[str, str | int]:
        """获取问题1,2,3...的键值对"""
        # 获取所有以 "ques" 开头的键值对
        questions_quesx = {
            key: value
            for key, value in self.questions.items()
            if key.startswith("ques") and key != "ques_count"
        }
        return questions_quesx

    def get_seq(self, ques_count: int) -> dict[str, str]:
        """获取论文章节顺序。

        Args:
            ques_count: 问题数量。

        Returns:
            以章节名为键的有序字典。
        """
        ques_str = [f"ques{i}" for i in range(1, ques_count + 1)]
        seq = [
            "firstPage",
            "RepeatQues",
            "analysisQues",
            "modelAssumption",
            "symbol",
            "eda",
            *ques_str,
            "sensitivity_analysis",
            "judge",
        ]
        return {key: "" for key in seq}
