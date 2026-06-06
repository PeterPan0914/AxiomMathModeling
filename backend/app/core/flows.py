"""工作流程定义模块，管理建模任务的求解和写作流程。"""

from app.models.user_output import UserOutput
from app.tools.base_interpreter import BaseCodeInterpreter
from app.core.agents.modeler_agent import ModelerToCoder


class Flows:
    """管理数学建模任务的求解流程和写作流程。"""
    def __init__(self, questions: dict[str, str | int]):
        self.flows: dict[str, dict] = {}
        self.questions: dict[str, str | int] = questions

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
        ques_flow = {
            key: {
                "coder_prompt": f"""
                        参考建模手给出的解决方案{solutions.get(key, "")}
                        完成如下问题{value}
                    """,
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
                        参考建模手给出的解决方案{solutions.get("sensitivity_analysis", "对模型进行灵敏度分析")}
                        完成敏感性分析
                    """,
            },
        }
        return flows

    def get_write_flows(
        self, user_output: UserOutput, config_template: dict, bg_ques_all: str
    ):
        """生成写作阶段的流程配置。

        Args:
            user_output: 用户输出对象，包含已求解的结果。
            config_template: 论文模板配置。
            bg_ques_all: 问题背景和题目信息。

        Returns:
            写作流程配置字典，键为章节名，值为写作提示。
        """
        model_build_solve = user_output.get_model_build_solve()
        flows = {
            "firstPage": f"""【任务】撰写论文的标题、摘要和关键词。

【问题背景】
{bg_ques_all}

【模型求解信息】
{model_build_solve}

【写作要求】
1. 标题：简洁明了，15-25字，体现核心方法和研究对象
2. 摘要：
   - 第一段：背景与整体方法（1-2句）
   - 中间段落：每个问题单独成段，必须包含具体方法和数值结果
   - 最后一段：敏感性分析与结论
   - 总字数400-600字
3. 关键词：4-5个，用空格分隔

【模板参考】
{config_template["firstPage"]}
""",
            "RepeatQues": f"""【任务】撰写论文的问题重述章节。

【问题背景】
{bg_ques_all}

【模型求解信息】
{model_build_solve}

【写作要求】
1. 问题背景（1.1）：一段话，150-200字，说明研究背景和实际意义
2. 问题重述（1.2）：以"本文基于以上信息建立数学模型来解决以下问题"开头，逐条列出

【模板参考】
{config_template["RepeatQues"]}
""",
            "analysisQues": f"""【任务】撰写论文的问题分析章节。

【问题背景】
{bg_ques_all}

【模型求解信息】
{model_build_solve}

【写作要求】
1. 每个问题用"## 2.X 问题X的分析"标题
2. 每个问题约200-300字，分两段：
   - 第一段：问题本质（类型、难点、约束）
   - 第二段：解题思路（模型选择、核心思想、预期结果）
3. 使用"针对问题X"开头
4. 必须说明模型选择理由

【模板参考】
{config_template["analysisQues"]}
""",
            "modelAssumption": f"""【任务】撰写论文的模型假设章节。

【问题背景】
{bg_ques_all}

【模型求解信息】
{model_build_solve}

【写作要求】
1. 写3-5条假设
2. 每条用编号格式：(1) 假设名称：假设内容。
3. 假设类型：数据有效性、环境稳定性、参数确定性、独立性等
4. 简明扼要，每条1-2句话

【模板参考】
{config_template["modelAssumption"]}
""",
            "symbol": f"""【任务】撰写论文的符号说明章节。

【模型求解信息】
{model_build_solve}

【写作要求】
1. 以表格形式列出主要符号
2. 包含：符号（$...$ 格式）、含义、单位
3. 按照出现顺序或逻辑分组排列
4. 符号应涵盖论文中所有重要变量

【模板参考】
{config_template["symbol"]}
""",
            "judge": f"""【任务】撰写论文的模型评价章节。

【模型求解信息】
{model_build_solve}

【写作要求】
1. 模型优点（7.1）：3-5个，每个有具体依据
2. 模型缺点（7.2）：2-3个，每个有具体分析
3. 改进与推广（7.3）：100-150字，针对缺点提出改进方案
4. 优点数量要多于缺点
5. 总字数300-400字

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

        quesx_writer_prompt = {
            key: f"""【任务】撰写{key.replace("ques", "问题")}的模型建立与求解章节。

【问题背景】
{bgc}

【代码手求解结果】
{coder_response}

【代码执行输出】
{code_output}

【写作要求】
1. 问题分析与模型选择：150-200字，说明问题类型和模型选择理由
2. 模型的建立：300-400字，包含完整数学公式（$$...$$ 格式），每个公式后用"其中"说明变量
3. 模型的求解：200-300字，引用具体数值结果
4. 结果分析：150-200字，深入解读结果
5. 总字数约800-1000字
6. 必须插入代码手生成的图片（![描述](文件名.png) 格式）
7. 每张图片后至少3行分析解读

【模板参考】
{config_template[key]}
"""
            for key in questions_quesx_keys
        }

        writer_prompt = {
            "eda": f"""【任务】撰写数据预处理与探索性分析章节。

【问题背景】
{bgc}

【代码手求解结果】
{coder_response}

【代码执行输出】
{code_output}

【写作要求】
1. 数据基本情况：来源、时间范围、变量个数、样本量
2. 数据质量：缺失值处理、异常值检测
3. 描述性统计：关键变量的均值、标准差、最大最小值
4. 可视化分析：分布特征、趋势、相关性
5. 总字数约200-300字
6. 使用"由表X可知""如图X所示"等句式引用图表
7. 必须插入代码手生成的图片

【模板参考】
{config_template["eda"]}
""",
            **quesx_writer_prompt,
            "sensitivity_analysis": f"""【任务】撰写灵敏度分析章节。

【问题背景】
{bgc}

【代码手求解结果】
{coder_response}

【代码执行输出】
{code_output}

【写作要求】
1. 参数选择与范围：选择2-3个关键参数，说明变化范围（±5%、±10%）
2. 分析方法与结果：用表格展示参数变化对结果的影响
3. 稳健性评估：分析模型对参数变化的敏感程度
4. 总字数约300-500字
5. 必须插入代码手生成的图表

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
