# MathModelAgent 优化实施路线图

> 基于 3 位专家（Professor 1/2/3）的 16 个 subagent 深度分析产出，汇总生成。
> 生成时间：2026-06-08

---

## 项目现状总结

当前系统已有 11 个 Agent（ProblemAnalyst, Literature, Coordinator, Modeler, Coder, ResultInterpreter, Writer, Critic, Outline, Consistency, Review/MultiReviewer），workflow.py 约 1100 行，包含 Phase 0-11 共 12 个阶段。

**最近一次完整运行（20260607-133515）的关键数据：**
- LLM 调用：18 次（严重不足，ModelerAgent 只调用 1 次处理全部 4 个子问题）
- 代码执行：140 次（成功 132，失败 8，成功率 94.3%）
- 总运行时间：133.6 分钟
- 最终评审得分：86 分（超过阈值 80）
- CriticAgent 决策：approve 1 / revise 3 / reject 2
- 评分量纲异常：初始 250-317 分，修正后 83-96 分

**核心差距诊断（来自 Professor 1/2 的共识）：**
> 获奖论文每一步都是**数据驱动的**：看到数据里 t² 显著就加 t²，看到交互项不显著就删。AI 论文是**先验驱动的**：根据"应该有交互项"就加交互项，从不问"数据说这个假设成立吗"。这不是算法层面的差距，是**建模哲学**的差距。

---

## Phase 1：紧急修复（3-5 天）

**目标**：消除当前系统中影响论文质量的硬伤，不改变整体架构。

### 任务 1.1：HTML 注释泄漏修复 🔴P0

- **涉及文件**：`writer.py`, `writer_agent.py`, `user_output.py`
- **预计工作量**：0.5 天
- **依赖关系**：无
- **具体做法**：
  1. 在 `writer.py` 的 `CHAPTER_PROMISES_RULES` 中将 HTML 注释格式改为 `~~~TRACKING_START` / `~~~TRACKING_END`
  2. 在 `writer_agent.py` 的 `run()` 中对 `response_content` 执行 `re.sub(r'<!--.*?-->', '', text)` 剥离
  3. 在 `user_output.py` 的 `save_result()` 中增加断言：若检测到 HTML 注释则强制剥离并 log warning

### 任务 1.2：图表路径验证（FigurePathResolver） 🔴P0

- **涉及文件**：`user_output.py`, `writer_agent.py`
- **预计工作量**：0.5 天
- **依赖关系**：无
- **具体做法**：
  1. 新增 `FigurePathResolver`，维护短文件名→完整路径映射
  2. 在 writer_agent.py 中注册 `coder_response.created_images`
  3. 论文拼接时自动修正图片引用路径，缺失图片标记 `[图片缺失]`

### 任务 1.3：WriterAgent 日志增强

- **涉及文件**：`writer_agent.py`
- **预计工作量**：0.25 天
- **依赖关系**：无

### 任务 1.4：公式自动编号

- **涉及文件**：`writer.py`, `writer_agent.py`
- **预计工作量**：0.25 天
- **依赖关系**：无
- **具体做法**：Prompt 要求 `\tag{N}` 编号 + 代码校验连续性

### 任务 1.5：PaperSanitizer 模块

- **涉及文件**：`writer_agent.py`
- **预计工作量**：0.5 天
- **依赖关系**：任务 1.1
- **具体做法**：移除 HTML 注释、`[thinking]` 标签、LLM 元叙述前缀

### 任务 1.6：LiteratureAgent 输出结构化

- **涉及文件**：`literature_agent.py`, `literature.py`
- **预计工作量**：0.5 天
- **依赖关系**：无
- **具体做法**：输出 Mainstream/Innovation/Why 三段结构，生成 `LiteratureResult` dataclass

### 任务 1.7：评分系统统一量纲

- **涉及文件**：`evaluation.py`, `review_agent.py`
- **预计工作量**：0.5 天
- **依赖关系**：无
- **具体做法**：
  1. QualityScore 改为三维度：method(0-40) + writing(0-30) + format(0-30) = 100
  2. 新增 `MandatoryMinimums`：method < 24 时标记必须重写
  3. 新增 `clamp_score()` 防止越界

### Phase 1 验收标准

- [ ] 论文中无 HTML 注释残留
- [ ] 所有图片引用可解析到实际文件
- [ ] 公式编号连续且无重复
- [ ] LiteratureAgent 输出包含 Mainstream/Innovation/Why
- [ ] 评分系统输出三维度分数，无越界

---

## Phase 2：核心升级（7-10 天）

**目标**：引入关键新 Agent 和数据结构，提升建模质量。

### 任务 2.1：ModelSpec 扩展 model_search_protocol

- **涉及文件**：`A2A.py`, `modeler_agent.py`
- **预计工作量**：0.5 天
- **依赖关系**：无
- **具体做法**：新增 `ModelSearchProtocol` 数据结构（search_strategy, candidate_models, selection_criteria）

### 任务 2.2：ModelSearchAgent（变量筛选） 🔴P0

- **涉及文件**：新建 `model_search_agent.py`, `model_search.py`, 修改 `workflow.py`
- **预计工作量**：1.5 天
- **依赖关系**：任务 2.1
- **具体做法**：
  1. 新建 ModelSearchAgent，执行系统性变量筛选
  2. 输出 AIC/BIC 对比表，推荐最优变量组合
  3. 领域知识规则：ICC > 0.4, R² 合理性, AIC 差异 > 2
  4. Phase 4 和 Phase 5 之间插入 Phase 4.5

### 任务 2.3：ProblemTypeAgent（问题类型诊断） 🟡P1

- **涉及文件**：新建 `problem_type_agent.py`, `problem_type.py`
- **预计工作量**：1 天
- **依赖关系**：无
- **具体做法**：
  1. 诊断决策树：结局变量 → 时间结构 → 删失 → 稀疏性
  2. 识别生存分析、纵向回归、组合优化等特殊类型
  3. SparsityAnalyzer（纯 Python，不消耗 LLM）

### 任务 2.4：GPR 和生存分析代码模板

- **涉及文件**：`coder_agent.py`, `coder.py`
- **预计工作量**：1 天
- **依赖关系**：任务 2.3
- **具体做法**：
  1. GPR 模板（sklearn GaussianProcessRegressor）
  2. 生存分析模板（Cox PH / Kaplan-Meier / lifelines）
  3. 三阶段决策流程：GPR 纵向建模 → 特征工程 → 生存分析

### 任务 2.5：联合优化与 GA 支持

- **涉及文件**：`modeler_agent.py`, `coder.py`, `modeler.py`
- **预计工作量**：1.5 天
- **依赖关系**：无
- **具体做法**：
  1. OptimizationDetector：检测耦合决策变量
  2. GA 代码模板：scipy.optimize.differential_evolution
  3. 多场景分析：50%/75%/90%/99% 准确率
  4. 风险函数 + 蒙特卡洛误差分析

### 任务 2.6：ProblemReformulationAgent 🟡P1

- **涉及文件**：新建 `problem_reformulation_agent.py`, `problem_reformulation.py`
- **预计工作量**：1.5 天
- **依赖关系**：任务 2.3
- **具体做法**：
  1. 10 种标准问题类型知识库（纵向回归/生存分析/组合优化/多分类不平衡等）
  2. 约束 ModelerAgent 只能从推荐模型家族中选择
  3. 创新包装机制（预测-优化联合框架、多专家协同诊断等）

### 任务 2.7：CriticAgent 拆分（ReviewerAgent + AwardJudgeAgent）

- **涉及文件**：`critic_agent.py`, `evaluation.py`, `workflow.py`
- **预计工作量**：1 天
- **依赖关系**：任务 1.7
- **具体做法**：
  1. ReviewerAgent（正确性守卫）：数学正确性 40 + 逻辑自洽性 35 + 数据一致性 25 = 100
  2. AwardJudgeAgent（国奖评审官）：方法 30 + 创新性 30 + 解释性 20 + 竞赛风格 20 = 100
  3. Innovation < 20 强制打回机制

### 任务 2.8：DependencyAgent（子问题依赖图）

- **涉及文件**：新建 `dependency_agent.py`, 修改 `workflow.py`
- **预计工作量**：1 天
- **依赖关系**：无
- **具体做法**：
  1. QuestionDependencyGraph（DAG 拓扑排序）
  2. 依赖边：source → target + what_to_use + how_to_use
  3. build_dependency_context()：注入前序问题结论
  4. Phase 5 按拓扑序执行

### Phase 2 验收标准

- [ ] ModelSearchAgent 输出 AIC/BIC 对比表
- [ ] ProblemTypeAgent 能识别生存分析等特殊类型
- [ ] 联合优化问题被正确识别并使用 GA 求解
- [ ] ProblemReformulationAgent 输出标准问题类型
- [ ] ReviewerAgent + AwardJudgeAgent 串行工作正常
- [ ] DependencyAgent 输出 DAG，子任务按拓扑序执行

---

## Phase 3：竞赛增强（10-15 天）

**目标**：引入评审强化、数据探查、不平衡处理等高级特性。

### 任务 3.1：Reflexion 改进追踪
- ImprovementRecord + RegressionAlert + verify_substantive_change()

### 任务 3.2：DataProfiler（纯 Python 数据探查）
- 缺失值统计、异常值检测、相关性矩阵摘要

### 任务 3.3：ImbalanceDetector
- 类别比例检测，SMOTE + 集成方法 + 类别权重

### 任务 3.4：本地引用数据库
- 25-30 种常见方法的标准引用（BibTeX 格式）

### 任务 3.5：章节级系统提示词
- 根据章节类型（摘要/模型/结果）动态切换 prompt

### 任务 3.6：AcademicFormatEnforcer
- 公式编号、图表编号、章节编号自动校验

### 任务 3.7：ReferenceManager
- 统一管理文献引用，防止重复和遗漏

### 任务 3.8：领域 Persona 提示词
- 根据问题类型注入对应领域专家视角

### 任务 3.9：Context Isolation（SubtaskContext）
- 隔离每个子任务的上下文，防止污染

### 任务 3.10：Checklist 评分系统
- 10 维布尔值检查，与现有评分并行

### Phase 3 验收标准

- [ ] Reflexion 能检测回归并自动回退
- [ ] DataProfiler 不消耗 LLM 调用
- [ ] 类别不平衡被自动检测
- [ ] 本地引用数据库覆盖 25+ 种方法
- [ ] 最终论文 Checklist 通过率 >= 80%

---

## 总体时间预估

| Phase | 工作量 | 日历天数 | LLM 调用变化 |
|-------|--------|----------|-------------|
| Phase 1：紧急修复 | 3 天 | 3-5 天 | 52 次（不变） |
| Phase 2：核心升级 | 9.5 天 | 7-10 天 | 52 → 55 次 |
| Phase 3：竞赛增强 | 8.5 天 | 10-15 天 | 55 → 68 次 |
| **总计** | **21 天** | **20-30 天** | **+16 次** |

---

## 风险评估

| 风险等级 | 任务 | 风险描述 | 缓解措施 |
|---------|------|---------|---------|
| 🔴高 | CriticAgent 拆分 | 两个 Agent 评分标准可能矛盾 | 用历史论文数据校准 |
| 🔴高 | ModelSearchAgent | 领域知识规则需大量调优 | 先实现基础版本迭代 |
| 🔴高 | DependencyAgent | DAG 正确性直接影响执行顺序 | 增加无环检测和异常处理 |
| 🟡中 | ProblemReformulationAgent | 知识库覆盖面可能不足 | 持续扩展 |
| 🟡中 | 联合优化 GA | 参数调优需要经验 | 提供默认参数模板 |
| 🟢低 | HTML 注释修复 | 纯正则替换 | 技术风险极低 |
| 🟢低 | DataProfiler | 纯 Python 实现 | 可控性强 |

---

## 关键文件清单

按修改频率排序：

1. `backend/app/core/workflow.py` — 几乎每个任务都涉及
2. `backend/app/core/agents/writer_agent.py` — Phase 1 多个任务
3. `backend/app/core/evaluation.py` — 评分系统重构
4. `backend/app/core/agents/modeler_agent.py` — 变量筛选 + 问题重构
5. `backend/app/core/agents/critic_agent.py` — 拆分
6. `backend/app/core/prompts/writer.py` — 格式规范
7. `backend/app/core/prompts/modeler.py` — 选型约束
8. `backend/app/core/prompts/coder.py` — 代码模板注入
9. `backend/app/schemas/A2A.py` — 数据结构扩展
10. `backend/app/core/global_state.py` — 状态管理扩展

---

## 一句话总结

**Phase 1 修硬伤，Phase 2 加 Agent，Phase 3 冲国奖。** 核心理念：从"先验驱动"转向"数据驱动"——每一个建模决策都必须被数据验证，不能被先验直觉代替。
