<h1 align="center">🤖 MathModelAgent (Enhanced) 📐</h1>

<p align="center">
    <img src="./docs/icon.png" height="250px">
</p>

<h4 align="center">
    基于 <a href="https://github.com/jihe520/MathModelAgent">jihe520/MathModelAgent</a> 二次开发<br>
    多 Agent 协作的数学建模自动化系统
</h4>

<p align="center">
    <a href="https://github.com/jihe520/MathModelAgent">原项目</a> ·
    <a href="OPTIMIZATION_ROADMAP.md">优化路线图</a> ·
    <a href="CLAUDE.md">开发文档</a>
</p>

---

## 📌 项目说明

本项目是 [MathModelAgent](https://github.com/jihe520/MathModelAgent) 的**二次开发版本**，由 [jihe520](https://github.com/jihe520) 原创开发的数学建模自动化系统。原项目通过多 Agent 协作（建模手、代码手、论文手等）自动完成数学建模竞赛的全流程。

**本二次开发版本**在原项目基础上，针对建模质量和论文学术性进行了系统性优化，主要改动包括：

- 🔬 **问题类型自动识别**：新增 ProblemTypeAgent，通过诊断决策树识别生存分析、纵向回归、组合优化等特殊问题类型
- 🧠 **问题重述机制**：新增 ProblemReformulationAgent，将竞赛题目的领域语言翻译为标准数学问题类型，约束模型选型
- 🔗 **子问题依赖图**：新增 DependencyAgent，构建 DAG 依赖图，确保后续子问题能看到前序问题的结论
- 📊 **系统性变量筛选**：新增 ModelSearchAgent，强制执行 AIC/BIC 比较和逐步筛选，而非直接指定最终模型
- 🏆 **双层评审体系**：ReviewerAgent（正确性守卫）+ AwardJudgeAgent（国奖评审官），Innovation<20 强制打回
- 📝 **论文格式修复**：HTML 注释泄漏修复、图表路径自动校验、公式自动编号、PaperSanitizer 内容净化
- 📚 **文献系统增强**：Mainstream/Innovation/Why 三段输出结构、25+ 种常见方法的本地引用数据库
- 📐 **数据探查**：纯 Python DataProfiler（不消耗 LLM 调用），自动检测数据稀疏性、类别不平衡、重复测量结构

详细改动说明请参阅 [OPTIMIZATION_ROADMAP.md](./OPTIMIZATION_ROADMAP.md)。

---

## 🙏 致谢与版权

**原项目作者**：[jihe520](https://github.com/jihe520)

本项目的全部基础架构、核心 Agent 设计、前端界面、Docker 部署方案等均来自原项目。二次开发仅在 Agent 层和提示词层进行了增强，未修改原项目的核心架构设计。

原项目版权和许可协议详见：[License](./docs/md/License.md)

**个人免费使用，请勿商业用途，商业用途请联系原项目作者。**

---

## 🚀 快速开始

与原项目部署方式完全一致：

### Docker 部署（推荐）

```bash
# 克隆本仓库
git clone <本仓库地址>
cd MathModelAgent

# 启动服务
docker-compose up -d

# 访问
# 前端：http://localhost:5173
# 后端：http://localhost:8000
```

### 本地部署

```bash
# 后端
cd backend
pip install uv
uv sync
ENV=DEV uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 前端
cd frontend
pnpm i
pnpm run dev
```

---

## 📐 二次开发改动概览

### 新增文件

| 文件 | 说明 |
|------|------|
| `backend/app/core/agents/dependency_agent.py` | 子问题依赖图 Agent |
| `backend/app/core/agents/problem_type_agent.py` | 问题类型识别 Agent |
| `backend/app/core/agents/problem_reformulation_agent.py` | 问题重述 Agent |
| `backend/app/core/agents/model_search_agent.py` | 变量筛选 Agent |
| `backend/app/core/agents/reviewer_agent.py` | 正确性审查 Agent（替代 CriticAgent） |
| `backend/app/core/agents/award_judge_agent.py` | 国奖评审 Agent |
| `backend/app/core/dependency_graph.py` | DAG 依赖图数据结构 |
| `backend/app/core/domain_rules.py` | 领域知识规则库 |
| `backend/app/core/data_profiler.py` | 数据探查模块（纯 Python） |
| `backend/app/core/citation_db.py` | 本地引用数据库（25+ 方法） |

### 工作流改动

```
原工作流：ProblemAnalyst → Literature → Coordinator → Modeler → Coder/Writer 循环 → 评审

新增 Phase：
  Phase 0.5  DataProfiler（纯 Python 数据探查）
  Phase 3.1  ProblemTypeAgent（问题类型识别）
  Phase 3.2  DependencyAgent（子问题依赖图）
  Phase 3.3  ProblemReformulationAgent（问题重述）
  Phase 4.5  ModelSearchAgent（变量筛选）
  Phase 5d-1 ReviewerAgent（正确性审查）
  Phase 5d-2 AwardJudgeAgent（国奖潜力评估）
```

---

## 📖 原项目功能特性

- 🔍 自动分析问题，数学建模，编写代码，纠正错误，撰写论文
- 💻 Code Interpreter（本地 Jupyter / E2B 云端沙箱）
- 📝 生成编排好格式的论文
- 🤝 Multi-agents：建模手、代码手、论文手等
- 🔄 Multi-LLMs：每个 agent 设置不同的模型
- 🤖 支持所有模型：[litellm](https://docs.litellm.ai/docs/providers)

更多功能请参阅 [原项目 README](https://github.com/jihe520/MathModelAgent)。

---

## 📄 License

沿用原项目许可协议：个人免费使用，请勿商业用途。

[License](./docs/md/License.md)
