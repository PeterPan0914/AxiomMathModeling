# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MathModelAgent 是数学建模竞赛自动化系统，通过多 Agent 协作完成建模、代码生成和论文撰写。

**核心工作流**：用户上传题目 → CoordinatorAgent 分析拆解 → ModelerAgent 建模 → CoderAgent 编码执行 → WriterAgent 撰写论文 → ReviewAgent 质量评审

**Reflexion 循环**：WriterAgent 和 ReviewAgent 构成迭代优化环——Writer 生成论文 → Reviewer 评审打分（百分制）→ 若低于阈值则生成反馈 → Writer 根据反馈改进 → 最多重复 N 轮。评分通过 `QualityTracker` 追踪每个章节的历次得分。

**诊断日志系统**：所有 LLM 交互（prompt、response、reasoning、tool_calls、token usage）和代码执行结果都通过 `DiagnosticLogger` 写入 JSONL 文件，用于质量分析和调试。

**学术知识检索**：通过 OpenAlex API 检索相关学术论文，为 WriterAgent 提供参考文献支持。

**LLM 提供商抽象**：通过 Strategy 模式支持 OpenAI Chat、OpenAI Responses、Anthropic 三种后端，由 `LLMFactory` 统一创建。

## Commands

### 后端

```bash
cd backend

# 安装依赖（uv 管理）
uv sync

# 启动开发服务器（需要先启动 Redis）
ENV=DEV uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120 --reload

# Lint（使用虚拟环境中的 ruff）
.\.venv\Scripts\python.exe -m ruff check app/
.\.venv\Scripts\python.exe -m ruff format app/

# 类型检查
npx pyright app/
```

### 前端

```bash
cd frontend

# 安装依赖
pnpm i

# 启动开发服务器
pnpm run dev

# 构建
pnpm run build

# Lint
npx biome check src/
npx biome check --write src/  # 自动修复
```

### Docker 部署

```bash
# 复制并填写环境变量
cp backend/.env.example backend/.env.dev

# 构建并启动所有服务（Redis + 后端 + 前端）
docker-compose up -d --build

# 查看日志
docker-compose logs -f backend

# 停止
docker-compose down
```

### 测试

```bash
cd backend
# 运行测试（如有 pytest）
.\.venv\Scripts\python.exe -m pytest app/tests/ -v

# 单独运行某测试
.\.venv\Scripts\python.exe -m pytest app/tests/test_common_utils.py -v
```

## 项目结构

```
MathModelAgent/
├── CLAUDE.md                   # 本文件
├── IMPROVEMENTS.md             # 改进记录
├── docker-compose.yml          # Docker 编排（Redis + backend + frontend）
├── 优秀论文示例/                # 竞赛优秀论文，作为 prompt 优化参考
│
├── backend/
│   ├── pyproject.toml          # Python 依赖（uv 管理）
│   ├── Dockerfile
│   ├── .env.example            # 环境变量模板
│   ├── .env.dev                # 本地开发环境变量（不提交）
│   ├── fonts/                  # 字体文件（simhei.ttf，论文生成用）
│   ├── project/work_dir/       # 任务运行时工作目录
│   │
│   └── app/
│     ├── main.py               # FastAPI 入口，CORS、路由注册、生命周期
│     │
│     ├── config/
│     │   ├── setting.py         # Pydantic Settings，所有配置项
│     │   ├── model_config.toml  # LLM 模型配置
│     │   ├── md_template.toml   # 论文章节模板（WriterAgent 使用）
│     │   └── template.md        # 论文整体模板
│     │
│     ├── core/
│     │   ├── workflow.py         # 工作流主入口，编排所有 Agent
│     │   ├── flows.py           # 任务拆解逻辑、write_flows 生成
│     │   ├── functions.py       # 公共函数
│     │   ├── evaluation.py      # QualityScore/Report/Tracker，Reflexion 评分
│     │   ├── knowledge_retrieval.py  # OpenAlex 学术论文检索
│     │   │
│     │   ├── agents/
│     │   │   ├── agent.py            # Agent 基类：对话历史、轮次控制、记忆压缩
│     │   │   ├── coordinator_agent.py # 任务分解，生成子任务列表
│     │   │   ├── modeler_agent.py     # 数学建模，生成 LaTeX 模型
│     │   │   ├── coder_agent.py       # 代码生成与 Jupyter 执行
│     │   │   ├── writer_agent.py      # 论文撰写，含搜索文献能力
│     │   │   └── review_agent.py      # 论文评审，结构化评分 + 反馈
│     │   │
│     │   ├── llm/
│     │   │   ├── llm.py           # LLM 客户端封装
│     │   │   ├── llm_factory.py   # 工厂模式创建 LLM 提供商
│     │   │   ├── types.py         # LLM 相关类型定义
│     │   │   └── providers/
│     │   │       ├── base.py           # 抽象基类
│     │   │       ├── openai_chat.py    # OpenAI Chat Completions
│     │   │       ├── openai_responses.py # OpenAI Responses API
│     │   │       └── anthropic.py      # Anthropic Claude
│     │   │
│     │   └── prompts/
│     │       ├── coordinator.py       # CoordinatorAgent 系统提示词
│     │       ├── modeler.py           # ModelerAgent 系统提示词
│     │       ├── coder.py             # CoderAgent 系统提示词
│     │       ├── writer.py            # WriterAgent 系统提示词（含学术写作规范）
│     │       ├── reviewer.py          # ReviewAgent 系统提示词（含评审标准）
│     │       ├── prompt_engineering.py # 提示词工程工具
│     │       └── shared.py            # 共享提示词片段
│     │
│     ├── routers/
│     │   ├── common_router.py    # 通用 API（健康检查等）
│     │   ├── files_router.py     # 文件上传/管理
│     │   ├── modeling_router.py  # 建模任务提交/查询
│     │   └── ws_router.py       # WebSocket 实时通信
│     │
│     ├── schemas/
│     │   ├── enums.py            # 枚举（任务状态、竞赛类型等）
│     │   ├── request.py          # 请求模型
│     │   ├── response.py         # 响应模型
│     │   ├── base.py            # 基础模型
│     │   ├── A2A.py             # Agent-to-Agent 通信协议
│     │   └── tool_result.py     # 工具调用结果
│     │
│     ├── services/
│     │   ├── redis_manager.py    # Redis 连接管理、发布/订阅
│     │   └── ws_manager.py      # WebSocket 连接管理
│     │
│     ├── tools/
│     │   ├── base.py            # 工具基类
│     │   ├── base_interpreter.py # 代码解释器抽象基类
│     │   ├── local_interpreter.py # 本地 Jupyter 内核执行
│     │   ├── e2b_interpreter.py  # E2B 云端沙箱执行
│     │   ├── interpreter_factory.py # 解释器工厂
│     │   ├── notebook_serializer.py # Notebook 序列化
│     │   └── openalex_scholar.py # OpenAlex API 封装
│     │
│     ├── utils/
│     │   ├── diagnostic_logger.py # 诊断日志（JSONL 格式记录所有 LLM 交互）
│     │   ├── data_recorder.py    # 数据记录器
│     │   ├── common_utils.py    # 通用工具函数
│     │   ├── log_util.py        # 日志配置
│     │   ├── cli.py            # CLI 工具
│     │   ├── RichPrinter.py    # 终端富文本输出
│     │   └── track.py          # 追踪工具
│     │
│     ├── models/
│     │   └── user_output.py     # 用户输出数据模型
│     │
│     ├── example/               # 示例竞赛题目和数据
│     │   ├── 2023华数杯C题/
│     │   ├── 2024高教杯C题/
│     │   └── 2025五一杯C题/
│     │
│     └── tests/                 # 测试文件
│
├── frontend/
│   ├── package.json            # 前端依赖（pnpm 管理）
│   ├── biome.json              # Biome lint/格式化配置
│   ├── Dockerfile
│   │
│   └── src/
│     ├── main.ts               # Vue 入口
│     ├── App.vue               # 根组件
│     │
│     ├── apis/                 # 后端 API 调用封装
│     │   ├── submitModelingApi.ts  # 建模任务提交
│     │   ├── filesApi.ts       # 文件管理
│     │   ├── apiKeyApi.ts      # API Key 管理
│     │   └── commonApi.ts      # 通用 API
│     │
│     ├── components/
│     │   ├── AppSidebar.vue     # 侧边栏导航
│     │   ├── ChatArea.vue      # 聊天区域
│     │   ├── Bubble.vue        # 消息气泡
│     │   ├── NotebookArea.vue  # Jupyter Notebook 展示
│     │   ├── NotebookCell.vue  # Notebook 单元格
│     │   ├── Files.vue         # 文件管理
│     │   ├── Tree.vue          # 树形结构
│     │   ├── AgentEditor/      # Agent 编辑器（Modeler/Coder/Writer）
│     │   └── ui/               # shadcn-vue UI 库（不要修改！）
│     │
│     ├── pages/
│     │   ├── chat/             # 主聊天页面
│     │   ├── task/             # 任务管理页面
│     │   ├── example/          # 示例展示页面
│     │   ├── login/            # 登录页面
│     │   └── test/             # 测试页面
│     │
│     ├── stores/               # Pinia 状态管理
│     │   ├── task.ts           # 任务状态
│     │   └── apiKeys.ts        # API Keys 状态
│     │
│     ├── router/index.ts       # Vue Router 路由配置
│     └── utils/                # 工具函数
│       ├── websocket.ts        # WebSocket 客户端
│       ├── markdown.ts         # Markdown 渲染
│       ├── request.ts          # HTTP 请求封装
│       ├── interface.ts        # TypeScript 接口
│       ├── enum.ts             # 前端枚举
│       ├── const.ts            # 常量
│       └── response.ts         # 响应处理
│
└── skills/                     # Claude Code 技能（交互式辅助）
  ├── 1start-mathmodel/        # 启动建模
  ├── 2analysis-modeling/       # 分析建模
  ├── 3coding-visual/          # 编码可视化
  ├── 4drawio/                 # 流程图
  ├── 5writing/                # 论文写作（含 Typst 模板）
  ├── 6verity/                 # 论文检查
  ├── doctor/                  # 诊断工具
  └── typst-author/            # Typst 文档生成
```

## 关键数据流

### 完整任务执行流程

```
用户提交题目 + 附件
  → POST /api/modeling/submit
  → 创建任务，写入 Redis
  → WebSocket 推送状态更新
  → Workflow.execute() 启动
    → CoordinatorAgent: 分析题目，拆解为子任务列表
    → 对每个子任务:
      → ModelerAgent: 建立数学模型（LaTeX）
      → CoderAgent: 生成 Python 代码，在 Jupyter 中执行
      → 若代码出错: 自动反馈给 CoderAgent 重试
    → 所有子任务完成后:
      → WriterAgent: 生成论文章节
      → ReviewAgent: 评审打分
      → 若分数 < 阈值且轮次未满:
        → 生成改进建议 → WriterAgent 改进 → 重新评审（Reflexion）
      → 保存最终论文（.md + .docx）
  → WebSocket 推送完成通知
```

### Agent 间通信

- **task_id** 隔离并发任务
- **Redis pub/sub** 广播状态到 WebSocket
- **DiagnosticLogger** 记录所有 LLM 交互到 `{work_dir}/diagnostic/`

### 输出产物

每次运行在 `backend/project/work_dir/{task_id}/` 下生成：
- `diagnostic/interactions.jsonl` — 所有 LLM 交互记录
- `diagnostic/tool_results.jsonl` — 工具调用结果
- `diagnostic/quality.json` — 质量评分历史
- `diagnostic/config.json` — 运行配置快照
- `*.ipynb` — Jupyter Notebook 代码
- `*.md` — 最终论文
- `*.docx` — Word 格式论文

## Code Style

### 后端（Python）

- 模块级、类级、公共方法均使用 Google 风格 docstring（Args/Returns/Raises）
- 类型注解：使用 `str | None` 而非 `Optional[str]`
- 循环导入处理：使用 `from __future__ import annotations` + `TYPE_CHECKING` 守卫
- 异步：全程 async/await，FastAPI 路由均为 async def
- 注释：中文，解释 WHY 而非 WHAT

```python
"""模块级 docstring：描述模块用途。"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.utils.diagnostic_logger import DiagnosticLogger


class ExampleAgent:
    """类级 docstring：简述职责。"""

    def __init__(self, diagnostic_logger: DiagnosticLogger | None = None):
        self.diagnostic_logger = diagnostic_logger

    async def run(self, prompt: str, system_prompt: str) -> str:
        """执行任务并返回结果。

        Args:
            prompt: 用户输入。
            system_prompt: 系统提示词。

        Returns:
            处理结果文本。

        Raises:
            ValueError: 输入参数无效时。
        """
```

### 前端（Vue 3 + TypeScript）

- SFC 使用 `<script setup lang="ts">`
- 代码按逻辑分组，用注释分隔：`// ---- Props ----`、`// ---- State ----`、`// ---- Computed ----`、`// ---- Methods ----`
- TypeScript 接口和 API 函数使用 JSDoc `/** */` 注释
- UI 库组件（`components/ui/`）为 shadcn-vue 生成代码，不要修改
- 格式：tab 缩进，双引号，Biome 管理 lint 和格式化

```vue
<script setup lang="ts">
import { ref, computed } from "vue";

// ---- Props ----

/** 组件属性 */
interface Props {
	/** 消息类型 */
	type: "agent" | "user";
	/** 消息内容 */
	content: string;
}
const props = withDefaults(defineProps<Props>(), { type: "user" });

// ---- Computed ----

const rendered = computed(() => marked.parse(props.content));
</script>
```

## Git Workflow

提交信息格式：`<type>: <描述>`，type 包括：

- `feat`: 新功能
- `fix`: 修复
- `refactor`: 重构
- `chore`: 杂项变更
- `enhance`: 增强
- `docs`: 文档

示例：`feat: 添加 OpenAlex API Key 支持并更新相关配置`

## Boundaries

### 不要修改的内容

- `frontend/src/components/ui/` — shadcn-vue 第三方 UI 库组件
- 已有的 `# type: ignore` 注释 — 这些是经过验证的类型抑制，非遗留问题
- `.env` 相关文件中的实际配置值

### 运行环境

- Python 3.12+，包管理用 uv（非 pip）
- Node.js，包管理用 pnpm（版本见 packageManager 字段）
- Redis 必须运行（任务队列和 WebSocket 广播）
- 后端虚拟环境路径：`backend/.venv/`

### 依赖管理注意事项

- 后端科学计算包（numpy、pandas、scikit-learn 等）是 Jupyter 内核依赖，CoderAgent 生成的代码在内核中执行时需要，不能移除
- `e2b-code-interpreter` 是可选的云端沙箱依赖，本地开发可移除
- 前端 `components/ui/` 下的 shadcn-vue 组件由工具生成，手动修改会被覆盖
