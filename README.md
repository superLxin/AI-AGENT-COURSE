# AI Agent 入门公开课

从零开始构建 AI Agent 的 10 讲公开课，配套微信公众号系列文章。

## 课程结构

| 讲次 | 主题 | 核心能力 |
|------|------|----------|
| 01 | 什么是 AI Agent？ | 理解 Agent 概念，聊天机器人 vs Agent 的区别 |
| 02 | 动手构建第一个 Agent | 原生 Python 实现 Agent 循环 + LangChain 入门 |
| 03 | Agent 的超能力：工具调用 | Function Calling 原理，定义工具，Agent 自动选择 |
| 04 | Agent 的知识库：Agentic RAG | Chroma 向量数据库，检索增强生成，迭代检索 |
| 05 | Agent 的思考方式：规划与元认知 | LangGraph 状态图，任务分解，自纠错 |
| 06 | 多 Agent 协作：团队作战 | Supervisor 模式，专业分工，并行协作 |
| 07 | 构建可信赖的 Agent | Human-in-the-Loop，上下文工程，安全护栏 |
| 08 | Agent 的记忆系统 | Chroma 记忆，Mem0，长期/短期记忆管理 |
| 09 | 从 Demo 到生产 | FastAPI 部署，Docker 容器化，LangFuse 可观测性 |
| 10 | 前沿探索与总结 | MCP 协议，Ollama 本地 Agent，Browser Use |

## 快速开始

### 环境要求

- Python 3.10+
- 一个 OpenAI 兼容的 API Key（OpenAI / DeepSeek / 阿里百炼 均可）

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourname/ai-agent-course.git
cd ai-agent-course

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY
```

### 运行

每讲代码独立可运行，进入对应目录直接执行：

```bash
cd lecture-02
python 02_agent_loop.py
```

## 技术栈

- **LLM 调用**: OpenAI SDK + LangChain
- **Agent 框架**: LangChain + LangGraph
- **向量数据库**: Chroma
- **记忆管理**: Mem0
- **部署**: FastAPI + Docker
- **可观测性**: LangFuse
- **协议**: MCP
- **本地模型**: Ollama

不绑定任何特定云服务商，所有代码只需一个 OpenAI 兼容 API Key 即可运行。
