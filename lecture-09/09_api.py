"""
第 9 讲：从 Demo 到生产 —— 部署、评估与可观测性
=====================================================
1. FastAPI 将 Agent 暴露为 REST API
2. LangFuse 可观测性追踪
3. 模型路由（小模型处理简单问题，大模型处理复杂问题）
4. 响应缓存

运行方式：
    uvicorn 09_api:app --reload
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"你好"}'
"""

import hashlib
import json
import os
import time
from datetime import datetime
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# 1. 模型路由：简单问题用小模型，复杂问题用大模型
# ============================================================

small_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
large_llm = ChatOpenAI(model="gpt-4o", temperature=0)

SIMPLE_KEYWORDS = ["你好", "谢谢", "再见", "帮助", "天气", "几点", "今天"]
COMPLEX_KEYWORDS = ["规划", "行程", "攻略", "对比", "推荐方案", "预订"]


def choose_model(user_input: str) -> ChatOpenAI:
    """根据问题复杂度选择模型"""
    is_complex = any(kw in user_input for kw in COMPLEX_KEYWORDS)
    is_simple = any(kw in user_input for kw in SIMPLE_KEYWORDS) and not is_complex

    if is_complex:
        print(f"  🧠 路由到: gpt-4o (复杂问题)")
        return large_llm
    else:
        print(f"  ⚡ 路由到: gpt-4o-mini (简单问题)")
        return small_llm


# ============================================================
# 2. 响应缓存
# ============================================================

cache: dict[str, dict] = {}


def get_cached_or_compute(user_input: str, func, **kwargs):
    """缓存：相同输入直接返回缓存结果"""
    cache_key = hashlib.md5(user_input.encode()).hexdigest()

    if cache_key in cache:
        print(f"  💾 缓存命中！")
        return cache_key, cache[cache_key]

    result = func(**kwargs)
    cache[cache_key] = result
    return cache_key, result


# ============================================================
# 3. 工具定义
# ============================================================


@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    data = {
        "北京": "晴，32°C", "上海": "多云，30°C",
        "杭州": "小雨，28°C", "成都": "阴，26°C",
    }
    return data.get(city, f"{city}：多云，25°C")


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """搜索航班"""
    flights = [
        {"flight": "CA1234", "departure": "08:00", "arrival": "10:30", "price": 580},
        {"flight": "MU5678", "departure": "14:00", "arrival": "16:30", "price": 720},
    ]
    return json.dumps(flights, ensure_ascii=False)


tools = [get_weather, search_flights]

# ============================================================
# 4. LangFuse 可观测性（可选）
# ============================================================

langfuse_handler = None
try:
    from langfuse.callback import CallbackHandler
    langfuse_handler = CallbackHandler(
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    print("✅ LangFuse 可观测性已启用")
except Exception:
    print("ℹ️  LangFuse 未配置，可观测性跳过（不影响运行）")

# ============================================================
# 5. FastAPI 应用
# ============================================================

app = FastAPI(
    title="旅行规划助手 API",
    description="一个 AI Agent 入门课程的示例 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"


class ChatResponse(BaseModel):
    reply: str
    model_used: str
    from_cache: bool = False
    tool_calls: list = []
    timestamp: str


class FeedbackRequest(BaseModel):
    chat_id: str
    rating: int  # 1-5
    comment: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """主要的 Agent 对话接口"""
    start_time = time.time()

    # 1. 选择模型
    model = choose_model(request.message)

    # 2. 创建 Agent
    system_prompt = "你是一个旅行助手。使用工具获取信息，给出友好、准确的回答。"
    agent = create_tool_calling_agent(model, tools, system_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 3. 执行（带缓存）
    callbacks = [langfuse_handler] if langfuse_handler else None

    try:
        cache_key, result = get_cached_or_compute(
            request.message,
            executor.invoke,
            input=request.message,
            config={"callbacks": callbacks} if callbacks else {},
        )

        elapsed = time.time() - start_time
        print(f"  ⏱️  耗时: {elapsed:.2f}s")

        # 提取工具调用信息
        tool_calls = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                tool_calls.append({
                    "tool": step[0].tool,
                    "input": step[0].tool_input,
                    "output": str(step[1])[:200],
                })

        return ChatResponse(
            reply=result["output"],
            model_used=model.model_name,
            from_cache=(cache_key in cache and cache_key != cache_key),
            tool_calls=tool_calls,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """收集用户反馈"""
    if langfuse_handler:
        # LangFuse 记录用户评分
        pass
    return {"status": "received", "rating": request.rating}


@app.get("/metrics")
async def metrics():
    """简单的指标端点"""
    return {
        "cache_size": len(cache),
        "uptime": "since server start",
    }


# ============================================================
# 启动命令：
#   uvicorn 09_api:app --reload
#
# 测试命令：
#   curl -X POST http://localhost:8000/chat \
#     -H "Content-Type: application/json" \
#     -d '{"message": "帮我查北京到杭州明天最便宜的航班"}'
# ============================================================
