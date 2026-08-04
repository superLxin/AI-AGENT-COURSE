"""
第 10 讲（终章）：完整旅行助手 —— 10 讲内容的最终形态
=========================================================
这个文件整合了课程中的核心概念，展示一个完整的旅行规划 Agent。

能力清单：
✅ 工具调用（查天气、搜航班、找酒店）
✅ 知识库检索（Chroma RAG）
✅ 规划与自检（元认知）
✅ 多 Agent 协作（Supervisor 模式）
✅ 安全护栏（Human-in-the-Loop）
✅ 记忆系统（用户偏好记忆）
"""

import json
import os
import uuid
from datetime import datetime
from typing import TypedDict, Annotated

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
import operator

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# ============================================================
# 1. 知识库（Chroma）
# ============================================================

chroma_client = chromadb.PersistentClient(path="./chroma_data")
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

kb_collection = chroma_client.get_or_create_collection(
    name="travel_knowledge", embedding_function=embed_fn,
)

# 初始化知识库
if len(kb_collection.get()["ids"]) == 0:
    docs = [
        "西湖免费开放，建议游玩半天。雷峰塔门票40元。最佳季节春秋两季。",
        "杭州美食：西湖醋鱼、龙井虾仁、东坡肉。推荐餐厅：楼外楼（人均150）、知味观（人均80）。",
        "杭州8月炎热多雨，28-35°C，常有午后雷阵雨。建议带伞，选择清晨和傍晚户外活动。",
        "杭州三天经典行程：Day1 西湖环湖，Day2 灵隐寺+龙井村，Day3 西溪湿地。",
    ]
    kb_collection.add(documents=docs, ids=[f"doc_{i}" for i in range(len(docs))])

# ============================================================
# 2. 记忆系统（Chroma）
# ============================================================

memory_collection = chroma_client.get_or_create_collection(
    name="user_memories", embedding_function=embed_fn,
)


def get_user_memories(user_input: str, user_id: str) -> str:
    """检索用户记忆"""
    results = memory_collection.query(
        query_texts=[user_input], where={"user_id": user_id}, n_results=3,
    )
    if results["documents"] and results["documents"][0]:
        return "用户偏好：" + "; ".join(results["documents"][0])
    return ""


def save_user_memory(conversation: str, user_id: str):
    """保存新记忆"""
    prompt = f"从对话中提取用户偏好（如'喜欢靠窗座位'），返回JSON数组。没有则返回[]。\n{conversation}"
    try:
        memories = json.loads(llm.invoke(prompt).content)
        for mem in memories:
            memory_collection.add(
                documents=[mem],
                metadatas=[{"user_id": user_id}],
                ids=[f"mem_{uuid.uuid4().hex[:8]}"],
            )
    except Exception:
        pass

# ============================================================
# 3. 工具定义
# ============================================================


@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：小雨，28°C（8月常有午后雷阵雨，建议带伞）"


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """搜索航班"""
    return json.dumps({
        "flights": [
            {"flight": "CA1234", "time": "08:00-10:30", "price": 580},
            {"flight": "MU5678", "time": "14:00-16:30", "price": 720},
            {"flight": "CZ9012", "time": "19:00-21:30", "price": 490},
        ]
    }, ensure_ascii=False)


@tool
def search_hotels(city: str, checkin: str, checkout: str) -> str:
    """搜索酒店"""
    return json.dumps({
        "hotels": [
            {"name": "湖畔花园", "price": 380, "rating": 4.8, "feature": "西湖边湖景房"},
            {"name": "便捷商务", "price": 180, "rating": 4.2, "feature": "市中心交通便利"},
        ]
    }, ensure_ascii=False)


@tool
def search_knowledge(query: str) -> str:
    """搜索杭州旅游攻略知识库"""
    results = kb_collection.query(query_texts=[query], n_results=2)
    return "\n".join(results["documents"][0]) if results["documents"] else "未找到"


# ============================================================
# 4. 专业 Agent
# ============================================================

flight_expert = create_react_agent(llm, [search_flights, get_weather],
    prompt="你是航班专家。帮助搜索航班，考虑价格、出发时间和用户偏好。")

hotel_expert = create_react_agent(llm, [search_hotels],
    prompt="你是酒店专家。帮助搜索酒店，推荐时说明理由。")

guide_expert = create_react_agent(llm, [get_weather, search_knowledge],
    prompt="你是行程专家。综合航班、酒店、天气、攻略信息，编排合理行程。")

# ============================================================
# 5. Supervisor 模式
# ============================================================


class State(TypedDict):
    user_input: str
    user_id: str
    flight_result: str
    hotel_result: str
    guide_result: str
    memories: str
    next_agent: str
    final_answer: str
    messages: Annotated[list, operator.add]


def supervisor(state: State) -> dict:
    """调度员"""
    has_f = bool(state.get("flight_result"))
    has_h = bool(state.get("hotel_result"))
    has_g = bool(state.get("guide_result"))
    if has_f and has_h and has_g:
        return {"next_agent": "FINISH"}
    if not has_f: return {"next_agent": "flight_expert"}
    if not has_h: return {"next_agent": "hotel_expert"}
    return {"next_agent": "guide_expert"}


def call_flight(state: State) -> dict:
    context = f"{state.get('memories','')}\n用户需求：{state['user_input']}"
    result = flight_expert.invoke({"messages": [("user", context)]})
    return {"flight_result": result["messages"][-1].content}


def call_hotel(state: State) -> dict:
    result = hotel_expert.invoke({"messages": [("user", state["user_input"])]})
    return {"hotel_result": result["messages"][-1].content}


def call_guide(state: State) -> dict:
    context = f"用户：{state['user_input']}\n航班：{state.get('flight_result','')}\n酒店：{state.get('hotel_result','')}"
    result = guide_expert.invoke({"messages": [("user", context)]})
    return {"guide_result": result["messages"][-1].content}


def human_approval(state: State) -> dict:
    """高危操作前请求审批（模拟）"""
    if "订" not in state["user_input"]:
        return {"final_answer": state["guide_result"]}
    print("\n⚠️  检测到预订请求，需要用户确认（在实际应用中这里会暂停等待审批）")
    return {"final_answer": f"[需要审批]\n\n方案：{state['guide_result']}\n\n请回复'确认'以继续预订。"}


# ============================================================
# 6. 构建状态图
# ============================================================

graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("flight_expert", call_flight)
graph.add_node("hotel_expert", call_hotel)
graph.add_node("guide_expert", call_guide)
graph.add_node("approval", human_approval)

graph.set_entry_point("supervisor")
graph.add_edge("flight_expert", "supervisor")
graph.add_edge("hotel_expert", "supervisor")
graph.add_edge("guide_expert", "supervisor")
graph.add_edge("approval", END)

graph.add_conditional_edges("supervisor", lambda s: s["next_agent"], {
    "flight_expert": "flight_expert",
    "hotel_expert": "hotel_expert",
    "guide_expert": "guide_expert",
    "FINISH": "approval",
})

app = graph.compile()


# ============================================================
# 7. 主入口
# ============================================================

def travel_agent(user_input: str, user_id: str = "anonymous") -> str:
    """完整旅行助手入口"""
    # 检索记忆
    memories = get_user_memories(user_input, user_id)

    print(f"\n{'='*60}")
    print(f"🧳 旅行助手启动")
    print(f"   用户: {user_id}")
    print(f"   记忆: {memories if memories else '无'}")
    print(f"{'='*60}")

    # 执行多 Agent 协作
    result = app.invoke({
        "user_input": user_input,
        "user_id": user_id,
        "flight_result": "",
        "hotel_result": "",
        "guide_result": "",
        "memories": memories,
        "next_agent": "",
        "final_answer": "",
        "messages": [],
    })

    # 保存新记忆
    conversation = f"用户: {user_input}\n助手: {result['final_answer']}"
    save_user_memory(conversation, user_id)

    return result["final_answer"]


if __name__ == "__main__":
    print("🏁 AI Agent 入门公开课 —— 终章演示")
    print("10 讲的全部能力整合在一个 Agent 中\n")

    answer = travel_agent(
        "我8月10日从北京去杭州玩3天，预算3000，喜欢自然风光。帮我规划。",
        user_id="final_demo",
    )
    print(f"\n📝 最终回答：\n{answer}")
    print(f"\n{'='*60}")
    print("✅ 10 讲课程至此结束！")
    print("你已经学会了如何从零构建一个完整的 AI Agent。")
    print(f"{'='*60}")
