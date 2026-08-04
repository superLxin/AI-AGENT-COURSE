"""
第 6 讲：多 Agent 协作 —— Supervisor 模式
=============================================
一个 Supervisor（调度员）+ 三个专业 Worker（航班/酒店/行程专家）。
Supervisor 动态决定哪个 Worker 来处理当前任务。
"""

import json
import os
from typing import TypedDict, Literal, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import operator

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# ============================================================
# 1. 定义工具
# ============================================================


@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
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
        {"flight": "CZ9012", "departure": "19:00", "arrival": "21:30", "price": 490},
    ]
    return json.dumps({"route": f"{origin}→{destination}", "flights": flights}, ensure_ascii=False)


@tool
def search_hotels(city: str, checkin: str, checkout: str) -> str:
    """搜索酒店"""
    hotels = [
        {"name": "湖畔花园", "price": 380, "rating": 4.8, "feature": "湖景房"},
        {"name": "便捷商务", "price": 180, "rating": 4.2, "feature": "交通便利"},
        {"name": "山间度假", "price": 520, "rating": 4.6, "feature": "自然环境好"},
    ]
    return json.dumps({"city": city, "hotels": hotels}, ensure_ascii=False)


# ============================================================
# 2. 创建专业 Worker Agent
# ============================================================

flight_agent = create_react_agent(
    llm, [search_flights, get_weather],
    prompt="你是航班专家。帮助搜索和比较航班，考虑价格和出发时间。"
)

hotel_agent = create_react_agent(
    llm, [search_hotels],
    prompt="你是酒店专家。根据用户预算和偏好推荐酒店。说明推荐理由。"
)

itinerary_agent = create_react_agent(
    llm, [get_weather],
    prompt="你是行程规划专家。综合航班、酒店、天气信息，编排合理行程。"
)

# ============================================================
# 3. 定义状态和 Supervisor
# ============================================================


class TeamState(TypedDict):
    user_input: str
    messages: Annotated[list, operator.add]
    next_agent: str
    flight_result: str
    hotel_result: str
    itinerary_result: str
    final_answer: str


SUPERVISOR_PROMPT = """你是旅行规划的调度员。你的团队有三位专家：
- flight_expert: 处理航班搜索和比价
- hotel_expert: 处理酒店搜索和推荐
- itinerary_expert: 处理行程规划和编排

根据用户需求和当前进度，决定下一步：
1. 如果需要航班信息 → flight_expert
2. 如果需要酒店推荐 → hotel_expert
3. 如果需要行程规划 → itinerary_expert
4. 如果所有信息收集完毕 → FINISH

请只回复一个词：flight_expert / hotel_expert / itinerary_expert / FINISH
"""


def supervisor(state: TeamState) -> dict:
    """调度员：根据当前状态决定下一步"""
    # 根据已有信息智能决策
    has_flight = bool(state.get("flight_result"))
    has_hotel = bool(state.get("hotel_result"))
    has_itinerary = bool(state.get("itinerary_result"))

    if has_flight and has_hotel and has_itinerary:
        print("  🎯 [Supervisor] 所有信息已收集，汇总输出")
        return {"next_agent": "FINISH"}

    # 让 LLM 做调度决策
    context = f"当前进度：航班={'✅' if has_flight else '❌'} 酒店={'✅' if has_hotel else '❌'} 行程={'✅' if has_itinerary else '❌'}\n用户需求：{state['user_input']}"
    response = llm.invoke(f"{SUPERVISOR_PROMPT}\n{context}")
    decision = response.content.strip()

    valid = ["flight_expert", "hotel_expert", "itinerary_expert", "FINISH"]
    if decision not in valid:
        # 智能 fallback
        if not has_flight:
            decision = "flight_expert"
        elif not has_hotel:
            decision = "hotel_expert"
        else:
            decision = "itinerary_expert"

    print(f"  🎯 [Supervisor] → {decision}")
    return {"next_agent": decision}


# ============================================================
# 4. Worker 节点
# ============================================================


def call_flight_agent(state: TeamState) -> dict:
    print("  ✈️  [Flight Expert] 正在搜索航班...")
    result = flight_agent.invoke({"messages": [("user", state["user_input"])]})
    last_msg = result["messages"][-1].content
    print(f"  ✈️  [Flight Expert] 完成")
    return {"flight_result": last_msg}


def call_hotel_agent(state: TeamState) -> dict:
    print("  🏨 [Hotel Expert] 正在搜索酒店...")
    result = hotel_agent.invoke({"messages": [("user", state["user_input"])]})
    last_msg = result["messages"][-1].content
    print(f"  🏨 [Hotel Expert] 完成")
    return {"hotel_result": last_msg}


def call_itinerary_agent(state: TeamState) -> dict:
    print("  📋 [Itinerary Expert] 正在编排行程...")
    context = f"用户需求：{state['user_input']}\n航班：{state.get('flight_result', '未查')}\n酒店：{state.get('hotel_result', '未查')}"
    result = itinerary_agent.invoke({"messages": [("user", context)]})
    last_msg = result["messages"][-1].content
    print(f"  📋 [Itinerary Expert] 完成")
    return {"itinerary_result": last_msg}


def summarize(state: TeamState) -> dict:
    """汇总所有专家的结果"""
    summary = llm.invoke(
        f"请将以下专家意见汇总为一份完整的旅行方案：\n\n"
        f"航班建议：{state['flight_result']}\n\n"
        f"酒店建议：{state['hotel_result']}\n\n"
        f"行程建议：{state['itinerary_result']}\n\n"
        f"用户需求：{state['user_input']}"
    )
    return {"final_answer": summary.content}


# ============================================================
# 5. 构建状态图
# ============================================================

graph = StateGraph(TeamState)

graph.add_node("supervisor", supervisor)
graph.add_node("flight_expert", call_flight_agent)
graph.add_node("hotel_expert", call_hotel_agent)
graph.add_node("itinerary_expert", call_itinerary_agent)
graph.add_node("summarize", summarize)

graph.set_entry_point("supervisor")

# 每个 Worker 完成后回到 Supervisor
graph.add_edge("flight_expert", "supervisor")
graph.add_edge("hotel_expert", "supervisor")
graph.add_edge("itinerary_expert", "supervisor")
graph.add_edge("summarize", END)

# Supervisor 的路由
graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {
        "flight_expert": "flight_expert",
        "hotel_expert": "hotel_expert",
        "itinerary_expert": "itinerary_expert",
        "FINISH": "summarize",
    },
)

app = graph.compile()

# ============================================================
# 6. 运行
# ============================================================

if __name__ == "__main__":
    query = "我8月10日从北京去杭州玩3天，预算3000元。帮我规划。"
    print(f"用户：{query}\n")
    print("=" * 60)

    result = app.invoke({
        "user_input": query,
        "messages": [],
        "next_agent": "",
        "flight_result": "",
        "hotel_result": "",
        "itinerary_result": "",
        "final_answer": "",
    })

    print("\n" + "=" * 60)
    print("📝 最终方案：")
    print("=" * 60)
    print(result["final_answer"])
