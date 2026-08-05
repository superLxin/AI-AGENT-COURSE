"""
第 3 讲：Agent 的超能力 —— 工具调用
======================================
给旅行助手装上三件工具：查天气、搜航班、找酒店。
Agent 自动选择正确的工具来回答用户问题。
"""

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# 旅行助手的三件工具
# ============================================================

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。city: 城市中文名称，如'杭州'、'北京'。"""
    weather_data = {
        "北京": {"weather": "晴", "temp": "32°C", "humidity": "45%"},
        "上海": {"weather": "多云", "temp": "30°C", "humidity": "60%"},
        "杭州": {"weather": "小雨", "temp": "28°C", "humidity": "75%"},
        "成都": {"weather": "阴", "temp": "26°C", "humidity": "55%"},
        "三亚": {"weather": "晴", "temp": "34°C", "humidity": "70%"},
    }
    data = weather_data.get(city, {"weather": "未知", "temp": "N/A", "humidity": "N/A"})
    return f"{city}：{data['weather']}，{data['temp']}，湿度{data['humidity']}"

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """
    搜索航班信息。
    origin: 出发城市中文名称
    destination: 目的城市中文名称
    date: 出发日期，格式YYYY-MM-DD
    """
    flights = [
        {"flight": "CA1234", "airline": "中国国航", "departure": "08:00", "arrival": "10:30", "price": 580, "duration": "2h30m"},
        {"flight": "MU5678", "airline": "东方航空", "departure": "14:00", "arrival": "16:30", "price": 720, "duration": "2h30m"},
        {"flight": "CZ9012", "airline": "南方航空", "departure": "19:00", "arrival": "21:30", "price": 490, "duration": "2h30m"},
        {"flight": "HU3456", "airline": "海南航空", "departure": "07:30", "arrival": "10:00", "price": 650, "duration": "2h30m"},
    ]
    result = {
        "route": f"{origin} → {destination}",
        "date": date,
        "flights": flights,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

@tool
def search_hotels(city: str, checkin: str, checkout: str) -> str:
    """
    搜索酒店信息。
    city: 城市中文名称
    checkin: 入住日期 YYYY-MM-DD
    checkout: 离店日期 YYYY-MM-DD
    """
    hotels = [
        {"name": "湖畔花园酒店", "price": 380, "rating": 4.8, "location": "西湖区，距西湖500m", "feature": "湖景房，含早餐"},
        {"name": "便捷商务酒店", "price": 180, "rating": 4.2, "location": "市中心，距地铁站200m", "feature": "性价比高，交通便利"},
        {"name": "山间度假村", "price": 520, "rating": 4.6, "location": "西溪湿地旁", "feature": "自然环境优美，适合度假"},
    ]
    result = {
        "city": city,
        "checkin": checkin,
        "checkout": checkout,
        "hotels": hotels,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

# ============================================================
# 系统提示词
# ============================================================

SYSTEM_PROMPT = """你是一个旅行规划助手。你可以使用工具来查询天气、航班和酒店信息。

工作规则：
1. 使用工具获取真实数据，不要编造信息
2. 当用户问及航班时，必须使用 search_flights 工具
3. 当用户问及酒店时，必须使用 search_hotels 工具
4. 综合多个工具的结果给出合理建议
5. 推荐时说明理由（价格、便利性、用户偏好等）
"""

# ============================================================
# 创建 Agent
# ============================================================

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

tools = [get_weather, search_flights, search_hotels]

graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    # 一个需要多种工具协作的复杂问题
    query = (
        "我计划2026年8月10日从北京去杭州玩3天，"
        "帮我查一下：1）杭州那几天的天气 2）北京到杭州的航班 "
        "3）8月10日到13日的酒店。"
        "然后给我一个综合建议，推荐最佳方案。"
    )

    print(f"用户：{query}\n")
    result = graph.invoke({"messages": [{"role": "user", "content": query}]})
    print(f"\n{'=' * 60}")
    print(f"Agent 最终回答：\n{result['messages'][-1].content}")
