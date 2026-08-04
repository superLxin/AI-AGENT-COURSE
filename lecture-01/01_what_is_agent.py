"""
第 1 讲：什么是 AI Agent？
============================
本文件仅做概念演示，展示聊天机器人 vs Agent 的核心区别。
无需安装任何依赖，直接运行即可理解概念。
"""

# ============================================================
# 场景：用户想知道"三亚明天天气怎么样？"
# ============================================================

# --- 聊天机器人的回答方式 ---
# 聊天机器人只能从训练数据中"回忆"，无法获取实时信息。
# 如果训练数据中没有明天的天气，它只能：
chatbot_response = (
    "抱歉，我的训练数据截止到某个日期，无法获取实时的天气信息。"
    "建议您查看天气预报网站。"
)
print("【聊天机器人】")
print(f"用户：三亚明天天气怎么样？")
print(f"机器人：{chatbot_response}")
print()

# --- AI Agent 的回答方式 ---
# Agent = LLM + 工具 + 行动循环
# 它有一个 get_weather 工具可以调用真实的天气 API

def get_weather(city: str) -> dict:
    """模拟天气 API 调用（实际项目中替换为真实 API）"""
    # 真实场景这里会调用 OpenWeatherMap 或和风天气等 API
    return {
        "city": city,
        "date": "2026-08-04",
        "weather": "晴",
        "temperature": 32,
        "humidity": "45%",
    }

# Agent 的执行过程（简化版）：
# 1. 用户提问
# 2. LLM 判断：需要调用 get_weather("三亚")
# 3. 执行 get_weather → 拿到数据
# 4. LLM 根据数据生成自然语言回答

weather_data = get_weather("三亚")
agent_response = (
    f"三亚明天（8月4日）天气{weather_data['weather']}，"
    f"气温{weather_data['temperature']}°C，"
    f"湿度{weather_data['humidity']}。适合出行！"
)
print("【AI Agent】")
print(f"用户：三亚明天天气怎么样？")
print(f"Agent：{agent_response}")
print()

# ============================================================
# 核心概念总结
# ============================================================
print("=" * 50)
print("聊天机器人 vs AI Agent")
print("=" * 50)
print("聊天机器人 = 只能生成文本")
print("AI Agent   = LLM（大脑）+ 工具（双手）+ 行动循环（执行力）")
print()
print("Agent 的三个核心组件：")
print("  1. Environment（环境）: Agent 工作的空间")
print("  2. Sensors（感知）:   读取环境状态（调用 API）")
print("  3. Actuators（行动）: 改变环境（发邮件、下单）")
