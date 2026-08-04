"""
第 2 讲：Agent 的本质 —— 一个 while 循环（原生 Python 实现）
==============================================================
不依赖任何 Agent 框架，用纯 OpenAI SDK 手写一个完整的 Agent 循环。
读完你会理解：Agent 的核心就是一个 while 循环。
"""

import json
import os
from datetime import datetime

import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# 1. 定义工具
# ============================================================

def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    # 模拟数据——实际项目中替换为真实天气 API
    weather_data = {
        "北京": "晴，32°C，湿度45%",
        "上海": "多云，30°C，湿度60%",
        "杭州": "小雨，28°C，湿度75%",
        "成都": "阴，26°C，湿度55%",
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"计算出错：{e}"

# 工具映射表
TOOL_MAP = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "calculate": calculate,
}

# ============================================================
# 2. 工具的"说明书"（OpenAI Function Schema）
# ============================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当用户问'现在几点'、'今天几号'时使用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气。当用户问天气相关问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如'北京'、'上海'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。当用户需要做数学计算时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如'3*4+2'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

# ============================================================
# 3. Agent 核心循环
# ============================================================

SYSTEM_PROMPT = "你是一个有帮助的助手。你可以使用工具来获取实时信息。"


def run_agent(user_message: str) -> str:
    """Agent 核心循环：思考 → 行动 → 观察 → 重复"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        # 向 LLM 发送当前消息
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=TOOLS_SCHEMA,
        )

        choice = response.choices[0]

        # finish_reason == "stop" → LLM 决定直接回答用户
        if choice.finish_reason == "stop":
            return choice.message.content

        # finish_reason == "tool_calls" → LLM 决定调用工具
        if choice.finish_reason == "tool_calls":
            message = choice.message

            # 执行每一个被调用的工具
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  🔧 调用工具: {tool_name}({tool_args})")

                # 执行真正的工具函数
                tool_func = TOOL_MAP[tool_name]
                result = tool_func(**tool_args)

                print(f"  📋 工具结果: {result}")

                # 将工具执行结果加入对话
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # 将 LLM 的 tool_calls 消息也加入对话
            messages.append(message)
            # 循环继续 —— LLM 收到工具结果后再次思考


# ============================================================
# 4. 运行测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 1：简单问题（不需要工具）")
    print("=" * 60)
    answer = run_agent("你好，请用一句话介绍你自己")
    print(f"Agent：{answer}\n")

    print("=" * 60)
    print("测试 2：需要调用工具")
    print("=" * 60)
    answer = run_agent("北京现在天气怎么样？上海的也查一下")
    print(f"Agent：{answer}\n")

    print("=" * 60)
    print("测试 3：需要多个工具组合")
    print("=" * 60)
    answer = run_agent("现在是几点？帮我把12345乘以6789算出来")
    print(f"Agent：{answer}\n")

    print("=" * 60)
    print("✅ 这就是 Agent 的核心！")
    print("while 循环 + finish_reason 判断 + 工具调用 = Agent")
    print("=" * 60)
