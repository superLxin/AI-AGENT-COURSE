"""
第 2 讲：用 LangChain 实现同样的 Agent
=========================================
对比 02_agent_loop_raw.py，看框架帮我们省了多少代码。
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# 1. 定义工具 —— @tool 装饰器自动生成 Schema
# ============================================================

@tool
def get_current_time() -> str:
    """获取当前日期和时间。当用户问'现在几点'、'今天几号'时使用。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。当用户问天气相关问题时使用。city: 城市名称。"""
    weather_data = {
        "北京": "晴，32°C，湿度45%",
        "上海": "多云，30°C，湿度60%",
        "杭州": "小雨，28°C，湿度75%",
        "成都": "阴，26°C，湿度55%",
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。expression: 数学表达式字符串。"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"计算出错：{e}"

# ============================================================
# 2. 系统提示词
# ============================================================

SYSTEM_PROMPT = "你是一个有帮助的助手。你可以使用工具来获取实时信息。"

# ============================================================
# 3. 一行创建 Agent ✨
# ============================================================

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

tools = [get_current_time, get_weather, calculate]

agent = create_tool_calling_agent(llm, tools, SYSTEM_PROMPT)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ============================================================
# 4. 运行测试
# ============================================================

if __name__ == "__main__":
    # 和原生版本同样的测试，但代码少了一半
    questions = [
        "北京和上海的天气分别怎么样？哪边更适合明天出游？",
        "现在几点？如果现在是下午3点，距离明天早上8点还有多少小时？帮我算一下。",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"用户：{q}")
        print("=" * 60)
        result = executor.invoke({"input": q})
        print(f"\nAgent：{result['output']}")

    print(f"\n{'=' * 60}")
    print("✅ 对比原生版本：")
    print("  - 不需要手动写 TOOLS_SCHEMA（@tool 自动生成）")
    print("  - 不需要手动写 while 循环（AgentExecutor 内置）")
    print("  - 不需要手动拼接 messages（框架自动管理）")
    print("  - verbose=True 自动打印思考过程")
    print("=" * 60)
