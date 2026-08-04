"""
第 10 讲：完全本地运行的 Agent（Ollama + Qwen）
==================================================
使用 Ollama 运行开源模型，数据完全不出本机。

前置条件：
1. 安装 Ollama: https://ollama.com (macOS: brew install ollama)
2. 拉取模型: ollama pull qwen2.5:7b
3. 启动服务: ollama serve

然后运行本文件即可。
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


def check_ollama() -> bool:
    """检查 Ollama 是否可用"""
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================
# 工具定义（和云端版本完全一样！）
# ============================================================


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴，32°C，湿度45%",
        "上海": "多云，30°C，湿度60%",
        "杭州": "小雨，28°C，湿度75%",
    }
    return weather_data.get(city, f"{city}：多云，25°C")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"计算出错：{e}"


# ============================================================
# 创建本地 Agent
# ============================================================


def create_local_agent(model_name: str = "qwen2.5:7b"):
    """
    创建本地 Agent。
    关键：base_url 指向 Ollama 的 OpenAI 兼容端点。
    其他代码和云端版完全一样！
    """
    llm = ChatOpenAI(
        model=model_name,
        base_url="http://localhost:11434/v1",  # ← 唯一区别！
        api_key="not-needed",                   # 本地不需要 key
        temperature=0,
    )

    tools = [get_current_time, get_weather, calculate]

    system_prompt = "你是一个有帮助的助手。使用工具来获取实时信息。用中文回答。"

    agent = create_tool_calling_agent(llm, tools, system_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return executor


# ============================================================
# 运行
# ============================================================


if __name__ == "__main__":
    print("=" * 60)
    print("🏠 本地 Agent 演示（Ollama + Qwen）")
    print("=" * 60)

    # 检查 Ollama
    if not check_ollama():
        print("""
❌ Ollama 未运行！

请先执行以下步骤：
  1. 安装 Ollama: brew install ollama  (或访问 https://ollama.com)
  2. 启动服务:     ollama serve
  3. 拉取模型:     ollama pull qwen2.5:7b

然后再运行本文件。
        """)
        exit(1)

    print("✅ Ollama 已连接")
    print(f"📦 使用模型: qwen2.5:7b")
    print()

    agent = create_local_agent()

    questions = [
        "现在几点了？杭州天气怎么样？",
        "帮我算一下 158 * 347 等于多少",
    ]

    for q in questions:
        print(f"\n用户：{q}")
        print("-" * 40)
        result = agent.invoke({"input": q})
        print(f"\nAgent：{result['output']}")

    print("\n" + "=" * 60)
    print("✅ 本地 Agent 演示完成！")
    print()
    print("💡 关键区别（vs 云端版本）：")
    print("  云端: base_url=https://api.openai.com/v1")
    print("  本地: base_url=http://localhost:11434/v1")
    print("  其他代码完全一样！")
    print()
    print("📊 本地模型的优缺点：")
    print("  ✅ 数据不出本机，完全私密")
    print("  ✅ 无 API 费用，随便用")
    print("  ✅ 离线可用")
    print("  ❌ 推理能力弱于 GPT-4")
    print("  ❌ 复杂任务可能出错")
    print()
    print("💡 策略：本地模型做调度 + 工具做重活")
    print("=" * 60)
