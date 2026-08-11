"""
第 8 讲：Agent 的记忆系统
============================
两种实现方式：
1. Chroma 手动记忆管理（提取 → 存储 → 检索 → 注入）
2. Mem0 自动记忆管理（一行代码提取+去重+更新）

让旅行助手记住用户偏好，第二次对话自动引用。
"""

import json
import os
import uuid
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# ============================================================
# 1. 初始化 Chroma 记忆存储
# ============================================================

chroma_client = chromadb.PersistentClient(path="./memory_data")

# Embedding 模型：优先用 API，不可用时自动降级为本地模型
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

try:
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )
    embedding_fn(["test"])
    print(f"✅ 使用 API Embedding 模型: {EMBEDDING_MODEL}")
except Exception:
    print(f"⚠️  API Embedding 模型 {EMBEDDING_MODEL} 不可用，降级为本地模型")
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    print("✅ 使用本地 Embedding 模型: all-MiniLM-L6-v2")
    try:
        chroma_client.delete_collection("user_memories")
        print("🔄 已清理旧集合（API embedding → 本地 embedding）")
    except Exception:
        pass

memory_collection = chroma_client.get_or_create_collection(
    name="user_memories",
    embedding_function=embedding_fn,
)


# ============================================================
# 2. 记忆管理函数
# ============================================================


def extract_memories(conversation_text: str, user_id: str) -> list[str]:
    """从对话中自动提取值得长期记住的信息"""
    prompt = f"""从以下对话中提取值得长期记住的、关于用户的偏好或事实信息。

对话内容：
{conversation_text}

提取规则：
1. 只提取可复用的偏好、习惯、事实（如"用户喜欢靠窗座位"、"用户预算上限5000元"）
2. 不要提取一次性信息（如"明天要下雨"、"CA1234航班价格580元"）
3. 每条记忆用第三人称简短描述
4. 如果对话中没有值得长期记住的信息，返回空数组 []

返回JSON格式：["记忆1", "记忆2", ...]
"""
    response = llm.invoke(prompt)
    try:
        memories = json.loads(response.content)
        return memories if isinstance(memories, list) else []
    except json.JSONDecodeError:
        return []


def save_memories(memories: list[str], user_id: str):
    """将记忆存入 Chroma（带去重）"""
    for mem in memories:
        existing = memory_collection.query(
            query_texts=[mem],
            where={"user_id": user_id},
            n_results=1,
        )
        if existing["distances"] and existing["distances"][0] and existing["distances"][0][0] < 0.3:
            old_id = existing["ids"][0][0]
            memory_collection.update(ids=[old_id], documents=[mem])
            print(f"  🔄 更新记忆: {mem}")
        else:
            memory_id = f"mem_{user_id}_{uuid.uuid4().hex[:8]}"
            memory_collection.add(
                documents=[mem],
                metadatas=[{"user_id": user_id, "timestamp": datetime.now().isoformat()}],
                ids=[memory_id],
            )
            print(f"  💾 新增记忆: {mem}")


def retrieve_memories(user_input: str, user_id: str, n: int = 5) -> list[str]:
    """检索与当前对话相关的用户记忆"""
    results = memory_collection.query(
        query_texts=[user_input],
        where={"user_id": user_id},
        n_results=n,
    )
    if results["documents"] and results["documents"][0]:
        return results["documents"][0]
    return []


def delete_all_memories(user_id: str):
    """遗忘权：删除用户的所有记忆"""
    existing = memory_collection.get(where={"user_id": user_id})
    if existing["ids"]:
        memory_collection.delete(ids=existing["ids"])
        print(f"  🗑️  已删除用户 {user_id} 的所有记忆")


# ============================================================
# 3. 旅行工具
# ============================================================


@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴，28°C"


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """搜索航班"""
    flights = [
        {"flight": "CA1234", "departure": "08:00", "arrival": "10:30", "price": 580, "seat_type": "靠窗"},
        {"flight": "MU5678", "departure": "14:00", "arrival": "16:30", "price": 720, "seat_type": "过道"},
    ]
    return json.dumps(flights, ensure_ascii=False)


# ============================================================
# 4. 创建带记忆的 Agent
# ============================================================


def create_agent_with_memory(user_id: str):
    """为指定用户创建带记忆注入的 Agent"""
    base_prompt = "你是一个旅行助手。你可以使用工具来查询天气和航班信息。回答时参考用户偏好，推荐时说明理由。"
    return create_agent(
        model=llm,
        tools=[get_weather, search_flights],
        system_prompt=base_prompt,
    )


# ============================================================
# 5. 对话引擎
# ============================================================


def chat_with_memory(user_input: str, user_id: str, graph) -> str:
    """一次完整的对话 + 记忆提取 + 记忆注入"""

    # 1. 检索相关记忆
    memories = retrieve_memories(user_input, user_id)

    # 2. 将记忆注入用户输入
    if memories:
        memory_context = "【关于该用户的已知偏好】\n" + "\n".join(f"- {m}" for m in memories)
        enhanced_input = f"{memory_context}\n\n【当前问题】{user_input}"
        print(f"  🧠 注入记忆: {memories}")
    else:
        enhanced_input = user_input
        print("  🧠 无相关记忆")

    # 3. 执行 Agent
    result = graph.invoke({"messages": [{"role": "user", "content": enhanced_input}]})
    answer = result["messages"][-1].content

    # 4. 从本轮对话中提取新记忆
    conversation = f"用户: {user_input}\n助手: {answer}"
    new_memories = extract_memories(conversation, user_id)
    if new_memories:
        save_memories(new_memories, user_id)

    return answer


# ============================================================
# 6. 运行
# ============================================================

if __name__ == "__main__":
    USER_ID = "user_demo"
    agent = create_agent_with_memory(USER_ID)

    print("=" * 60)
    print("第 1 轮对话：用户表露偏好")
    print("=" * 60)
    answer = chat_with_memory(
        "帮我查一下8月10日北京到杭州的航班。对了，我喜欢早上的航班，偏好靠窗座位。",
        USER_ID, agent,
    )
    print(f"\nAgent：{answer}\n")

    print("=" * 60)
    print("第 2 轮对话：Agent 自动引用记忆")
    print("=" * 60)
    answer = chat_with_memory(
        "再帮我查一次8月15日北京到杭州的航班",
        USER_ID, agent,
    )
    print(f"\nAgent：{answer}\n")

    # 查看已存储的记忆
    print("=" * 60)
    print("📋 用户记忆存档：")
    all_memories = memory_collection.get(where={"user_id": USER_ID})
    for doc in all_memories.get("documents", []):
        print(f"  - {doc}")

    # 演示遗忘权
    print("\n演示遗忘权...")
    delete_all_memories(USER_ID)
