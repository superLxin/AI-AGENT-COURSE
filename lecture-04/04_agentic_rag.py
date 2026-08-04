"""
第 4 讲：Agent 的知识库 —— Agentic RAG
=========================================
1. 用 Chroma 构建本地旅游攻略知识库
2. 把知识库检索包装为 Agent 工具
3. Agent 自主决定何时检索、如何改写查询
"""

import json
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# 1. 初始化 Chroma 向量数据库
# ============================================================

# 持久化存储到本地目录
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# 使用 OpenAI Embedding（也支持免费的本地模型）
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

# 创建或获取集合
collection = chroma_client.get_or_create_collection(
    name="travel_knowledge",
    embedding_function=embedding_fn,
)

# ============================================================
# 2. 导入旅游攻略文档
# ============================================================

TRAVEL_DOCUMENTS = [
    "西湖是杭州最著名的景点，免费开放。建议游玩时间半天到一天。最佳游览季节是春秋两季。雷峰塔门票40元。",
    "西溪湿地门票80元，以自然生态闻名，适合徒步和观鸟。建议游玩时间半天。距离市中心约10公里。",
    "灵隐寺始建于东晋咸和元年（公元326年），门票45元，是杭州最古老的佛教寺庙。飞来峰造像群是国保单位。",
    "龙井村位于西湖西南，是中国十大名茶龙井茶的原产地。可以免费参观茶园，体验采茶和品茶。最佳季节是春季（3-5月）。",
    "杭州美食推荐：西湖醋鱼、龙井虾仁、东坡肉、叫花鸡、片儿川。推荐餐厅：楼外楼（西湖边，人均150）、知味观（多家分店，人均80）。",
    "杭州交通：地铁覆盖主要景区，一号线可到西湖和龙翔桥。公交2元起步。打车起步价14元。推荐使用支付宝扫码乘车。",
    "杭州住宿建议：西湖周边酒店价格较高但风景好（400-800元/晚），市区商务酒店性价比高（200-400元/晚）。旅游旺季（4-5月、9-10月）建议提前1-2周预订。",
    "杭州三天经典行程：Day1 西湖环湖（断桥→白堤→孤山→雷峰塔），Day2 灵隐寺+龙井村+九溪烟树，Day3 西溪湿地或宋城。",
    "杭州8月天气炎热多雨，平均温度28-35°C，常有午后雷阵雨。出行建议带伞和防晒，选择清晨和傍晚户外活动。",
    "宋城景区：大型主题公园，门票320元（含宋城千古情演出），以宋代文化为主题。建议游玩半天到一天。距离市中心约15公里。",
]

# 检查是否已导入过（避免重复导入）
existing = collection.get()
if len(existing["ids"]) == 0:
    collection.add(
        documents=TRAVEL_DOCUMENTS,
        ids=[f"doc_{i}" for i in range(len(TRAVEL_DOCUMENTS))],
    )
    print(f"✅ 已导入 {len(TRAVEL_DOCUMENTS)} 条旅游攻略文档\n")
else:
    print(f"✅ 知识库中已有 {len(existing['ids'])} 条文档\n")

# ============================================================
# 3. 将 Chroma 检索包装为 Agent 工具
# ============================================================


@tool
def search_travel_knowledge(query: str) -> str:
    """
    搜索杭州旅游攻略知识库。
    当用户询问杭州旅游相关的景点、美食、交通、行程、住宿问题时使用。
    query: 自然语言搜索查询
    """
    results = collection.query(query_texts=[query], n_results=3)
    documents = results["documents"][0]
    distances = results.get("distances", [[1, 1, 1]])[0]

    # 格式化检索结果
    output = []
    for i, (doc, dist) in enumerate(zip(documents, distances)):
        relevance = "★★★★★" if dist < 1.0 else "★★★☆☆"
        output.append(f"[相关度:{relevance}] {doc}")

    return "\n\n".join(output)


# ============================================================
# 4. 系统提示词（含自检指令）
# ============================================================

SYSTEM_PROMPT = """你是杭州旅游助手。你可以使用 search_travel_knowledge 工具来查询旅游攻略。

工作规则：
1. 涉及杭州景点、美食、交通、住宿、行程的问题，必须先调用 search_travel_knowledge 检索
2. 检索结果不充分时，尝试用不同的关键词再查一次
3. 如果多次检索仍无满意结果，诚实告知用户并给出通用建议
4. 回答时引用具体信息并说明来源
5. 不要编造景点信息或价格
"""

# ============================================================
# 5. 创建 Agent
# ============================================================

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

agent = create_tool_calling_agent(llm, [search_travel_knowledge], SYSTEM_PROMPT)
executor = AgentExecutor(agent=agent, tools=[search_travel_knowledge], verbose=True)

# ============================================================
# 6. 运行
# ============================================================

if __name__ == "__main__":
    questions = [
        "西湖有什么好玩的？要门票吗？",
        "帮我推荐杭州三天行程，我8月份去，喜欢自然风光",
        "杭州有什么必吃的美食？去哪家餐厅比较好？",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"用户：{q}")
        print("=" * 60)
        result = executor.invoke({"input": q})
        print(f"\n{result['output']}\n")
