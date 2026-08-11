"""
第 4 讲：Agent 的知识库 —— Agentic RAG 与 Skill
===================================================
1. 用 Chroma 构建本地旅游攻略知识库（RAG）
2. 定义 Skill 专业知识包（规划方法论、美食指南）
3. 把知识库检索 + Skill 加载包装为 Agent 工具
4. Agent 自主决定何时检索事实、何时加载方法论
5. Tool vs Skill vs RAG：三者分工明确
"""

import json
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# 1. 初始化 Chroma 向量数据库
# ============================================================

# 持久化存储到本地目录
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# Embedding 模型：优先用 API，不可用时自动降级为本地模型
# Agnes-AI / DeepSeek 不提供 embedding 接口，需降级到本地 sentence-transformers
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

try:
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )
    # 验证 API 是否可用（一次轻量调用）
    embedding_fn(["test"])
    print(f"✅ 使用 API Embedding 模型: {EMBEDDING_MODEL}")
except Exception:
    print(f"⚠️  API Embedding 模型 {EMBEDDING_MODEL} 不可用，降级为本地模型")
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    print("✅ 使用本地 Embedding 模型: all-MiniLM-L6-v2")
    # 如果之前用 API embedding 创建过集合，需删除重建（embedding 函数不兼容）
    try:
        chroma_client.delete_collection("travel_knowledge")
        print("🔄 已清理旧集合（API embedding → 本地 embedding）")
    except Exception:
        pass

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
# 3. 定义 Skill（专业知识包）
# ============================================================
# Skill 和 Tool 的区别：
#   Tool = 执行动作的函数（查天气、搜航班）
#   Skill = 提示词驱动的专业知识包（告诉模型"怎么思考"）
# 两者都通过 Tool 接口暴露给 Agent，但本质不同。

SKILLS = {
    "travel_planning": """
【旅行规划专家指南】
1. 行程节奏：一天不超过 3 个核心景点，景点间交通控制在 1 小时内
2. 偏好匹配：优先考虑用户偏好（自然风光 vs 人文历史 vs 美食购物）
3. 预算估算：交通 + 住宿 + 门票 + 餐饮，预留 10% 机动预算
4. 季节适配：8 月杭州炎热多雨（28-35°C），户外景点安排在上午，下午准备室内备选
5. 推荐策略：给出"首选"和"备选"两套方案，说明各自的取舍
""",
    "foodie_guide": """
【美食探店指南】
1. 本地特色优先：推荐当地独有菜品，避开全国连锁
2. 分层推荐：高端宴请（人均 150+）、本地老字号（人均 60-120）、街边小吃（人均 <50）
3. 地理就近：推荐的餐厅应尽量靠近用户当天的行程路线
4. 避坑提醒：知名但不值得排长队的店要诚实标注
""",
}

# ============================================================
# 4. 将 Chroma 检索包装为 Agent 工具
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


@tool
def load_skill(skill_name: str) -> str:
    """
    加载专业领域知识/技能包。当需要特定领域的专业指导、方法论或判断规则时使用。
    skill_name: 技能名称，可选 'travel_planning'（旅行规划）、'foodie_guide'（美食探店）
    """
    return SKILLS.get(skill_name, f"未找到技能'{skill_name}'。可用技能：{', '.join(SKILLS.keys())}")


# ============================================================
# 5. 系统提示词（含自检指令）
# ============================================================

SYSTEM_PROMPT = """你是杭州旅游助手。你可以使用以下工具：

- search_travel_knowledge：搜索杭州旅游攻略（景点、美食、交通、住宿等事实信息）
- load_skill：加载专业领域知识包（规划方法论、判断规则等指导性知识）

工作规则：
1. 涉及杭州景点、美食、交通、住宿、行程的问题，必须先调用 search_travel_knowledge 检索
2. 需要做行程规划、预算评估、综合推荐时，先调用 load_skill 加载专业指南
3. 检索结果不充分时，尝试用不同的关键词再查一次
4. 如果多次检索仍无满意结果，诚实告知用户并给出通用建议
5. 回答时引用具体信息并说明来源
6. 不要编造景点信息或价格
"""

# ============================================================
# 6. 创建 Agent
# ============================================================

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

graph = create_agent(
    model=llm,
    tools=[search_travel_knowledge, load_skill],
    system_prompt=SYSTEM_PROMPT,
)

# ============================================================
# 7. 运行
# ============================================================

if __name__ == "__main__":
    questions = [
        "西湖有什么好玩的？要门票吗？",
        "帮我推荐杭州三天行程，我8月份去，喜欢自然风光",
        "杭州有什么必吃的美食？去哪家餐厅比较好？",
        "帮我规划一个杭州两日游，预算1500，我喜欢深度人文体验，不喜欢走马观花",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"用户：{q}")
        print("=" * 60)
        result = graph.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"\n{result['messages'][-1].content}\n")
