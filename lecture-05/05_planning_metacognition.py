"""
第 5 讲：Agent 的思考方式 —— 规划与元认知
=============================================
用 LangGraph 实现 Plan → Execute → Evaluate → Loop 三段式循环。
Agent 先规划分步计划，再执行，最后自我评估，不满意则重试。
"""

import json
import os
from typing import TypedDict, List, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# ============================================================
# 1. 定义结构化输出（规划结果用 Pydantic 约束格式）
# ============================================================


class PlanStep(BaseModel):
    step: int
    task: str
    reason: str


class TravelPlan(BaseModel):
    goal: str
    steps: List[PlanStep]


structured_llm = llm.with_structured_output(TravelPlan)

# ============================================================
# 2. 定义 Agent 状态
# ============================================================


class AgentState(TypedDict):
    user_input: str
    plan: List[dict]
    step_results: List[str]
    current_step: int
    final_answer: str
    retry_count: int


# ============================================================
# 3. Planner 节点：拆解任务
# ============================================================


def planner(state: AgentState) -> dict:
    """将用户需求拆解为分步执行计划"""
    print("  🧠 [Planner] 正在分析任务...")

    prompt = f"""你是一个旅行规划专家。用户的需求是：
{state["user_input"]}

请制定一个分步执行计划。每个步骤应该是一个独立的查询任务。
步骤示例："查询目的地天气"、"搜索合适的航班"、"推荐行程安排"。
"""
    plan = structured_llm.invoke(prompt)
    steps = [s.dict() for s in plan.steps]

    print(f"  📋 制定了 {len(steps)} 步计划：")
    for s in steps:
        print(f"     {s['step']}. {s['task']}（{s['reason']}）")

    return {"plan": steps, "current_step": 0, "step_results": [], "retry_count": 0}


# ============================================================
# 4. Executor 节点：执行每个步骤
# ============================================================


def executor(state: AgentState) -> dict:
    """模拟执行计划中的每个步骤"""
    step_idx = state["current_step"]
    step = state["plan"][step_idx]

    print(f"  ⚡ [Executor] 执行第{step['step']}步: {step['task']}")

    # 模拟执行（实际项目中将不同类型的步骤路由到对应的工具）
    result = llm.invoke(
        f"模拟执行以下任务（生成合理的模拟结果）：{step['task']}\n"
        f"原始用户需求：{state['user_input']}"
    )

    new_results = state["step_results"] + [f"[{step['task']}] {result.content}"]
    new_step = step_idx + 1

    return {"step_results": new_results, "current_step": new_step}


def should_continue_execution(state: AgentState) -> str:
    """判断是否还有步骤需要执行"""
    if state["current_step"] < len(state["plan"]):
        return "continue"
    return "evaluate"


# ============================================================
# 5. Evaluator 节点：评估 + 元认知自检
# ============================================================


def evaluator(state: AgentState) -> dict:
    """评估执行结果是否充分，不足则触发重试"""
    print("  🔍 [Evaluator] 正在评估结果...")

    summary = "\n".join(state["step_results"])

    eval_prompt = f"""请评估以下旅行规划任务的执行结果：

用户需求：{state["user_input"]}

执行结果：
{summary}

请做以下自检：
1. 是否覆盖了用户的所有需求？
2. 信息是否具体可用？（有具体的航班、酒店、景点名称和价格）
3. 是否有逻辑矛盾？（如推荐的户外活动在下雨天）
4. 如果有遗漏，明确指出需要补充什么。

如果结果充分，请用"## 旅行方案"开头，综合所有信息生成最终回答。
如果有重大遗漏，请说明"需要补充：[具体内容]"。
"""
    response = llm.invoke(eval_prompt)
    content = response.content

    print(f"  📊 评估结果（前200字）：{content[:200]}...")

    # 判断是否需要重试
    need_retry = "需要补充" in content and state.get("retry_count", 0) < 2

    if need_retry:
        new_retry_count = state.get("retry_count", 0) + 1
        print(f"  🔄 信息不充分，第{new_retry_count}次重试...")
        return {
            "final_answer": content,
            "retry_count": new_retry_count,
        }

    return {"final_answer": content}


def should_retry(state: AgentState) -> str:
    """判断是否需要重试"""
    if "需要补充" in state.get("final_answer", "") and state.get("retry_count", 0) < 3:
        return "retry"
    return "finish"


# ============================================================
# 6. 构建 LangGraph 状态图
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("evaluator", evaluator)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    should_continue_execution,
    {"continue": "executor", "evaluate": "evaluator"},
)
workflow.add_conditional_edges(
    "evaluator",
    should_retry,
    {"retry": "executor", "finish": END},
)

app = workflow.compile()

# ============================================================
# 7. 运行
# ============================================================

if __name__ == "__main__":
    query = "我8月10-12日从北京去杭州旅游，预算3000元，喜欢自然风光和美食。帮我规划行程。"

    print(f"用户需求：{query}\n")
    print("=" * 60)

    result = app.invoke({
        "user_input": query,
        "plan": [],
        "step_results": [],
        "current_step": 0,
        "final_answer": "",
        "retry_count": 0,
    })

    print("\n" + "=" * 60)
    print("📝 最终方案：")
    print("=" * 60)
    print(result["final_answer"])
