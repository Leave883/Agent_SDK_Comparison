import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from dotenv import load_dotenv

# 加载 .env 文件中的变量到环境变量中
load_dotenv()

# DeepSeek 提供 OpenAI 兼容接口，直接用 ChatOpenAI 改 base_url 即可
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    api_key=os.environ["API_KEY"],
)

# 整张图共享的状态，LangGraph 中的各个节点通过一个共享的 state 来传数据
class State(TypedDict):
    topic: str       # 邮件主题
    draft: str       # 当前草稿
    feedback: str    # 上一轮的修改意见
    approved: bool   # 是否通过审批

# 节点 1：根据主题（和上一轮反馈）生成邮件草稿
def write_email(state: State) -> dict:
    if state.get("feedback"):
        # 已有上一版草稿和修改意见，基于原文做修订
        prompt = (
            f"下面是上一版邮件草稿：\n{state['draft']}\n\n"
            f"修改意见：{state['feedback']}\n\n"
            "请基于上面的草稿和意见改写，要求 3 句话以内。"
        )
    else:
        # 首次生成
        prompt = f"写一封简短的邮件，主题：{state['topic']}。要求：3 句话以内。"
    resp = llm.invoke(prompt) # 调用模型
    return {"draft": resp.content}

# 节点 2：暂停等人工审批
# 人工要么打 approve 通过，要么给修改意见让模型重写
def human_review(state: State) -> dict:
    decision = interrupt({"draft": state["draft"]}) # 停止
    if decision == "approve":
        return {"approved": True}
    return {"feedback": decision, "approved": False}

# 条件边：通过则结束，否则回到 write_email 重写
def route_after_review(state: State) -> str:
    return END if state["approved"] else "write_email"

# 创建 LangGraph
builder = StateGraph(State)
# 创建图节点
# START 和 END 是 LangGraph 预定义的特殊节点，分别表示图的开始和结束，不需要我们手动创建
builder.add_node("write_email", write_email)
builder.add_node("human_review", human_review)

# 创建边连接节点
# 从 START 节点首先走到 write_email 节点
builder.add_edge(START, "write_email")
# 从 write_email 节点走到 human_review 节点
builder.add_edge("write_email", "human_review")
# 从 human_review 节点根据 route_after_review 函数的返回值
# 决定下一步走向哪个节点
builder.add_conditional_edges("human_review", route_after_review)

# InMemorySaver 保存执行进度，让 interrupt 暂停后能恢复
# checkpointer 可能同时保存很多流程，thread_id 告知需要恢复的是哪一个
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "email-1"}}

# 第一次调用，跑到 interrupt 处会自动暂停
# 通过 result["__interrupt__"] 可以获取到模型输出的 draft 字段
result = graph.invoke({"topic": "祝团队周末愉快"}, config=config)
print("=== 第一版草稿 ===")
print(result["__interrupt__"][0].value["draft"])

# 模拟人工反馈，要求改口语化
result = graph.invoke(Command(resume="太正式了，改得更口语化、更轻松"), config=config)
print("\n=== 第二版草稿（按反馈重写）===")
print(result["__interrupt__"][0].value["draft"])

# 模拟人工通过
result = graph.invoke(Command(resume="approve"), config=config)
print("\n=== 已通过，最终邮件 ===")
print(result["draft"])