import asyncio
import os
import sys

from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

# 加载 .env 文件中的变量到环境变量中
load_dotenv()

API_KEY = os.environ["API_KEY"]
# 关闭默认的 OpenAI tracing 上报
set_tracing_disabled(True)

# 统一用 DeepSeek 模型，通过 LiteLLM 接入
model = LitellmModel(model="deepseek/deepseek-chat", api_key=API_KEY)

# 计费专员
billing_agent = Agent(
    name="Billing",
    handoff_description="处理扣款、退款、订阅相关问题",
    instructions="你是计费专员，只回答付款、退款、订阅相关问题，回答简短具体。",
    model=model,
)

# 技术支持
technical_agent = Agent(
    name="Technical",
    handoff_description="处理 APP 故障、登录问题、bug 反馈",
    instructions="你是技术支持，只回答 APP 使用、bug 排查问题，最多给 3 条排查建议，每条一句话。",
    model=model,
)

# 分流入口
triage_agent = Agent(
    name="Triage",
    instructions=(
        "你是客服调度员，根据用户问题类型把请求转交给合适的同事："
        "计费相关交给 Billing，技术问题交给 Technical。"
    ),
    handoffs=[billing_agent, technical_agent],
    model=model,
)

async def main():
    # 从命令行参数读用户问题
    question = sys.argv[1]
    print(f"用户: {question}")
    result = await Runner.run(triage_agent, question)
    # last_agent 是最终接管对话的那个 Agent
    print(f"接管者: {result.last_agent.name}")
    print(f"回复: {result.final_output}")

asyncio.run(main())