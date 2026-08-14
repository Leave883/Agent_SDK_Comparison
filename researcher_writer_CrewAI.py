import os

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

# 加载 .env 文件中的变量到环境变量中
load_dotenv()

llm = LLM(model="deepseek/deepseek-chat", api_key=os.environ["API_KEY"])

# 角色 1：研究员
researcher = Agent(
    role="技术研究员",
    goal="搜集某个技术主题的关键事实，整理成简明要点",
    backstory="你是经验丰富的技术研究员，擅长抓住一个主题的本质，不啰嗦。",
    llm=llm,
)

# 角色 2：写作者
writer = Agent(
    role="技术写作者",
    goal="把研究员的要点改写成易懂的短文",
    backstory="你擅长用通俗的语言解释技术概念，文字简洁有趣，避免堆砌术语。",
    llm=llm,
)

# 任务 1：调研
research_task = Task(
    description="调研 HTTP/2 相对 HTTP/1.1 的主要改进，整理 3-5 条要点。",
    expected_output="一个简短的要点列表，每条一句话。",
    agent=researcher,
)

# 任务 2：基于调研写一段短文
writing_task = Task(
    description="基于研究员的要点写一段 150 字以内的通俗介绍，不要列表，用一段流畅文字。",
    expected_output="一段 150 字以内的中文短文。",
    agent=writer,
    # 依赖 task 1 的输出
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
)

crew.kickoff()
print("=== 研究员的要点 ===")
print(research_task.output.raw)
print("\n=== 写作者的成稿 ===")
print(writing_task.output.raw)