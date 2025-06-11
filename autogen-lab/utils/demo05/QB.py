import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseAgentEvent
from autogen_agentchat.teams import SelectorGroupChat
import json
import os
load_dotenv()  

api_key = os.environ["OPENAI_API_KEY"]
model_client = OpenAIChatCompletionClient(
    model = "Qwen/Qwen3-30B-A3B",   
    api_key = api_key,
    base_url = "https://api.siliconflow.cn/v1/",
    model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "qwen",
        "structured_output": True,
        "multiple_system_messages": True,
    }
)

user_proxy = UserProxyAgent("user_proxy", input_func=input, description="是获取用户反馈的工具") 
# 创建收集代理
Fst_agent = AssistantAgent(
    name="Fst_agnet",
    model_client=model_client,
    description="辩论的正方发言",
    reflect_on_tool_use = True,
    system_message="""
    你需要围绕用户的议题进行辩论, 你作为辩论的正方提供证据
    ​立论举证​​：围绕议题提出支持己方的核心论点，并附带​​实证​​（数据/案例/权威来源）。
    ​主动驳斥​​：针对反方的每个观点，​​即时识别逻辑漏洞或证据弱点​​，提出反驳。
    ​攻守兼备​​：防守时——化解反方对己方论点的质疑；进攻时——揭露反方论据的不合理之处
    """
)

Sed_agent = AssistantAgent(
    name="Sed_agnet",
    model_client=model_client,
    description="辩论的反方",
    reflect_on_tool_use = True,
    system_message="""
    你需要围绕用户的议题进行辩论, 你作为辩论的反方提供证据
    ​​挑战质疑​​：直接针对正方论点提出​​针对性质疑​​，指出证据不足或逻辑矛盾。
    ​反向举证​​：提出反对议题的独立论据，并用​​反例或数据​​强化立场。
    ​压制推进​​：持续迫使正方回应关键质疑，同时避免己方立场被正方带偏
    """ 
)


# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("完毕")        #检测到完毕时停止

selector_prompt = """
选择一个agent去完成任务, 这是agent的职责, Fst_agent是正方辩论的工具, Sed_agent是反方辩论的工具, 你要让他们轮流进行辩论

    {roles}

    这是当前任务的上下文消息:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
"""
termination = TextMentionTermination("已解决")
# Create a team with the primary and critic agents.
team = SelectorGroupChat(
    [Fst_agent, Sed_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False,  
)
mention = BaseAgentEvent
# async def main() -> None:
#     async for message in team.run_stream(task="我上个月买的一件T恤掉色了, 我想要获取赔偿"):  # type: ignore
#         if isinstance(message, TaskResult):
#             print("Stop Reason:", message.stop_reason)
#         else:
#             print(message)
#     await model_client.close()


async def main() -> None:
    stream = team.run_stream(task="钱是王八蛋")
    await Console(stream) 
    # Save the state of the agent team.
    # Load team state.
    


asyncio.run(main())
