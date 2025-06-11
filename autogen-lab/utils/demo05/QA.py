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
before_agnet = AssistantAgent(
    name="before_agnet",
    model_client=model_client,
    description="是一个服装店的售前客服",
    system_message="""
    你是一个服装店的售前客服, 你的目标是帮助用户了解服装, 包括材料柔软度,售价和销售量, 完成推销
    """
)

after_agent = AssistantAgent(
    name="after_agent",
    model_client=model_client,
    description="这是一个服装店的售后客服",
    reflect_on_tool_use = True,
    system_message="""
    你是一个服装店的售后客服, 你的目标是帮助用户解决使用上遇到的一些问题, 以及一些退换货请求, 解决用户的问题
    """ 
)


# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("完毕")        #检测到完毕时停止

selector_prompt = """选择一个agent去完成任务, 这是agent的职责, after_agent和before_agent回答完后需要获得用户的反馈

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
    [after_agent, before_agnet, user_proxy],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,  
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
    with open("learning-autogen/autogen-lab/utils/coding/team_state1.json", "r") as f:
        team_state = json.load(f)
    await team.load_state(team_state)
    stream = team.run_stream(task="我的上一个问题是什么")
    await Console(stream)
    team_state = await team.save_state()
    with open("learning-autogen/autogen-lab/utils/coding/team_state1.json", "w") as f:
        json.dump(team_state, f)

    

    # Save the state of the agent team.
    # Load team state.
    


asyncio.run(main())
