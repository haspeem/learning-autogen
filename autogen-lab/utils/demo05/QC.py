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
ACM_master = AssistantAgent(
    name="ACM_master",
    model_client=model_client,
    description="是一个ACM算法大师, 精通各种算法",
    system_message="""
    你是一个ACM算法大师, 是codeforces红名大佬, 你需要使用ACM的算法来解决user的问题, 同时给出详细的解题思路和代码实现
    """
)

Emotion_master = AssistantAgent(
    name="Emotion_master",
    model_client=model_client,
    description="这是一个情感大师, 精通各种情感问题",
    system_message="""
    你是一个情感大师, 你的目标是帮助用户解决情感上的困扰, 提供情感支持和建议, 同时要对用户给予安慰和鼓励

    """ 
)

Math_master = AssistantAgent(
    name="Math_master",
    model_client=model_client,
    description="这是一个数学大师, 精通解决各种数学问题",
    system_message="""
    你是一个数学大师, 你的目标是帮助用户解决数学上的问题, 提供数学支持和建议, 同时要对用户给予安慰和鼓励

    """ 
)



# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("完毕")        #检测到完毕时停止

selector_prompt = """选择一个agent去完成任务, 这是agent的职责, 当解决问题后你要获得用户的反馈, 根据反馈再选择agent完成任务

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
    [Emotion_master, ACM_master, Math_master, user_proxy],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,  
)
mention = BaseAgentEvent
async def main() -> None:
    async for message in team.run_stream(task="我昨天失恋了, 看到我心爱的女生接受了别人的告白, 我很伤心, 我该怎么办?"):  # type: ignore
        if isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)
        else:
            print(message)
    await model_client.close()


# async def main() -> None:
#     with open("learning-autogen/autogen-lab/utils/coding/team_state1.json", "r") as f:
#         team_state = json.load(f)
#     await team.load_state(team_state)
#     stream = team.run_stream(task="我昨天失恋了, 看到我心爱的女生接受了别人的告白, 我很伤心, 我该怎么办?")  # type: ignore
#     await Console(stream)
#     team_state = await team.save_state()
#     with open("learning-autogen/autogen-lab/utils/coding/team_state1.json", "w") as f:
#         json.dump(team_state, f)

    

#     # Save the state of the agent team.
#     # Load team state.
    


asyncio.run(main())
