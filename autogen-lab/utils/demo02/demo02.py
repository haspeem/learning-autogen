import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
load_dotenv()  
api_key = os.environ["OPENAI_API_KEY"]
model_client = OpenAIChatCompletionClient(
model="deepseek-ai/DeepSeek-V2.5", # 必须与官方给的模型名称一致
    base_url="https://api.siliconflow.cn/v1", # 调用API地址
    api_key=api_key,
    model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "deepseek",
        "structured_output": True,
        "multiple_system_messages": False,
    }
)

# Create the primary agent.                 
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent.                  
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",   #提示ai使用approve
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")        #检测到approve时停止

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

async def main() -> None:
    await Console(team.run_stream(task="Write a short poem about the fall season."))
    # Close the connection to the model client.
    await model_client.close()

asyncio.run(main())