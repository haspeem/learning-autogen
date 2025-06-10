from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()  
api_key = os.environ["OPENAI_API_KEY"]  
model_client = OpenAIChatCompletionClient(
    model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        base_url="https://api.siliconflow.cn/v1",
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
assistant = AssistantAgent("assistant", model_client=model_client)
user_proxy = UserProxyAgent("user_proxy", input_func=input) 

# Create the termination condition which will end the conversation when the user says "APPROVE".
termination = TextMentionTermination("APPROVE")

# Create the team.
team = RoundRobinGroupChat([assistant, user_proxy], termination_condition=termination)

# Run the conversation and stream to the console.
stream = team.run_stream(task="Write a 4-line poem about the ocean.")
# Use asyncio.run(...) when running in a script.

async def main() -> None:
    await Console(stream)
    # Close the connection to the model client.
    await model_client.close()

asyncio.run(main())

