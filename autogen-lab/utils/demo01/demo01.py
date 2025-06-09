from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
from dotenv import load_dotenv
import os
import random

load_dotenv()  
api_key = os.environ["OPENAI_API_KEY"]

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
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


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.

async def get_humidity(location: str) -> str:
    humidity = random.randint(40, 80)  # 随机湿度40%-80%
    status = "舒适" if humidity < 60 else "潮湿"
    return f"{location}当前空气湿度：{humidity}%（{status}）"

async def get_weather(city: str) -> str:
    weather_types = ["晴天", "多云", "小雨", "阴天"]
    temperature = random.randint(15, 35)  # 随机温度15-35℃
    return f"{city}天气预报：{random.choice(weather_types)}，{temperature}℃"

# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_humidity ,get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(agent.run_stream(task="杭州今天天气怎么样, 要戴口罩吗?"))
    # Close the connection to the model client.
    await model_client.close()


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).


asyncio.run(main())
