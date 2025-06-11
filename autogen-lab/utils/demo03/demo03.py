import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from autogen_agentchat.messages import BaseAgentEvent
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

user_proxy = UserProxyAgent("user_proxy", input_func=input) 
# 创建收集代理
collect_agent = AssistantAgent(
    name="collect_agent",
    model_client=model_client,
    system_message="""
    你是日报信息收集助手，你的任务是：
    1. 引导用户完成信息收集
    2. 要求用户以(时间, 人物, 地点, 行为, 事件)的格式收集信息
    3. 将收集到的完整信息交给generate_agent
    """
)

# 创建生成代理
generate_agent = AssistantAgent(
    name="generate_agent",
    model_client=model_client,
    system_message="""
    你是日报生成专家，你的任务是：
    1. 根据收集到的用户信息生成结构化的日报 
    2. 使用generate_daily_report函数创建标准日报
    3. 将标准日报交给style_agent进行风格转换
    """
)

# 创建风格转换代理
style_agent = AssistantAgent(
    name="style_agent",
    model_client=model_client,
    system_message='''
    根据`generate_xiaohongshu_report`函数的信息将标准日报转换为小红书风格
    **执行流程**：
    1. 接收用户输入的标准日报
    2. 生成小红书风格的日报
    3. 继续让collect_agent收集信息
    '''
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("完毕")        #检测到完毕时停止

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([collect_agent, generate_agent, style_agent, user_proxy], termination_condition=text_termination)

mention = BaseAgentEvent
async def main() -> None:
    await Console(team.run_stream(task="Report the user's experiences and output them in Xiaohongshu (Little Red Book) format."))
    # Close the connection to the model client.
    await model_client.close()

asyncio.run(main())