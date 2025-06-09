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
model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", # 必须与官方给的模型名称一致
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

# 创建收集代理
collect_agent = AssistantAgent(
    name="collect_agent",
    model_client=model_client,
    system_message="""
    你是日报信息收集助手，你的任务是：
    1. 引导用户完成信息收集
    2. 要求用户给出时间, 任务, 事件等信息
    3. 使用collect_user_info函数获取用户数据
    4. 将收集到的完整信息交给generate_agent
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
    system_message="""
    你是小红书内容创作专家，你的任务是：
    1. 使用generate_xiaohongshu_report函数将标准日报转换为小红书风格
    2. 确保内容轻松活泼、带emoji和有吸引力的标题
    3. 添加5个相关话题标签
    4. 当你认为条件够了的时候在答案的最后加上完毕
    """
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("完毕")        #检测到approve时停止

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([collect_agent, generate_agent, style_agent], termination_condition=text_termination)

async def main() -> None:
    await Console(team.run_stream(task="Report the user's experiences and output them in Xiaohongshu (Little Red Book) format."))
    # Close the connection to the model client.
    await model_client.close()

asyncio.run(main())