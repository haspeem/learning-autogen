from typing import Any, Dict, List
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
load_dotenv()  
api_key = os.environ["OPENAI_API_KEY"]  
model_client = OpenAIChatCompletionClient(
    model="Qwen/Qwen3-30B-A3B",
        base_url="https://api.siliconflow.cn/v1",
        api_key=api_key,
        model_info = {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "qwen",
            "structured_output": True,
            "multiple_system_messages": False,
        }
)


#完成一个消费者热线投诉服务平台系统

#  user投诉  --> 消费者投诉平台 --> APP --- > 调用工具(客服介入)

def tuikuan(Transaction_Number: str) -> str:
    """退款工具"""
    return f"关于编号为 {Transaction_Number} 的这项交易, 客服已介入已经要求退款并道歉"


def tuohuo(Transaction_Number: str) -> str:
    """"""
    return f"关于编号为 {Transaction_Number} 的这项交易, 客服已介入已经要求退货并道歉"



Customer_service = AssistantAgent(          #消费者热线投诉服务平台
    "Customer_service",
    model_client=model_client,
    handoffs=["APP_leader", "user"],
    system_message="""
        你是消费者服务平台，刚刚收到user的投诉, 你要帮助用户解决问题
        如果你需要user的更多信息必须先发送信息，然后才能转交给user。
        请以简洁、礼貌的语言表示对user的歉意，如果你需要user提供信息, 你需要转交给user
        当你觉得信息足够了, 转交给App_leader
        当你认为服务结束了使用'很高兴为您服务'来终止对话
        """,
)

APP_leader = AssistantAgent(
    "APP_leader",
    model_client=model_client,
    handoffs=["user"],
    tools=[tuikuan],
    system_message="""
        你是消费 APP 的客服，你需要解决消费者服务平台给出的的投诉问题。
        如果你需要与user进行沟通, 你必须先发送信息，然后才能转交给user。
        你必须要先知道user的交易编号以及退款原因,才能调用工具解决问题
        当你认为服务结束了输出'很高兴为您服务'来终止对话
        """,
)

termination = HandoffTermination(target="user") | TextMentionTermination("很高兴为您服务")
team = Swarm([Customer_service, APP_leader], termination_condition=termination)
task = "我上个星期在拼多多平台买了一件衬衫, 昨天到货了, 但是我觉得衣服尺寸太小了, 我想要退货, 但是商家拒绝了, 我要投诉这个商家, 商家的名称叫'拼衣服'"

async def run_team_stream() -> None:
    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]     # 获取最后一条消息

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]        



async def main() -> None:
    await run_team_stream()
    
    await model_client.close()

asyncio.run(main())