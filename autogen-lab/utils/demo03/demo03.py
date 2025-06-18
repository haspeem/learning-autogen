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

#用户代理
user_proxy = UserProxyAgent("user_proxy", input_func=input) 

# 创建收集代理
collect_agent = AssistantAgent(
    name="collect_agent",
    model_client=model_client,
    system_message="""
    你是日报信息收集助手，你的任务是：
    1.询问用户是否对生成的小红书风格日报感兴趣满意
    2. 引导用户完成信息收集
    3. 要求用户以(时间, 人物, 地点, 行为, 事件)的格式收集信息
    """
)

# 创建生成代理
generate_agent = AssistantAgent(
    name="generate_agent",
    model_client=model_client,
    system_message="""
    你是日报生成专家，你的任务是：
    根据收集到的信息生成结构化的日报 
    """
)

# 创建风格转换代理
style_agent = AssistantAgent(
    name="style_agent",
    model_client=model_client,
    system_message='''
    将标准日报转换为小红书风格
    **执行流程**：
        1.接收输出的标准日报
        2.生成小红书风格的日报
        3.结束对话
    '''
)

text_termination = TextMentionTermination("完毕")        #检测到完毕时停止

#RoundRobinGroupChat模式
team = RoundRobinGroupChat([generate_agent, style_agent, collect_agent, user_proxy], termination_condition=text_termination)

mention = BaseAgentEvent
async def main() -> None:
    await Console(team.run_stream(task="""
       我是曹滨, 我上周去了黄山旅游, 这是我的旅游经历     Day 1：抵达黄山，夜宿山脚​​
        早上从​​杭州​​出发，乘坐高铁到​​黄山北站​​（约2小时），再转乘景区大巴直达​​汤口镇​​（黄山脚下）。入住提前预订的民宿，老板热情推荐了登山路线，并帮忙寄存了大件行李（山上住宿条件有限，建议轻装上山）。

        傍晚在汤口镇闲逛，尝了当地特色菜​​臭鳜鱼​​和​​毛豆腐​​，味道独特但值得一试！

        ​​Day 2：登山！迎客松与光明顶​​
        ​​路线​​：

        ​​前山上（玉屏索道）​​ → ​​迎客松​​ → ​​莲花峰​​（黄山最高峰，海拔1864米）→ ​​一线天​​ → ​​光明顶​​（看日落）→ ​​夜宿白云宾馆​​
        ​​体验亮点​​：

        ​​迎客松​​：黄山的标志，比想象中更挺拔，周围挤满拍照的游客。
        ​​莲花峰​​：台阶陡峭，爬得腿抖，但登顶后视野无敌，云海在脚下翻滚。
        ​​一线天​​：仅容一人通过的狭窄石缝，抬头只见“一线”蓝天，刺激！
        ​​日落​​：在光明顶守到傍晚，夕阳把云海染成金色，像泼墨画。
        ​​Tips​​：

        山上物价高（矿泉水15元/瓶），建议自带零食和保温杯（宾馆可接热水）。
        白云宾馆床位紧张，提前一周预订，房间虽小但干净，有暖气。
        ​​Day 3：日出、西海大峡谷与下山​​
        ​​凌晨4:30​​摸黑出发，跟着人群到​​光明顶​​等日出。清晨山顶寒风刺骨，但看到太阳从云海中跃出的瞬间，一切值得！

        ​​下山路线​​：

        ​​光明顶​​ → ​​飞来石​​（《红楼梦》取景地）→ ​​西海大峡谷​​（坐网红小火车）→ ​​后山徒步下山​​
        ​​西海大峡谷​​是此行最大惊喜！乘小火车穿越峡谷，两侧峭壁如刀削，谷底云雾缭绕，仿佛仙境。

        ​​后记​​
        ​​体力消耗​​：两天徒步约15公里，腿酸了三天，但回味无穷。
        ​​必带物品​​：登山杖、防风外套、充电宝（山上插座少）、防晒霜。
        ​​遗憾​​：没看到“猴子观海”奇石，留给下次吧！
        ​​黄山归来不看岳​​——果然名不虚传！"""))
    # Close the connection to the model client.
    await model_client.close()

asyncio.run(main())