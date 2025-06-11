import autogen
import numpy as np
from typing import List, Dict, Optional
# -----------------运行失败-----------
# --------------------------------------------------------------------------
# 1. 游戏核心逻辑 (五子棋)
# 我们将游戏逻辑与 Agent 定义分开，以保持代码清晰。
# --------------------------------------------------------------------------

BOARD_SIZE = 9  # 棋盘大小 (为了快速演示，使用 9x9)

def create_board() -> np.ndarray:
    """创建一个空的棋盘"""
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

def print_board(board: np.ndarray) -> str:
    """将棋盘状态转换为字符串，以便在对话中显示"""
    header = "   " + " ".join([f"{i:<2}" for i in range(BOARD_SIZE)])
    board_str = [header]
    for i in range(BOARD_SIZE):
        row_str = f"{i:<2} "
        for j in range(BOARD_SIZE):
            if board[i, j] == 1:
                row_str += " X "  # Alice (黑棋)
            elif board[i, j] == 2:
                row_str += " O "  # Bob (白棋)
            else:
                row_str += " . "
        board_str.append(row_str)
    return "\n".join(board_str)

def is_valid_move(board: np.ndarray, row: int, col: int) -> bool:
    """检查落子是否有效"""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row, col] == 0

def check_win(board: np.ndarray, player: int) -> bool:
    """检查玩家是否赢得比赛（连成五个子）"""
    # 检查行
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 4):
            if np.all(board[r, c:c+5] == player):
                return True
    # 检查列
    for r in range(BOARD_SIZE - 4):
        for c in range(BOARD_SIZE):
            if np.all(board[r:r+5, c] == player):
                return True
    # 检查对角线 (从左上到右下)
    for r in range(BOARD_SIZE - 4):
        for c in range(BOARD_SIZE - 4):
            if np.all(np.diag(board[r:r+5, c:c+5]) == player):
                return True
    # 检查对角线 (从右上到左下)
    for r in range(BOARD_SIZE - 4):
        for c in range(4, BOARD_SIZE):
            if np.all(np.diag(np.fliplr(board[r:r+5, c-4:c+1])) == player):
                return True
    return False

def is_board_full(board: np.ndarray) -> bool:
    """检查棋盘是否已满"""
    return not np.any(board == 0)

# --------------------------------------------------------------------------
# 2. Autogen Agent 配置
# --------------------------------------------------------------------------

# **重要**: 请在这里填入你的 LLM API 配置
# 例如:
# config_list = [
#     {
#         "model": "gpt-4",
#         "api_key": "sk-...",
#     }
# ]
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["gpt-4", "gpt-4-32k"]},
)

llm_config = {
    "config_list": config_list,
    "cache_seed": 42, # 使用缓存以获得可重复的结果
    "temperature": 0.1,
}

# --------------------------------------------------------------------------
# 3. 定义我们的智能体
# --------------------------------------------------------------------------

# --- 裁判 (Referee) ---
class RefereeAgent(autogen.ConversableAgent):
    def __init__(self, name="Referee"):
        super().__init__(
            name=name,
            system_message="You are the referee. You manage the game state, validate moves, and declare the winner.",
            llm_config=False,  # 裁判不使用 LLM，只执行代码逻辑
            human_input_mode="NEVER",
        )
        self.board = create_board()
        self.current_player_id = 1 # Alice (黑棋) 先手
        self.game_over = False
        self.winner = 0

    def process_move(self, message: str, sender: autogen.Agent) -> Optional[str]:
        """处理玩家的移动并更新游戏状态"""
        if self.game_over:
            return f"GAME OVER. Winner: Player {self.winner}."

        # 解析玩家的移动
        try:
            # LLM 的输出可能不完美，我们尝试从中提取坐标
            move_str = message.split("MOVE:")[1].strip()
            row, col = map(int, move_str.split(','))
        except (ValueError, IndexError):
            return "Invalid move format. Please use 'MOVE: row,col'. Try again."

        # 验证移动
        if not is_valid_move(self.board, row, col):
            return f"Invalid move at ({row}, {col}). The cell is already taken or out of bounds. Try again."

        # 更新棋盘
        self.board[row, col] = self.current_player_id
        current_board_str = print_board(self.board)

        # 检查游戏是否结束
        if check_win(self.board, self.current_player_id):
            self.game_over = True
            self.winner = self.current_player_id
            player_name = "Alice" if self.current_player_id == 1 else "Bob"
            return f"{current_board_str}\n\nGAME OVER! {player_name} (Player {self.winner}) wins! TERMINATE"

        if is_board_full(self.board):
            self.game_over = True
            return f"{current_board_str}\n\nGAME OVER! It's a draw. TERMINATE"

        # 切换玩家
        self.current_player_id = 2 if self.current_player_id == 1 else 1
        next_player_name = "Alice" if self.current_player_id == 1 else "Bob"
        
        return f"Move accepted. Board updated.\n{current_board_str}\n\nIt's {next_player_name}'s turn."

# --- 玩家 (Alice 和 Bob) ---
alice_system_message = """You are a player in a Gomoku (five-in-a-row) game named Alice.
Your goal is to win the game by placing five of your pieces ('X') in a row (horizontally, vertically, or diagonally).
You are Player 1, and your pieces are represented by the number 1. Your opponent's pieces ('O') are represented by 2.
The board is a 9x9 grid. 0 represents an empty cell.
You will be shown the current board state. Analyze it carefully and choose your best move.
You MUST respond with your move in the exact format: `MOVE: row,col` where `row` and `col` are integers from 0 to 8.
Do not add any other text before or after the move command.
Think strategically. Try to form your own lines of five while blocking your opponent.
"""

bob_system_message = """You are a player in a Gomoku (five-in-a-row) game named Bob.
Your goal is to win the game by placing five of your pieces ('O') in a row (horizontally, vertically, or diagonally).
You are Player 2, and your pieces are represented by the number 2. Your opponent's pieces ('X') are represented by 1.
The board is a 9x9 grid. 0 represents an empty cell.
You will be shown the current board state. Analyze it carefully and choose your best move.
You MUST respond with your move in the exact format: `MOVE: row,col` where `row` and `col` are integers from 0 to 8.
Do not add any other text before or after the move command.
Think strategically. Try to form your own lines of five while blocking your opponent.
"""

alice = autogen.ConversableAgent(
    name="Alice",
    system_message=alice_system_message,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

bob = autogen.ConversableAgent(
    name="Bob",
    system_message=bob_system_message,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

referee = RefereeAgent()

# --------------------------------------------------------------------------
# 4. 设置群聊和管理者
# --------------------------------------------------------------------------

# 定义一个自定义的说话者选择函数来强制执行游戏顺序
def custom_speaker_selection_func(last_speaker: autogen.Agent, groupchat: autogen.GroupChat) -> autogen.Agent:
    """自定义轮流顺序: Referee -> Alice -> Referee -> Bob -> ..."""
    if last_speaker is groupchat.agent_by_name("User"):
        # 游戏开始，裁判先发言
        return groupchat.agent_by_name("Referee")
    
    if last_speaker is groupchat.agent_by_name("Referee"):
        # 裁判发言后，轮到当前玩家
        if referee.current_player_id == 1:
            return groupchat.agent_by_name("Alice")
        else:
            return groupchat.agent_by_name("Bob")
    else:
        # 玩家发言后，轮到裁判处理
        return groupchat.agent_by_name("Referee")

# 注册回复函数
# 当裁判收到来自 Alice 或 Bob 的消息时，调用 process_move 函数
referee.register_reply(
    [autogen.Agent, None],  # 当收到任何 Agent 或用户的消息时触发
    reply_func=RefereeAgent.process_move,
    config=None, # reply_func不需要llm_config
    reset_config=False
)

groupchat = autogen.GroupChat(
    agents=[alice, bob, referee],
    messages=[],
    max_round=50, # 9*9=81个格子，最多41步黑棋40步白棋
    speaker_selection_method=custom_speaker_selection_func,
    # 当检测到 "TERMINATE" 时，对话将结束
    send_introductions=False # 我们不需要介绍性消息
)

# UserProxyAgent 用于启动对话
# 它不会实际参与游戏，只是作为对话的发起者
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False,
    default_auto_reply="",
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    # 当消息中包含 "TERMINATE" 时，对话会终止
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
)

# --------------------------------------------------------------------------
# 5. 开始游戏
# --------------------------------------------------------------------------

initial_message = f"""Let the Gomoku game begin!
The board is {BOARD_SIZE}x{BOARD_SIZE}.
Alice is Player 1 (X), Bob is Player 2 (O).
{print_board(referee.board)}
It's Alice's turn to make the first move.
"""

user_proxy.initiate_chat(
    manager,
    message=initial_message,
)
