import os
import json
import datetime
from typing import cast, List, Dict, Any, Optional
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam, 
    ChatCompletionToolParam, 
    ChatCompletionMessageToolCall
)
from dotenv import load_dotenv
from utils.find_root_dir import get_project_root
from utils.json_clean import parse_json_from_llm 



class AgentMemory:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._ensure_file_validity() # 启动时自检

    def _ensure_file_validity(self):
        """确保文件存在且是合法的 JSON,否则重置"""
        if not os.path.exists(self.file_path):
            self._reset_memory()
            return
        
        # 如果文件存在，尝试读取，看是不是坏的
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content: # 如果是空文件
                    raise ValueError("File is empty")
                json.loads(content) # 尝试解析
        except (json.JSONDecodeError, ValueError):
            print(f"⚠️ 警告：记忆文件 {self.file_path} 损坏或为空，已重置为 []。")
            self._reset_memory()

    def _reset_memory(self):
        """重置记忆文件"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

    def read(self, query: str = "") -> str:
        """
        注意，这个read方法只是一个半成品，是将记忆全扔给了模型，之后可能需要实现更复杂的检索逻辑。
        """
        self._ensure_file_validity() # 读之前再检查一次
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False)

    def write(self, content: str) -> str:
        self._ensure_file_validity() # 写之前再检查一次
        
        # 读取现有数据
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 追加新数据
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        data.append({"time": timestamp, "content": content})
        
        # 重新写入
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return f"已记录: {content}"

class Agent:
    def __init__(self, api_key: str, base_url: str, model_name: str, memory_path: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.memory = AgentMemory(memory_path) # 拥有一个记忆模块
        self.max_history = 5  # 【新增】只保留最近 5 轮对话
        self.max_tool_iterations = 5  # 最大工具调用轮数，防止死循环    
        
        # 上下文历史
        self.messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "你是一个拥有长期记忆的研究型Agent。"}
        ]
        
        # 注册工具
        self.tools_schema = self._get_tools_schema()
        self.available_functions = {
            "manage_memory": self._tool_manage_memory, # 把工具绑定到类方法上
            "get_time": self._tool_get_time
        }

    def _get_tools_schema(self) -> List[ChatCompletionToolParam]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "manage_memory",
                    "description": "记忆管理。如果你觉得有些信息你应该知道但却不知道，他有可能在你的记忆存储里面，试试在记忆中找找。如果你觉得有重要信息，也请写入记忆中。action='read'读取相关记忆，action='write'写入重要信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["read", "write"]},
                            "content": {"type": "string", "description": "写入的内容或读取的查询词"}
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "获取当前时间",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    
    def _manage_history(self):
        """
        鲁棒的历史记录管理：
        1. 始终保留 System Message。
        2. 基于 'user' 角色进行切分，而不是简单的列表切片。
        3. 确保截断后的第一条非 System 消息一定是 User 发起的，避免 Tool/Assistant 消息孤立。
        """
        # 如果消息总数没超过阈值，直接跳过 (这里稍微放宽一点阈值，避免频繁操作)
        # 假设平均一轮 3-4 条消息 (User + AI + Tool + AI)，给个 buffer
        if len(self.messages) < self.max_history * 4:
            return

        # 1. 提取 System Message (如果有)
        system_msg = None
        if self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]

        # 2. 找到所有 User 消息的索引,认为 "User" 发言是一轮对话的逻辑起点
        user_indices = [
            i for i, msg in enumerate(self.messages) 
            if msg["role"] == "user"
        ]

        # 3. 判断是否需要截断,如果 User 发言次数超过了 max_history，就需要截断
        if len(user_indices) > self.max_history:
            # 找到需要保留的那轮 User 对话的起始索引
            # 例如保留最近 5 轮，就取倒数第 5 个 User 消息的索引
            cutoff_index = user_indices[-self.max_history]
            
            # 4. 构建新的消息列表
            # 保留 System + 从 cutoff_index 开始的所有后续消息
            new_history = self.messages[cutoff_index:]
            
            if system_msg:
                self.messages = [system_msg] + new_history
            else:
                self.messages = new_history
            
            print(f"🧹 [History] 已执行清理，当前保留最近 {self.max_history} 轮对话，剩余消息数: {len(self.messages)}")
        else:
            # 如果 User 轮次还不够多，说明可能是 Tool 消息太多导致长度增加
            # 这种情况下通常不建议硬切，除非总 Token 超标（那是另一个 Token 计算的问题）
            pass
           

    # --- 工具具体实现 ---
    def _tool_manage_memory(self, args: Dict[str, Any]):
        action = args.get("action")
        content = args.get("content", "")
        if action == "write":
            return self.memory.write(content)
        elif action == "read":
            return self.memory.read(content) # 这里传入 content 作为 query
        return "未知操作"

    def _tool_get_time(self, args):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    
    def chat(self, user_input: str) -> str:
        """
        执行一次对话交互。
        支持循环调用工具 (ReAct 循环)，直到模型认为任务完成。
        """
        # 1. 历史管理 (防止上下文无限增长)
        if hasattr(self, '_manage_history'):
            self._manage_history()

        self.messages.append({"role": "user", "content": user_input})
     
        # 记录循环步数
        iteration = 0

        while iteration < self.max_tool_iterations:
            # 2. 思考 (调用大模型)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=self.tools_schema,
                tool_choice="auto"
            )
            response_msg = response.choices[0].message
            
            # 3. 无论是否调用工具，都要先把 Assistant 的回复加入历史
            self.messages.append(cast(ChatCompletionMessageParam, response_msg.model_dump()))

            # 4. 判断是否需要调用工具
            if response_msg.tool_calls:
                iteration += 1
                
                # 处理本轮所有的工具调用
                for tool_call in response_msg.tool_calls:
                    tool_call = cast(ChatCompletionMessageToolCall, tool_call)
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments # 获取原始字符串，用于报错回显
                    
                    func_result = ""

                    
                    
                    # A. 尝试解析参数
                    func_args = parse_json_from_llm(raw_args)

                    # B. 检查解析是否成功 (None 表示格式错误)
                    if func_args is None:
                        # 构造错误回显：告诉 LLM 它给的 JSON 是坏的，强迫它在下一轮修正
                        func_result = f"Error: Invalid JSON format in arguments. You provided: '{raw_args}'. Please output a valid JSON object."
                        print(f"🔄 [Self-Correction] 第{iteration}轮: JSON解析失败，已反馈给 Agent 要求重试...")
                    
                    # C. 解析成功，正常执行工具
                    else:
                        try:
                            print(f"⚙️ [第{iteration}轮] 调用工具: {func_name} | 参数: {func_args}")
                            
                            if func_name in self.available_functions:
                                func_result = self.available_functions[func_name](func_args)
                            else:
                                func_result = f"Error: Tool '{func_name}' not found."
                                
                        except Exception as e:
                            # 捕获工具内部运行错误（如数据库连接失败等）
                            func_result = f"Error executing tool '{func_name}': {str(e)}"
                            print(f"❌ 工具运行时错误: {e}")

                

                    # 5. 将执行结果（无论是成功的返回值，还是格式错误的报错信息）追加到历史
                    self.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": str(func_result)
                    })
                
                # 关键：continue 让循环继续，LLM 会读取上面的 "Error: Invalid JSON" 然后尝试重新生成
                continue 
            
            else:
                # 6. 如果没有 tool_calls，说明模型输出了最终回答
                return str(response_msg.content)

        return "⚠️ 任务过长，强制终止循环。"
    


    


if __name__ == "__main__":
    load_dotenv()
    # 配置
    API_KEY = os.getenv("DEEPSEEK_API_KEY")

    if not API_KEY:
        raise ValueError("❌ 严重错误：未找到 API Key！请检查 .env 文件是否存在，以及变量名是否正确。")

    BASE_URL = os.getenv("DEEPSEEK_URL")

    if not BASE_URL:
        raise ValueError("❌ 严重错误：未找到 URL！请检查 .env 文件是否存在，以及变量名是否正确。")
    
  
    project_root_dir = get_project_root()
    memory_path = os.path.join(project_root_dir, "memory", "main_memory.json")

   
    my_agent = Agent(
        api_key=API_KEY, 
        base_url=BASE_URL, 
        model_name="deepseek-chat",
        memory_path=memory_path
    )

    print(" Agent 已启动...")
    while True:
        q = input("\n👤 你: ")
        if q.lower() in ['q', 'exit']: break
        
        ans = my_agent.chat(q)
        print(f"🤖 Agent: {ans}")