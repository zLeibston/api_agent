# core/agent.py
import datetime
from typing import cast, List, Dict, Any
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam, 
    ChatCompletionToolParam, 
    ChatCompletionMessageToolCall
)

# 引入你的工具模块和配置
import config
from utils.json_clean import parse_json_from_llm
from memory.navie_memory import AgentMemory# 引用刚才拆分出去的记忆模块

class Agent:
    def __init__(self):
        # 从 config 直接读取配置，参数更少更干净
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.model_name = config.MODEL_NAME
        
        # 初始化记忆模块
        self.memory = AgentMemory(config.DEFAULT_MEMORY_PATH)
        
        self.max_history = 5
        self.max_tool_iterations = 5  
        
        self.messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "你是一个拥有长期记忆的研究型Agent。"}
        ]
        
        self.tools_schema = self._get_tools_schema()
        self.available_functions = {
            "manage_memory": self._tool_manage_memory,
            "get_time": self._tool_get_time
        }

    def _get_tools_schema(self) -> List[ChatCompletionToolParam]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "manage_memory",
                    "description": "记忆管理...",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["read", "write"]},
                            "content": {"type": "string"}
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
        # ... (这里保持你原来的历史管理逻辑不变，代码太长我就省略了，直接拷贝过来即可) ...
        # 建议直接把原来的 _manage_history 完整代码贴在这里
        if len(self.messages) < self.max_history * 4:
            return
        system_msg = None
        if self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
        user_indices = [i for i, msg in enumerate(self.messages) if msg["role"] == "user"]
        if len(user_indices) > self.max_history:
            cutoff_index = user_indices[-self.max_history]
            new_history = self.messages[cutoff_index:]
            if system_msg:
                self.messages = [system_msg] + new_history
            else:
                self.messages = new_history
            print(f"🧹 [History] 已清理，保留最近 {self.max_history} 轮。")

    def _tool_manage_memory(self, args: Dict[str, Any]):
        action = args.get("action")
        content = args.get("content", "")
        if action == "write":
            return self.memory.write(content)
        elif action == "read":
            return self.memory.read(content)
        return "未知操作"

    def _tool_get_time(self, args):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def chat(self, user_input: str) -> str:
        # ... (这里保持你原来的 chat 逻辑不变，直接拷贝过来) ...
        # 记得引入 parse_json_from_llm
        if hasattr(self, '_manage_history'):
            self._manage_history()

        self.messages.append({"role": "user", "content": user_input})
        iteration = 0

        while iteration < self.max_tool_iterations:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=self.tools_schema,
                tool_choice="auto"
            )
            response_msg = response.choices[0].message
            self.messages.append(cast(ChatCompletionMessageParam, response_msg.model_dump()))

            if response_msg.tool_calls:
                iteration += 1
                for tool_call in response_msg.tool_calls:
                    tool_call = cast(ChatCompletionMessageToolCall, tool_call)
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    
                    func_args = parse_json_from_llm(raw_args)
                    func_result = ""

                    if func_args is None:
                        func_result = f"Error: Invalid JSON format. You provided: '{raw_args}'"
                        print(f"🔄 [Self-Correction] JSON解析失败...")
                    else:
                        try:
                            print(f"⚙️ [Tool] {func_name} | Args: {func_args}")
                            if func_name in self.available_functions:
                                func_result = self.available_functions[func_name](func_args)
                            else:
                                func_result = f"Error: Tool not found."
                        except Exception as e:
                            func_result = f"Error: {str(e)}"

                    self.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": str(func_result)
                    })
                continue 
            else:
                return str(response_msg.content)

        return "⚠️ 任务过长终止。"