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
        只保留最近的 max_history 轮对话，防止上下文过长
        """
        if len(self.messages) > self.max_history * 2 + 1: # *2 是因为(User+Assistant)成对，+1是System
      
            system_msg = self.messages[0]
           
            recent_msgs = self.messages[-(self.max_history * 2):]
           
            self.messages = [system_msg] + recent_msgs
           

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

    # --- 核心思考步 (Step) ---
    def chat(self, user_input: str) -> str:
        """
        执行一次对话交互。
        这把原来的大 while 循环拆解成了单次函数调用，方便评测。
        """
        self._manage_history()

        self.messages.append({"role": "user", "content": user_input})
        
        # 1. 思考
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            tools=self.tools_schema,
            tool_choice="auto"
        )
        response_msg = response.choices[0].message
        
        # 2. 决策与工具调用
        if response_msg.tool_calls:
            self.messages.append(cast(ChatCompletionMessageParam, response_msg.model_dump()))
            
            for tool_call in response_msg.tool_calls:
                tool_call = cast(ChatCompletionMessageToolCall, tool_call)
                func_name = tool_call.function.name
                func_args = parse_json_from_llm(tool_call.function.arguments)

                
                
                print(f"⚙️ 调用工具: {func_name} | 参数: {func_args}")
                
                # 执行
                func_result = self.available_functions[func_name](func_args)
                
                self.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": str(func_result)
                })
            
            # 3. 拿到工具结果后的二次回复
            final_res = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )
            reply = final_res.choices[0].message.content
        else:
            reply = response_msg.content

        # 记录助手回复
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
        
        return str(reply)
    


    


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