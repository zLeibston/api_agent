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



class AgentMemory:
    """
    未来做研究的主战场。
    可以继承这个类，写 VectorMemory, GraphMemory 等等。
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._init_storage()

    def _init_storage(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def read(self, query: str = "") -> str:
        """
        科研点：这里目前是全量读取。
        以后可以改为：根据 query 计算向量相似度，只返回 Top-k 记忆。
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False)

    def write(self, content: str) -> str:
        """
        科研点：这里目前是直接追加。
        以后可以改为：记忆压缩、遗忘机制、实体提取存入图谱。
        """
        with open(self.file_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            data.append({"time": timestamp, "content": content})
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"已记录: {content}"


class Agent:
    def __init__(self, api_key: str, base_url: str, model_name: str, memory_path: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.memory = AgentMemory(memory_path) # 拥有一个记忆模块
        
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
                    "description": "记忆管理。action='read'读取相关记忆，action='write'写入重要信息。",
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
                func_args = json.loads(tool_call.function.arguments)

                
                
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
    
def get_project_root():
    """
    聪明地找到项目根目录：
    从当前脚本所在目录开始向上找，只要看到 '.env' 或 '.gitignore' 文件，
    就认定那是根目录。
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    
    # 定义根目录的特征文件（找到其中一个就行）
    root_markers = [".env", ".gitignore", ".git"]
    
    while True:
        # 看看当前目录下有没有标志文件
        for marker in root_markers:
            if os.path.exists(os.path.join(current_path, marker)):
                return current_path
        
        # 尝试向上一级
        parent_path = os.path.dirname(current_path)
        
        # 如果已经到了硬盘的根目录(比如 C:\ 或 /)还没找到
        if parent_path == current_path:
            # 没办法，这就当做根目录吧，或者报错
            print("⚠️ 警告：没找到项目根目录，将使用脚本所在目录")
            return os.path.dirname(os.path.abspath(__file__))
        
        current_path = parent_path

    


if __name__ == "__main__":
    load_dotenv()
    # 配置
    API_KEY = os.getenv("SILICON_API_KEY")

    if not API_KEY:
        raise ValueError("❌ 严重错误：未找到 API Key！请检查 .env 文件是否存在，以及变量名是否正确。")

    BASE_URL = "https://api.siliconflow.cn/v1"
    
    # 锁定记忆文件路径
    project_root_dir = get_project_root()
    memory_path = os.path.join(project_root_dir, "memory", "main_memory.json")

    # 实例化 Agent
    my_agent = Agent(
        api_key=API_KEY, 
        base_url=BASE_URL, 
        model_name="Qwen/Qwen2.5-72B-Instruct",
        memory_path=memory_path
    )

    print(" Agent 已启动...")
    while True:
        q = input("\n👤 你: ")
        if q.lower() in ['q', 'exit']: break
        
        ans = my_agent.chat(q)
        print(f"🤖 Agent: {ans}")