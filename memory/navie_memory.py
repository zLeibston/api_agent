import os
import json
import datetime


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