import json
import re
from typing import Dict, Any, Union ,Optional# 引入类型提示

# 尝试导入 json_repair 库 (推荐安装: pip install json_repair)
# 如果没安装，代码会自动降级使用内置的正则处理，不会报错
try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

def extract_json_block(text: str) -> str:
    # ... (这部分保持你原来的代码不变，写得很好) ...
    # 稍微优化一下正则，兼容 markdown 没写 json 字样的情况
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

def parse_json_from_llm(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    清洗并解析大模型输出的 JSON。
    返回 Dict: 解析成功
    返回 None: 解析失败或类型错误 (用于触发外部 Retry)
    """
    if not raw_text:
        return {} # 空字符串视为无参数，返回空字典是安全的

    # === 方案 A: 使用 json_repair (最强) ===
    if HAS_JSON_REPAIR:
        try:
            # repair_json 可能返回 dict, list, str, float 等
            parsed = repair_json(raw_text, return_objects=True)
            
            if isinstance(parsed, dict):
                return parsed
            elif isinstance(parsed, list):
                # 如果是列表，尝试取第一个元素，且该元素必须是字典
                if len(parsed) > 0 and isinstance(parsed[0], dict):
                    return parsed[0]
            
            # 关键修改：类型不对（比如解析出字符串、数字），返回 None，而不是 {}
            # 这样 main.py 才知道格式错了，才能触发 Retry
            return None
            
        except Exception as e:
            print(f"⚠️ json_repair 解析异常: {e}，尝试手动清洗...")

    # === 方案 B: 手动正则清洗 (备用) ===
    clean_text = extract_json_block(raw_text)
    clean_text = clean_text.strip()

    try:
        res = json.loads(clean_text)
        if isinstance(res, dict):
            return res
        # 关键修改：类型不对，返回 None
        return None
    except json.JSONDecodeError:
        pass 

    print(f"❌ JSON 解析彻底失败。原始文本:\n{raw_text}")
    return None

# 简单的测试代码 (当你直接运行这个文件时执行)
if __name__ == "__main__":
    test_cases = [
        '{"tool": "get_time"}',                       # 正常
        '```json\n{"tool": "get_time"}\n```',        # Markdown
        'Sure! Here is the code: {"tool": "get_time"}', # 包含废话
        '{"tool": "get_time",}',           
        'uedu{"tool":"rtyu"},}'         # 尾部多余逗号 (标准json不支持，json_repair支持)
    ]
    
    print("=== 开始测试清洗功能 ===")
    for t in test_cases:
        res = parse_json_from_llm(t)
        print(f"原文: {t}")
        print(f"结果: {res} (类型: {type(res)})\n")