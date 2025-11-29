import os
from dotenv import load_dotenv
from utils.find_root_dir import get_project_root

# 1. 加载环境变量
load_dotenv()

# 2. 项目根目录定位
PROJECT_ROOT = get_project_root()

# 3. LLM 配置
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_URL")
MODEL_NAME = "deepseek-chat"

if not API_KEY or not BASE_URL:
    raise ValueError("❌ 严重错误：未找到 API Key 或 URL！请检查 .env 文件。")

# 4. 记忆路径 (暂时沿用 JSON，后面换 ChromaDB)
MEMORY_DIR = os.path.join(PROJECT_ROOT, "data","naive_memory")
DEFAULT_MEMORY_PATH = os.path.join(MEMORY_DIR, "main_memory.json")

# 确保 memory 文件夹存在
os.makedirs(MEMORY_DIR, exist_ok=True)