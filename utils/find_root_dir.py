import os

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