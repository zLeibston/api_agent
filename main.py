from core.agent import Agent
import config

def main():
    print(f"Agent 正在启动...")
    print(f"模型: {config.MODEL_NAME}")
    print(f"记忆路径: {config.DEFAULT_MEMORY_PATH}")
    
    # 实例化 Agent
    try:
        my_agent = Agent()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return

    print("✅ 系统就绪。输入 'q' 或 'exit' 退出。")
    
    # 交互循环
    while True:
        try:
            q = input("\n👤 你: ").strip()
            if not q: continue
            if q.lower() in ['q', 'exit']: 
                print("👋 再见！")
                break
            
            # 调用 Agent
            ans = my_agent.chat(q)
            print(f"🤖 Agent: {ans}")
            
        except KeyboardInterrupt:
            print("\n👋 强制退出")
            break
        except Exception as e:
            print(f"❌ 运行时错误: {e}")

if __name__ == "__main__":
    main()