import os
import requests
from dotenv import load_dotenv

#1.加载.env文件中的环境变量
load_dotenv()


def chat_with_glm(user_input,messages):
    #调用智谱的API
    api_key = os.getenv('ZHIPU_API_KEY')
    if not api_key:
        return "错误：请检查 .env 文件中的 ZHIPU_API_KEY 配置"

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "glm-4-flash",  # 使用快速版本，免费额度足够
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API调用错误: {e}"

def main():
    print("欢迎使用 Big A Agent!基于智谱AI GLM模型。")
    print("请输入你的问题（输入'quit'或'退出'即开结束程序）")

    #初始化对话历史，可以给模型一个系统角色
    messages = [
        {
            "role":"system",
            "content":"你是一个乐于助人的AI助手。"
        }
    ]

    while True:
        #获取用户输入
        user_input = input("\n请输入你的问题：").strip()

        #检查退出条件
        if user_input.lower() in ['quit','exit','退出']:
            print("再见！")
            break


        #如果用户没有输入内容，则提示后续循环
        if not user_input:
            print("问题不能为空，请重新输入。")
            continue

        #将用户输入添加到对话历史中
        messages.append({
            "role":"user",
            "content":user_input
        })

        print("思考中...")  # 添加提示，让用户知道程序在运行

        # 调用智谱AI API
        ai_response = chat_with_glm(user_input, messages)

        # 将AI回复添加到对话历史中
        messages.append({
            "role": "assistant",
            "content": ai_response
        })


         #打印模型的回复
        print(f"\nAI: {ai_response}")


if __name__ == "__main__":
    main()