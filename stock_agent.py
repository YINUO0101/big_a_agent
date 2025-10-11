"""
LangGraph是什么？
类似于一个工作流管理器，帮助我们把复杂的AI任务分解成一步一步小任务。
数据流动？
用户问题--分析结果--工具参数--原始数据--格式化回答
工作模式？
用户问题--LangGraph工作流--MCP工具--数据--回答
"""
import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# 定义状态类 描述智能体的状态结构
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 初始化MCP工具
async def setup_mcp_tools():
    """设置MCP工具"""
    try:
        #创建MCP客户端，连接本地运行的MCP服务器
        client = MultiServerMCPClient({
            "tushare_mcp_server": {
                "command": "python",
                "args": ["mcp_server/tushare_mcp_server.py"],
                "transport": "stdio",
            }
        })
        #获取工具列表
        return await client.get_tools()
    except Exception as e:
        print(f"工具初始化失败: {e}")
        return None


# 创建智能体 构建完整的LangGraph工作流
def create_stock_agent(tools):
    """创建股票查询智能体
    参数：tools--从MCP服务器获取的工具列表
    返回：编译后的智能体图
    """
    # 初始化大模型
    llm = ChatDeepSeek(model="deepseek-chat", api_key=DEEPSEEK_API_KEY)
    llm_with_tools = llm.bind_tools(tools)

    # 定义节点函数
    def agent_node(state: AgentState):
        """智能体节点--用来处理用户输入
        参数：state--当前的状态，包含消息历史
        返回：新消息的字典
        """
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 创建工具节点，用来执行工具调用
    tool_node = ToolNode(tools)

    # 判断是否需要调用工具
    def should_use_tools(state: AgentState):
        """决定下一步是调用工具还是结束
        参数：state--当前状态
        返回：use_tools--需要调用工具
             end--可以结束对话
        """
        last_message = state["messages"][-1]
        # 检查最后一条信息是否有工具调用
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "use_tools"
        return "end"

    # 构建图，指定状态类型
    workflow = StateGraph(AgentState)

    # 添加节点到图中
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 设置边（节点直接的连接关系），开始节点到智能体节点
    workflow.add_edge(START, "agent")

    #添加条件边 - 根据条件决定下一步走向哪个节点
    workflow.add_conditional_edges(
        "agent",
        should_use_tools,
        {
            "use_tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")

    # 编译图--创建内存检查点保存器，用于保存对话状态
    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)


# 处理用户查询
async def handle_user_query(query, agent):
    """处理用户查询并显示结果
    参数：query -- 用户输入的查询文本
         agent -- 编译后的智能体图
    """
    config = {"configurable": {"thread_id": "user_session"}}

    try:
        #使用astream方法流式处理用户查询，实时看到处理过程，而不是等待全部完成
        async for event in agent.astream(
                {"messages": [{"role": "user", "content": query}]},
                config
        ):
            #遍历事件中每个节点输出
            for node_name, node_output in event.items():
                if "messages" in node_output:
                    message = node_output["messages"][-1]

                    #如果是智能体节点的输出-显示AI的文本回复-显示工具调用信息
                    if node_name == "agent":
                        # 显示AI回复
                        if hasattr(message, 'content') and message.content:
                            print(f"AI: {message.content}")

                        # 显示工具调用信息
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                print(f"调用工具: {tool_call['name']}")

                    #如果是工具调用信息
                    elif node_name == "tools":
                        print("工具执行完成")

    except Exception as e:
        print(f"处理查询时出错: {e}")


# 主函数
async def main():
    """主程序"""
    print("正在启动股票查询系统...")

    # 初始化工具
    tools = await setup_mcp_tools()
    if not tools:
        print("无法启动系统：工具初始化失败")
        return

    # 创建智能体
    agent = create_stock_agent(tools)
    print("系统启动完成！")

    # 交互循环 -- 持续接收用户输入
    while True:
        try:
            user_input = input("\n请输入您的查询 (输入 'quit' 退出): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if user_input:
                await handle_user_query(user_input, agent)

        except Exception as e:
            print(f"发生错误: {e}")

#Python程序入口点
if __name__ == "__main__":
    asyncio.run(main())
