"""
LangGraph是什么？
类似于一个工作流管理器，帮助我们把复杂的AI任务分解成一步一步小任务。
数据流动？
用户问题--分析结果--工具参数--原始数据--格式化回答
工作模式？
用户问题--LangGraph工作流--MCP工具--数据--回答
"""
from langgraph.graph import StateGraph, END     #工作流管理工具
from typing import TypedDict, List, Dict, Any   #类型定义工具
from langchain_core.messages import HumanMessage, SystemMessage #AI对话消息处理
from langchain_openai import ChatOpenAI   #AI大脑连接工具
#一些辅助工具
import asyncio
import json
import os
from dotenv import load_dotenv

# 加载环境变量，读取.env中的密钥
load_dotenv()


class AgentState(TypedDict):
    question: str              #用户问题
    tool_name: str              #要使用哪个工具
    tool_args: Dict[str, Any]   #工具的设置参数：字典[键，值]
    tool_result: Any            #工具返回的结果数据
    success: bool               #是否成功（T/F）
    answer: str                 #最终给用户的回答
    conversation_history: List  #对话历史记录

class StockQueryAgent:
    def __init__(self):
        # 设置AI大脑
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,        #回答准确度：0表很准确，不随机
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        # 整理工具，贴上标签     基本信息，价格数据，实时报价，财务指标
        self.tools = {
            "get_stock_basic": "获取股票基本信息",
            "get_stock_price": "获取股票历史价格数据",
            "get_realtime_price": "获取股票实时报价",
            "get_financial_indicator": "获取财务指标数据"
        }

    def create_workflow(self):
        """创建LangGraph工作流"""
        # 创建一条生产线，用来规定产品规格
        workflow = StateGraph(AgentState)
        # 定义节点  三站式：分析要调用的工具--执行工具--包装返回给用户
        workflow.add_node("analyze_query", self.analyze_user_query)
        workflow.add_node("execute_tool", self.execute_tool_call)
        workflow.add_node("generate_response", self.generate_final_response)
        # 定义边 分析查询--执行工具--生成响应--结束
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "execute_tool")
        workflow.add_edge("execute_tool", "generate_response")
        workflow.add_edge("generate_response", END)
        # 启动生产线
        return workflow.compile()
    # 工作站1
    def analyze_user_query(self, state: AgentState):
        """分析用户查询，决定要调用什么工具"""
        tool_call = self.parse_tool_call(state["question"]) #调用问题解析器来解析用户查询
        return {
            "tool_name": tool_call["tool_name"],
            "tool_args": tool_call["tool_args"],
            "conversation_history": state.get("conversation_history", []) + [HumanMessage(content=state["question"])]
            # 记录对话
        }

    # 工作站2：执行工具调用
    def execute_tool_call(self, state):
        """执行工具调用 - 模拟数据用于测试"""
        try:
            tool_name = state["tool_name"]
            stock_code = state["tool_args"]["stock_code"]

            print(f"调用工具: {self.tools.get(tool_name, tool_name)}")
            print(f"股票代码: {stock_code}")

            # 模拟数据 - 用于测试交互
            mock_data = self.get_mock_stock_data(tool_name, stock_code)
            return {"tool_result": mock_data, "success": True}

        except Exception as e:
            error_msg = f"工具调用失败: {str(e)}"
            print(f"{error_msg}")
            return {
                "tool_result": error_msg,
                "success": False
            }

    #测试
    def get_mock_stock_data(self, tool_name: str, stock_code: str):
        """提供模拟股票数据用于测试"""
        if tool_name == "get_stock_basic":
            return {
                "ts_code": stock_code,
                "name": "贵州茅台" if stock_code == "600519.SH" else "平安银行",
                "area": "贵州" if stock_code == "600519.SH" else "广东",
                "industry": "白酒",
                "market": "主板",
                "exchange": "SSE" if stock_code.endswith(".SH") else "SZSE",
                "list_date": "2001-08-27",
                "fullname": "贵州茅台酒股份有限公司" if stock_code == "600519.SH" else "平安银行股份有限公司"
            }
        elif tool_name == "get_stock_price":
            return {
                "ts_code": stock_code,
                "trade_date": "2024-01-15",
                "open": 1650.0,
                "high": 1680.0,
                "low": 1645.0,
                "close": 1675.5,
                "vol": 5000000,
                "amount": 8375000000
            }
        elif tool_name == "get_realtime_price":
            return {
                "ts_code": stock_code,
                "name": "贵州茅台" if stock_code == "600519.SH" else "平安银行",
                "price": 1675.5,
                "change": 25.5,
                "change_percent": 1.55,
                "volume": 50000,
                "amount": 83750000,
                "time": "14:30:00"
            }
        elif tool_name == "get_financial_indicator":
            return {
                "ts_code": stock_code,
                "ann_date": "2023-10-28",
                "end_date": "2023-09-30",
                "eps": 15.88,
                "bps": 125.36,
                "roe": 12.67,
                "profit_margin": 52.3
            }
        else:
            return {"error": "未知工具类型"}
    # 工作站3：生成最终回答
    def generate_final_response(self, state):
        """生成最终回答"""
        # 1.成功了吗？
        if not state.get("success", False):
            return {"answer": f"查询失败：{state['tool_result']}"}

        #2.打包原始数据给AI
        data_summary = json.dumps(state["tool_result"], ensure_ascii=False, indent=2)
        # 3.让AI把原始数据变成自然语言
        messages = [
            SystemMessage(content="请将股票数据结果转化为用户容易理解的自然语言描述。"),
            HumanMessage(content=f"用户问题：{state['question']}"),
            HumanMessage(content=f"原始数据：{data_summary}"),
            HumanMessage(content="请用中文回答，突出重点信息。")
        ]
        # 获取AI生成的回答，并返回答案
        try:
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        except Exception as e:
            # 如果AI调用失败，返回原始数据
            return {"answer": f"查询结果：\n{data_summary}"}

    def parse_tool_call(self, question: str) -> dict:
        """智能分析用户问题，决定使用哪个工具"""
        print(f"正在分析问题...") # 添加开始标记
        # 简单的规则匹配（默认）
        stock_code = "000001.SZ"

        # 识别股票
        if any(keyword in question for keyword in ["茅台", "600519"]):
            stock_code = "600519.SH"
        elif any(keyword in question for keyword in ["平安", "000001"]):
            stock_code = "000001.SZ"
        elif any(keyword in question for keyword in ["招商", "600036"]):
            stock_code = "600036.SH"
        elif any(keyword in question for keyword in ["万科", "000002"]):
            stock_code = "000002.SZ"

        # 识别工具类型（默认）
        tool_name = "get_stock_basic"

        if any(keyword in question for keyword in ["实时", "当前", "现在", "最新"]):
            tool_name = "get_realtime_price"
        elif any(keyword in question for keyword in ["价格", "股价", "走势", "k线", "行情"]):
            tool_name = "get_stock_price"
        elif any(keyword in question for keyword in ["财务", "指标", "业绩", "盈利", "收益"]):
            tool_name = "get_financial_indicator"
        elif any(keyword in question for keyword in ["信息", "基本", "概况", "介绍", "公司"]):
            tool_name = "get_stock_basic"

        print(f"分析完成 - 使用工具: {self.tools[tool_name]}")
        return {
            "tool_name": tool_name,
            "tool_args": {"stock_code": stock_code}
        }

#实现交互
async def chat_with_agent():
    """与Agent进行交互的主函数"""
    print("=" * 60)
    print("我可以帮您查询：")
    print("股票基本信息（公司概况、上市信息等）")
    print("历史价格数据")
    print("实时报价")
    print("财务指标")
    print("eg：")
    print("贵州茅台的基本信息")
    print("查看平安银行的实时价格")
    print("招商银行的历史行情")
    print("万科的财务指标")
    print("=" * 60)
    # 初始化Agent
    try:
        agent = StockQueryAgent()
        workflow = agent.create_workflow()
        print("Agent初始化成功！")
    except Exception as e:
        print(f"初始化失败了")
        return

    # 交互循环
    while True:
        try:
            print("-" * 40)
            user_input = input("请输入问题（输入'退出'结束）: ").strip()

            if user_input.lower() in ['退出', 'quit', 'exit']:
                print("再见！")
                break

            if not user_input:
                print("请输入有效的问题")
                continue

            # 处理用户查询
            print("正在处理请求...")
            result = await workflow.ainvoke({
                "question": user_input,
                "tool_name": "",
                "tool_args": {},
                "tool_result": None,
                "success": False,
                "answer": "",
                "conversation_history": []
            })

            # 显示结果
            print("=" * 50)
            print("回答:")
            print(result.get("answer", "抱歉，没有生成回答"))
            print("=" * 50)

        except KeyboardInterrupt:
            print("再见！")
            break
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            continue


if __name__ == '__main__':
    # 启动交互式聊天
    asyncio.run(chat_with_agent())
