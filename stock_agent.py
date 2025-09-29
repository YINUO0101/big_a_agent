"""
LangGraph是什么？
类似于一个工作流管理器，帮助我们把复杂的AI任务分解成一步一步小任务。
数据流动？
用户问题--分析结果--工具参数--原始数据--格式化回答
工作模式？
用户问题--LangGraph工作流--MCP工具--数据--回答
"""
from langgraph.graph import StateGraph, END  #工作流管理工具
from typing import TypedDict, List, Dict, Any  #类型定义工具
from langchain_core.messages import HumanMessage, SystemMessage #AI对话消息处理
from langchain_openai import ChatOpenAI  #AI大脑连接工具
#一些辅助工具
import asyncio
import json
import os
from dotenv import load_dotenv

# 加载环境变量，读取.env中的密钥
load_dotenv()

# 关键修改1：添加状态模式
class AgentState(TypedDict):
    question: str               #用户问题
    tool_name: str              #要使用哪个工具
    tool_args: Dict[str, Any]   #工具的设置参数：字典[键，值]
    tool_result: Any            #工具返回的结果数据
    success: bool               #是否成功（T/F）
    answer: str                 #最终给用户的回答
    conversation_history: List  #对话历史记录

class StockQueryAgent:
    def __init__(self):
        #设置AI大脑
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,        #回答准确度：0表很准确，不随机
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

        #整理工具，贴上标签     基本信息，价格数据，实时报价，财务指标
        self.tools = {
            "get_stock_basic": "获取股票基本信息",
            "get_stock_price": "获取股票历史价格数据",
            "get_realtime_price":  "获取股票实时报价",
            "get_financial_indicator": "获取财务指标数据"
        }

        #self.system_prompt = """你是一个股票查询助手。请根据用户问题选择合适工具。"""

    def create_workflow(self):
        """创建LangGraph工作流"""
        # 创建一条生产线，用来规定产品规格
        workflow = StateGraph(AgentState)

        # 定义节点  三站式：分析要调用的工具--执行工具--包装返回给用户
        workflow.add_node("analyze_query", self.analyze_user_query)
        workflow.add_node("execute_tool", self.execute_tool_call)
        workflow.add_node("generate_response", self.generate_final_response)
        #定义边 分析查询--执行工具--生成响应--结束
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "execute_tool")
        workflow.add_edge("execute_tool", "generate_response")
        workflow.add_edge("generate_response", END)  # 使用END
        #启动生产线
        return workflow.compile()
    #工作站1
    def analyze_user_query(self, state: AgentState):
        """分析用户查询，决定要调用什么工具"""
        tool_call = self.parse_tool_call(state["question"]) #调用问题解析器来解析用户查询

        return {
            "tool_name": tool_call["tool_name"],
            "tool_args": tool_call["tool_args"],
            "conversation_history": state.get("conversation_history", []) + [HumanMessage(content=state["question"])]
            #记录对话
        }
    #工作站2：执行工具调用
    def execute_tool_call(self, state):
        """执行MCP工具调用"""
        try:
            # 直接调用MCP服务器中的函数
            from mcp_server.tushare_mcp_server import get_stock_basic, get_stock_price, get_realtime_price, get_financial_indicator

            tool_name = state["tool_name"]
            stock_code = state["tool_args"]["stock_code"]

            if tool_name == "get_stock_basic":
                result = get_stock_basic(stock_code)
            elif tool_name == "get_stock_price":
                result = get_stock_price(stock_code)
            elif tool_name == "get_realtime_price":
                result = get_realtime_price(stock_code)
            elif tool_name == "get_financial_indicator":
                result = get_financial_indicator(stock_code)
            else:
                return {"tool_result": "未知工具", "success": False}

            data = json.loads(result)
            return {"tool_result": data, "success": True}

        except Exception as e:
                error_msg = f"工具调用失败: {str(e)}"
                return {
                    "tool_result": error_msg,
                    "success": False
                }
    #工作站3：生成最终回答
    def generate_final_response(self, state):
        """生成最终的自然语言响应"""
        #1.成功了吗？
        if not state.get("success", False):
            return {"answer": f"查询失败：{state['tool_result']}"}
        #2.打包原始数据给AI
        data_summary = json.dumps(state["tool_result"], ensure_ascii=False)
        #3.让AI把原始数据变成自然语言
        messages = [
            SystemMessage(content="请将股票数据结果转化为用户容易理解的自然语言描述。"),
            HumanMessage(content=f"用户问题：{state['question']}"),
            HumanMessage(content=f"查询结果：{data_summary}"),
            HumanMessage(content="请用中文总结这些数据，突出重点信息。")
        ]
        #获取AI生成的回答，并返回答案
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def parse_tool_call(self, question: str) -> dict:
        """智能分析用户问题，决定使用哪个工具"""
        print(f"=== 开始分析问题: {question} ===")  # 添加开始标记
        #工具描述
        tools_description = """
        可用工具：
        1. get_stock_basic - 查询股票基本信息
        2. get_stock_price - 查询股票历史价格数据
        3. get_realtime_price - 查询股票实时报价
        4. get_financial_indicator - 查询财务指标
        
        请根据用户问题选择最合适的工具，并提取股票代码。
        """
        #用AI分析用户意图
        messages = [
            SystemMessage(content=question),
            HumanMessage(content=tools_description)
            ]
        response = self.llm.invoke(messages)

        try:
            import re
            json_match = re.search(r'\{.*\}', response.content)
            if json_match:
                tool_info = json.loads(json_match.group())
                stock_code = tool_info.get("stock_code", "000001.SZ")
                tool_name = tool_info.get("tool_name", "get_stock_basic")
                print(f"解析成功 - 工具: {tool_name}, 股票: {stock_code}")#调试
                #返回对应参数数据
                if tool_name == "get_stock_basic":
                    return {
                        "tool_name": tool_name,
                        "tool_args": {"stock_code": stock_code}
                    }
                else:
                    return {
                        "tool_name": tool_name,
                        "tool_args": {"stock_code": stock_code}
                    }
        except:
            print("AI工具选择失败，使用默认查询")
            #默认返回基本信息查询
        stock_code = "000001.SZ"
        if "茅台" in question or "600519" in question:
            stock_code = "600519.SH"
        return {
            "tool_name": "get_stock_basic",
            "tool_args": {"stock_code": stock_code}
        }



#主程序-启动整个系统
async def main():
    agent = StockQueryAgent()
    workflow = agent.create_workflow()

    # 测试查询
    result = await workflow.ainvoke({
        "question": "贵州茅台的基本信息",
        "tool_name": "",
        "tool_args": {},
        "tool_result": None,
        "success": False,
        "answer": "",
        "conversation_history": []
    })

    print("回答:", result["answer"])

#程序入口
if __name__ == '__main__':
    asyncio.run(main())
