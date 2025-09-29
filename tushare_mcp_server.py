"""
#目标：创建一个自然语言股票查询系统，让用户可以用普通中文提问（比如”查询贵州茅台最近股价“）
系统就能自动调用Tushare数据并返回答案。
#为什么用MCP?
让大模型能够安全，标准化的调用外部工具（比如TushareAPI）
#怎么实现呢？
1.导入必要的库
2.加载环境变量（从.env文件读取Tushare的token）
3.初始化MCP服务器
4.设置Tushare API
5.定义工具函数（四个工具）
6.启动服务器
"""
import tushare as ts
from mcp.server.fastmcp import FastMCP
import sys
import json
import os
from dotenv import load_dotenv

#加载环境变量，读取.env
load_dotenv()

#1.创建MCP服务器，名为StockServer
mcp = FastMCP("StockServer")

#2.设置Tushare API，用token创建一个TushareAPI连接，把它赋值给api变量
token = os.getenv('TUSHARE_TOKEN')
api = ts.pro_api(token)

#3.定义一个工具函数:获取股票基本信息
@mcp.tool()
def get_stock_basic(stock_code):
    """
    获取股票基本信息
    Args:stock_code:股票代码，000001.SZ,值传递给ts_code
    Return:JSON格式的股票基本信息
    步骤：装饰器给工具贴标签--调用Tushare API获取数据--把数据打包成JSON格式--异常处理
    api.stock_basic：调用Tushare API的stock_basic方法
    ts_code:API 函数定义的参数名
    str(e)：把异常对象e转换成字符串
    """
    try:
        result = api.stock_basic(ts_code=stock_code)
        return result.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return json.dumps({'error': str(e)})

#4.添加第二个工具函数:获取股票价格
@mcp.tool()
def get_stock_price(stock_code):
   """
   获取股票最近的价格数据
   调用api.daily获取每日价格数据
   错误返回直接是字符串而不是JSON
   """
   try:
       result = api.daily(ts_code=stock_code)
       return result.to_json(orient='records', force_ascii=False)
   except Exception as e:
       return f"错误；{str(e)}"

#5.第三个工具函数：获取股票实时报价（需要相应权限）
@mcp.tool()
def get_realtime_price(stock_code):
    """获取股票实时报价"""
    try:
        result = api.realtime_price(ts_code=stock_code)
        return result.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return json.dumps({'error': str(e)})

#6.第四个工具函数：获取财务指标数据
@mcp.tool()
def get_financial_indicator(stock_code, period='20231231'):
    """
    获取财务指标数据
    Args:stock_code,period
    period='20231231' = 默认参数，如果不传period，就用'20231231'
    """
    try:
        result = api.financial_indicator(ts_code=stock_code, period=period)
        return result.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return json.dumps({'error': str(e)})

#7.启动服务器
if __name__ == "__main__":
    mcp.run(transport="stdio")

#用标准输入输出方式启动MCP服务器

