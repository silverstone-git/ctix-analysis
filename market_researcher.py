import os
from smolagents import GradioUI, LiteLLMModel, ToolCallingAgent
from smolagents.default_tools import FinalAnswerTool, PythonInterpreterTool, UserInputTool, VisitWebpageTool
from localtools import output_save_to_file

from search import brave_search_tool
from stock_price_daily import get_yahoo_finance_data

python_interpretor_tool= PythonInterpreterTool()
final_answer_tool= FinalAnswerTool()
user_input_tool= UserInputTool()
visit_webpage_tool= VisitWebpageTool()

model = LiteLLMModel(model_id="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))
agent = ToolCallingAgent(tools=[
    get_yahoo_finance_data,
    brave_search_tool,
    #get_nse_india_data,
    python_interpretor_tool,
    final_answer_tool, user_input_tool,
    visit_webpage_tool, output_save_to_file
], model=model)

company= "Tata Consultancy Services, listed on NSE India as TCS"
fundamental_time_period= "q1 and q2 2025"
quantitative_time_period="June 10th 2025"
# agent.run(

#     "Could you make a detailed report about this company: {}, on the aspects of fundamental research, and market sentiment analysis during {} and quantitative research for {}? Save the final report."
#     .format(company, fundamental_time_period, quantitative_time_period),
# )

ui= GradioUI(agent)
ui.launch()
