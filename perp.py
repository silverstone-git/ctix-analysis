import os
from smolagents import LiteLLMModel, ToolCallingAgent
from search import brave_search_tool
from smolagents.default_tools import FinalAnswerTool, UserInputTool, VisitWebpageTool
from localtools import output_save_to_file
from smolagents import GradioUI


def main():
    # python_interpretor_tool= PythonInterpreterTool()
    final_answer_tool= FinalAnswerTool()
    user_input_tool= UserInputTool()
    visit_webpage_tool= VisitWebpageTool()

    model = LiteLLMModel(model_id="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))
    agent = ToolCallingAgent(tools=[brave_search_tool,
        final_answer_tool, user_input_tool,
        visit_webpage_tool, output_save_to_file], model=model, add_base_tools= False)

    # prompt= "Make a python function for getting the nse expiries for a given equity stock symbol from NSE api using nse library BennyThadikaran/NseIndiaApi. the output should be a list of datetime objects"
    # prompt= "Does investing in Indian Solar Companies make sense, as of June 10, 2025 ? Do some analysis and give a detailed report. Also list some highly performant and promising companies"
    # prompt="Do you think North India pollution problem can be solved, as of June 11th 2025. What are the recommended steps for urban cities according to research? What steps are being taken? Generate a report and save the output in the end"
    # prompt= "What were the conclusions of the India Pakistan Conflict of May 2025 ? How did it end? What do security experts say about this? Who won, diplomatically speaking? Generate a report and save the output."
    # prompt= "How to do detailed market analysis using fundamental and quantitative strategies on Indian Companies. Implement top 5 strategies in Python using Machine Learning and Black Scholes Model. Use a trustworthy API / SDK for data fetching from Indian Stock Exchange like NSE and BSE. Store the outputs in the end."
    ui= GradioUI(agent)
    ui.launch()


if __name__ == "__main__":
    main()
