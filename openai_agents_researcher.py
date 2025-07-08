import datetime
import json
import os
import asyncio
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from agents import Agent, function_tool, Runner, trace, set_default_openai_client
from get_market_sentiment import get_nse_india_data
from stock_price_daily import get_yahoo_finance_data
from search import brave_search_tool
from openai import AsyncOpenAI


@function_tool(name_override="predict_stock_trend_gb", description_override="")
def predict_stock_trend_gb(json_data: str) -> str:
    """
    Uses Gradient Boosting to predict future stock price trends.
    Performs trend prediction on stock data using a Gradient Boosting Regressor.
    """
    try:
        print("Performing Gradient Boosting trend prediction...")
        df = pd.read_json(json_data)
        df['Date'] = df.index
        df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())

        X = df[['Date_ordinal']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        gb.fit(X_train, y_train)

        future_dates = pd.to_datetime(np.arange(df.index[-1].toordinal() + 1, df.index[-1].toordinal() + 31))
        future_dates_ordinal = [d.toordinal() for d in future_dates]
        predictions = gb.predict(np.array(future_dates_ordinal).reshape(-1, 1))

        result = {
            "prediction_summary": f"Predicted trend for the next 30 days shows an average price of ${np.mean(predictions):.2f}.",
            "predictions": list(predictions),
            "future_dates": [d.strftime('%Y-%m-%d') for d in future_dates]
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Failed during trend prediction: {e}"})



@function_tool(name_override="create_financial_charts", description_override="Generates and saves financial charts.")
def create_financial_charts(json_data: str, prediction_data: str | None = None) -> str:
    """
    Creates and saves a price chart and optionally a trend prediction chart.
    """
    try:
        print("Generating financial charts...")
        os.makedirs("./outputs", exist_ok=True)
        df = pd.read_json(json_data)
        df.index = pd.to_datetime(df.index)

        # Create and save Price Chart
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Close Price', color='skyblue')
        ax.set_title('Historical Stock Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        price_chart_path = "./outputs/price_chart.png"
        plt.savefig(price_chart_path)
        plt.close(fig)

        chart_paths = [price_chart_path]

        # Create and save Prediction Chart if data is available
        if prediction_data:
            pred_dict = json.loads(prediction_data)
            future_dates = pd.to_datetime(pred_dict['future_dates'])
            predictions = pred_dict['predictions']

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df['Close'], label='Historical Close', color='skyblue')
            ax.plot(future_dates, predictions, label='Predicted Trend', color='coral', linestyle='--')
            ax.set_title('Stock Price Trend Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            prediction_chart_path = "./outputs/trend_prediction.png"
            plt.savefig(prediction_chart_path)
            plt.close(fig)
            chart_paths.append(prediction_chart_path)

        return json.dumps({"chart_paths": chart_paths})
    except Exception as e:
        return json.dumps({"error": f"Failed to create charts: {e}"})

# --- Agent Definitions ---

def build_agents():
    """Builds and returns the set of specialist agents and the portfolio manager."""

    # Fundamental Agent
    fundamental_agent = Agent(
        name="Fundamental Agent",
        instructions="Analyze the market sentiment for a stock by searching for recent news. Provide a summary of the news and a sentiment score (Positive, Neutral, Negative).",
        model="deepseek/deepseek-r1-0528-qwen3-8b",
        tools=[brave_search_tool]
    )

    # Quantitative Agent
    quantitative_agent = Agent(
        name="Quantitative Agent",
        instructions="Perform quantitative analysis on a stock. Fetch historical data, predict future trends using gradient boosting, and generate relevant charts.",
        model="deepseek/deepseek-r1-0528-qwen3-8b",
        tools=[get_yahoo_finance_data, get_nse_india_data, predict_stock_trend_gb, create_financial_charts]
    )

    # Mocking the `make_agent_tool` and parallel execution logic from the provided context
    def make_agent_tool(agent, name, description):
        @function_tool(name_override=name, description_override=description)
        async def agent_tool(input_query):
            # In a real SDK, you'd run the agent here. We'll simulate a response.
            print(f"--- Simulating call to Specialist Agent: {agent.name} ---")
            print(f"Input: {input_query}")
            if agent.name == "Fundamental Agent":
                return brave_search_tool(query=input_query)
            elif agent.name == "Quantitative Agent":
                # Simulate the quant workflow
                today = datetime.date.today()
                start_date = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
                end_date = today.strftime('%Y-%m-%d')

                stock_data = get_yahoo_finance_data(input_query, start_date, end_date)
                prediction = predict_stock_trend_gb(stock_data)
                charts = create_financial_charts(stock_data, prediction)

                return json.dumps({
                    "historical_data": json.loads(stock_data),
                    "trend_prediction": json.loads(prediction),
                    "charts": json.loads(charts)
                })
            return json.dumps({"error": "Unknown agent for tool call"})
        return agent_tool

    fundamental_tool = make_agent_tool(fundamental_agent, "fundamental_analysis", "Generate the Fundamental Analysis section.")
    quant_tool = make_agent_tool(quantitative_agent, "quantitative_analysis", "Generate the Quantitative Analysis section.")


    # Head Portfolio Manager Agent
    head_pm_agent = Agent(
        name="Head Portfolio Manager Agent",
        instructions="You are a world-class portfolio manager. Your goal is to create a concise and insightful financial report on a given stock. Use your specialist agents to gather fundamental and quantitative analysis, then synthesize their findings into a final report. The report must contain a summary, a fundamental section with news sentiment, a quantitative section with trend predictions, and references to generated charts.",
        model= "deepseek/deepseek-r1-0528-qwen3-8b",
        tools=[fundamental_tool, quant_tool]
    )

    return {
        "fundamental": fundamental_agent,
        "quant": quantitative_agent,
        "head_pm": head_pm_agent,
    }

# --- Main Workflow ---

async def run_workflow():
    """Sets up the environment and runs the multi-agent workflow."""

    custom_client = AsyncOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=os.getenv("HF_TOKEN"))
    set_default_openai_client(custom_client)


    today_str = datetime.date.today().strftime("%B %d, %Y")
    question = (
        f"Today is {today_str}. "
        "Please provide a full financial report on Google (GOOGL). "
        "I need to understand the current market sentiment based on news, and a quantitative analysis of its price trend."
    )

    agents = build_agents()
    head_pm = agents["head_pm"]

    print("Running multi-agent workflow...\n")
    with trace("Financial Report Workflow", metadata={"question": question[:512]}) as workflow_trace:
        print(
            f"\n🔗 Mock Trace ID: {workflow_trace.trace_id}\n"
        )

        response = None
        try:
            # The Runner would internally orchestrate the tool calls to the specialist agents
            response = await asyncio.wait_for(
                Runner.run(head_pm, question, max_turns=10),
                timeout=1200
            )
        except asyncio.TimeoutError:
            print("\n❌ Workflow timed out.")

        print(f"\n✅ Workflow Completed.")
        if response and 'final_output' in response:
            print("\n--- Final Generated Report ---")
            # Pretty print the final JSON report
            report_data = json.loads(response['final_output'])
            print(json.dumps(report_data, indent=4))

            report_path = "./outputs/financial_report.json"
            os.makedirs("./outputs", exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            print(f"\nReport saved to: {report_path}")

            if report_data.get("graphs"):
                 print("\nGenerated Charts:")
                 for path in report_data["graphs"]:
                     print(f"- {path}")
        else:
            print("No final output was generated.")


if __name__ == "__main__":
    # In a Jupyter notebook cell, you would run: await run_workflow()
    # For a standard Python script, we use asyncio.run()

    asyncio.run(run_workflow())
