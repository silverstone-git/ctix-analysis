# CTIX Analysis Project

This project is designed for analyzing stock market data, particularly focusing on the Indian stock market (NSE). It includes tools for data acquisition, sentiment analysis, and generating trading signals.

## Overview

The project leverages various Python libraries and APIs to gather stock data, perform technical analysis, and create reports. It's structured to facilitate both fundamental and quantitative analysis, incorporating machine learning models for prediction.

## Key Components

-   **`nseutils_breeze.py`:** Contains utilities for fetching data from the ICICI Direct Breeze API.
-   **`nseutils_kite.py`:** Contains utilities for fetching data from the Zerodha Kite Connect API.
-   **`kite_init.py`:** Manages the initialization and authentication with the Kite Connect API.
-   **`openai_agents_researcher.py`:** Defines agents for performing comprehensive financial analysis, incorporating sentiment analysis and stock price prediction.
-   **`stock_price_daily.py`:** Provides functions for fetching historical stock data from Yahoo Finance and Alpha Vantage APIs.
-   **`search.py`:** Implements web search functionality using the Brave Search API.
-   **`localtools.py`:** Provides utility functions, such as saving output to files.
-   **`main.py`:** Sets up and launches the Gradio interface for interacting with the analysis tools.
-   **`perp.py`:** An alternative entry point with different agent configurations, designed for perpetual execution to continue running.
-   **`requirements.txt`:** Lists the Python packages required to run the project.
-   **`outputs/`:** A directory where the generated reports and charts are stored.
-   **`pyrightconfig.json`:** Configuration file for the Pyright type checker.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Variables:**
    Create a `.env` file in the root directory and set the following environment variables:
    -   `BRAVE_API_KEY`: API key for the Brave Search API.
    -   `KITE_API_KEY`: API key for the Zerodha Kite Connect API.
    -   `KITE_API_SECRET`: API secret for the Zerodha Kite Connect API.
    -   `ALPHA_VANTAGE_API_KEY`: API key for the Alpha Vantage API.
    -   `GEMINI_API_KEY`: API key for the Google Gemini API (if using Gemini models).
    -   `HF_TOKEN`: API key for the Hugging Face Inference API (if using Hugging Face models).
    -   `MODEL_NAME`: Choose LLM Model name. Default to `gemini/gemini-2.0-flash`

    Example `.env` file:
    ```
    BRAVE_API_KEY=YOUR_BRAVE_API_KEY
    KITE_API_KEY=YOUR_KITE_API_KEY
    KITE_API_SECRET=YOUR_KITE_API_SECRET
    ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_API_KEY
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    HF_TOKEN=YOUR_HF_TOKEN
    MODEL_NAME=gemini/gemini-2.0-flash
    ```

## Usage

1.  **Run the main script:**

    ```bash
    python main.py
    ```

    This will launch the Gradio interface, allowing you to interact with the agent through a chat interface or voice input.
2.  **Run the perpetual agent script:**

    ```bash
    python perp.py
    ```

    This will launch Gradio.
3.  **Accessing the Interface:**
    Open your web browser and navigate to the address provided in the console output (e.g., `http://127.0.0.1:7860`).

## Notes

-   The `get_data()` function in `NSE_CTIX_ANALYSIS.ipynb` is designed for use within Google Colab and requires a Google Drive connection.
-   Some notebook cells are specific to the Google Colab environment and may not work directly in other environments.
-   The project uses a mix of hardcoded parameters and external dependencies, so results may vary depending on the availability and quality of external data sources.
-   The Kite API key is handled via environment variables for security. Be sure to set these up correctly.

## Directory Structure

```
ctix-analysis/
├── .gitignore
├── outputs/
├── .ipynb_checkpoints/
├── .gradio/
├── logs/
├── venv_ctix/
├── prompt.txt
├── LICENSE
├── NIFTY_PREDICTION.ipynb
├── NSE_CTIX_ANALYSIS.ipynb
├── get_market_sentiment.py
├── kite_init.py
├── localtools.py
├── main.py
├── openai_agents_researcher.py
├── openai_utils.py
├── pyrightconfig.json
└── requirements.txt
```

## Contact

For questions or issues, please contact [cyto/aryan_sidhwani@protonmail.com].
