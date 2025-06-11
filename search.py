import os
import requests
import json
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from smolagents import tool

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _fetch_and_extract_text(url: str, timeout: int = 10) -> dict:
    """
    Fetches the content of a URL and extracts all visible text (excluding HTML/CSS).

    Args:
        url (str): The URL to fetch.
        timeout (int): Timeout in seconds for the HTTP request.

    Returns:
        dict: A dictionary containing the URL and its extracted text, or an error.
    """
    try:
        logging.info(f"Attempting to fetch and parse URL: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style tags
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get text, strip whitespace, and handle line breaks
        text = soup.get_text(separator='\n', strip=True)

        return {
            "url": url,
            "extracted_text": text
        }
    except requests.exceptions.Timeout:
        logging.warning(f"Timeout fetching URL: {url}")
        return {"url": url, "error": f"Timeout after {timeout} seconds"}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return {"url": url, "error": f"Failed to fetch: {e}"}
    except Exception as e:
        logging.error(f"An unexpected error occurred while parsing {url}: {e}")
        return {"url": url, "error": f"Error parsing content: {e}"}


@tool
def brave_search_tool(query: str, num_results: int = 5, fetch_full_text: bool = False, full_text_timeout: int = 10) -> str:
    """
    Performs a web search using the Brave Search API and returns the results.
    Optionally fetches and extracts text from the top search results.

    Args:
        query (str): The search query.
        num_results (int): The maximum number of search results to return from Brave Search.
                           Defaults to 5.
        fetch_full_text (bool): If True, attempts to fetch and extract text from the URLs
                                of the top results. Defaults to False.
        full_text_timeout (int): Timeout in seconds for fetching each full text.
                                Defaults to 10 seconds.

    Returns:
        str: A JSON string of the search results, optionally including extracted text,
             or an error message.
             JSON output is of the form:
             {"title", "url", "snippet", "full_text"}
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return json.dumps({"error": "Brave Search API key not found. Please set the BRAVE_API_KEY environment variable."})

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": num_results,
        "offset": 0,
        "country": "us",
        "search_lang": "en"
    }

    try:
        logging.info(f"Initiating Brave Search for query: '{query}' with {num_results} results.")
        response = requests.get(url, headers=headers, params=params, timeout=15) # Brave API call timeout
        response.raise_for_status()

        data = response.json()
        raw_web_results = []
        if 'web' in data and 'results' in data['web']:
            raw_web_results = data['web']['results']
        else:
            logging.warning("No web results found in Brave Search response.")
            return json.dumps({"error": "No web results found in Brave Search response.", "raw_response": data})

        formatted_results = []
        urls_to_fetch = []

        for result in raw_web_results:
            formatted_item = {
                "title": result.get("title"),
                "url": result.get("url"),
                "snippet": result.get("description")
            }
            formatted_results.append(formatted_item)
            if fetch_full_text and result.get("url"):
                urls_to_fetch.append(result["url"])

        if fetch_full_text and urls_to_fetch:
            logging.info(f"Fetching full text for {len(urls_to_fetch)} URLs with {full_text_timeout}s timeout per URL.")
            # Use ThreadPoolExecutor to fetch URLs concurrently for efficiency
            with ThreadPoolExecutor(max_workers=min(len(urls_to_fetch), 5)) as executor: # Limit concurrent fetches
                future_to_url = {executor.submit(_fetch_and_extract_text, url, full_text_timeout): url for url in urls_to_fetch}
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        extracted_data = future.result()
                        # Find the corresponding result in formatted_results and add extracted text
                        for item in formatted_results:
                            if item["url"] == url:
                                item["full_text"] = extracted_data.get("extracted_text")
                                if "error" in extracted_data:
                                    item["full_text_error"] = extracted_data["error"]
                                break
                    except Exception as exc:
                        logging.error(f"URL {url} generated an exception during full text fetch: {exc}")
                        for item in formatted_results:
                            if item["url"] == url:
                                item["full_text_error"] = f"Failed to get full text due to internal error: {exc}"
                                break

        return json.dumps(formatted_results, indent=2)

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred during Brave Search: {http_err} - Status: {response.status_code}")
        return json.dumps({"error": f"HTTP error occurred with Brave Search: {http_err}", "status_code": response.status_code, "response_text": response.text})
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred during Brave Search: {conn_err}")
        return json.dumps({"error": f"Connection error occurred with Brave Search: {conn_err}"})
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred during Brave Search API call: {timeout_err}")
        return json.dumps({"error": f"Timeout error occurred with Brave Search API: {timeout_err}"})
    except requests.exceptions.RequestException as req_err:
        logging.error(f"An unexpected request error occurred during Brave Search: {req_err}")
        return json.dumps({"error": f"An unexpected request error occurred with Brave Search: {req_err}"})
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON response from Brave Search API.")
        return json.dumps({"error": "Failed to decode JSON response from Brave Search API."})
    except Exception as e:
        logging.error(f"An unexpected error occurred in brave_search_tool: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred in brave_search_tool: {e}"})

# Example usage (for testing the tool function independently)
if __name__ == "__main__":
    # For testing, you might temporarily set the API key here or ensure it's in your env
    # os.environ["BRAVE_API_KEY"] = "YOUR_BRAVE_API_KEY" # REMOVE IN PRODUCTION
    # If not set, the tool will return an error about missing key

    print("--- Testing Brave Search Tool with Full Text Fetch ---")
    search_query = "Impact of AI on job market latest research"

    # Test 1: Basic search (no full text)
    print("\n--- Test 1: Basic Search ---")
    results_basic = brave_search_tool(search_query, num_results=2, fetch_full_text=False)
    print(results_basic)

    # Test 2: Search with full text fetching
    print("\n--- Test 2: Search with Full Text Fetch (num_results=2, timeout=5s) ---")
    results_full_text = brave_search_tool(search_query, num_results=2, fetch_full_text=True, full_text_timeout=5)
    print(results_full_text)

    print("\n--- Test 3: Search with Full Text Fetch (num_results=1, very short timeout) ---")
    results_short_timeout = brave_search_tool("impact of climate change on agriculture", num_results=1, fetch_full_text=True, full_text_timeout=1)
    print(results_short_timeout)

    print("\n--- Testing with missing API Key (example of error handling) ---")
    original_key = os.getenv("BRAVE_API_KEY")
    if original_key:
        del os.environ["BRAVE_API_KEY"] # Temporarily unset for test
    missing_key_results = brave_search_tool("test")
    print(missing_key_results)
    if original_key:
        os.environ["BRAVE_API_KEY"] = original_key # Restore for other parts of the script
