import os
from breeze_connect import BreezeConnect
import logging
from datetime import datetime

# Generate Session (replace with your secret key and session token)
# You only need to generate a new session when your session_token expires.

# Initialize BreezeConnect
breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))

def generate_session(breeze, session_token: str):
    try:
        breeze.generate_session(api_secret=os.getenv("BREEZE_SECRET_KEY"), session_token= session_token)
        print("Breeze API session established successfully.")
    except Exception as e:
        print(f"Error establishing Breeze API session: {e}")
        print("Please ensure your API Key, Secret Key, and Session Token are correct and active.")
        print("You might need to generate a new session token from the ICICIdirect API portal.")
        raise(e)






def _safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        if value is None or value == '':
            return default
        return int(float(value))  # Convert to float first to handle string decimals
    except (ValueError, TypeError):
        return default

def nse_eq(symbol, breeze_session_token):
    """
    Finds the equity data for the given symbol
    Args:
        symbol (str): The symbol of the stock
        breeze_session_token (str): The session token of the ICICI Direct Breeze API

    Returns:
        dict | None
        dict is of form {info: {lastPrice, open, dayHigh, dayLow, totalTradedVolume}}
    """
    generate_session(breeze, breeze_session_token)
    try:
        quote = breeze.get_quotes(stock_code=symbol,
                                exchange_code="NSE",
                                product_type="cash")

        if not quote or quote['Success']:
            return None

        return {
            "info": {
                "lastPrice": quote['Success']['ltp'],
                "open": quote['Success']['open'],
                "dayHigh": quote['Success']['high'],
                "dayLow": quote['Success']['low'],
                "totalTradedVolume": quote['Success']['total_buy_qty'] + quote['Success']['total_sell_qty']
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        return None



def nse_fno(symbol, expiry, breeze_session_token):
    """
    Fetch NSE F&O options data using Breeze Connect API

    Args:
        symbol (str): Stock symbol (e.g., 'RELIANCE', 'TCS')
        expiry (str): Expiry date in 'YYYY-MM-DD' or 'DD-MM-YYYY' format
        breeze_session: Initialized BreezeConnect session object

    Returns:
        dict: Dictionary with 'filtered' key containing options data
              Returns None if no data found or error occurs
    """
    generate_session(breeze, breeze_session_token)
    try:

        # Convert expiry to the format expected by Breeze API
        expiry_date = _format_expiry_date(expiry)
        if not expiry_date:
            logging.error(f"Invalid expiry date format: {expiry}")
            return None

        # Fetch options chain data from Breeze API
        try:
            # Get options chain data
            options_data = breeze.get_option_chain_quotes(
                stock_code=symbol,
                exchange_code="NFO",  # NSE F&O segment
                expiry_date=expiry_date,
                product_type="options",
                right="both"  # Both CE and PE
            )

            if not options_data or options_data.get('Status') != 200:
                logging.warning(f"No options data found for {symbol} expiry {expiry}")
                return None

        except Exception as api_error:
            logging.error(f"Breeze API error for {symbol}: {str(api_error)}")
            return None

        # Extract and process the options data
        if 'Success' in options_data and options_data['Success']:
            raw_data = options_data['Success']

            # Process the raw data into the expected format
            processed_data = _process_options_data(raw_data, symbol)

            if processed_data:
                return {
                    'filtered': {
                        'data': processed_data
                    },
                    'symbol': symbol,
                    'expiry': expiry,
                    'timestamp': datetime.now().isoformat()
                }

        logging.warning(f"No valid options data processed for {symbol}")
        return None

    except Exception as e:
        logging.error(f"Error in nse_fno for {symbol}: {str(e)}")
        return None


def _format_expiry_date(expiry_str):
    """
    Convert expiry date to format expected by Breeze API

    Args:
        expiry_str (str): Date in various formats

    Returns:
        str: Formatted date string or None if invalid
    """
    try:
        # Try different date formats
        date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d']

        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(expiry_str, fmt)
                # Return in DD-MM-YYYY format (commonly used by Breeze)
                return date_obj.strftime('%d-%m-%Y')
            except ValueError:
                continue

        return None

    except Exception:
        return None


def _process_options_data(raw_data, symbol):
    """
    Process raw options data from Breeze API into expected format

    Args:
        raw_data: Raw data from Breeze API
        symbol (str): Stock symbol

    Returns:
        list: List of dictionaries with processed options data
    """
    try:
        processed_data = []

        # Handle different possible data structures from Breeze API
        if isinstance(raw_data, list):
            options_list = raw_data
        elif isinstance(raw_data, dict) and 'data' in raw_data:
            options_list = raw_data['data']
        else:
            options_list = [raw_data] if raw_data else []

        for option in options_list:
            if not isinstance(option, dict):
                continue

            # Map Breeze API fields to expected column names
            processed_option = {
                'symbol': symbol,
                'strikePrice': _safe_float(option.get('strike_price', option.get('strikePrice', 0))),
                'lastPrice': _safe_float(option.get('ltp', option.get('lastPrice', option.get('last_price', 0)))),
                'change': _safe_float(option.get('change', option.get('net_change', 0))),
                'pChange': _safe_float(option.get('percentage_change', option.get('pChange', option.get('per_change', 0)))),
                'openInterest': _safe_int(option.get('open_interest', option.get('openInterest', 0))),
                'changeinOpenInterest': _safe_int(option.get('oi_change', option.get('changeinOpenInterest',
                                                            option.get('change_in_oi', 0)))),
                'totalTradedVolume': _safe_int(option.get('volume', option.get('totalTradedVolume',
                                                        option.get('total_traded_volume', 0)))),
                'impliedVolatility': _safe_float(option.get('iv', option.get('impliedVolatility',
                                                          option.get('implied_volatility', 0)))),
                'optionType': option.get('right', option.get('optionType', option.get('option_type', ''))),
                'expiryDate': option.get('expiry_date', option.get('expiryDate', '')),

                # Additional fields that might be useful
                'bid': _safe_float(option.get('bid', 0)),
                'ask': _safe_float(option.get('ask', 0)),
                'high': _safe_float(option.get('high', 0)),
                'low': _safe_float(option.get('low', 0)),
                'open': _safe_float(option.get('open', 0)),
                'close': _safe_float(option.get('close', 0)),
            }

            # Only add options with valid strike prices
            if processed_option['strikePrice'] > 0:
                processed_data.append(processed_option)

        return processed_data if processed_data else None

    except Exception as e:
        logging.error(f"Error processing options data: {str(e)}")
        return None


# def expiry_list(symbol):
#     """Fetch expiries from futures historical data [5]"""
#     try:
#         fut_symbol = f"{symbol}FUT"
#         instrument_token = get_instrument_token(fut_symbol, "NFO")

#         # Get historical data to extract expiries
#         data = kite.historical_data(
#             instrument_token=instrument_token,
#             from_date="2025-01-01",
#             to_date="2025-06-30",
#             interval="day"
#         )

#         expiries = list(set([d['date'].strftime("%Y-%m-%d") for d in data]))
#         return sorted(expiries)

#     except Exception as e:
#         print(f"Expiry fetch failed: {e}")
#         return None





def expiries_list(stock_code: str, breeze_session_token: str):
    """
    Fetches the available expiry dates for a given stock's Futures and Options
    contracts from NSE using the Breeze Connect API.

    Args:
        stock_code (str): The NSE stock symbol (e.g., "RELIANCE", "SBIN").
        breeze_session_token (str): The session token of the ICICI Direct Breeze API

    Returns:
        dict: A dictionary containing lists of unique expiry dates for 'futures'
              and 'options' (sorted). Returns empty lists if no data is found.
    """
    generate_session(breeze, breeze_session_token)
    expiry_dates = {
        "futures": [],
        "options": []
    }

    try:
        # --- Get Expiry Dates for Options ---
        # The get_option_chain_quotes method is excellent for this.
        # By providing only the stock_code and exchange_code, it typically returns
        # the entire option chain including available expiry dates.
        # We don't need strike_price or right initially to get all expiries.

        print(f"\nFetching options chain for {stock_code} to find expiry dates...")
        option_chain_response = breeze.get_option_chain_quotes(
            stock_code=stock_code,
            exchange_code="NFO",  # NFO is the exchange for Futures & Options
            product_type="options",
            expiry_date="",       # Leave empty to get all available expiries
            right="",             # Leave empty for both calls and puts
            strike_price=""       # Leave empty for all strike prices
        )

        if option_chain_response and option_chain_response.get('Success'):
            option_data = option_chain_response['Success']

            # Extract unique expiry dates from the options data
            unique_option_expiries = set()
            for item in option_data:
                if 'expiry_date' in item and item['expiry_date']:
                    # Breeze API returns expiry_date as a string like "2025-06-26T06:00:00.000Z"
                    # We need to parse this into a datetime object and then just the date part.
                    exp_datetime_str = item['expiry_date']
                    # Using string split for efficiency if only date part is needed
                    exp_date_only_str = exp_datetime_str.split('T')[0]
                    unique_option_expiries.add(exp_date_only_str)

            # Sort the expiry dates
            expiry_dates["options"] = sorted(list(unique_option_expiries))

            print(f"Found {len(expiry_dates['options'])} option expiries.")
        elif option_chain_response and option_chain_response.get('Error'):
            print(f"Error fetching options chain for {stock_code}: {option_chain_response['Error']}")
        else:
            print(f"No successful response for options chain for {stock_code}.")

        # --- Get Expiry Dates for Futures ---
        # Futures data might be fetched differently or implicitly included.
        # Breeze API's get_option_chain_quotes seems to be primarily for options.
        # For futures, we might need to use get_historical_data or check if
        # get_option_chain_quotes can return futures info.
        # Based on documentation, 'get_historical_data' requires a specific expiry date for futures.
        # So, we might infer futures expiries from options expiries (as they are usually the same)
        # or from a different method if Breeze has one specifically for futures contract listings.

        print(f"\nFetching futures chain for {stock_code} to find expiry dates...")
        futures_chain_response = breeze.get_option_chain_quotes(
            stock_code=stock_code,
            exchange_code="NFO",
            product_type="futures",
            expiry_date="",
            right="",
            strike_price=""
        )

        if futures_chain_response and futures_chain_response.get('Success'):
            future_data = futures_chain_response['Success']
            unique_future_expiries = set()
            for item in future_data:
                if 'expiry_date' in item and item['expiry_date']:
                    exp_datetime_str = item['expiry_date']
                    exp_date_only_str = exp_datetime_str.split('T')[0]
                    unique_future_expiries.add(exp_date_only_str)

            expiry_dates["futures"] = sorted(list(unique_future_expiries))
            print(f"Found {len(expiry_dates['futures'])} futures expiries.")
        elif futures_chain_response and futures_chain_response.get('Error'):
            print(f"Error fetching futures chain for {stock_code}: {futures_chain_response['Error']}")
        else:
            print(f"No successful response for futures chain for {stock_code}.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return expiry_dates

    # --- Example Usage ---
    # stock_symbol = "SBIN"
    # expiries = get_stock_future_options_expiries(stock_symbol)
    # print("Futures Expiries:", expiries["futures"])
    # print("Options Expiries:", expiries["options"])
