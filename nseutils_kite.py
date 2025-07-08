import logging
from datetime import datetime
import os
from kite_init import kite_init
import pandas as pd

def nse_eq(symbol: str, kite_session=None):
    """
    Finds the equity data for the given symbol using Kite Connect

    Args:
        symbol (str): The symbol of the stock (e.g., 'RELIANCE', 'TCS')
        kite_session: Initialized KiteConnect session object

    Returns:
        dict | None: Dictionary with equity data or None if error/no data
        dict is of form {info: {lastPrice, open, dayHigh, dayLow, totalTradedVolume}}
    """
    try:
        if not kite_session:
            logging.error("Kite session not provided")
            return None

        # Get instrument token for the symbol
        instrument_token = _get_nse_instrument_token(kite_session, symbol)
        logging.info("instrument token is : ", instrument_token)
        if not instrument_token:
            logging.error(f"Could not find instrument token for {symbol}")
            return None

        # Get quote data from Kite
        try:
            quote_data = kite_session.quote([instrument_token])
            logging.info("quote was : ", quote_data)

            if not quote_data or instrument_token not in quote_data:
                logging.warning(f"No quote data found for {symbol}")
                return None

            stock_quote = quote_data[instrument_token]

        except Exception as api_error:
            logging.error(f"Kite API error for {symbol}: {str(api_error)}")
            return None

        # Extract OHLC and other data
        ohlc = stock_quote.get('ohlc', {})

        # Build response in the expected format
        equity_data = {
            "info": {
                "lastPrice": _safe_float(stock_quote.get('last_price', ohlc.get('close', 0))),
                "open": _safe_float(ohlc.get('open', 0)),
                "dayHigh": _safe_float(ohlc.get('high', 0)),
                "dayLow": _safe_float(ohlc.get('low', 0)),
                "totalTradedVolume": _safe_int(stock_quote.get('volume', 0))
            }
        }

        # Validate that we have meaningful data
        if equity_data["info"]["lastPrice"] <= 0:
            logging.warning(f"Invalid price data for {symbol}")
            return None

        return equity_data

    except Exception as e:
        logging.error(f"Error in nse_eq for {symbol}: {str(e)}")
        return None


def _get_nse_instrument_token(kite_session, symbol):
    """
    Get instrument token for NSE equity symbol

    Args:
        kite_session: KiteConnect session
        symbol (str): Stock symbol

    Returns:
        str: Instrument token or None
    """
    try:
        # Get NSE instruments
        instruments = kite_session.instruments("NSE")

        # Find the exact symbol match
        for inst in instruments:
            if (inst['tradingsymbol'] == symbol and
                inst['segment'] == 'NSE' and
                inst['instrument_type'] == 'EQ'):
                return str(inst['instrument_token'])

        # If exact match not found, try case-insensitive search
        symbol_upper = symbol.upper()
        for inst in instruments:
            if (inst['tradingsymbol'].upper() == symbol_upper and
                inst['segment'] == 'NSE' and
                inst['instrument_type'] == 'EQ'):
                return str(inst['instrument_token'])

        return None

    except Exception as e:
        logging.error(f"Error getting NSE instrument token for {symbol}: {str(e)}")
        return None



# Enhanced version with additional data
def nse_eq_detailed(symbol, kite_session=None):
    """
    Enhanced version of nse_eq with additional market data

    Args:
        symbol (str): The symbol of the stock
        kite_session: Initialized KiteConnect session object

    Returns:
        dict | None: Dictionary with detailed equity data
    """
    try:
        if not kite_session:
            logging.error("Kite session not provided")
            return None

        # Get instrument token
        instrument_token = _get_nse_instrument_token(kite_session, symbol)
        if not instrument_token:
            logging.error(f"Could not find instrument token for {symbol}")
            return None

        # Get detailed quote data
        try:
            quote_data = kite_session.quote([instrument_token])

            if not quote_data or instrument_token not in quote_data:
                logging.warning(f"No quote data found for {symbol}")
                return None

            stock_quote = quote_data[instrument_token]

        except Exception as api_error:
            logging.error(f"Kite API error for {symbol}: {str(api_error)}")
            return None

        # Extract all available data
        ohlc = stock_quote.get('ohlc', {})
        depth = stock_quote.get('depth', {})

        # Build comprehensive response
        detailed_data = {
            "info": {
                # Basic OHLC data (matching original format)
                "lastPrice": _safe_float(stock_quote.get('last_price', ohlc.get('close', 0))),
                "open": _safe_float(ohlc.get('open', 0)),
                "dayHigh": _safe_float(ohlc.get('high', 0)),
                "dayLow": _safe_float(ohlc.get('low', 0)),
                "totalTradedVolume": _safe_int(stock_quote.get('volume', 0)),

                # Additional data
                "change": _safe_float(stock_quote.get('net_change', 0)),
                "pChange": _safe_float(stock_quote.get('net_change', 0) / ohlc.get('close', 1) * 100 if ohlc.get('close', 0) != 0 else 0),
                "averagePrice": _safe_float(stock_quote.get('average_price', 0)),
                "openInterest": _safe_int(stock_quote.get('oi', 0)),  # For F&O stocks
                "oiDayChange": _safe_int(stock_quote.get('oi_day_change', 0)),

                # Bid/Ask data
                "bid": _safe_float(depth.get('buy', [{}])[0].get('price', 0) if depth.get('buy') else 0),
                "ask": _safe_float(depth.get('sell', [{}])[0].get('price', 0) if depth.get('sell') else 0),
                "bidQty": _safe_int(depth.get('buy', [{}])[0].get('quantity', 0) if depth.get('buy') else 0),
                "askQty": _safe_int(depth.get('sell', [{}])[0].get('quantity', 0) if depth.get('sell') else 0),

                # Metadata
                "instrumentToken": instrument_token,
                "timestamp": stock_quote.get('timestamp', datetime.now().isoformat()),
                "lastTradeTime": stock_quote.get('last_trade_time', ''),
            }
        }

        return detailed_data

    except Exception as e:
        logging.error(f"Error in nse_eq_detailed for {symbol}: {str(e)}")
        return None


# Bulk equity data function
def nse_eq_bulk(symbols, kite_session=None):
    """
    Get equity data for multiple symbols at once

    Args:
        symbols (list): List of stock symbols
        kite_session: Initialized KiteConnect session object

    Returns:
        dict: Dictionary with symbol as key and equity data as value
    """
    try:
        if not kite_session or not symbols:
            logging.error("Kite session or symbols not provided")
            return {}

        # Get instrument tokens for all symbols
        instruments = kite_session.instruments("NSE")
        symbol_token_map = {}
        token_symbol_map = {}

        for symbol in symbols:
            for inst in instruments:
                if (inst['tradingsymbol'].upper() == symbol.upper() and
                    inst['segment'] == 'NSE' and
                    inst['instrument_type'] == 'EQ'):
                    token = str(inst['instrument_token'])
                    symbol_token_map[symbol] = token
                    token_symbol_map[token] = symbol
                    break

        if not symbol_token_map:
            logging.warning("No valid instrument tokens found for provided symbols")
            return {}

        # Get quotes for all tokens
        try:
            all_quotes = kite_session.quote(list(symbol_token_map.values()))
        except Exception as api_error:
            logging.error(f"Bulk quote API error: {str(api_error)}")
            return {}

        # Process results
        results = {}
        for token, quote_data in all_quotes.items():
            symbol = token_symbol_map.get(token)
            if not symbol or not quote_data:
                continue

            ohlc = quote_data.get('ohlc', {})

            results[symbol] = {
                "info": {
                    "lastPrice": _safe_float(quote_data.get('last_price', ohlc.get('close', 0))),
                    "open": _safe_float(ohlc.get('open', 0)),
                    "dayHigh": _safe_float(ohlc.get('high', 0)),
                    "dayLow": _safe_float(ohlc.get('low', 0)),
                    "totalTradedVolume": _safe_int(quote_data.get('volume', 0))
                }
            }

        return results

    except Exception as e:
        logging.error(f"Error in nse_eq_bulk: {str(e)}")
        return {}


# Example usage:
def nse_fno(symbol, expiry, kite_session=None):
    """
    Fetch NSE F&O options data using Kite Connect API

    Args:
        symbol (str): Stock symbol (e.g., 'RELIANCE', 'TCS')
        expiry (str): Expiry date in 'YYYY-MM-DD' or 'DD-MM-YYYY' format
        kite_session: Initialized KiteConnect session object

    Returns:
        dict: Dictionary with 'filtered' key containing options data
              Returns None if no data found or error occurs
    """
    try:
        if not kite_session:
            logging.error("Kite session not provided")
            return None

        # Convert expiry to the format expected by Kite API
        expiry_date = _format_expiry_date(expiry)
        if not expiry_date:
            logging.error(f"Invalid expiry date format: {expiry}")
            return None

        # Get instrument token for the symbol
        instrument_token = _get_instrument_token(kite_session, symbol)
        if not instrument_token:
            logging.error(f"Could not find instrument token for {symbol}")
            return None

        # Fetch options chain data from Kite API
        try:
            # Get all instruments for NFO segment
            instruments = kite_session.instruments("NFO")

            # Filter options for the specific symbol and expiry
            symbol_options = [
                inst for inst in instruments
                if (inst['name'] == symbol and
                    inst['expiry'].strftime('%Y-%m-%d') == expiry_date and
                    inst['instrument_type'] in ['CE', 'PE'])
            ]

            if not symbol_options:
                logging.warning(f"No options found for {symbol} expiry {expiry}")
                return None

            # Get quotes for all option strikes
            option_tokens = [str(inst['instrument_token']) for inst in symbol_options]

            # Fetch quotes in batches (Kite has limits on number of instruments per call)
            all_quotes = {}
            batch_size = 100  # Kite allows up to 100 instruments per quote call

            for i in range(0, len(option_tokens), batch_size):
                batch_tokens = option_tokens[i:i + batch_size]
                try:
                    batch_quotes = kite_session.quote(batch_tokens)
                    all_quotes.update(batch_quotes)
                except Exception as batch_error:
                    logging.warning(f"Error fetching batch {i//batch_size + 1}: {str(batch_error)}")
                    continue

            if not all_quotes:
                logging.warning(f"No quote data retrieved for {symbol}")
                return None

        except Exception as api_error:
            logging.error(f"Kite API error for {symbol}: {str(api_error)}")
            return None

        # Process the options data
        processed_data = _process_kite_options_data(symbol_options, all_quotes, symbol)

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


def _get_instrument_token(kite_session, symbol):
    """
    Get instrument token for the underlying symbol

    Args:
        kite_session: KiteConnect session
        symbol (str): Stock symbol

    Returns:
        str: Instrument token or None
    """
    try:
        instruments = kite_session.instruments("NSE")
        for inst in instruments:
            if inst['tradingsymbol'] == symbol:
                return inst['instrument_token']
        return None
    except Exception as e:
        logging.error(f"Error getting instrument token: {str(e)}")
        return None


def _format_expiry_date(expiry_str):
    """
    Convert expiry date to format expected by Kite API

    Args:
        expiry_str (str): Date in various formats

    Returns:
        str: Formatted date string (YYYY-MM-DD) or None if invalid
    """
    try:
        # Try different date formats
        date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d']

        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(expiry_str, fmt)
                # Return in YYYY-MM-DD format (used by Kite)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue

        return None

    except Exception:
        return None


def _process_kite_options_data(instruments, quotes, symbol):
    """
    Process Kite options data into expected format

    Args:
        instruments: List of instrument data from Kite
        quotes: Dictionary of quote data from Kite
        symbol (str): Stock symbol

    Returns:
        list: List of dictionaries with processed options data
    """
    try:
        processed_data = []

        for instrument in instruments:
            token = str(instrument['instrument_token'])
            quote_data = quotes.get(token, {})

            if not quote_data:
                continue

            # Extract OHLC and other data
            ohlc = quote_data.get('ohlc', {})

            # Map Kite API fields to expected column names
            processed_option = {
                'symbol': symbol,
                'strikePrice': _safe_float(instrument.get('strike', 0)),
                'lastPrice': _safe_float(quote_data.get('last_price', ohlc.get('close', 0))),
                'change': _safe_float(quote_data.get('net_change', 0)),
                'pChange': _safe_float(quote_data.get('net_change', 0) / ohlc.get('close', 1) * 100 if ohlc.get('close', 0) != 0 else 0),
                'openInterest': _safe_int(quote_data.get('oi', 0)),
                'changeinOpenInterest': _safe_int(quote_data.get('oi_day_change', 0)),
                'totalTradedVolume': _safe_int(quote_data.get('volume', 0)),
                'impliedVolatility': _safe_float(0),  # Kite doesn't provide IV directly
                'optionType': instrument.get('instrument_type', ''),
                'expiryDate': instrument.get('expiry', '').strftime('%Y-%m-%d') if instrument.get('expiry') else '',

                # Additional OHLC data
                'open': _safe_float(ohlc.get('open', 0)),
                'high': _safe_float(ohlc.get('high', 0)),
                'low': _safe_float(ohlc.get('low', 0)),
                'close': _safe_float(ohlc.get('close', 0)),

                # Bid/Ask data
                'bid': _safe_float(quote_data.get('depth', {}).get('buy', [{}])[0].get('price', 0) if quote_data.get('depth', {}).get('buy') else 0),
                'ask': _safe_float(quote_data.get('depth', {}).get('sell', [{}])[0].get('price', 0) if quote_data.get('depth', {}).get('sell') else 0),

                # Additional fields
                'instrumentToken': token,
                'tradingSymbol': instrument.get('tradingsymbol', ''),
                'lotSize': _safe_int(instrument.get('lot_size', 0)),
            }

            # Only add options with valid strike prices
            if processed_option['strikePrice'] > 0:
                processed_data.append(processed_option)

        # Sort by strike price and option type for better organization
        processed_data.sort(key=lambda x: (x['strikePrice'], x['optionType']))

        return processed_data if processed_data else None

    except Exception as e:
        logging.error(f"Error processing Kite options data: {str(e)}")
        return None


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



def expiries_list(symbol, kite_session = None):
    """Fetch F&O expiry dates using Kite Connect API."""
    try:
        if not kite_session:
            logging.error("Kite session not provided")
            return None

        return_format='list'

        # Fetch all NFO instruments
        instruments = kite_session.instruments("NFO")

        # Filter instruments for the given symbol and derivative type
        filtered = [
            inst for inst in instruments
            if inst['tradingsymbol'].startswith(symbol)
            and inst['instrument_type'] in ['CE', 'PE', 'FUT']
        ]

        # Extract and sort unique expiry dates
        expiries = sorted({inst['expiry'] for inst in filtered})

        # Format dates to strings
        expiries = [datetime.strftime(e, "%Y-%m-%d") for e in expiries]

        if not expiries:
            return None

        # Return in requested format
        return pd.DataFrame({'ExpiryDate': expiries}) if return_format.lower() == 'df' else expiries

    except Exception as e:
        logging.error(f"Expiry fetch error: {str(e)}")
        return None


if __name__ == "__main__":
    # Setup Kite session (use the setup from previous artifact)

    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    callback_url = "http://localhost:8000/callback"
    kite = kite_init(api_key, api_secret, callback_url)

    # Single equity data (matches your original function signature)
    symbol = "RELIANCE"
    equity_data = nse_eq(symbol, kite_session=kite)

    if equity_data:
        print(f"Last Price: {equity_data['info']['lastPrice']}")
        print(f"Open: {equity_data['info']['open']}")
        print(f"Day High: {equity_data['info']['dayHigh']}")
        print(f"Day Low: {equity_data['info']['dayLow']}")
        print(f"Volume: {equity_data['info']['totalTradedVolume']}")

    # Enhanced version with more data
    detailed_data = nse_eq_detailed(symbol, kite_session=kite)

    # Bulk data for multiple symbols
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
    bulk_data = nse_eq_bulk(symbols, kite_session=kite)
