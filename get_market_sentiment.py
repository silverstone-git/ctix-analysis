import json
from datetime import date, datetime
from typing import Dict, List
import pandas as pd
import logging
import numpy as np
from smolagents import tool
import nseutils_breeze
import nseutils_kite
import asyncio
from kite_init import kite_init
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@tool
def get_nse_india_data(symbol: str) -> str:
    """
    Fetches current market data and performs comprehensive sentiment analysis
    using futures and options data for a given symbol from NSE India.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'RELIANCE', 'TCS').
                        Note: NSE symbols are typically uppercase.
    Returns:
        str: A JSON string containing the fetched data and comprehensive sentiment analysis,
             or an error message.
    """

    breeze_session_token: str = ""
    # breeze_session_token (str): (Optional) The session token of the ICICI Direct Breeze API, to be input by the user
    #                 Note: defaults to empty string. If not entered, Zerodha Kite Connect is used instead of ICICI Breeze API

    kite = None
    if breeze_session_token == "":
        api_key = os.getenv("KITE_API_KEY")
        api_secret = os.getenv("KITE_API_SECRET")
        callback_url = "http://localhost:8000/callback"
        print("GET MARKET SENTIMENT: getting kite session")
        kite = asyncio.run(kite_init(api_key, api_secret, callback_url))
        print("GET MARKET SENTIMENT: got kite session")

    logging.info(f"Fetching NSE India data for {symbol}...")
    result_data = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "price": None,
        "open": None,
        "high": None,
        "low": None,
        "volume": None,
        "change_percent": None,
        "sentiment_score": 0,  # -100 to +100 scale
        "sentiment_label": "NEUTRAL",
        "confidence_level": "LOW",
        "key_insights": [],
        "options_analysis": {},
        "futures_analysis": {},
        "risk_metrics": {},
        "trading_signals": []
    }

    try:
        # --- 1. Fetch Current Equity Data ---
        if breeze_session_token != "":
            equity_data = nseutils_breeze.nse_eq(symbol, breeze_session_token)
        else:
            equity_data = nseutils_kite.nse_eq(symbol, kite_session = kite)
        print("nse eq results: ", equity_data)

        if equity_data and 'info' in equity_data and 'lastPrice' in equity_data['info']:
            info = equity_data['info']
            result_data["price"] = float(info.get("lastPrice", 0))
            result_data["open"] = float(info.get("open", 0))
            result_data["high"] = float(info.get("dayHigh", 0))
            result_data["low"] = float(info.get("dayLow", 0))
            result_data["volume"] = int(info.get("totalTradedVolume", 0))

            # Calculate intraday change percentage
            if result_data["open"] > 0:
                result_data["change_percent"] = ((result_data["price"] - result_data["open"]) / result_data["open"]) * 100

            logging.info(f"Successfully fetched equity data for {symbol}.")
        else:
            logging.warning(f"Could not fetch live equity data for {symbol}.")
            result_data["key_insights"].append("⚠️ Could not retrieve live equity data")

        # --- 2. Enhanced Futures & Options Analysis ---
        if breeze_session_token != "":
            expiries = nseutils_breeze.expiries_list(symbol, breeze_session_token= breeze_session_token)
        else:
            expiries = nseutils_kite.expiries_list(symbol, kite_session = kite)
        print("expiries are: ", expiries)

        if expiries == None or (isinstance(expiries, pd.DataFrame) and expiries.empty):
            logging.warning(f"No expiry dates found for {symbol}.")
            result_data["key_insights"].append("❌ No F&O data available - analysis limited to spot price")
            return json.dumps(result_data, indent=2)

        # Analyze multiple expiries for better sentiment understanding
        nearest_expiry = expiries[0]
        next_expiry = expiries[1] if len(expiries) > 1 else None

        logging.info(f"Analyzing F&O data for {symbol} - Expiries: {nearest_expiry}, {next_expiry}")

        # Get F&O data for nearest expiry
        #
        if breeze_session_token != "":
            fno_data = nseutils_breeze.nse_fno(symbol, expiry=nearest_expiry, breeze_session_token= breeze_session_token)
        else:
            fno_data = nseutils_kite.nse_fno(symbol, expiry= nearest_expiry, kite_session = kite)
        print("fno results: ", fno_data)


        if fno_data and 'filtered' in fno_data and 'data' in fno_data['filtered']:
            options_df = pd.DataFrame(fno_data['filtered']['data'])

            # Enhanced data cleaning and preprocessing
            options_df = _clean_options_data(options_df)

            # Perform comprehensive options analysis
            options_analysis = _analyze_options_comprehensive(options_df, result_data["price"])
            result_data["options_analysis"] = options_analysis

            # Attempt futures analysis
            futures_analysis = _analyze_futures_sentiment(symbol, nearest_expiry, result_data["price"])
            result_data["futures_analysis"] = futures_analysis

            # Calculate composite sentiment score
            sentiment_components = _calculate_sentiment_score(options_analysis, futures_analysis, result_data)
            result_data.update(sentiment_components)

            # Generate risk metrics
            result_data["risk_metrics"] = _calculate_risk_metrics(options_df, result_data["price"])

            # Generate actionable trading signals
            result_data["trading_signals"] = _generate_trading_signals(
                options_analysis, futures_analysis, result_data["price"], sentiment_components
            )

            # Add key insights summary
            result_data["key_insights"] = _generate_key_insights(
                options_analysis, futures_analysis, sentiment_components, result_data
            )

        else:
            logging.warning(f"No F&O data found for {symbol}.")
            result_data["key_insights"].append("❌ No F&O data available for comprehensive analysis")

    except Exception as e:
        logging.error(f"Error fetching NSE data for {symbol}: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to retrieve NSE data for {symbol}: {e}"})

    return json.dumps(result_data, indent=2)


def _clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess options data for analysis."""
    numeric_cols = [
        'openInterest', 'changeinOpenInterest', 'totalTradedVolume',
        'lastPrice', 'strikePrice', 'impliedVolatility', 'change', 'pChange'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Remove rows with invalid strike prices
    res = df[df['strikePrice'] > 0]

    # Add derived columns
    res['moneyness'] = res['strikePrice'] / res['lastPrice'].replace(0, np.nan)
    res['volume_oi_ratio'] = res['totalTradedVolume'] / res['openInterest'].replace(0, np.nan)

    return res


def _analyze_options_comprehensive(df: pd.DataFrame, spot_price: float) -> Dict:
    """Perform comprehensive options chain analysis."""
    if df.empty or spot_price <= 0:
        return {"error": "Insufficient data for options analysis"}

    calls = df[df['optionType'] == 'CE'].copy()
    puts = df[df['optionType'] == 'PE'].copy()

    analysis = {
        "pcr_oi": None,
        "pcr_volume": None,
        "total_call_oi": int(calls['openInterest'].sum()),
        "total_put_oi": int(puts['openInterest'].sum()),
        "total_call_volume": int(calls['totalTradedVolume'].sum()),
        "total_put_volume": int(puts['totalTradedVolume'].sum()),
        "call_oi_change": int(calls['changeinOpenInterest'].sum()),
        "put_oi_change": int(puts['changeinOpenInterest'].sum()),
    }

    # Put-Call Ratios
    analysis["pcr_oi"] = analysis["total_put_oi"] / max(analysis["total_call_oi"], 1)
    analysis["pcr_volume"] = analysis["total_put_volume"] / max(analysis["total_call_volume"], 1)

    # Support and Resistance Analysis
    support_resistance = _find_support_resistance_levels(calls, puts, spot_price)
    analysis.update(support_resistance)

    # Volatility Analysis
    volatility_analysis = _analyze_implied_volatility(calls, puts, spot_price)
    analysis.update(volatility_analysis)

    # Greeks Analysis (if available)
    greeks_analysis = _analyze_options_greeks(calls, puts)
    analysis.update(greeks_analysis)

    # Pain Point Analysis
    pain_analysis = _calculate_max_pain(calls, puts)
    analysis.update(pain_analysis)

    # Activity Concentration
    activity_analysis = _analyze_strike_activity(calls, puts, spot_price)
    analysis.update(activity_analysis)

    return analysis


def _find_support_resistance_levels(calls: pd.DataFrame, puts: pd.DataFrame, spot_price: float) -> Dict:
    """Identify key support and resistance levels from options data."""
    analysis = {}

    if not calls.empty:
        # Major resistance (highest call OI)
        max_call_oi_idx = calls['openInterest'].idxmax()
        analysis["major_resistance"] = float(calls.loc[max_call_oi_idx, 'strikePrice'])
        analysis["resistance_oi"] = int(calls.loc[max_call_oi_idx, 'openInterest'])

        # Call writing analysis (negative change in OI indicates writing)
        call_writing = calls[calls['changeinOpenInterest'] > 1000].sort_values('changeinOpenInterest', ascending=False)
        if not call_writing.empty:
            analysis["fresh_call_writing"] = call_writing[['strikePrice', 'changeinOpenInterest']].head(3).to_dict('records')

    if not puts.empty:
        # Major support (highest put OI)
        max_put_oi_idx = puts['openInterest'].idxmax()
        analysis["major_support"] = float(puts.loc[max_put_oi_idx, 'strikePrice'])
        analysis["support_oi"] = int(puts.loc[max_put_oi_idx, 'openInterest'])

        # Put writing analysis
        put_writing = puts[puts['changeinOpenInterest'] > 1000].sort_values('changeinOpenInterest', ascending=False)
        if not put_writing.empty:
            analysis["fresh_put_writing"] = put_writing[['strikePrice', 'changeinOpenInterest']].head(3).to_dict('records')

    # Calculate distance from current levels
    if "major_resistance" in analysis:
        analysis["resistance_distance_pct"] = ((analysis["major_resistance"] - spot_price) / spot_price) * 100

    if "major_support" in analysis:
        analysis["support_distance_pct"] = ((spot_price - analysis["major_support"]) / spot_price) * 100

    return analysis


def _analyze_implied_volatility(calls: pd.DataFrame, puts: pd.DataFrame, spot_price: float) -> Dict:
    """Analyze implied volatility patterns for sentiment insights."""
    analysis = {}

    # Get ATM options (closest to spot price)
    atm_calls = calls.iloc[(calls['strikePrice'] - spot_price).abs().argsort()[:3]]
    atm_puts = puts.iloc[(puts['strikePrice'] - spot_price).abs().argsort()[:3]]

    if not atm_calls.empty and 'impliedVolatility' in atm_calls.columns:
        analysis["atm_call_iv"] = float(atm_calls['impliedVolatility'].mean())

    if not atm_puts.empty and 'impliedVolatility' in atm_puts.columns:
        analysis["atm_put_iv"] = float(atm_puts['impliedVolatility'].mean())

        # IV skew analysis
        if "atm_call_iv" in analysis:
            analysis["iv_skew"] = analysis["atm_put_iv"] - analysis["atm_call_iv"]
            analysis["skew_interpretation"] = (
                "Put premium higher - bearish bias" if analysis["iv_skew"] > 2
                else "Call premium higher - bullish bias" if analysis["iv_skew"] < -2
                else "Neutral IV bias"
            )

    return analysis


def _analyze_options_greeks(calls: pd.DataFrame, puts: pd.DataFrame) -> Dict:
    """Analyze options Greeks if available (delta, gamma, theta, vega)."""
    # Note: NSE data might not always include Greeks
    # This is a placeholder for when Greeks data is available
    analysis = {}

    greek_columns = ['delta', 'gamma', 'theta', 'vega']
    available_greeks = [col for col in greek_columns if col in calls.columns or col in puts.columns]

    if available_greeks:
        analysis["greeks_available"] = available_greeks
        # Add specific Greeks analysis when data is available
    else:
        analysis["greeks_available"] = []

    return analysis


def _calculate_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> Dict:
    """Calculate Max Pain point - strike price where option writers lose least money."""
    analysis = {}

    try:
        # Get unique strike prices
        strikes = sorted(set(calls['strikePrice'].tolist() + puts['strikePrice'].tolist()))

        pain_values = []

        for strike in strikes:
            # Calculate total pain at this strike
            call_pain = calls[calls['strikePrice'] < strike]['openInterest'].sum() * (strike - calls[calls['strikePrice'] < strike]['strikePrice']).sum()
            put_pain = puts[puts['strikePrice'] > strike]['openInterest'].sum() * (puts[puts['strikePrice'] > strike]['strikePrice'] - strike).sum()

            total_pain = call_pain + put_pain
            pain_values.append((strike, total_pain))

        if pain_values:
            max_pain_strike, min_pain_value = min(pain_values, key=lambda x: x[1])
            analysis["max_pain_strike"] = float(max_pain_strike)
            analysis["max_pain_value"] = float(min_pain_value)

    except Exception as e:
        analysis["max_pain_error"] = str(e)

    return analysis


def _analyze_strike_activity(calls: pd.DataFrame, puts: pd.DataFrame, spot_price: float) -> Dict:
    """Analyze activity concentration around specific strikes."""
    analysis = {}

    # Define ATM range (±5% of spot price)
    atm_range = spot_price * 0.05
    atm_strikes = [strike for strike in set(calls['strikePrice'].tolist() + puts['strikePrice'].tolist())
                   if abs(strike - spot_price) <= atm_range]

    if atm_strikes:
        atm_calls = calls[calls['strikePrice'].isin(atm_strikes)]
        atm_puts = puts[puts['strikePrice'].isin(atm_strikes)]

        analysis["atm_activity"] = {
            "total_call_oi": int(atm_calls['openInterest'].sum()),
            "total_put_oi": int(atm_puts['openInterest'].sum()),
            "total_call_volume": int(atm_calls['totalTradedVolume'].sum()),
            "total_put_volume": int(atm_puts['totalTradedVolume'].sum()),
            "strike_range": f"{min(atm_strikes)}-{max(atm_strikes)}"
        }

    return analysis


def _analyze_futures_sentiment(symbol: str, expiry: str, spot_price: float) -> Dict:
    """Enhanced futures sentiment analysis."""
    analysis = {
        "futures_premium": None,
        "basis": None,
        "futures_sentiment": "UNAVAILABLE",
        "cost_of_carry": None
    }

    try:
        # Attempt to get futures data using derivative quote
        # This requires constructing the futures symbol properly
        current_year = str(date.today().year)[-2:]  # Last 2 digits of year

        # Convert month name to format (this is approximate and may need adjustment)
        month_map = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
        }

        # Note: This is a simplified approach. Real futures symbol construction
        # requires more sophisticated logic based on NSE naming conventions

        analysis["note"] = "Futures analysis requires specific contract symbol construction"

    except Exception as e:
        analysis["error"] = str(e)

    return analysis


def _calculate_sentiment_score(options_analysis: Dict, futures_analysis: Dict, equity_data: Dict) -> Dict:
    """Calculate composite sentiment score from various indicators."""
    sentiment_components = {
        "sentiment_score": 0,
        "sentiment_label": "NEUTRAL",
        "confidence_level": "LOW",
        "scoring_breakdown": {}
    }

    total_score = 0
    max_possible_score = 0
    confidence_factors = []

    # PCR-based scoring (40% weight)
    if "pcr_oi" in options_analysis:
        pcr_oi = options_analysis["pcr_oi"]
        pcr_score = 0

        if pcr_oi > 1.5:  # Very bullish
            pcr_score = 40
        elif pcr_oi > 1.2:  # Bullish
            pcr_score = 25
        elif pcr_oi < 0.6:  # Very bearish
            pcr_score = -40
        elif pcr_oi < 0.8:  # Bearish
            pcr_score = -25
        else:  # Neutral
            pcr_score = 0

        total_score += pcr_score
        max_possible_score += 40
        sentiment_components["scoring_breakdown"]["pcr_oi_score"] = pcr_score
        confidence_factors.append("PCR_OI")

    # OI Change Analysis (30% weight)
    if "call_oi_change" in options_analysis and "put_oi_change" in options_analysis:
        call_change = options_analysis["call_oi_change"]
        put_change = options_analysis["put_oi_change"]

        oi_change_score = 0
        if put_change > call_change * 1.5 and put_change > 100000:  # Strong put buildup
            oi_change_score = 30
        elif call_change > put_change * 1.5 and call_change > 100000:  # Strong call buildup
            oi_change_score = -30
        elif put_change > call_change and put_change > 50000:
            oi_change_score = 15
        elif call_change > put_change and call_change > 50000:
            oi_change_score = -15

        total_score += oi_change_score
        max_possible_score += 30
        sentiment_components["scoring_breakdown"]["oi_change_score"] = oi_change_score
        confidence_factors.append("OI_CHANGE")

    # Support/Resistance Analysis (20% weight)
    if "resistance_distance_pct" in options_analysis and "support_distance_pct" in options_analysis:
        res_dist = options_analysis["resistance_distance_pct"]
        sup_dist = options_analysis["support_distance_pct"]

        level_score = 0
        if sup_dist < 2:  # Very close to support - potential bounce
            level_score = 15
        elif sup_dist < 5:  # Near support
            level_score = 10
        elif res_dist < 2:  # Very close to resistance - potential rejection
            level_score = -15
        elif res_dist < 5:  # Near resistance
            level_score = -10

        total_score += level_score
        max_possible_score += 20
        sentiment_components["scoring_breakdown"]["support_resistance_score"] = level_score
        confidence_factors.append("SUPPORT_RESISTANCE")

    # Price momentum (10% weight)
    if equity_data.get("change_percent"):
        momentum_score = min(max(equity_data["change_percent"] * 2, -10), 10)  # Cap at ±10
        total_score += momentum_score
        max_possible_score += 10
        sentiment_components["scoring_breakdown"]["momentum_score"] = momentum_score
        confidence_factors.append("MOMENTUM")

    # Normalize score to -100 to +100
    if max_possible_score > 0:
        sentiment_components["sentiment_score"] = int((total_score / max_possible_score) * 100)

    # Determine sentiment label
    score = sentiment_components["sentiment_score"]
    if score >= 60:
        sentiment_components["sentiment_label"] = "STRONGLY BULLISH"
    elif score >= 30:
        sentiment_components["sentiment_label"] = "BULLISH"
    elif score >= 10:
        sentiment_components["sentiment_label"] = "MILDLY BULLISH"
    elif score <= -60:
        sentiment_components["sentiment_label"] = "STRONGLY BEARISH"
    elif score <= -30:
        sentiment_components["sentiment_label"] = "BEARISH"
    elif score <= -10:
        sentiment_components["sentiment_label"] = "MILDLY BEARISH"
    else:
        sentiment_components["sentiment_label"] = "NEUTRAL"

    # Determine confidence level
    confidence_score = len(confidence_factors)
    if confidence_score >= 4:
        sentiment_components["confidence_level"] = "HIGH"
    elif confidence_score >= 2:
        sentiment_components["confidence_level"] = "MEDIUM"
    else:
        sentiment_components["confidence_level"] = "LOW"

    sentiment_components["confidence_factors"] = confidence_factors

    return sentiment_components


def _calculate_risk_metrics(options_df: pd.DataFrame, spot_price: float) -> Dict:
    """Calculate risk metrics from options data."""
    risk_metrics = {}

    try:
        # Volatility clustering
        if 'impliedVolatility' in options_df.columns:
            iv_data = options_df['impliedVolatility'].dropna()
            if not iv_data.empty:
                risk_metrics["avg_implied_volatility"] = float(iv_data.mean())
                risk_metrics["iv_range"] = {
                    "min": float(iv_data.min()),
                    "max": float(iv_data.max())
                }

        # Liquidity metrics
        total_volume = options_df['totalTradedVolume'].sum()
        total_oi = options_df['openInterest'].sum()

        risk_metrics["liquidity_metrics"] = {
            "total_volume": int(total_volume),
            "total_oi": int(total_oi),
            "volume_oi_ratio": float(total_volume / max(total_oi, 1))
        }

        # Market depth (number of active strikes)
        active_strikes = len(options_df[options_df['totalTradedVolume'] > 0])
        risk_metrics["market_depth"] = {
            "active_strikes": active_strikes,
            "total_strikes": len(options_df)
        }

    except Exception as e:
        risk_metrics["error"] = str(e)

    return risk_metrics


def _generate_trading_signals(options_analysis: Dict, futures_analysis: Dict,
                            spot_price: float, sentiment_data: Dict) -> List[str]:
    """Generate actionable trading signals based on analysis."""
    signals = []

    sentiment_score = sentiment_data.get("sentiment_score", 0)
    confidence = sentiment_data.get("confidence_level", "LOW")

    # Primary trend signals
    if sentiment_score >= 30 and confidence in ["HIGH", "MEDIUM"]:
        signals.append(f"🟢 BULLISH SIGNAL: Strong upward bias detected (Score: {sentiment_score})")

        if "major_resistance" in options_analysis:
            resistance = options_analysis["major_resistance"]
            signals.append(f"🎯 TARGET: Watch for breakout above ₹{resistance:.2f}")

    elif sentiment_score <= -30 and confidence in ["HIGH", "MEDIUM"]:
        signals.append(f"🔴 BEARISH SIGNAL: Strong downward bias detected (Score: {sentiment_score})")

        if "major_support" in options_analysis:
            support = options_analysis["major_support"]
            signals.append(f"🎯 WATCH: Critical support at ₹{support:.2f}")

    # Specific strategy signals
    if "pcr_oi" in options_analysis:
        pcr = options_analysis["pcr_oi"]
        if pcr > 1.5:
            signals.append("📈 STRATEGY: Consider bullish strategies (PCR indicates oversold)")
        elif pcr < 0.6:
            signals.append("📉 STRATEGY: Consider bearish strategies (PCR indicates overbought)")

    # Risk management signals
    if "resistance_distance_pct" in options_analysis:
        res_dist = options_analysis["resistance_distance_pct"]
        if res_dist < 2:
            signals.append("⚠️ CAUTION: Near major resistance - watch for rejection")

    if "support_distance_pct" in options_analysis:
        sup_dist = options_analysis["support_distance_pct"]
        if sup_dist < 2:
            signals.append("💪 OPPORTUNITY: Near major support - potential bounce zone")

    # Max pain insights
    if "max_pain_strike" in options_analysis:
        max_pain = options_analysis["max_pain_strike"]
        distance_from_pain = ((spot_price - max_pain) / spot_price) * 100

        if abs(distance_from_pain) < 3:
            signals.append(f"🎯 MAX PAIN: Price near max pain level ₹{max_pain:.2f} - expect consolidation")
        elif distance_from_pain > 5:
            signals.append(f"⬇️ GRAVITATION: Price may move toward max pain ₹{max_pain:.2f}")

    if not signals:
        signals.append("😐 NEUTRAL: No strong directional signals detected - range-bound trading expected")

    return signals


def _generate_key_insights(options_analysis: Dict, futures_analysis: Dict,
                         sentiment_data: Dict, equity_data: Dict) -> List[str]:
    """Generate key insights summary for quick decision making."""
    insights = []

    # Market sentiment summary
    sentiment_label = sentiment_data.get("sentiment_label", "NEUTRAL")
    confidence = sentiment_data.get("confidence_level", "LOW")
    score = sentiment_data.get("sentiment_score", 0)

    insights.append(f"📊 MARKET SENTIMENT: {sentiment_label} (Score: {score}, Confidence: {confidence})")

    # Key levels
    if "major_support" in options_analysis and "major_resistance" in options_analysis:
        support = options_analysis["major_support"]
        resistance = options_analysis["major_resistance"]
        current_price = equity_data.get("price", 0)

        insights.append(f"🎯 KEY LEVELS: Support ₹{support:.2f} | Current ₹{current_price:.2f} | Resistance ₹{resistance:.2f}")

    # PCR interpretation
    if "pcr_oi" in options_analysis:
        pcr = options_analysis["pcr_oi"]
        if pcr > 1.2:
            insights.append(f"📈 PUT-CALL RATIO: {pcr:.2f} - Bullish bias (more put protection)")
        elif pcr < 0.8:
            insights.append(f"📉 PUT-CALL RATIO: {pcr:.2f} - Bearish bias (more call hedging)")
        else:
            insights.append(f"⚖️ PUT-CALL RATIO: {pcr:.2f} - Balanced positioning")

    # Activity insights
    if "call_oi_change" in options_analysis and "put_oi_change" in options_analysis:
        call_change = options_analysis["call_oi_change"]
        put_change = options_analysis["put_oi_change"]

        if abs(call_change) > 100000 or abs(put_change) > 100000:
            dominant_change = "PUT" if abs(put_change) > abs(call_change) else "CALL"
            insights.append(f"🔥 ACTIVITY SPIKE: Major {dominant_change} activity detected")

    # Risk assessment
    if "avg_implied_volatility" in equity_data.get("risk_metrics", {}):
        iv = equity_data["risk_metrics"]["avg_implied_volatility"]
        if iv > 30:
            insights.append(f"⚠️ HIGH VOLATILITY: IV at {iv:.1f}% - expect larger price swings")
        elif iv < 15:
            insights.append(f"😴 LOW VOLATILITY: IV at {iv:.1f}% - expect range-bound movement")

    return insights


if __name__ == "__main__":
    print(get_nse_india_data("TCS"))
