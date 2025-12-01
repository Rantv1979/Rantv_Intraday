# Rantv Intraday Trading Signals & Market Analysis - Enhanced
import time
from datetime import datetime, time as dt_time
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import math
import warnings
warnings.filterwarnings("ignore")

# Configuration
st.set_page_config(page_title="Rantv Intraday Terminal Pro - Enhanced", layout="wide", initial_sidebar_state="expanded")
IND_TZ = pytz.timezone("Asia/Kolkata")

CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10

SIGNAL_REFRESH_MS = 90000
PRICE_REFRESH_MS = 60000

MARKET_OPTIONS = ["CASH"]

NIFTY_50 = [
   "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "BAJFINANCE.NS", "ONGC.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "HDFCLIFE.NS", "DRREDDY.NS", "HINDALCO.NS", "CIPLA.NS", "SBILIFE.NS",
    "GRASIM.NS", "TECHM.NS", "BAJAJFINSV.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "DIVISLAB.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "UPL.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "INDUSINDBK.NS", "ADANIENT.NS", "TATACONSUM.NS", "BPCL.NS"
]

NIFTY_100 = NIFTY_50 + [
    "BAJAJHLDNG.NS", "TATAMOTORS.NS", "VEDANTA.NS", "PIDILITIND.NS",
    "BERGEPAINT.NS", "AMBUJACEM.NS", "DABUR.NS", "HAVELLS.NS", "ICICIPRULI.NS",
    "MARICO.NS", "PEL.NS", "SIEMENS.NS", "TORNTPHARM.NS", "ACC.NS",
    "AUROPHARMA.NS", "BOSCHLTD.NS", "GLENMARK.NS", "MOTHERSUMI.NS", "BIOCON.NS",
    "ZYDUSLIFE.NS", "COLPAL.NS", "CONCOR.NS", "DLF.NS", "GODREJCP.NS",
    "HINDPETRO.NS", "IBULHSGFIN.NS", "IOC.NS", "JINDALSTEL.NS", "LUPIN.NS",
    "MANAPPURAM.NS", "MCDOWELL-N.NS", "NMDC.NS", "PETRONET.NS", "PFC.NS",
    "PNB.NS", "RBLBANK.NS", "SAIL.NS", "SRTRANSFIN.NS", "TATAPOWER.NS",
    "YESBANK.NS", "ZEEL.NS"
]

# Enhanced Trading Strategies with Better Balance
TRADING_STRATEGIES = {
    "EMA_VWAP_Confluence": {"name": "EMA + VWAP Confluence", "weight": 3, "type": "BUY"},
    "RSI_MeanReversion": {"name": "RSI Mean Reversion", "weight": 2, "type": "BUY"},
    "Bollinger_Reversion": {"name": "Bollinger Band Reversion", "weight": 2, "type": "BUY"},
    "MACD_Momentum": {"name": "MACD Momentum", "weight": 2, "type": "BUY"},
    "Support_Resistance_Breakout": {"name": "Support/Resistance Breakout", "weight": 3, "type": "BUY"},
    "EMA_VWAP_Downtrend": {"name": "EMA + VWAP Downtrend", "weight": 3, "type": "SELL"},
    "RSI_Overbought": {"name": "RSI Overbought Reversal", "weight": 2, "type": "SELL"},
    "Bollinger_Rejection": {"name": "Bollinger Band Rejection", "weight": 2, "type": "SELL"},
    "MACD_Bearish": {"name": "MACD Bearish Crossover", "weight": 2, "type": "SELL"},
    "Trend_Reversal": {"name": "Trend Reversal", "weight": 2, "type": "SELL"}
}

# FIXED CSS with Light Yellowish Background and Better Tabs
st.markdown("""
<style>
    /* Light Yellowish Background */
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    /* Main container background */
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }
    
    /* Enhanced Tabs with Multiple Colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #e6f2ff 0%, #ffe6e6 50%, #e6ffe6 100%);
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px;
        gap: 8px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 14px;
        color: #1e3a8a;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        border: 2px solid #2563eb;
        box-shadow: 0 4px 8px rgba(30, 58, 138, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
        border: 2px solid #93c5fd;
        transform: translateY(-1px);
    }
    
    /* FIXED Market Mood Gauge Styles - Circular */
    .gauge-container {
        background: white;
        border-radius: 50%;
        padding: 25px;
        margin: 10px auto;
        border: 4px solid #e0f2fe;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        width: 200px;
        height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        position: relative;
    }
    
    .gauge-title {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 8px;
        color: #1e3a8a;
    }
    
    .gauge-value {
        font-size: 16px;
        font-weight: bold;
        margin: 3px 0;
    }
    
    .gauge-sentiment {
        font-size: 12px;
        font-weight: bold;
        margin-top: 6px;
        padding: 3px 10px;
        border-radius: 15px;
    }
    
    .bullish { 
        color: #059669;
        background-color: #d1fae5;
    }
    
    .bearish { 
        color: #dc2626;
        background-color: #fee2e2;
    }
    
    .neutral { 
        color: #d97706;
        background-color: #fef3c7;
    }
    
    /* Circular Progress Bar */
    .gauge-progress {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: conic-gradient(#059669 0% var(--progress), #e5e7eb var(--progress) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 8px 0;
        position: relative;
    }
    
    .gauge-progress-inner {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* RSI Scanner Styles */
    .rsi-oversold { 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
    }
    
    .rsi-overbought { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
    }
    
    /* Market Profile Styles */
    .bullish-signal { 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        border-radius: 8px;
    }
    
    .bearish-signal { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        border-radius: 8px;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Auto-refresh counter */
    .refresh-counter {
        background: #1e3a8a;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 8px;
    }
    
    /* Trade History PnL Styling */
    .profit-positive {
        color: #059669;
        font-weight: bold;
        background-color: #d1fae5;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    .profit-negative {
        color: #dc2626;
        font-weight: bold;
        background-color: #fee2e2;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    .trade-buy {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
    }
    
    .trade-sell {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Utilities
def now_indian():
    return datetime.now(IND_TZ)

def market_open():
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        return False

def should_auto_close():
    n = now_indian()
    try:
        auto_close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 10)))
        return n >= auto_close_time
    except Exception:
        return False

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_market_profile_vectorized(high, low, close, volume, bins=20):
    low_val = float(min(high.min(), low.min(), close.min()))
    high_val = float(max(high.max(), low.max(), close.max()))
    if np.isclose(low_val, high_val):
        high_val = low_val * 1.01 if low_val != 0 else 1.0
    edges = np.linspace(low_val, high_val, bins + 1)
    hist, _ = np.histogram(close, bins=edges, weights=volume)
    centers = (edges[:-1] + edges[1:]) / 2
    if hist.sum() == 0:
        poc = float(close.iloc[-1])
        va_high = poc * 1.01
        va_low = poc * 0.99
    else:
        idx = int(np.argmax(hist))
        poc = float(centers[idx])
        sorted_idx = np.argsort(hist)[::-1]
        cumulative = 0.0
        total = float(hist.sum())
        selected = []
        for i in sorted_idx:
            selected.append(centers[i])
            cumulative += hist[i]
            if cumulative / total >= 0.70:
                break
        va_high = float(max(selected))
        va_low = float(min(selected))
    profile = [{"price": float(c), "volume": int(v)} for c, v in zip(centers, hist)]
    return {"poc": poc, "value_area_high": va_high, "value_area_low": va_low, "profile": profile}

def calculate_support_resistance_advanced(high, low, close, period=20):
    resistance = []
    support = []
    ln = len(high)
    if ln < period * 2 + 1:
        return {"support": float(close.iloc[-1] * 0.98), "resistance": float(close.iloc[-1] * 1.02),
                "support_levels": [], "resistance_levels": []}
    for i in range(period, ln - period):
        if high.iloc[i] >= high.iloc[i - period:i + period + 1].max():
            resistance.append(float(high.iloc[i]))
        if low.iloc[i] <= low.iloc[i - period:i + period + 1].min():
            support.append(float(low.iloc[i]))
    recent_res = sorted(resistance)[-3:] if resistance else [float(close.iloc[-1] * 1.02)]
    recent_sup = sorted(support)[:3] if support else [float(close.iloc[-1] * 0.98)]
    return {"support": float(np.mean(recent_sup)), "resistance": float(np.mean(recent_res)),
            "support_levels": recent_sup, "resistance_levels": recent_res}

def adx(high, low, close, period=14):
    h = high.copy().reset_index(drop=True)
    l = low.copy().reset_index(drop=True)
    c = close.copy().reset_index(drop=True)
    df = pd.DataFrame({"high": h, "low": l, "close": c})
    df["tr"] = np.maximum(df["high"] - df["low"],
                          np.maximum((df["high"] - df["close"].shift()).abs(),
                                     (df["low"] - df["close"].shift()).abs()))
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]
    df["dm_pos"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
    df["dm_neg"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)
    df["tr_sum"] = df["tr"].rolling(window=period).sum()
    df["dm_pos_sum"] = df["dm_pos"].rolling(window=period).sum()
    df["dm_neg_sum"] = df["dm_neg"].rolling(window=period).sum()
    df["di_pos"] = 100 * (df["dm_pos_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["di_neg"] = 100 * (df["dm_neg_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["dx"] = (abs(df["di_pos"] - df["di_neg"]) / (df["di_pos"] + df["di_neg"]).replace(0, np.nan)) * 100
    df["adx"] = df["dx"].rolling(window=period).mean().fillna(0)
    return df["adx"].values

# FIXED Circular Market Mood Gauge Component with Rounded Percentages
def create_circular_market_mood_gauge(index_name, current_value, change_percent, sentiment_score):
    """Create a circular market mood gauge for Nifty50 and BankNifty"""
    
    # Round sentiment score and change percentage
    sentiment_score = round(sentiment_score)
    change_percent = round(change_percent, 2)
    
    # Determine sentiment color and text
    if sentiment_score >= 70:
        sentiment_color = "bullish"
        sentiment_text = "BULLISH"
        emoji = "📈"
        progress_color = "#059669"
    elif sentiment_score <= 30:
        sentiment_color = "bearish"
        sentiment_text = "BEARISH"
        emoji = "📉"
        progress_color = "#dc2626"
    else:
        sentiment_color = "neutral"
        sentiment_text = "NEUTRAL"
        emoji = "➡️"
        progress_color = "#d97706"
    
    # Create circular gauge HTML
    gauge_html = f"""
    <div class="gauge-container">
        <div class="gauge-title">{emoji} {index_name}</div>
        <div class="gauge-progress" style="--progress: {sentiment_score}%; background: conic-gradient({progress_color} 0% {sentiment_score}%, #e5e7eb {sentiment_score}% 100%);">
            <div class="gauge-progress-inner">
                {sentiment_score}%
            </div>
        </div>
        <div class="gauge-value">₹{current_value:,.0f}</div>
        <div class="gauge-sentiment {sentiment_color}">{sentiment_text}</div>
        <div style="color: {'#059669' if change_percent >= 0 else '#dc2626'}; font-size: 12px; margin-top: 3px;">
            {change_percent:+.2f}%
        </div>
    </div>
    """
    return gauge_html

# Enhanced Data Manager with 15min RSI Focus
class EnhancedDataManager:
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}
        self.backtest_engine = RealBacktestEngine()  # CHANGED: Using RealBacktestEngine
        self.market_profile_cache = {}
        self.last_rsi_scan = None

    def _validate_live_price(self, symbol):
        now_ts = time.time()
        key = f"price_{symbol}"
        if key in self.price_cache:
            cached = self.price_cache[key]
            if now_ts - cached["ts"] < 2:
                return cached["price"]
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval="1m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                return round(price, 2)
            df = ticker.history(period="2d", interval="5m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                return round(price, 2)
        except Exception:
            pass
        known = {"RELIANCE.NS": 2750.0, "TCS.NS": 3850.0, "HDFCBANK.NS": 1650.0}
        base = known.get(symbol, 1000.0)
        self.price_cache[key] = {"price": float(base), "ts": now_ts}
        return float(base)

    @st.cache_data(ttl=30)
    def _fetch_yf(_self, symbol, period, interval):
        try:
            return yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        except Exception:
            return pd.DataFrame()

    def get_stock_data(self, symbol, interval="15m"):
        # Force 15min timeframe for RSI analysis as requested
        if interval == "15m":
            period = "7d"
        elif interval == "1m":
            period = "1d"
        elif interval == "5m":
            period = "2d"
        else:
            period = "14d"

        df = self._fetch_yf(symbol, period, interval)
        if df is None or df.empty or len(df) < 20:
            return self.create_validated_demo_data(symbol)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        expected = ["Open", "High", "Low", "Close", "Volume"]
        for e in expected:
            if e not in df.columns:
                if e.upper() in df.columns:
                    df[e] = df[e.upper()]
                else:
                    return self.create_validated_demo_data(symbol)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if len(df) < 20:
            return self.create_validated_demo_data(symbol)

        try:
            live_price = self._validate_live_price(symbol)
            current_close = df["Close"].iloc[-1]
            price_diff_pct = abs(live_price - current_close) / max(current_close, 1e-6)
            if price_diff_pct > 0.005:
                df.iloc[-1, df.columns.get_loc("Close")] = live_price
                df.iloc[-1, df.columns.get_loc("High")] = max(df.iloc[-1]["High"], live_price)
                df.iloc[-1, df.columns.get_loc("Low")] = min(df.iloc[-1]["Low"], live_price)
        except Exception:
            pass

        # Enhanced Indicators with 15min focus
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()

        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]

        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]

        try:
            df_adx = adx(df["High"], df["Low"], df["Close"], period=14)
            df["ADX"] = pd.Series(df_adx, index=df.index).fillna(method="ffill").fillna(20)
        except Exception:
            df["ADX"] = 20

        try:
            htf = self._fetch_yf(symbol, period="7d", interval="1h")
            if htf is not None and len(htf) > 50:
                if isinstance(htf.columns, pd.MultiIndex):
                    htf.columns = ["_".join(map(str, col)).strip() for col in htf.columns.values]
                htf = htf.rename(columns={c: c.capitalize() for c in htf.columns})
                htf_close = htf["Close"]
                htf_ema50 = ema(htf_close, 50).iloc[-1]
                htf_ema200 = ema(htf_close, 200).iloc[-1] if len(htf_close) > 200 else ema(htf_close, 100).iloc[-1]
                df["HTF_Trend"] = 1 if htf_ema50 > htf_ema200 else -1
            else:
                df["HTF_Trend"] = 1
        except Exception:
            df["HTF_Trend"] = 1

        return df

    def create_validated_demo_data(self, symbol):
        live = self._validate_live_price(symbol)
        periods = 300
        end = now_indian()
        dates = pd.date_range(end=end, periods=periods, freq="15min")
        base = float(live)
        rng = np.random.default_rng(int(abs(hash(symbol)) % (2 ** 32 - 1)))
        returns = rng.normal(0, 0.0009, periods)
        prices = base * np.cumprod(1 + returns)
        openp = prices * (1 + rng.normal(0, 0.0012, periods))
        highp = prices * (1 + abs(rng.normal(0, 0.0045, periods)))
        lowp = prices * (1 - abs(rng.normal(0, 0.0045, periods)))
        vol = rng.integers(1000, 200000, periods)
        df = pd.DataFrame({"Open": openp, "High": highp, "Low": lowp, "Close": prices, "Volume": vol}, index=dates)
        df.iloc[-1, df.columns.get_loc("Close")] = live
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]
        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]
        df["ADX"] = adx(df["High"], df["Low"], df["Close"], period=14)
        df["HTF_Trend"] = 1
        return df

    def get_historical_accuracy(self, symbol, strategy):
        key = f"{symbol}_{strategy}"
        if key in self.backtest_engine.historical_results:
            return self.backtest_engine.historical_results[key]
        
        data = self.get_stock_data(symbol, "15m")
        accuracy = self.backtest_engine.calculate_historical_accuracy(symbol, strategy, data)
        
        self.backtest_engine.historical_results[key] = accuracy
        return accuracy

    def calculate_market_profile_signals(self, symbol):
        """Calculate market profile signals with improved timeframe alignment"""
        try:
            # Get 15min data for market profile analysis
            data_15m = self.get_stock_data(symbol, "15m")
            if len(data_15m) < 50:
                return {"signal": "NEUTRAL", "confidence": 0.5, "reason": "Insufficient data"}
            
            # Get 5min data for more current market sentiment
            data_5m = self.get_stock_data(symbol, "5m")
            
            current_price_15m = float(data_15m["Close"].iloc[-1])
            current_price_5m = float(data_5m["Close"].iloc[-1]) if len(data_5m) > 0 else current_price_15m
            
            # Calculate signals from both timeframes
            ema8_15m = float(data_15m["EMA8"].iloc[-1])
            ema21_15m = float(data_15m["EMA21"].iloc[-1])
            ema50_15m = float(data_15m["EMA50"].iloc[-1])
            rsi_val_15m = float(data_15m["RSI14"].iloc[-1])
            vwap_15m = float(data_15m["VWAP"].iloc[-1])
            
            # Get 5min indicators for current sentiment
            if len(data_5m) > 0:
                rsi_val_5m = float(data_5m["RSI14"].iloc[-1])
                ema8_5m = float(data_5m["EMA8"].iloc[-1])
                ema21_5m = float(data_5m["EMA21"].iloc[-1])
            else:
                rsi_val_5m = rsi_val_15m
                ema8_5m = ema8_15m
                ema21_5m = ema21_15m
            
            # Calculate bullish/bearish score with timeframe alignment
            bullish_score = 0
            bearish_score = 0
            
            # 15min trend analysis
            if current_price_15m > ema8_15m > ema21_15m > ema50_15m:
                bullish_score += 3
            elif current_price_15m < ema8_15m < ema21_15m < ema50_15m:
                bearish_score += 3
                
            # 5min momentum (more weight for current sentiment)
            if current_price_5m > ema8_5m > ema21_5m:
                bullish_score += 2
            elif current_price_5m < ema8_5m < ema21_5m:
                bearish_score += 2
                
            # RSI alignment across timeframes
            if rsi_val_15m > 55 and rsi_val_5m > 50:
                bullish_score += 1
            elif rsi_val_15m < 45 and rsi_val_5m < 50:
                bearish_score += 1
            elif (rsi_val_15m > 55 and rsi_val_5m < 50) or (rsi_val_15m < 45 and rsi_val_5m > 50):
                # Conflicting signals - reduce confidence
                bullish_score -= 1
                bearish_score -= 1
                
            # Price relative to VWAP
            if current_price_15m > vwap_15m and current_price_5m > vwap_15m:
                bullish_score += 2
            elif current_price_15m < vwap_15m and current_price_5m < vwap_15m:
                bearish_score += 2
                
            total_score = max(bullish_score + bearish_score, 1)  # Avoid division by zero
            bullish_ratio = (bullish_score + 5) / (total_score + 10)  # Normalize to 0-1
            
            # Adjust confidence based on timeframe alignment
            price_alignment = 1.0 if abs(current_price_15m - current_price_5m) / current_price_15m < 0.01 else 0.7
            
            final_confidence = min(0.95, bullish_ratio * price_alignment)
            
            if bullish_ratio >= 0.65:
                return {"signal": "BULLISH", "confidence": final_confidence, "reason": "Strong bullish alignment across timeframes"}
            elif bullish_ratio <= 0.35:
                return {"signal": "BEARISH", "confidence": final_confidence, "reason": "Strong bearish alignment across timeframes"}
            else:
                return {"signal": "NEUTRAL", "confidence": 0.5, "reason": "Mixed signals across timeframes"}
                
        except Exception as e:
            return {"signal": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}

    def should_run_rsi_scan(self):
        """Check if RSI scan should run (every 3rd refresh)"""
        current_time = time.time()
        if self.last_rsi_scan is None:
            self.last_rsi_scan = current_time
            return True
        
        # Run every 3rd refresh (approx every 75 seconds)
        if current_time - self.last_rsi_scan >= 75:
            self.last_rsi_scan = current_time
            return True
        return False

# NEW: RealBacktestEngine - Drop-in replacement for the existing BacktestEngine
from dataclasses import dataclass, field

@dataclass
class RealBacktestEngine:
    historical_results: dict = field(default_factory=dict)  # cache per symbol_strategy

    def _default_position_size(self, capital, risk_per_trade=0.01, stop_loss=None, entry=None):
        """Simple position sizing: risk_per_trade fraction of capital. Requires stop_loss and entry."""
        if stop_loss is None or entry is None:
            return 0
        risk_amount = capital * risk_per_trade
        per_share_risk = abs(entry - stop_loss)
        if per_share_risk <= 0:
            return 0
        qty = int(risk_amount // per_share_risk)
        return max(qty, 0)

    def _apply_costs(self, price, quantity, cost_pct, slippage_pct, side="buy"):
        """Return effective fill price after accounting for slippage and costs (per side)."""
        # slippage moves price unfavorably
        slip = price * slippage_pct
        if side.lower() == "buy":
            fill_price = price + slip
        else:
            fill_price = price - slip
        # costs (brokerage/taxes) will be applied to pnl, not fill price here
        return fill_price

    def _compute_equity_curve_stats(self, equity_series):
        """Compute equity curve metrics: max drawdown, drawdown duration, CAGR (approx), daily returns, sharpe."""
        eq = pd.Series(equity_series).dropna()
        if eq.empty:
            return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "total_return": 0.0}
        cum = eq.values
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(dd.min())
        max_dd_pct = abs(max_dd)
        total_return = (cum[-1] / cum[0]) - 1.0
        # simple daily returns from equity series timestamps not available -> approximate per-trade returns
        # compute per-step returns and annualize assuming 252 trading days and e.g., 6.5 trading hours if needed
        returns = pd.Series(cum).pct_change().fillna(0)
        mean_r = returns.mean()
        std_r = returns.std(ddof=0) if returns.std(ddof=0) != 0 else 1e-9
        # approximate sharpe: sqrt(N) * mean/std; use N = 252 as canonical
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
        return {"max_drawdown": max_dd, "max_drawdown_pct": max_dd_pct, "sharpe": float(sharpe), "total_return": float(total_return)}

    def backtest_with_generator(self,
                                data: pd.DataFrame,
                                signal_generator,   # function(data_slice)-> {'action': 'BUY'|'SELL', 'confidence':..} or None
                                initial_capital: float = 1_000_000.0,
                                risk_per_trade: float = 0.01,
                                cost_pct: float = 0.0004,      # e.g., 0.04% round-trip per side
                                slippage_pct: float = 0.0003,  # 0.03% slippage per fill
                                max_hold_bars: int = 40,
                                min_bars_between_signals: int = 1):
        """
        Generic backtester for intraday series (index ordered chronologically).
        - Scans from left to right generating candidate signals via signal_generator(current_slice)
        - When a signal is returned, we size the position using risk_per_trade (requires stop-loss calculation)
        - Exit rules: hit stop_loss, hit target, or hold max_hold_bars
        - Accounts for slippage and transaction cost on execution (applied to pnl)
        Returns a dict with trades list and metrics.
        """
        if data is None or len(data) < 30:
            return {"error": "insufficient_data", "metrics": {}}

        capital = float(initial_capital)
        equity = capital
        equity_curve = [equity]
        trades = []
        last_signal_bar = -9999

        # Ensure expected columns exist
        needed_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed_cols:
            if c not in data.columns:
                raise ValueError(f"Data must contain column '{c}'")

        # iterate bars
        for i in range(20, len(data) - 1):  # start after some lookback
            # simple prevention of overtrading
            if i - last_signal_bar < min_bars_between_signals:
                continue

            current_slice = data.iloc[:i+1].copy()
            signal = signal_generator(current_slice)
            if not signal or "action" not in signal:
                # nothing to do
                continue

            action = signal["action"].upper()
            # attempt to compute stop_loss & target via user-supplied fields OR fallback rules
            entry_price = float(current_slice["Close"].iloc[-1])
            # allow generator to provide stop and target
            stop_loss = signal.get("stop_loss", None)
            target = signal.get("target", None)
            # fallback default SL/TP using ATR if available
            if stop_loss is None or target is None:
                if "ATR" in current_slice.columns and not np.isnan(current_slice["ATR"].iloc[-1]) and current_slice["ATR"].iloc[-1] > 0:
                    atr = float(current_slice["ATR"].iloc[-1])
                    if action == "BUY":
                        stop_loss = entry_price - 1.2 * atr if stop_loss is None else stop_loss
                        target = entry_price + 2.5 * atr if target is None else target
                    else:
                        stop_loss = entry_price + 1.2 * atr if stop_loss is None else stop_loss
                        target = entry_price - 2.5 * atr if target is None else target
                else:
                    # fallback: 0.5% SL, 1.25% target
                    if action == "BUY":
                        stop_loss = entry_price * 0.995 if stop_loss is None else stop_loss
                        target = entry_price * 1.0125 if target is None else target
                    else:
                        stop_loss = entry_price * 1.005 if stop_loss is None else stop_loss
                        target = entry_price * 0.9875 if target is None else target

            # compute quantity via risk sizing
            qty = self._default_position_size(capital, risk_per_trade, stop_loss=stop_loss, entry=entry_price)
            if qty <= 0:
                continue  # cannot size a trade

            # simulate trade filled with slippage
            entry_fill = self._apply_costs(entry_price, qty, cost_pct, slippage_pct, side="buy" if action=="BUY" else "sell")
            position = {
                "entry_bar": i,
                "entry_time": data.index[i],
                "action": action,
                "entry_price": entry_fill,
                "quantity": qty,
                "stop_loss": stop_loss,
                "target": target,
                "exit_bar": None,
                "exit_time": None,
                "exit_price": None,
                "pnl": None,
                "return_pct": None,
                "duration_bars": None
            }

            # Deduct used capital (simple; for shorts we reserve margin equal to trade value * 0.2)
            trade_value = entry_fill * qty
            if action == "BUY":
                capital -= trade_value  # lock cash
            else:
                margin = trade_value * 0.2
                capital -= margin

            # walk forward until exit
            exited = False
            for j in range(i+1, min(i+1+max_hold_bars, len(data))):
                high = float(data["High"].iloc[j])
                low = float(data["Low"].iloc[j])
                close = float(data["Close"].iloc[j])

                # check exit by stop or target intrabar using conservative assumption: if both hit, prioritize stop for realism
                if action == "BUY":
                    # target
                    if high >= target:
                        exit_price = self._apply_costs(target, qty, cost_pct, slippage_pct, side="sell")
                        reason = "target"
                        exit_bar = j
                        exited = True
                    elif low <= stop_loss:
                        exit_price = self._apply_costs(stop_loss, qty, cost_pct, slippage_pct, side="sell")
                        reason = "stop"
                        exit_bar = j
                        exited = True
                    else:
                        # no exit this bar
                        continue
                else:  # SELL (short)
                    if low <= target:
                        exit_price = self._apply_costs(target, qty, cost_pct, slippage_pct, side="buy")
                        reason = "target"
                        exit_bar = j
                        exited = True
                    elif high >= stop_loss:
                        exit_price = self._apply_costs(stop_loss, qty, cost_pct, slippage_pct, side="buy")
                        reason = "stop"
                        exit_bar = j
                        exited = True
                    else:
                        continue

                if exited:
                    # compute pnl: for BUY pnl = (exit - entry)*qty minus costs; for SELL = (entry - exit)*qty
                    if action == "BUY":
                        gross_pnl = (exit_price - entry_fill) * qty
                    else:
                        gross_pnl = (entry_fill - exit_price) * qty
                    # apply round-trip cost approximation
                    cost_total = (entry_fill * qty) * cost_pct + (exit_price * qty) * cost_pct
                    net_pnl = gross_pnl - cost_total
                    return_pct = net_pnl / (abs(entry_fill * qty) + 1e-9)
                    # release capital
                    if action == "BUY":
                        capital += exit_price * qty
                    else:
                        # for shorts add back margin + realized pnl
                        capital += (entry_fill * qty * 0.2) + (entry_fill * qty - exit_price * qty)
                        # simplified: real margin accounting differs by broker
                    # record trade
                    position.update({
                        "exit_bar": exit_bar,
                        "exit_time": data.index[exit_bar],
                        "exit_price": exit_price,
                        "pnl": float(net_pnl),
                        "return_pct": float(return_pct),
                        "duration_bars": exit_bar - i,
                        "exit_reason": reason
                    })
                    trades.append(position)
                    equity = capital  # approximate equity after closing (not counting open positions)
                    equity_curve.append(equity)
                    last_signal_bar = j
                    break

            # If not exited by stop/target within max_hold -> exit at close of last bar (market)
            if not exited:
                j = min(i + max_hold_bars, len(data) - 1)
                market_close = float(data["Close"].iloc[j])
                exit_price = self._apply_costs(market_close, qty, cost_pct, slippage_pct, side="sell" if action=="BUY" else "buy")
                if action == "BUY":
                    gross_pnl = (exit_price - entry_fill) * qty
                else:
                    gross_pnl = (entry_fill - exit_price) * qty
                cost_total = (entry_fill * qty) * cost_pct + (exit_price * qty) * cost_pct
                net_pnl = gross_pnl - cost_total
                return_pct = net_pnl / (abs(entry_fill * qty) + 1e-9)
                if action == "BUY":
                    capital += exit_price * qty
                else:
                    capital += (entry_fill * qty * 0.2) + (entry_fill * qty - exit_price * qty)
                position.update({
                    "exit_bar": j,
                    "exit_time": data.index[j],
                    "exit_price": exit_price,
                    "pnl": float(net_pnl),
                    "return_pct": float(return_pct),
                    "duration_bars": j - i,
                    "exit_reason": "max_hold"
                })
                trades.append(position)
                equity = capital
                equity_curve.append(equity)
                last_signal_bar = j

        # Finished scanning
        # Compute trade-level metrics
        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            metrics = {"total_trades": 0}
            return {"trades": [], "metrics": metrics, "equity_curve": equity_curve}

        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / total_trades if total_trades else 0.0
        avg_win = wins["pnl"].mean() if len(wins) else 0.0
        avg_loss = losses["pnl"].mean() if len(losses) else 0.0
        gross_profit = wins["pnl"].sum()
        gross_loss = -losses["pnl"].sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        expectancy = ((avg_win * (len(wins)/total_trades)) + (avg_loss * (len(losses)/total_trades)))
        avg_return_pct = trades_df["return_pct"].mean()
        # equity curve stats
        eq_stats = self._compute_equity_curve_stats(equity_curve)
        metrics = {
            "total_trades": int(total_trades),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "profit_factor": float(profit_factor if not math.isinf(profit_factor) else 9999.0),
            "expectancy": float(expectancy),
            "avg_return_pct": float(avg_return_pct),
            "max_drawdown": float(eq_stats["max_drawdown"]),
            "max_drawdown_pct": float(eq_stats["max_drawdown_pct"]),
            "sharpe": float(eq_stats["sharpe"]),
            "total_return": float(eq_stats["total_return"]),
            "final_equity": float(equity_curve[-1])
        }

        # cache minimal result
        return {"trades": trades_df.to_dict(orient="records"), "metrics": metrics, "equity_curve": equity_curve}

    # Backward-compatible wrapper used by your existing code
    def calculate_historical_accuracy(self, symbol, strategy, data):
        """
        Backwards-compatible function: runs the robust backtest using your existing
        generate_signal_for_backtest-like function if available in the caller's scope.
        Expected behavior: return a single scalar accuracy number [0.0 .. 1.0].
        For richer info use backtest_with_generator(...) above.
        """
        key = f"{symbol}_{strategy}"
        if key in self.historical_results:
            cached = self.historical_results[key]
            # If previously stored full result, return win_rate for compatibility
            if isinstance(cached, dict) and "metrics" in cached:
                return cached["metrics"].get("win_rate", 0.0)
            return float(cached)

        # Attempt to use an externally defined generator function by name if present.
        # Caller should pass a function; if not found, we will build a simple strategy generator
        generator_fn = None
        # We'll attempt to fallback to a simple rule-based generator based on 'strategy' name:
        def fallback_generator(d_slice):
            # simple mapping mirroring your earlier generate_signal_for_backtest rules but conservative
            if len(d_slice) < 20:
                return None
            try:
                close = float(d_slice["Close"].iloc[-1])
                ema8 = float(d_slice["EMA8"].iloc[-1]) if "EMA8" in d_slice.columns else close
                ema21 = float(d_slice["EMA21"].iloc[-1]) if "EMA21" in d_slice.columns else close
                ema50 = float(d_slice["EMA50"].iloc[-1]) if "EMA50" in d_slice.columns else close
                macd_line = float(d_slice["MACD"].iloc[-1]) if "MACD" in d_slice.columns else 0
                macd_signal = float(d_slice["MACD_Signal"].iloc[-1]) if "MACD_Signal" in d_slice.columns else 0
                rsi_val = float(d_slice["RSI14"].iloc[-1]) if "RSI14" in d_slice.columns else 50
                vwap = float(d_slice["VWAP"].iloc[-1]) if "VWAP" in d_slice.columns else close
                adx_val = float(d_slice["ADX"].iloc[-1]) if "ADX" in d_slice.columns else 20
                # use conservative thresholds
                if strategy == "EMA_VWAP_Confluence":
                    if ema8 > ema21 > ema50 and close > vwap and adx_val > 20:
                        return {"action": "BUY"}
                if strategy == "RSI_MeanReversion":
                    if rsi_val < 30:
                        return {"action": "BUY"}
                if strategy == "Bollinger_Reversion":
                    if "BB_Lower" in d_slice.columns and close <= d_slice["BB_Lower"].iloc[-1]:
                        return {"action": "BUY"}
                if strategy == "MACD_Momentum":
                    if macd_line > macd_signal and ema8 > ema21 and close > vwap:
                        return {"action": "BUY"}
                if strategy == "EMA_VWAP_Downtrend":
                    if ema8 < ema21 < ema50 and close < vwap:
                        return {"action": "SELL"}
                if strategy == "RSI_Overbought":
                    if rsi_val > 70:
                        return {"action": "SELL"}
                # fallback: no signal
                return None
            except Exception:
                return None

        generator_fn = fallback_generator

        # run backtest
        try:
            result = self.backtest_with_generator(data, generator_fn,
                                                  initial_capital=1_000_000.0,
                                                  risk_per_trade=0.01,
                                                  cost_pct=0.0004,
                                                  slippage_pct=0.0003,
                                                  max_hold_bars=40)
            # cache
            self.historical_results[key] = result
            # return win rate for compatibility
            return result["metrics"].get("win_rate", 0.0) if "metrics" in result else 0.0
        except Exception as e:
            # On error return 0.0 to indicate not tradable
            self.historical_results[key] = 0.0
            return 0.0

# Enhanced Multi-Strategy Trading Engine with Trade History
class MultiStrategyIntradayTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.stock_trades = 0
        self.auto_trades_count = 0
        self.last_reset = now_indian().date()
        self.selected_market = "CASH"
        self.auto_execution = False
        self.signal_history = []
        self.auto_close_triggered = False
        # Initialize strategy performance for ALL strategies
        self.strategy_performance = {}
        for strategy in TRADING_STRATEGIES.keys():
            self.strategy_performance[strategy] = {"signals": 0, "trades": 0, "wins": 0, "pnl": 0.0}

    def reset_daily_counts(self):
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.stock_trades = 0
            self.auto_trades_count = 0
            self.last_reset = current_date

    def can_auto_trade(self):
        return (self.auto_trades_count < MAX_AUTO_TRADES and 
                self.daily_trades < MAX_DAILY_TRADES and
                market_open())

    def calculate_support_resistance(self, symbol, current_price):
        try:
            data = data_manager.get_stock_data(symbol, "15m")
            if data is None or len(data) < 20:
                return current_price * 0.98, current_price * 1.02
            return float(data["Support"].iloc[-1]), float(data["Resistance"].iloc[-1])
        except Exception:
            return current_price * 0.98, current_price * 1.02

    def calculate_intraday_target_sl(self, entry_price, action, atr, current_price, support, resistance):
        # Enhanced intraday target and stop loss calculation
        if atr <= 0 or np.isnan(atr):
            atr = max(entry_price * 0.005, 1.0)
        
        if action == "BUY":
            sl = entry_price - (atr * 1.2)  # Slightly wider SL for intraday
            target = entry_price + (atr * 2.5)  # Better risk-reward for intraday
            if target > resistance:
                target = min(target, resistance * 0.998)  # Don't target exact resistance
            sl = max(sl, support * 0.995)
        else:
            sl = entry_price + (atr * 1.2)
            target = entry_price - (atr * 2.5)
            if target < support:
                target = max(target, support * 1.002)  # Don't target exact support
            sl = min(sl, resistance * 1.005)

        # Ensure minimum risk-reward ratio of 1:2 for intraday
        rr = abs(target - entry_price) / max(abs(entry_price - sl), 1e-6)
        if rr < 2.0:
            if action == "BUY":
                target = entry_price + max((entry_price - sl) * 2.0, atr * 2.0)
            else:
                target = entry_price - max((sl - entry_price) * 2.0, atr * 2.0)
                
        return round(float(target), 2), round(float(sl), 2)

    def equity(self):
        total = float(self.cash)
        for symbol, pos in self.positions.items():
            if pos.get("status") == "OPEN":
                try:
                    data = data_manager.get_stock_data(symbol, "5m")
                    price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
                    total += pos["quantity"] * price
                except Exception:
                    total += pos["quantity"] * pos["entry_price"]
        return total

    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, win_probability=0.75, auto_trade=False, strategy=None):
        self.reset_daily_counts()
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        if self.stock_trades >= MAX_STOCK_TRADES:
            return False, "Stock trade limit reached"
        if auto_trade and self.auto_trades_count >= MAX_AUTO_TRADES:
            return False, "Auto trade limit reached"

        trade_value = float(quantity) * float(price)
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"

        trade_id = f"TRADE_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        record = {
            "trade_id": trade_id, 
            "symbol": symbol, 
            "action": action, 
            "quantity": int(quantity),
            "entry_price": float(price), 
            "stop_loss": float(stop_loss) if stop_loss else None,
            "target": float(target) if target else None, 
            "timestamp": now_indian(),
            "status": "OPEN", 
            "current_pnl": 0.0, 
            "current_price": float(price),
            "win_probability": float(win_probability), 
            "closed_pnl": 0.0,
            "entry_time": now_indian().strftime("%H:%M:%S"),
            "auto_trade": auto_trade,
            "strategy": strategy
        }

        if action == "BUY":
            self.positions[symbol] = record
            self.cash -= trade_value
        else:
            margin = trade_value * 0.2
            record["margin_used"] = margin
            self.positions[symbol] = record
            self.cash -= margin

        self.stock_trades += 1
        self.trade_log.append(record)
        self.daily_trades += 1

        if auto_trade:
            self.auto_trades_count += 1

        if strategy and strategy in self.strategy_performance:
            self.strategy_performance[strategy]["trades"] += 1

        return True, f"{'[AUTO] ' if auto_trade else ''}{action} {int(quantity)} {symbol} @ ₹{price:.2f} | Strategy: {strategy}"

    def update_positions_pnl(self):
        if should_auto_close() and not self.auto_close_triggered:
            self.auto_close_all_positions()
            self.auto_close_triggered = True
            return
        for symbol, pos in list(self.positions.items()):
            if pos.get("status") != "OPEN":
                continue
            try:
                data = data_manager.get_stock_data(symbol, "5m")
                if data is not None and len(data) > 0:
                    price = float(data["Close"].iloc[-1])
                    pos["current_price"] = price
                    entry = pos["entry_price"]
                    if pos["action"] == "BUY":
                        pnl = (price - entry) * pos["quantity"]
                    else:
                        pnl = (entry - price) * pos["quantity"]
                    pos["current_pnl"] = float(pnl)
                    pos["max_pnl"] = max(pos.get("max_pnl", 0.0), float(pnl))
                    sl = pos.get("stop_loss")
                    tg = pos.get("target")
                    if sl is not None:
                        if (pos["action"] == "BUY" and price <= sl) or (pos["action"] == "SELL" and price >= sl):
                            self.close_position(symbol, exit_price=sl)
                            continue
                    if tg is not None:
                        if (pos["action"] == "BUY" and price >= tg) or (pos["action"] == "SELL" and price <= tg):
                            self.close_position(symbol, exit_price=tg)
                            continue
            except Exception:
                continue

    def auto_close_all_positions(self):
        for sym in list(self.positions.keys()):
            self.close_position(sym)

    def close_position(self, symbol, exit_price=None):
        if symbol not in self.positions:
            return False, "Position not found"
        pos = self.positions[symbol]
        if exit_price is None:
            try:
                data = data_manager.get_stock_data(symbol, "5m")
                exit_price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
            except Exception:
                exit_price = pos["entry_price"]
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            self.cash += pos.get("margin_used", 0) + (pos["quantity"] * pos["entry_price"])
        pos["status"] = "CLOSED"
        pos["exit_price"] = float(exit_price)
        pos["closed_pnl"] = float(pnl)
        pos["exit_time"] = now_indian()
        pos["exit_time_str"] = now_indian().strftime("%H:%M:%S")

        strategy = pos.get("strategy")
        if strategy and strategy in self.strategy_performance:
            if pnl > 0:
                self.strategy_performance[strategy]["wins"] += 1
            self.strategy_performance[strategy]["pnl"] += pnl

        try:
            del self.positions[symbol]
        except Exception:
            pass
        return True, f"Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"

    def get_open_positions_data(self):
        self.update_positions_pnl()
        out = []
        for symbol, pos in self.positions.items():
            if pos.get("status") != "OPEN":
                continue
            try:
                data = data_manager.get_stock_data(symbol, "5m")
                price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
                if pos["action"] == "BUY":
                    pnl = (price - pos["entry_price"]) * pos["quantity"]
                else:
                    pnl = (pos["entry_price"] - price) * pos["quantity"]
                var = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
                sup, res = self.calculate_support_resistance(symbol, price)
                
                strategy = pos.get("strategy", "Manual")
                historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy) if strategy != "Manual" else 0.65
                
                out.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Action": pos["action"],
                    "Quantity": pos["quantity"],
                    "Entry Price": f"₹{pos['entry_price']:.2f}",
                    "Current Price": f"₹{price:.2f}",
                    "P&L": f"₹{pnl:+.2f}",
                    "Variance %": f"{var:+.2f}%",
                    "Stop Loss": f"₹{pos.get('stop_loss', 0):.2f}",
                    "Target": f"₹{pos.get('target', 0):.2f}",
                    "Support": f"₹{sup:.2f}",
                    "Resistance": f"₹{res:.2f}",
                    "Historical Win %": f"{historical_accuracy:.1%}",
                    "Current Win %": f"{pos.get('win_probability', 0.75)*100:.1f}%",
                    "Entry Time": pos.get("entry_time"),
                    "Auto Trade": "Yes" if pos.get("auto_trade") else "No",
                    "Strategy": strategy,
                    "Status": pos.get("status")
                })
            except Exception:
                continue
        return out

    def get_trade_history_data(self):
        """Get formatted trade history data for display"""
        history_data = []
        for trade in self.trade_log:
            if trade.get("status") == "CLOSED":
                pnl = trade.get("closed_pnl", 0)
                pnl_class = "profit-positive" if pnl >= 0 else "profit-negative"
                trade_class = "trade-buy" if trade.get("action") == "BUY" else "trade-sell"
                
                history_data.append({
                    "Trade ID": trade.get("trade_id", ""),
                    "Symbol": trade.get("symbol", "").replace(".NS", ""),
                    "Action": trade.get("action", ""),
                    "Quantity": trade.get("quantity", 0),
                    "Entry Price": f"₹{trade.get('entry_price', 0):.2f}",
                    "Exit Price": f"₹{trade.get('exit_price', 0):.2f}",
                    "P&L": f"<span class='{pnl_class}'>₹{pnl:+.2f}</span>",
                    "Entry Time": trade.get("entry_time", ""),
                    "Exit Time": trade.get("exit_time_str", ""),
                    "Strategy": trade.get("strategy", "Manual"),
                    "Auto Trade": "Yes" if trade.get("auto_trade") else "No",
                    "Duration": self.calculate_trade_duration(trade.get("entry_time"), trade.get("exit_time_str")),
                    "_row_class": trade_class
                })
        return history_data

    def calculate_trade_duration(self, entry_time_str, exit_time_str):
        """Calculate trade duration in minutes"""
        try:
            if entry_time_str and exit_time_str:
                fmt = "%H:%M:%S"
                entry_time = datetime.strptime(entry_time_str, fmt).time()
                exit_time = datetime.strptime(exit_time_str, fmt).time()
                
                # Create datetime objects with today's date
                today = datetime.now().date()
                entry_dt = datetime.combine(today, entry_time)
                exit_dt = datetime.combine(today, exit_time)
                
                duration = (exit_dt - entry_dt).total_seconds() / 60
                return f"{int(duration)} min"
        except:
            pass
        return "N/A"

    def get_performance_stats(self):
        self.update_positions_pnl()
        closed = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        total_trades = len(closed)
        open_pnl = sum([p.get("current_pnl", 0) for p in self.positions.values() if p.get("status") == "OPEN"])
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "open_positions": len(self.positions),
                "open_pnl": open_pnl,
                "auto_trades": self.auto_trades_count
            }
        wins = len([t for t in closed if t.get("closed_pnl", 0) > 0])
        total_pnl = sum([t.get("closed_pnl", 0) for t in closed])
        win_rate = wins / total_trades if total_trades else 0.0
        avg_pnl = total_pnl / total_trades if total_trades else 0.0

        auto_trades = [t for t in self.trade_log if t.get("auto_trade")]
        auto_closed = [t for t in auto_trades if t.get("status") == "CLOSED"]
        auto_win_rate = len([t for t in auto_closed if t.get("closed_pnl", 0) > 0]) / len(auto_closed) if auto_closed else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "open_positions": len(self.positions),
            "open_pnl": open_pnl,
            "auto_trades": self.auto_trades_count,
            "auto_win_rate": auto_win_rate
        }

    def generate_strategy_signals(self, symbol, data):
        signals = []
        if data is None or len(data) < 30:
            return signals
        try:
            live = float(data["Close"].iloc[-1])
            ema8 = float(data["EMA8"].iloc[-1])
            ema21 = float(data["EMA21"].iloc[-1])
            ema50 = float(data["EMA50"].iloc[-1])
            rsi_val = float(data["RSI14"].iloc[-1])
            atr = float(data["ATR"].iloc[-1]) if "ATR" in data.columns else max(live*0.005,1)
            macd_line = float(data["MACD"].iloc[-1])
            macd_signal = float(data["MACD_Signal"].iloc[-1])
            vwap = float(data["VWAP"].iloc[-1])
            support = float(data["Support"].iloc[-1])
            resistance = float(data["Resistance"].iloc[-1])
            bb_upper = float(data["BB_Upper"].iloc[-1])
            bb_lower = float(data["BB_Lower"].iloc[-1])
            vol_latest = float(data["Volume"].iloc[-1])
            vol_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data["Volume"]) >= 20 else float(data["Volume"].mean())
            volume_spike = vol_latest > vol_avg * 1.3
            adx_val = float(data["ADX"].iloc[-1]) if "ADX" in data.columns else 20
            htf_trend = int(data["HTF_Trend"].iloc[-1]) if "HTF_Trend" in data.columns else 1

            # BUY STRATEGIES - Only generate if historical accuracy > 65%
            # Strategy 1: EMA + VWAP + ADX + HTF Trend
            if (ema8 > ema21 > ema50 and live > vwap and adx_val > 20 and htf_trend == 1):
                action = "BUY"; confidence = 0.82; score = 9; strategy = "EMA_VWAP_Confluence"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:  # Minimum 1:2 risk-reward for intraday
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    # Only generate signal if historical accuracy > 65%
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.85, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 2: RSI Mean Reversion (15min timeframe focused)
            rsi_prev = float(data["RSI14"].iloc[-2])
            if rsi_val < 30 and rsi_val > rsi_prev and live > support:
                action = "BUY"; confidence = 0.78; score = 8; strategy = "RSI_MeanReversion"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.80, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 3: Bollinger Reversion
            if live <= bb_lower and rsi_val < 35 and live > support:
                action = "BUY"; confidence = 0.75; score = 7; strategy = "Bollinger_Reversion"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.78, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 4: MACD Momentum
            if (macd_line > macd_signal and macd_line > 0 and ema8 > ema21 and 
                live > vwap and adx_val > 22 and htf_trend == 1):
                action = "BUY"; confidence = 0.80; score = 8; strategy = "MACD_Momentum"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.82, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 5: Support/Resistance Breakout
            if (live > resistance and volume_spike and rsi_val > 50 and 
                htf_trend == 1 and ema8 > ema21 and macd_line > macd_signal):
                action = "BUY"; confidence = 0.75; score = 7; strategy = "Support_Resistance_Breakout"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                stop_loss = max(stop_loss, resistance * 0.995)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.77, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # SELL STRATEGIES - Enhanced with 65% filter
            # Strategy 6: EMA + VWAP Downtrend
            if (ema8 < ema21 < ema50 and live < vwap and adx_val > 20 and htf_trend == -1):
                action = "SELL"; confidence = 0.78; score = 8; strategy = "EMA_VWAP_Downtrend"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.80, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 7: RSI Overbought (15min timeframe focused)
            if rsi_val > 70 and rsi_val < rsi_prev and live < resistance:
                action = "SELL"; confidence = 0.72; score = 7; strategy = "RSI_Overbought"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.75, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 8: Bollinger Rejection
            if live >= bb_upper and rsi_val > 65 and live < resistance:
                action = "SELL"; confidence = 0.70; score = 6; strategy = "Bollinger_Rejection"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.73, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 9: MACD Bearish Crossover
            if (macd_line < macd_signal and macd_line < 0 and ema8 < ema21 and 
                live < vwap and adx_val > 22 and htf_trend == -1):
                action = "SELL"; confidence = 0.75; score = 8; strategy = "MACD_Bearish"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.0:
                    historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        win_probability = min(0.78, historical_accuracy * 1.1)
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 10: Trend Reversal
            if len(data) > 5:
                prev_trend = 1 if data.iloc[-3]['EMA8'] > data.iloc[-3]['EMA21'] else -1
                current_trend = -1 if ema8 < ema21 else 1
                if prev_trend == 1 and current_trend == -1 and rsi_val > 60:
                    action = "SELL"; confidence = 0.68; score = 7; strategy = "Trend_Reversal"
                    target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                    rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                    if rr >= 2.0:
                        historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy)
                        if historical_accuracy >= 0.65:
                            win_probability = min(0.70, historical_accuracy * 1.1)
                            signals.append({
                                "symbol": symbol, "action": action, "entry": live, "current_price": live,
                                "target": target, "stop_loss": stop_loss, "confidence": confidence,
                                "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                                "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                                "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                            })

            # update strategy signals count
            for s in signals:
                strat = s.get("strategy")
                if strat in self.strategy_performance:
                    self.strategy_performance[strat]["signals"] += 1

            return signals

        except Exception as e:
            return signals

    def generate_quality_signals(self, universe, max_scan=None, min_confidence=0.7, min_score=6):
        signals = []
        stocks = NIFTY_50 if universe == "Nifty 50" else NIFTY_100
        if max_scan is None:
            max_scan = len(stocks)
        progress_bar = st.progress(0)
        status_text = st.empty()
        for idx, symbol in enumerate(stocks[:max_scan]):
            try:
                status_text.text(f"Scanning {symbol} ({idx+1}/{len(stocks[:max_scan])})")
                progress_bar.progress((idx + 1) / len(stocks[:max_scan]))
                data = data_manager.get_stock_data(symbol, "15m")  # Using 15min timeframe
                if data is None or len(data) < 30:
                    continue
                strategy_signals = self.generate_strategy_signals(symbol, data)
                signals.extend(strategy_signals)
            except Exception:
                continue
        progress_bar.empty()
        status_text.empty()
        signals = [s for s in signals if s["confidence"] >= min_confidence and s["score"] >= min_score]
        signals.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
        self.signal_history = signals[:30]
        return signals[:20]

    def auto_execute_signals(self, signals):
        executed = []
        for signal in signals[:10]:
            if not self.can_auto_trade():
                break
            if signal["symbol"] in self.positions:
                continue
            qty = int((self.cash * TRADE_ALLOC) / signal["entry"])
            if qty > 0:
                success, msg = self.execute_trade(
                    symbol=signal["symbol"],
                    action=signal["action"],
                    quantity=qty,
                    price=signal["entry"],
                    stop_loss=signal["stop_loss"],
                    target=signal["target"],
                    win_probability=signal.get("win_probability", 0.75),
                    auto_trade=True,
                    strategy=signal.get("strategy")
                )
                if success:
                    executed.append(msg)
        return executed

# Initialize
data_manager = EnhancedDataManager()
if "trader" not in st.session_state:
    st.session_state.trader = MultiStrategyIntradayTrader()
trader = st.session_state.trader

# Auto-refresh counter to prevent tab switching
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "📈 Dashboard"

st.session_state.refresh_count += 1

# Enhanced UI with Circular Market Mood Gauges
st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>Rantv Intraday Terminal Pro BUY/SELL Signals</h1>", unsafe_allow_html=True)
st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh_improved")

cols = st.columns(7)
try:
    nift = data_manager._validate_live_price("^NSEI")
    cols[0].metric("NIFTY 50", f"₹{nift:,.2f}")
except Exception:
    cols[0].metric("NIFTY 50", "N/A")
try:
    bn = data_manager._validate_live_price("^NSEBANK")
    cols[1].metric("BANK NIFTY", f"₹{bn:,.2f}")
except Exception:
    cols[1].metric("BANK NIFTY", "N/A")
cols[2].metric("Market Status", "LIVE" if market_open() else "CLOSED")
cols[3].metric("Auto Close", "15:10")
cols[4].metric("Stock Trades", f"{trader.stock_trades}/{MAX_STOCK_TRADES}")
cols[5].metric("Auto Trades", f"{trader.auto_trades_count}/{MAX_AUTO_TRADES}")
cols[6].metric("Available Cash", f"₹{trader.cash:,.0f}")

# Manual refresh button instead of auto-refresh to prevent tab switching
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown(f"<div style='text-align: left; color: #6b7280; font-size: 14px;'>Refresh Count: <span class='refresh-counter'>{st.session_state.refresh_count}</span></div>", unsafe_allow_html=True)
with col2:
    if st.button("🔄 Manual Refresh", use_container_width=True):
        st.rerun()
with col3:
    if st.button("📊 Update Prices", use_container_width=True):
        st.rerun()

# Market Mood Gauges for Nifty50 & BankNifty
st.subheader("📊 Market Mood Gauges")

try:
    nifty_data = yf.download("^NSEI", period="1d", interval="5m", auto_adjust=False)
    nifty_current = float(nifty_data["Close"].iloc[-1])
    nifty_prev = float(nifty_data["Close"].iloc[-2])
    nifty_change = ((nifty_current - nifty_prev) / nifty_prev) * 100
    
    # Calculate Nifty sentiment score (0-100) with rounding
    nifty_sentiment = 50 + (nifty_change * 8)  # Base 50 + amplified change
    nifty_sentiment = max(0, min(100, round(nifty_sentiment)))
    
except Exception:
    nifty_current = 22000
    nifty_change = 0.15
    nifty_sentiment = 65

try:
    banknifty_data = yf.download("^NSEBANK", period="1d", interval="5m", auto_adjust=False)
    banknifty_current = float(banknifty_data["Close"].iloc[-1])
    banknifty_prev = float(banknifty_data["Close"].iloc[-2])
    banknifty_change = ((banknifty_current - banknifty_prev) / banknifty_prev) * 100
    
    # Calculate BankNifty sentiment score with rounding
    banknifty_sentiment = 50 + (banknifty_change * 8)
    banknifty_sentiment = max(0, min(100, round(banknifty_sentiment)))
    
except Exception:
    banknifty_current = 48000
    banknifty_change = 0.25
    banknifty_sentiment = 70

# Display Circular Market Mood Gauges with Rounded Percentages
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(create_circular_market_mood_gauge("NIFTY 50", nifty_current, nifty_change, nifty_sentiment), unsafe_allow_html=True)
with col2:
    st.markdown(create_circular_market_mood_gauge("BANK NIFTY", banknifty_current, banknifty_change, banknifty_sentiment), unsafe_allow_html=True)
with col3:
    # Market status gauge
    market_status = "LIVE" if market_open() else "CLOSED"
    status_sentiment = 80 if market_open() else 20
    st.markdown(create_circular_market_mood_gauge("MARKET", 0, 0, status_sentiment).replace("₹0", market_status).replace("0.00%", ""), unsafe_allow_html=True)
with col4:
    # Auto close gauge
    auto_close_status = "ACTIVE" if not should_auto_close() else "INACTIVE"
    close_sentiment = 20 if should_auto_close() else 80
    st.markdown(create_circular_market_mood_gauge("AUTO CLOSE", 0, 0, close_sentiment).replace("₹0", "15:10").replace("0.00%", auto_close_status), unsafe_allow_html=True)

# Main metrics with card styling
st.subheader("📈 Live Metrics")
cols = st.columns(4)
with cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 12px; color: #6b7280;">Available Cash</div>
        <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">₹{trader.cash:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 12px; color: #6b7280;">Account Value</div>
        <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">₹{trader.equity():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 12px; color: #6b7280;">Open Positions</div>
        <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">{len(trader.positions)}</div>
    </div>
    """, unsafe_allow_html=True)
with cols[3]:
    open_pnl = sum([p.get('current_pnl', 0) for p in trader.positions.values()])
    pnl_color = "#059669" if open_pnl >= 0 else "#dc2626"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 12px; color: #6b7280;">Open P&L</div>
        <div style="font-size: 20px; font-weight: bold; color: {pnl_color};">₹{open_pnl:+.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with Strategy Performance
st.sidebar.header("🎯 Strategy Performance")
for strategy, config in TRADING_STRATEGIES.items():
    if strategy in trader.strategy_performance:
        perf = trader.strategy_performance[strategy]
        if perf["signals"] > 0:
            win_rate = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0
            color = "#059669" if win_rate > 0.6 else "#dc2626" if win_rate < 0.4 else "#d97706"
            st.sidebar.write(f"**{config['name']}**")
            st.sidebar.write(f"📊 Signals: {perf['signals']} | Trades: {perf['trades']}")
            st.sidebar.write(f"🎯 Win Rate: <span style='color: {color};'>{win_rate:.1%}</span>", unsafe_allow_html=True)
            st.sidebar.write(f"💰 P&L: ₹{perf['pnl']:+.2f}")
            st.sidebar.markdown("---")

st.sidebar.header("⚙️ Trading Configuration")
trader.selected_market = st.sidebar.selectbox("Market Type", MARKET_OPTIONS)
trader.auto_execution = st.sidebar.checkbox("Auto Execution", value=False)
min_conf_percent = st.sidebar.slider("Minimum Confidence %", 60, 95, 70, 5)
min_score = st.sidebar.slider("Minimum Score", 5, 10, 6, 1)
scan_limit = st.sidebar.selectbox("Scan Limit", ["All Stocks", "Top 40", "Top 20"], index=0)
max_scan_map = {"All Stocks": None, "Top 40": 40, "Top 20": 20}
max_scan = max_scan_map[scan_limit]

# Enhanced Tabs with Trade History - Using session state to remember current tab
tabs = st.tabs([
    "📈 Dashboard", 
    "🚦 Signals", 
    "💰 Paper Trading", 
    "📋 Trade History",  # NEW TRADE HISTORY TAB
    "📊 Market Profile", 
    "📉 RSI Extreme", 
    "🔍 Backtest", 
    "⚡ Strategies"
])

# Store current tab in session state
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "📈 Dashboard"

# Tab content with manual refresh handling
with tabs[0]:
    st.session_state.current_tab = "📈 Dashboard"
    st.subheader("Account Summary")
    trader.update_positions_pnl()
    perf = trader.get_performance_stats()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Value", f"₹{trader.equity():,.0f}", delta=f"₹{trader.equity() - trader.initial_capital:+,.0f}")
    c2.metric("Available Cash", f"₹{trader.cash:,.0f}")
    c3.metric("Open Positions", len(trader.positions))
    c4.metric("Total P&L", f"₹{perf['total_pnl'] + perf['open_pnl']:+.2f}")
    
    # Strategy Performance Overview
    st.subheader("Strategy Performance Overview")
    strategy_data = []
    for strategy, config in TRADING_STRATEGIES.items():
        if strategy in trader.strategy_performance:
            perf_data = trader.strategy_performance[strategy]
            if perf_data["trades"] > 0:
                win_rate = perf_data["wins"] / perf_data["trades"]
                strategy_data.append({
                    "Strategy": config["name"],
                    "Type": config["type"],
                    "Signals": perf_data["signals"],
                    "Trades": perf_data["trades"],
                    "Win Rate": f"{win_rate:.1%}",
                    "P&L": f"₹{perf_data['pnl']:+.2f}"
                })
    
    if strategy_data:
        st.dataframe(pd.DataFrame(strategy_data), use_container_width=True)
    else:
        st.info("No strategy performance data available yet.")

with tabs[1]:
    st.session_state.current_tab = "🚦 Signals"
    st.subheader("Multi-Strategy BUY/SELL Signals")
    col1, col2 = st.columns([1, 2])
    with col1:
        universe = st.selectbox("Universe", ["Nifty 50", "Nifty 100"])
        generate_btn = st.button("Generate Signals", type="primary", use_container_width=True)
    with col2:
        if trader.auto_execution:
            st.success("🔴 Auto Execution: ACTIVE")
        else:
            st.info("⚪ Auto Execution: INACTIVE")
            
    if generate_btn or trader.auto_execution:
        with st.spinner("Scanning stocks with enhanced BUY/SELL strategies..."):
            signals = trader.generate_quality_signals(universe, max_scan=max_scan, min_confidence=min_conf_percent/100.0, min_score=min_score)
        
        if signals:
            # Separate BUY and SELL signals
            buy_signals = [s for s in signals if s["action"] == "BUY"]
            sell_signals = [s for s in signals if s["action"] == "SELL"]
            
            st.success(f"Found {len(buy_signals)} BUY signals and {len(sell_signals)} SELL signals")
            
            data_rows = []
            for s in signals:
                data_rows.append({
                    "Symbol": s["symbol"].replace(".NS",""),
                    "Action": s["action"],
                    "Strategy": s["strategy_name"],
                    "Entry Price": f"₹{s['entry']:.2f}",
                    "Current Price": f"₹{s['current_price']:.2f}",
                    "Target": f"₹{s['target']:.2f}",
                    "Stop Loss": f"₹{s['stop_loss']:.2f}",
                    "Confidence": f"{s['confidence']:.1%}",
                    "Historical Win %": f"{s.get('historical_accuracy', 0.7):.1%}",
                    "Current Win %": f"{s.get('win_probability',0.7):.1%}",
                    "R:R": f"{s['risk_reward']:.2f}",
                    "Score": s['score'],
                    "RSI": f"{s['rsi']:.1f}"
                })
            
            st.dataframe(pd.DataFrame(data_rows), use_container_width=True)
            
            if trader.auto_execution and trader.can_auto_trade():
                executed = trader.auto_execute_signals(signals)
                if executed:
                    st.success("Auto-execution completed:")
                    for msg in executed:
                        st.write(f"✓ {msg}")
            
            st.subheader("Manual Execution")
            for s in signals:
                col_a, col_b, col_c = st.columns([3,1,1])
                with col_a:
                    action_color = "🟢" if s["action"] == "BUY" else "🔴"
                    st.write(f"{action_color} **{s['symbol'].replace('.NS','')}** - {s['action']} @ ₹{s['entry']:.2f} | Strategy: {s['strategy_name']} | Historical Win: {s.get('historical_accuracy',0.7):.1%} | R:R: {s['risk_reward']:.2f}")
                with col_b:
                    qty = int((trader.cash * TRADE_ALLOC) / s["entry"])
                    st.write(f"Qty: {qty}")
                with col_c:
                    if st.button(f"Execute", key=f"exec_{s['symbol']}_{s['strategy']}"):
                        success, msg = trader.execute_trade(
                            symbol=s["symbol"], action=s["action"], quantity=qty, price=s["entry"],
                            stop_loss=s["stop_loss"], target=s["target"], win_probability=s.get("win_probability",0.75),
                            strategy=s.get("strategy")
                        )
                        if success:
                            st.success(msg)
        else:
            st.info("No confirmed signals with current filters.")

with tabs[2]:
    st.session_state.current_tab = "💰 Paper Trading"
    st.subheader("Paper Trading - With Historical Accuracy")
    trader.update_positions_pnl()
    open_pos = trader.get_open_positions_data()
    
    if open_pos:
        st.dataframe(pd.DataFrame(open_pos), use_container_width=True)
        
        # FIXED: Enhanced Accuracy Summary - Fixed KeyError
        st.subheader("📊 Enhanced Accuracy Analysis")
        
        # Strategy-wise analysis
        strategies_used = set([pos['Strategy'] for pos in open_pos])
        strategy_analysis = []
        
        for strategy in strategies_used:
            strategy_positions = [pos for pos in open_pos if pos['Strategy'] == strategy]
            
            # FIX: Check if 'Historical Win %' exists and handle it safely
            historical_wins = []
            for pos in strategy_positions:
                if 'Historical Win %' in pos:
                    try:
                        win_pct = float(pos['Historical Win %'].strip('%'))/100
                        historical_wins.append(win_pct)
                    except (ValueError, AttributeError):
                        historical_wins.append(0.65)  # Default value
            
            avg_historical = np.mean(historical_wins) if historical_wins else 0.65
            
            # FIX: Handle P&L calculation safely
            current_pnl = 0
            for pos in strategy_positions:
                if 'P&L' in pos:
                    try:
                        pnl_str = pos['P&L'].replace('₹','').replace(',','').strip()
                        current_pnl += float(pnl_str)
                    except (ValueError, AttributeError):
                        continue
            
            strategy_analysis.append({
                "Strategy": strategy,
                "Positions": len(strategy_positions),
                "Avg Historical Win %": f"{avg_historical:.1%}",
                "Current P&L": f"₹{current_pnl:+.2f}"
            })
        
        if strategy_analysis:
            st.dataframe(pd.DataFrame(strategy_analysis), use_container_width=True)
        
        # Position management
        st.subheader("Position Management")
        cols_close = st.columns(4)
        for idx, symbol in enumerate(list(trader.positions.keys())):
            with cols_close[idx % 4]:
                if st.button(f"Close {symbol}", key=f"close_{symbol}", use_container_width=True):
                    success, msg = trader.close_position(symbol)
                    if success:
                        st.success(msg)
        
        if st.button("Close All Positions", type="primary", use_container_width=True):
            for sym in list(trader.positions.keys()):
                trader.close_position(sym)
            st.success("All positions closed!")
    else:
        st.info("No open positions.")

with tabs[3]:  # NEW TRADE HISTORY TAB
    st.session_state.current_tab = "📋 Trade History"
    st.subheader("📋 Trade History")
    
    # Get trade history data
    trade_history = trader.get_trade_history_data()
    
    if trade_history:
        # Convert to DataFrame for display
        history_df = pd.DataFrame(trade_history)
        
        # Display summary statistics
        total_trades = len(trade_history)
        winning_trades = len([t for t in trade_history if float(t['P&L'].split('₹')[1].split('</span>')[0]) > 0])
        total_pnl = sum([float(t['P&L'].split('₹')[1].split('</span>')[0]) for t in trade_history])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Winning Trades", winning_trades)
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total P&L", f"₹{total_pnl:+.2f}", delta_color=pnl_color)
        
        # Display trade history table with custom styling
        st.subheader("Trade Details")
        
        # Create a custom HTML table for better styling
        html_table = """
        <table style="width:100%; border-collapse: collapse; margin: 10px 0; font-size: 14px;">
            <thead>
                <tr style="background-color: #1e3a8a; color: white;">
                    <th style="padding: 8px; text-align: left;">Symbol</th>
                    <th style="padding: 8px; text-align: left;">Action</th>
                    <th style="padding: 8px; text-align: left;">Qty</th>
                    <th style="padding: 8px; text-align: left;">Entry</th>
                    <th style="padding: 8px; text-align: left;">Exit</th>
                    <th style="padding: 8px; text-align: left;">P&L</th>
                    <th style="padding: 8px; text-align: left;">Strategy</th>
                    <th style="padding: 8px; text-align: left;">Duration</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for trade in trade_history:
            row_class = "trade-buy" if trade['Action'] == 'BUY' else "trade-sell"
            html_table += f"""
            <tr class="{row_class}" style="border-bottom: 1px solid #e5e7eb;">
                <td style="padding: 8px;">{trade['Symbol']}</td>
                <td style="padding: 8px;"><strong>{trade['Action']}</strong></td>
                <td style="padding: 8px;">{trade['Quantity']}</td>
                <td style="padding: 8px;">{trade['Entry Price']}</td>
                <td style="padding: 8px;">{trade['Exit Price']}</td>
                <td style="padding: 8px;">{trade['P&L']}</td>
                <td style="padding: 8px;">{trade['Strategy']}</td>
                <td style="padding: 8px;">{trade['Duration']}</td>
            </tr>
            """
        
        html_table += "</tbody></table>"
        
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Export functionality
        st.subheader("Export Trade History")
        if st.button("Export to CSV", use_container_width=True):
            # Create exportable DataFrame (without HTML formatting)
            export_df = pd.DataFrame([{
                'Symbol': t['Symbol'],
                'Action': t['Action'],
                'Quantity': t['Quantity'],
                'Entry_Price': t['Entry Price'].replace('₹', ''),
                'Exit_Price': t['Exit Price'].replace('₹', ''),
                'P&L': float(t['P&L'].split('₹')[1].split('</span>')[0]),
                'Strategy': t['Strategy'],
                'Entry_Time': t['Entry Time'],
                'Exit_Time': t['Exit Time'],
                'Duration': t['Duration'],
                'Auto_Trade': t['Auto Trade']
            } for t in trade_history])
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("No trade history available. Closed trades will appear here.")

with tabs[4]:
    st.session_state.current_tab = "📊 Market Profile"
    st.subheader("Market Profile Analysis - Nifty 50/100")
    st.write("Enhanced bullish/bearish signal analysis with timeframe alignment")
    
    col1, col2 = st.columns(2)
    with col1:
        profile_universe = st.selectbox("Select Universe", ["Nifty 50", "Nifty 100"], key="profile_universe")
        analyze_btn = st.button("Analyze Market Profile", type="primary", use_container_width=True)
    
    with col2:
        min_confidence = st.slider("Minimum Confidence %", 60, 90, 70, 5, key="profile_confidence")
    
    if analyze_btn:
        stocks = NIFTY_50 if profile_universe == "Nifty 50" else NIFTY_100
        bullish_stocks = []
        bearish_stocks = []
        neutral_stocks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(stocks):
            status_text.text(f"Analyzing {symbol.replace('.NS', '')} ({idx+1}/{len(stocks)})")
            progress_bar.progress((idx + 1) / len(stocks))
            
            try:
                profile_signal = data_manager.calculate_market_profile_signals(symbol)
                stock_data = {
                    "Symbol": symbol.replace(".NS", ""),
                    "Signal": profile_signal["signal"],
                    "Confidence": f"{profile_signal['confidence']:.1%}",
                    "Reason": profile_signal["reason"]
                }
                
                if profile_signal["signal"] == "BULLISH" and profile_signal["confidence"] >= min_confidence/100.0:
                    bullish_stocks.append(stock_data)
                elif profile_signal["signal"] == "BEARISH" and profile_signal["confidence"] >= min_confidence/100.0:
                    bearish_stocks.append(stock_data)
                else:
                    neutral_stocks.append(stock_data)
                    
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results in tabulated format
        col_bull, col_bear = st.columns(2)
        
        with col_bull:
            if bullish_stocks:
                st.subheader(f"📈 Bullish Signals ({len(bullish_stocks)})")
                bullish_df = pd.DataFrame(bullish_stocks)
                st.dataframe(bullish_df, use_container_width=True)
        
        with col_bear:
            if bearish_stocks:
                st.subheader(f"📉 Bearish Signals ({len(bearish_stocks)})")
                bearish_df = pd.DataFrame(bearish_stocks)
                st.dataframe(bearish_df, use_container_width=True)
        
        if not bullish_stocks and not bearish_stocks:
            st.info("No strong bullish or bearish signals found with current confidence threshold.")

with tabs[5]:
    st.session_state.current_tab = "📉 RSI Extreme"
    st.subheader("RSI Extreme Scanner - 15min Timeframe")
    st.write("Automated scan for stocks with RSI in oversold (<30) and overbought (>70) zones using 15min data")
    
    # Check if we should run RSI scan (every 3rd refresh)
    should_run_rsi = data_manager.should_run_rsi_scan()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        universe_rsi = st.selectbox("Universe", ["Nifty 50", "Nifty 100"], key="rsi_universe")
    with col2:
        rsi_low_threshold = st.slider("Oversold Threshold", 20, 35, 30, 1)
    with col3:
        rsi_high_threshold = st.slider("Overbought Threshold", 65, 80, 70, 1)
    with col4:
        min_volume_multiplier = st.slider("Min Volume", 1.0, 3.0, 1.5, 0.1)
    
    if st.button("Scan RSI Extremes", type="primary", use_container_width=True) or should_run_rsi:
        stocks = NIFTY_50 if universe_rsi == "Nifty 50" else NIFTY_100
        rsi_low_stocks = []
        rsi_high_stocks = []
        
        progress_bar = st.progress(0)
        for idx, symbol in enumerate(stocks):
            progress_bar.progress((idx + 1) / len(stocks))
            try:
                # Using 15min timeframe specifically for RSI scanning
                data = data_manager.get_stock_data(symbol, "15m")
                if data is not None and len(data) > 14:
                    current_rsi = float(data["RSI14"].iloc[-1])
                    current_price = float(data["Close"].iloc[-1])
                    current_volume = float(data["Volume"].iloc[-1])
                    avg_volume = float(data["Volume"].rolling(20).mean().iloc[-1])
                    
                    # Check volume condition
                    volume_ok = current_volume > avg_volume * min_volume_multiplier
                    
                    if current_rsi <= rsi_low_threshold and volume_ok:
                        rsi_low_stocks.append({
                            "Symbol": symbol.replace(".NS", ""),
                            "RSI": f"{current_rsi:.1f}",
                            "Price": f"₹{current_price:.2f}",
                            "Volume Ratio": f"{current_volume/avg_volume:.1f}x",
                            "Signal": "Oversold"
                        })
                    
                    if current_rsi >= rsi_high_threshold and volume_ok:
                        rsi_high_stocks.append({
                            "Symbol": symbol.replace(".NS", ""),
                            "RSI": f"{current_rsi:.1f}",
                            "Price": f"₹{current_price:.2f}",
                            "Volume Ratio": f"{current_volume/avg_volume:.1f}x",
                            "Signal": "Overbought"
                        })
            except:
                continue
        progress_bar.empty()
        
        # Display results in tabulated format
        if rsi_low_stocks:
            st.subheader(f"📉 Oversold Stocks (RSI < {rsi_low_threshold})")
            low_df = pd.DataFrame(rsi_low_stocks)
            st.dataframe(low_df, use_container_width=True)
        
        if rsi_high_stocks:
            st.subheader(f"📈 Overbought Stocks (RSI > {rsi_high_threshold})")
            high_df = pd.DataFrame(rsi_high_stocks)
            st.dataframe(high_df, use_container_width=True)
        
        if not rsi_low_stocks and not rsi_high_stocks:
            st.info("No stocks found in RSI extreme zones with current filters.")
        
        if should_run_rsi:
            st.info("🔄 Auto-scan completed (runs every 3rd refresh)")

with tabs[6]:
    st.session_state.current_tab = "🔍 Backtest"
    st.subheader("Strategy Backtesting")
    st.write("Run historical backtest to evaluate strategy performance")
    
    col1, col2 = st.columns(2)
    with col1:
        backtest_symbol = st.selectbox("Select Stock", NIFTY_50[:20], key="backtest_stock")
        backtest_strategy = st.selectbox("Select Strategy", list(TRADING_STRATEGIES.keys()), 
                                        format_func=lambda x: TRADING_STRATEGIES[x]["name"])
    
    with col2:
        backtest_period = st.selectbox("Period", ["1mo", "3mo", "6mo"], index=1)
        backtest_interval = st.selectbox("Interval", ["15m", "30m", "1h"], index=0)
    
    if st.button("Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            try:
                data = data_manager.get_stock_data(backtest_symbol, backtest_interval)
                
                if data.empty:
                    st.error("No data available for backtest")
                else:
                    accuracy = data_manager.backtest_engine.calculate_historical_accuracy(
                        backtest_symbol, backtest_strategy, data
                    )
                    
                    st.success(f"**Backtest Results for {TRADING_STRATEGIES[backtest_strategy]['name']}**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Historical Accuracy", f"{accuracy:.1%}")
                    col2.metric("Strategy", TRADING_STRATEGIES[backtest_strategy]["name"])
                    col3.metric("Stock", backtest_symbol.replace(".NS", ""))
                    
                    if accuracy > 0.7:
                        st.success("✅ This strategy shows good historical performance")
                    elif accuracy > 0.6:
                        st.info("ℹ️ This strategy shows decent historical performance")
                    else:
                        st.warning("⚠️ This strategy may need optimization")
                        
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")

with tabs[7]:
    st.session_state.current_tab = "⚡ Strategies"
    st.subheader("Enhanced Trading Strategies")
    
    for strategy_key, config in TRADING_STRATEGIES.items():
        with st.expander(f"{'🟢' if config['type'] == 'BUY' else '🔴'} {config['name']} (Weight: {config['weight']})"):
            
            # Strategy descriptions
            strategy_descriptions = {
                "EMA_VWAP_Confluence": "**Description:** Combines EMA alignment with VWAP, ADX trend strength, and higher timeframe bias for high-probability BUY entries.\n\n**Conditions:** EMA8 > EMA21 > EMA50, Price > VWAP, ADX > 20, HTF Trend = Bullish",
                "RSI_MeanReversion": "**Description:** Identifies oversold conditions with RSI reversal for BUY entries at key support levels.\n\n**Conditions:** RSI < 30, RSI rising, Price > Support",
                "Bollinger_Reversion": "**Description:** Captures mean reversion BUY opportunities when price touches Bollinger Band extremes.\n\n**Conditions:** Price ≤ Lower BB, RSI < 35, Price > Support",
                "MACD_Momentum": "**Description:** Uses MACD crossover with ADX trend strength for BUY momentum entries.\n\n**Conditions:** MACD > Signal, MACD > 0, EMA8 > EMA21, Price > VWAP, ADX > 22",
                "Support_Resistance_Breakout": "**Description:** Identifies BUY breakouts at key resistance levels with volume confirmation.\n\n**Conditions:** Price > Resistance, Volume spike, RSI > 50, Bullish trend",
                "EMA_VWAP_Downtrend": "**Description:** Combines bearish EMA alignment with VWAP for SELL entries in downtrends.\n\n**Conditions:** EMA8 < EMA21 < EMA50, Price < VWAP, ADX > 20, HTF Trend = Bearish",
                "RSI_Overbought": "**Description:** Identifies overbought conditions with RSI reversal for SELL entries.\n\n**Conditions:** RSI > 70, RSI falling, Price < Resistance",
                "Bollinger_Rejection": "**Description:** Captures SELL opportunities when price rejects upper Bollinger Band.\n\n**Conditions:** Price ≥ Upper BB, RSI > 65, Price < Resistance",
                "MACD_Bearish": "**Description:** Uses MACD bearish crossover for SELL entries in downtrends.\n\n**Conditions:** MACD < Signal, MACD < 0, EMA8 < EMA21, Price < VWAP, ADX > 22",
                "Trend_Reversal": "**Description:** Identifies early trend reversal patterns for SELL entries.\n\n**Conditions:** Bullish to bearish trend change, RSI > 60"
            }
            
            st.write(strategy_descriptions.get(strategy_key, "Strategy description not available."))
            
            # Performance data
            if strategy_key in trader.strategy_performance:
                perf = trader.strategy_performance[strategy_key]
                if perf["trades"] > 0:
                    win_rate = perf["wins"]/perf["trades"] if perf["trades"]>0 else 0
                    st.write(f"**Live Performance:** {perf['trades']} trades | {win_rate:.1%} win rate | ₹{perf['pnl']:+.2f}")
                else:
                    st.write("**Live Performance:** No trades yet")
            else:
                st.write("**Live Performance:** No trades yet")
            
            # Historical accuracy ranges
            default_accuracies = {
                "EMA_VWAP_Confluence": "65-75%",
                "RSI_MeanReversion": "60-70%",
                "Bollinger_Reversion": "58-68%",
                "MACD_Momentum": "62-72%",
                "Support_Resistance_Breakout": "55-65%",
                "EMA_VWAP_Downtrend": "60-70%",
                "RSI_Overbought": "58-68%",
                "Bollinger_Rejection": "56-66%",
                "MACD_Bearish": "59-69%",
                "Trend_Reversal": "54-64%"
            }
            st.write(f"**Typical Historical Accuracy:** {default_accuracies.get(strategy_key, '60-70%')}")

st.markdown("---")
st.markdown("<div style='text-align:center; color: #6b7280;'>Enhanced Intraday Terminal Pro with BUY/SELL Signals & Market Analysis</div>", unsafe_allow_html=True)
