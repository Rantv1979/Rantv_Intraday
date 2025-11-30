# Rantv Intraday Trading Signals & Market Analysis - ENHANCED FINAL VERSION
import time
from datetime import datetime, time as dt_time, timedelta
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
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import logging

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Rantv Intraday Terminal Pro - Enhanced", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

IND_TZ = pytz.timezone("Asia/Kolkata")
CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10
SIGNAL_REFRESH_MS = 90000
PRICE_REFRESH_MS = 60000
MARKET_OPTIONS = ["CASH"]

# Circuit Breaker Settings
MAX_DAILY_LOSS = 50000  # ₹50,000 max loss per day
MAX_DRAWDOWN = 0.15  # 15% max drawdown
MAX_CONSECUTIVE_LOSSES = 3

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

# =====================================================================
# LOGGING SETUP
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_terminal.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================================
# ENHANCED CSS
# =====================================================================
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

/* Circular Market Mood Gauge */
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

/* Alert Boxes */
.alert-danger {
    background-color: #fee2e2;
    border-left: 4px solid #dc2626;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
    color: #991b1b;
}

.alert-warning {
    background-color: #fef3c7;
    border-left: 4px solid #d97706;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
    color: #92400e;
}

.alert-success {
    background-color: #d1fae5;
    border-left: 4px solid #059669;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
    color: #065f46;
}

/* Card Styling */
.metric-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #1e3a8a;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def now_indian():
    """Get current time in Indian timezone"""
    return datetime.now(IND_TZ)

def market_open():
    """Check if market is open"""
    n = now_indian()
    try:
        # Skip weekends
        if n.weekday() >= 5:
            return False
        
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        return False

def should_auto_close():
    """Check if positions should be auto-closed"""
    n = now_indian()
    try:
        auto_close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 10)))
        return n >= auto_close_time
    except Exception:
        return False

# =====================================================================
# TECHNICAL INDICATORS (Optimized)
# =====================================================================

def ema(series, span):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
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

def calculate_support_resistance_advanced(high, low, close, period=20):
    """Calculate support and resistance levels"""
    resistance = []
    support = []
    ln = len(high)
    
    if ln < period * 2 + 1:
        return {
            "support": float(close.iloc[-1] * 0.98),
            "resistance": float(close.iloc[-1] * 1.02),
            "support_levels": [],
            "resistance_levels": []
        }
    
    for i in range(period, ln - period):
        if high.iloc[i] >= high.iloc[i - period:i + period + 1].max():
            resistance.append(float(high.iloc[i]))
        if low.iloc[i] <= low.iloc[i - period:i + period + 1].min():
            support.append(float(low.iloc[i]))
    
    recent_res = sorted(resistance)[-3:] if resistance else [float(close.iloc[-1] * 1.02)]
    recent_sup = sorted(support)[:3] if support else [float(close.iloc[-1] * 0.98)]
    
    return {
        "support": float(np.mean(recent_sup)),
        "resistance": float(np.mean(recent_res)),
        "support_levels": recent_sup,
        "resistance_levels": recent_res
    }

def calculate_market_profile_vectorized(high, low, close, volume, bins=20):
    """Calculate market profile with POC and Value Area"""
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
    
    return {
        "poc": poc,
        "value_area_high": va_high,
        "value_area_low": va_low,
        "profile": profile
    }

# =====================================================================
# ENHANCED DATA MANAGER WITH RETRY LOGIC
# =====================================================================

class EnhancedDataManager:
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}
        self.backtest_engine = RealBacktestEngine()
        self.market_profile_cache = {}
        self.last_rsi_scan = None
        self.api_call_count = 0
        self.last_api_reset = time.time()
        
    def _validate_live_price(self, symbol, max_retries=3):
        """Validate and fetch live price with retry logic"""
        now_ts = time.time()
        key = f"price_{symbol}"
        
        # Check cache first
        if key in self.price_cache:
            cached = self.price_cache[key]
            if now_ts - cached["ts"] < 5:  # 5 second cache
                return cached["price"]
        
        # Try fetching with exponential backoff
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                
                # Try 1-minute data first
                df = ticker.history(period="1d", interval="1m")
                if df is not None and not df.empty:
                    price = float(df["Close"].iloc[-1])
                    self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                    return round(price, 2)
                
                # Fallback to 5-minute data
                df = ticker.history(period="2d", interval="5m")
                if df is not None and not df.empty:
                    price = float(df["Close"].iloc[-1])
                    self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                    return round(price, 2)
                
                # Last resort: ticker.info
                info = ticker.info
                if 'currentPrice' in info and info['currentPrice']:
                    price = float(info['currentPrice'])
                    self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                    return round(price, 2)
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        # If all attempts fail, log error and return None
        logger.error(f"Failed to fetch price for {symbol} after {max_retries} attempts")
        return None
    
    @st.cache_data(ttl=300)  # 5 minute cache
    def _fetch_yf(_self, symbol, period, interval, max_retries=3):
        """Fetch Yahoo Finance data with retry logic"""
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    prepost=False
                )
                
                if not data.empty:
                    return data
                    
            except Exception as e:
                logger.warning(f"Fetch attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to fetch {symbol} after {max_retries} attempts")
        
        return pd.DataFrame()
    
    def get_stock_data(self, symbol, interval="15m"):
        """Get stock data with all technical indicators"""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_api_reset > 60:
            self.api_call_count = 0
            self.last_api_reset = current_time
        
        if self.api_call_count > 100:
            logger.warning("API rate limit approaching, using cached data")
            return self.create_validated_demo_data(symbol)
        
        self.api_call_count += 1
        
        # Determine period based on interval
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
            logger.warning(f"Insufficient data for {symbol}, using demo data")
            return self.create_validated_demo_data(symbol)
        
        # Clean column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        
        # Ensure required columns exist
        expected = ["Open", "High", "Low", "Close", "Volume"]
        for e in expected:
            if e not in df.columns:
                if e.upper() in df.columns:
                    df[e] = df[e.upper()]
                else:
                    logger.error(f"Missing column {e} for {symbol}")
                    return self.create_validated_demo_data(symbol)
        
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        
        if len(df) < 20:
            return self.create_validated_demo_data(symbol)
        
        # Update with live price
        try:
            live_price = self._validate_live_price(symbol)
            if live_price is not None:
                current_close = df["Close"].iloc[-1]
                price_diff_pct = abs(live_price - current_close) / max(current_close, 1e-6)
                
                if price_diff_pct > 0.005:
                    df.iloc[-1, df.columns.get_loc("Close")] = live_price
                    df.iloc[-1, df.columns.get_loc("High")] = max(df.iloc[-1]["High"], live_price)
                    df.iloc[-1, df.columns.get_loc("Low")] = min(df.iloc[-1]["Low"], live_price)
        except Exception as e:
            logger.warning(f"Failed to update live price for {symbol}: {e}")
        
        # Calculate all indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        
        # Market Profile
        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]
        
        # Support/Resistance
        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]
        
        # ADX
        try:
            df_adx = adx(df["High"], df["Low"], df["Close"], period=14)
            df["ADX"] = pd.Series(df_adx, index=df.index).fillna(method="ffill").fillna(20)
        except Exception:
            df["ADX"] = 20
        
        # Higher Timeframe Trend
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
        """Create demo data when real data is unavailable"""
        logger.info(f"Creating demo data for {symbol}")
        
        live = self._validate_live_price(symbol)
        if live is None:
            live = 1000.0  # Ultimate fallback
        
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
        
        df = pd.DataFrame({
            "Open": openp,
            "High": highp,
            "Low": lowp,
            "Close": prices,
            "Volume": vol
        }, index=dates)
        
        df.iloc[-1, df.columns.get_loc("Close")] = live
        
        # Calculate indicators
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
    
    def get_comprehensive_accuracy(self, symbol, strategy):
        """Get comprehensive accuracy metrics with confidence intervals"""
        key = f"{symbol}_{strategy}"
        
        if key in self.backtest_engine.historical_results:
            return self.backtest_engine.historical_results[key]
        
        data = self.get_stock_data(symbol, "15m")
        result = self.backtest_engine.calculate_comprehensive_accuracy(symbol, strategy, data)
        
        self.backtest_engine.historical_results[key] = result
        return result
    
    def should_run_rsi_scan(self):
        """Check if RSI scan should run"""
        current_time = time.time()
        if self.last_rsi_scan is None:
            self.last_rsi_scan = current_time
            return True
        
        if current_time - self.last_rsi_scan >= 75:
            self.last_rsi_scan = current_time
            return True
        return False

# =====================================================================
# ENHANCED BACKTEST ENGINE
# =====================================================================

@dataclass
class RealBacktestEngine:
    historical_results: dict = field(default_factory=dict)
    
    def _default_position_size(self, capital, risk_per_trade=0.01, stop_loss=None, entry=None):
        """Calculate position size based on risk"""
        if stop_loss is None or entry is None:
            return 0
        
        risk_amount = capital * risk_per_trade
        per_share_risk = abs(entry - stop_loss)
        
        if per_share_risk <= 0:
            return 0
        
        qty = int(risk_amount // per_share_risk)
        return max(qty, 0)
    
    def _apply_costs(self, price, quantity, cost_pct, slippage_pct, side="buy"):
        """Apply slippage and costs to fill price"""
        slip = price * slippage_pct
        if side.lower() == "buy":
            fill_price = price + slip
        else:
            fill_price = price - slip
        return fill_price
    
    def _compute_equity_curve_stats(self, equity_series):
        """Compute equity curve statistics"""
        eq = pd.Series(equity_series).dropna()
        if eq.empty:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
                "total_return": 0.0
            }
        
        cum = eq.values
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(dd.min())
        max_dd_pct = abs(max_dd)
        
        total_return = (cum[-1] / cum,[object Object],) - 1.0
        
        returns = pd.Series(cum).pct_change().fillna(0)
        mean_r = returns.mean()
        std_r = returns.std(ddof=0) if returns.std(ddof=0) != 0 else 1e-9
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
        
        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "sharpe": float(sharpe),
            "total_return": float(total_return)
        }
    
    def backtest_with_generator(self,
                                data: pd.DataFrame,
                                signal_generator,
                                initial_capital: float = 1_000_000.0,
                                risk_per_trade: float = 0.01,
                                cost_pct: float = 0.0004,
                                slippage_pct: float = 0.0003,
                                max_hold_bars: int = 40,
                                min_bars_between_signals: int = 1):
        """Comprehensive backtesting engine"""
        if data is None or len(data) < 30:
            return {"error": "insufficient_data", "metrics": {}}
        
        capital = float(initial_capital)
        equity = capital
        equity_curve = [equity]
        trades = []
        last_signal_bar = -9999
        
        needed_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed_cols:
            if c not in data.columns:
                raise ValueError(f"Data must contain column '{c}'")
        
        for i in range(20, len(data) - 1):
            if i - last_signal_bar < min_bars_between_signals:
                continue
            
            current_slice = data.iloc[:i+1].copy()
            signal = signal_generator(current_slice)
            
            if not signal or "action" not in signal:
                continue
            
            action = signal["action"].upper()
            entry_price = float(current_slice["Close"].iloc[-1])
            
            stop_loss = signal.get("stop_loss", None)
            target = signal.get("target", None)
            
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
                    if action == "BUY":
                        stop_loss = entry_price * 0.995 if stop_loss is None else stop_loss
                        target = entry_price * 1.0125 if target is None else target
                    else:
                        stop_loss = entry_price * 1.005 if stop_loss is None else stop_loss
                        target = entry_price * 0.9875 if target is None else target
            
            qty = self._default_position_size(capital, risk_per_trade, stop_loss=stop_loss, entry=entry_price)
            if qty <= 0:
                continue
            
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
            
            trade_value = entry_fill * qty
            if action == "BUY":
                capital -= trade_value
            else:
                margin = trade_value * 0.2
                capital -= margin
            
            exited = False
            for j in range(i+1, min(i+1+max_hold_bars, len(data))):
                high = float(data["High"].iloc[j])
                low = float(data["Low"].iloc[j])
                close = float(data["Close"].iloc[j])
                
                if action == "BUY":
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
                        continue
                else:
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
                        "exit_bar": exit_bar,
                        "exit_time": data.index[exit_bar],
                        "exit_price": exit_price,
                        "pnl": float(net_pnl),
                        "return_pct": float(return_pct),
                        "duration_bars": exit_bar - i,
                        "exit_reason": reason
                    })
                    
                    trades.append(position)
                    equity = capital
                    equity_curve.append(equity)
                    last_signal_bar = j
                    break
            
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
        
        return {"trades": trades_df.to_dict(orient="records"), "metrics": metrics, "equity_curve": equity_curve}
    
    def calculate_comprehensive_accuracy(self, symbol, strategy, data):
        """Calculate comprehensive accuracy with confidence intervals"""
        key = f"{symbol}_{strategy}"
        
        if key in self.historical_results:
            cached = self.historical_results[key]
            if isinstance(cached, dict) and "composite_accuracy" in cached:
                return cached
        
        def fallback_generator(d_slice):
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
                
                return None
            except Exception:
                return None
        
        try:
            result = self.backtest_with_generator(
                data,
                fallback_generator,
                initial_capital=1_000_000.0,
                risk_per_trade=0.01,
                cost_pct=0.0004,
                slippage_pct=0.0003,
                max_hold_bars=40
            )
            
            metrics = result.get("metrics", {})
            win_rate = metrics.get("win_rate", 0.0)
            profit_factor = metrics.get("profit_factor", 0.0)
            sharpe = metrics.get("sharpe", 0.0)
            sample_size = metrics.get("total_trades", 0)
            
            # Calculate confidence interval
            if sample_size >= 30:
                std_error = np.sqrt((win_rate * (1 - win_rate)) / sample_size)
                confidence_95 = 1.96 * std_error
                lower_bound = max(0, win_rate - confidence_95)
            else:
                lower_bound = win_rate * 0.7  # Conservative adjustment for small samples
            
            # Composite accuracy score
            composite_accuracy = (
                win_rate * 0.4 +
                min(profit_factor / 3.0, 1.0) * 0.3 +
                min(max(sharpe, 0) / 2.0, 1.0) * 0.3
            )
            
            comprehensive_result = {
                "composite_accuracy": float(composite_accuracy),
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor),
                "sharpe": float(sharpe),
                "sample_size": int(sample_size),
                "confidence_lower_bound": float(lower_bound)
            }
            
            self.historical_results[key] = comprehensive_result
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Backtest error for {symbol} {strategy}: {e}")
            return {
                "composite_accuracy": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "sample_size": 0,
                "confidence_lower_bound": 0.0
            }

# =====================================================================
# CIRCUIT BREAKER & SAFETY
# =====================================================================

class CircuitBreaker:
    def __init__(self):
        self.max_daily_loss = MAX_DAILY_LOSS
        self.max_drawdown = MAX_DRAWDOWN
        self.max_consecutive_losses = MAX_CONSECUTIVE_LOSSES
        self.triggered = False
        self.trigger_reason = None
    
    def check_safety(self, trader):
        """Perform safety checks before auto-execution"""
        # Reset if new day
        current_date = now_indian().date()
        if hasattr(self, 'last_check_date') and self.last_check_date != current_date:
            self.triggered = False
            self.trigger_reason = None
        self.last_check_date = current_date
        
        if self.triggered:
            return False, self.trigger_reason
        
        # Check daily loss limit
        daily_pnl = trader.get_daily_pnl()
        if daily_pnl < -self.max_daily_loss:
            self.triggered = True
            self.trigger_reason = f"🚨 CIRCUIT BREAKER: Daily loss limit (₹{self.max_daily_loss:,.0f}) reached!"
            logger.critical(self.trigger_reason)
            return False, self.trigger_reason
        
        # Check drawdown
        current_equity = trader.equity()
        peak_equity = trader.get_peak_equity()
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown > self.max_drawdown:
                self.triggered = True
                self.trigger_reason = f"⚠️ HIGH DRAWDOWN: {drawdown:.1%} exceeds limit ({self.max_drawdown:.1%})"
                logger.warning(self.trigger_reason)
                return False, self.trigger_reason
        
        # Check consecutive losses
        consecutive_losses = trader.get_consecutive_losses()
        if consecutive_losses >= self.max_consecutive_losses:
            self.triggered = True
            self.trigger_reason = f"⚠️ {consecutive_losses} consecutive losses. Auto-execution paused."
            logger.warning(self.trigger_reason)
            return False, self.trigger_reason
        
        return True, "All safety checks passed"
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.triggered = False
        self.trigger_reason = None
        logger.info("Circuit breaker manually reset")

# =====================================================================
# ENHANCED TRADING ENGINE
# =====================================================================

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
        self.peak_equity = float(capital)
        self.daily_start_equity = float(capital)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Strategy performance
        self.strategy_performance = {}
        for strategy in TRADING_STRATEGIES.keys():
            self.strategy_performance[strategy] = {
                "signals": 0,
                "trades": 0,
                "wins": 0,
                "pnl": 0.0
            }
    
    def reset_daily_counts(self):
        """Reset daily counters"""
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.stock_trades = 0
            self.auto_trades_count = 0
            self.last_reset = current_date
            self.daily_start_equity = self.equity()
            logger.info(f"Daily counters reset for {current_date}")
    
    def can_auto_trade(self):
        """Check if auto-trading is allowed"""
        return (
            self.auto_trades_count < MAX_AUTO_TRADES and
            self.daily_trades < MAX_DAILY_TRADES and
            market_open()
        )
    
    def get_peak_equity(self):
        """Get peak equity value"""
        current = self.equity()
        self.peak_equity = max(self.peak_equity, current)
        return self.peak_equity
    
    def get_daily_pnl(self):
        """Calculate daily P&L"""
        return self.equity() - self.daily_start_equity
    
    def get_consecutive_losses(self):
        """Count consecutive losing trades"""
        closed_trades = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        if not closed_trades:
            return 0
        
        consecutive = 0
        for trade in reversed(closed_trades):
            if trade.get("closed_pnl", 0) < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def calculate_volatility_adjusted_position_size(self, signal, risk_pct=0.01):
        """Calculate position size adjusted for volatility"""
        entry = signal["entry"]
        stop_loss = signal["stop_loss"]
        atr = signal.get("atr", entry * 0.02)
        
        # Risk amount
        risk_amount = self.cash * risk_pct
        
        # Per-share risk
        per_share_risk = abs(entry - stop_loss)
        if per_share_risk <= 0:
            return 0
        
        # Base quantity
        base_qty = int(risk_amount / per_share_risk)
        
        # Volatility adjustment
        volatility_ratio = atr / entry
        if volatility_ratio > 0.03:  # High volatility
            base_qty = int(base_qty * 0.7)
        elif volatility_ratio < 0.01:  # Low volatility
            base_qty = int(base_qty * 1.2)
        
        # Max position size (10% of capital)
        max_position_value = self.cash * 0.10
        max_qty = int(max_position_value / entry)
        
        return min(base_qty, max_qty)
    
    def calculate_support_resistance(self, symbol, current_price):
        """Calculate support and resistance levels"""
        try:
            data = data_manager.get_stock_data(symbol, "15m")
            if data is None or len(data) < 20:
                return current_price * 0.98, current_price * 1.02
            return float(data["Support"].iloc[-1]), float(data["Resistance"].iloc[-1])
        except Exception:
            return current_price * 0.98, current_price * 1.02
    
    def calculate_intraday_target_sl(self, entry_price, action, atr, current_price, support, resistance):
        """Calculate intraday target and stop loss"""
        if atr <= 0 or np.isnan(atr):
            atr = max(entry_price * 0.005, 1.0)
        
        if action == "BUY":
            sl = entry_price - (atr * 1.2)
            target = entry_price + (atr * 2.5)
            
            if target > resistance:
                target = min(target, resistance * 0.998)
            sl = max(sl, support * 0.995)
        else:
            sl = entry_price + (atr * 1.2)
            target = entry_price - (atr * 2.5)
            
            if target < support:
                target = max(target, support * 1.002)
            sl = min(sl, resistance * 1.005)
