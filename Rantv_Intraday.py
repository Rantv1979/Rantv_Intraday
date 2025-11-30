# Rantv Intraday Trading Signals & Market Analysis - PRODUCTION READY
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
import os
from dataclasses import dataclass
from typing import Optional, Dict, List
import requests
import json

# Try to import optional dependencies with fallbacks
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
@dataclass
class AppConfig:
    database_url: str = 'sqlite:///trading_journal.db'
    risk_tolerance: str = 'MODERATE'
    max_daily_loss: float = 50000.0
    enable_ml: bool = True
    
    @classmethod
    def from_env(cls):
        return cls()

# Initialize configuration
config = AppConfig.from_env()

st.set_page_config(page_title="Rantv Intraday Terminal Pro - Enhanced", layout="wide", initial_sidebar_state="expanded")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading Constants
CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 15
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10

SIGNAL_REFRESH_MS = 90000
PRICE_REFRESH_MS = 60000

MARKET_OPTIONS = ["CASH"]

# Stock Universes - COMBINED ALL STOCKS
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

# MIDCAP STOCKS - High Potential for Intraday
NIFTY_MIDCAP_150 = [
    "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "AUBANK.NS", "AIAENG.NS",
    "APLAPOLLO.NS", "ASTRAL.NS", "AARTIIND.NS", "BALKRISIND.NS", "BANKBARODA.NS",
    "BANKINDIA.NS", "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS",
    "BIOCON.NS", "BOSCHLTD.NS", "BRIGADE.NS", "CANBK.NS", "CANFINHOME.NS",
    "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS",
    "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS",
    "DALBHARAT.NS", "DEEPAKNTR.NS", "DELTACORP.NS", "DIVISLAB.NS", "DIXON.NS",
    "DLF.NS", "DRREDDY.NS", "EDELWEISS.NS", "EICHERMOT.NS", "ESCORTS.NS",
    "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS", "GODREJCP.NS",
    "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS",
    "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIPRULI.NS",
    "IDEA.NS", "IDFCFIRSTB.NS", "IGL.NS", "INDIACEM.NS", "INDIAMART.NS",
    "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IPCALAB.NS", "JINDALSTEL.NS",
    "JSWENERGY.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", "L&TFH.NS", "LICHSGFIN.NS",
    "LT.NS", "LTTS.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MFSL.NS",
    "MGL.NS", "MINDTREE.NS", "MOTHERSUMI.NS", "MPHASIS.NS", "MRF.NS",
    "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NESTLEIND.NS", "NMDC.NS",
    "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS",
    "PEL.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS",
    "PNB.NS", "POWERGRID.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RBLBANK.NS",
    "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS",
    "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SRTRANSFIN.NS", "SUNPHARMA.NS",
    "SUNTV.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
    "TORNTPHARM.NS", "TRENT.NS", "UPL.NS", "VOLTAS.NS", "WIPRO.NS",
    "YESBANK.NS", "ZEEL.NS"
]

# COMBINED ALL STOCKS - NEW UNIVERSES
ALL_STOCKS = list(set(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

# Enhanced Trading Strategies with Better Balance - ALL STRATEGIES ENABLED
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

# HIGH ACCURACY STRATEGIES FOR ALL STOCKS - ENABLED FOR ALL UNIVERSES
HIGH_ACCURACY_STRATEGIES = {
    "Multi_Confirmation": {"name": "Multi-Confirmation Ultra", "weight": 5, "type": "BOTH"},
    "Enhanced_EMA_VWAP": {"name": "Enhanced EMA-VWAP", "weight": 4, "type": "BOTH"},
    "Volume_Breakout": {"name": "Volume Weighted Breakout", "weight": 4, "type": "BOTH"},
    "RSI_Divergence": {"name": "RSI Divergence", "weight": 3, "type": "BOTH"},
    "MACD_Trend": {"name": "MACD Trend Momentum", "weight": 3, "type": "BOTH"}
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
    
    /* High Accuracy Strategy Cards */
    .high-accuracy-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
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
    
    /* Alert Styles */
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #d97706;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    /* Midcap Specific Styles */
    .midcap-signal {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-left: 4px solid #0369a1;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Dependencies Warning Styling */
    .dependencies-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #d97706;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #f59e0b;
    }
    
    .dependencies-warning h4 {
        color: #92400e;
        margin-bottom: 10px;
    }
    
    .dependencies-warning ul {
        color: #92400e;
        margin-left: 20px;
    }
    
    .dependencies-warning code {
        background: #fef3c7;
        padding: 2px 6px;
        border-radius: 4px;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# Display dependencies warning in a proper format
if not SQLALCHEMY_AVAILABLE or not JOBLIB_AVAILABLE:
    st.markdown("""
    <div class="dependencies-warning">
        <h4>🔧 Missing Dependencies Detected</h4>
        <p>To enable all features, install the required packages:</p>
        <code>pip install sqlalchemy joblib</code>
        <p><strong>Current limitations:</strong></p>
        <ul>
            <li>Database features disabled</li>
            <li>ML model persistence disabled</li>
            <li>Some advanced features limited</li>
        </ul>
        <p><em>Basic functionality remains available. Install dependencies for full features.</em></p>
    </div>
    """, unsafe_allow_html=True)

# [Rest of your existing code remains the same until the universe selection part...]

# Enhanced Multi-Strategy Trading Engine with ALL NEW features
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
        
        # Initialize high accuracy strategies
        for strategy in HIGH_ACCURACY_STRATEGIES.keys():
            self.strategy_performance[strategy] = {"signals": 0, "trades": 0, "wins": 0, "pnl": 0.0}
        
        # NEW: Integrated systems
        self.data_manager = EnhancedDataManager()
        self.risk_manager = AdvancedRiskManager()
        self.ml_enhancer = MLSignalEnhancer()
        self.regime_detector = MarketRegimeDetector()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.alert_manager = AlertManager()

    # [Rest of your existing MultiStrategyIntradayTrader methods...]

    def generate_quality_signals(self, universe, max_scan=None, min_confidence=0.7, min_score=6, use_high_accuracy=True):  # CHANGED: Default to True for all
        signals = []
        if universe == "Nifty 50":
            stocks = NIFTY_50
        elif universe == "Nifty 100":
            stocks = NIFTY_100
        elif universe == "Midcap 150":
            stocks = NIFTY_MIDCAP_150
        elif universe == "All Stocks":  # NEW: All Stocks universe
            stocks = ALL_STOCKS
        else:
            stocks = NIFTY_50
            
        if max_scan is None:
            max_scan = len(stocks)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # NEW: Get market regime for context
        market_regime = self.data_manager.get_market_regime()
        st.info(f"📊 Current Market Regime: **{market_regime}** | Universe: **{universe}** | High Accuracy: **{'ON' if use_high_accuracy else 'OFF'}**")
        
        for idx, symbol in enumerate(stocks[:max_scan]):
            try:
                status_text.text(f"Scanning {symbol} ({idx+1}/{len(stocks[:max_scan])})")
                progress_bar.progress((idx + 1) / len(stocks[:max_scan]))
                data = self.data_manager.get_stock_data(symbol, "15m")
                if data is None or len(data) < 30:
                    continue
                
                # CHANGED: Always use high accuracy strategies when enabled
                if use_high_accuracy:
                    # Use high accuracy strategies for all stocks
                    high_acc_signals = self.generate_high_accuracy_signals(symbol, data)
                    signals.extend(high_acc_signals)
                
                # Also include standard strategies
                standard_signals = self.generate_strategy_signals(symbol, data)
                signals.extend(standard_signals)
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        progress_bar.empty()
        status_text.empty()
        signals = [s for s in signals if s["confidence"] >= min_confidence and s["score"] >= min_score]
        signals.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
        self.signal_history = signals[:30]
        return signals[:20]

# [Rest of your existing code...]

# Initialize with error handling
try:
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
    st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>Rantv Intraday Terminal Pro - PRODUCTION READY</h1>", unsafe_allow_html=True)
    st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh_improved")

    # Market overview with enhanced metrics
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
    
    # NEW: Market Regime Display
    market_regime = data_manager.get_market_regime()
    regime_color = {
        "TRENDING": "🟢",
        "VOLATILE": "🟡", 
        "MEAN_REVERTING": "🔵",
        "NEUTRAL": "⚪"
    }.get(market_regime, "⚪")
    cols[3].metric("Market Regime", f"{regime_color} {market_regime}")
    
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
        
        nifty_sentiment = 50 + (nifty_change * 8)
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
        market_status = "LIVE" if market_open() else "CLOSED"
        status_sentiment = 80 if market_open() else 20
        st.markdown(create_circular_market_mood_gauge("MARKET", 0, 0, status_sentiment).replace("₹0", market_status).replace("0.00%", ""), unsafe_allow_html=True)
    with col4:
        auto_close_status = "ACTIVE" if not should_auto_close() else "INACTIVE"
        close_sentiment = 20 if should_auto_close() else 80
        st.markdown(create_circular_market_mood_gauge("AUTO CLOSE", 0, 0, close_sentiment).replace("₹0", "15:10").replace("0.00%", auto_close_status), unsafe_allow_html=True)

    # Sidebar with Universe Selection - UPDATED with All Stocks
    st.sidebar.header("⚙️ Trading Configuration")
    trader.selected_market = st.sidebar.selectbox("Market Type", MARKET_OPTIONS)
    
    # UPDATED: Universe Selection with All Stocks
    universe = st.sidebar.selectbox("Trading Universe", ["All Stocks", "Nifty 50", "Nifty 100", "Midcap 150"])
    
    # NEW: High Accuracy Toggle for All Universes
    enable_high_accuracy = st.sidebar.checkbox("Enable High Accuracy Strategies", value=True, 
                                              help="Enable high accuracy strategies for all stock universes")
    
    trader.auto_execution = st.sidebar.checkbox("Auto Execution", value=False)
    
    # NEW: Risk Management Settings
    st.sidebar.subheader("🎯 Risk Management")
    enable_ml = st.sidebar.checkbox("Enable ML Enhancement", value=True)
    kelly_sizing = st.sidebar.checkbox("Kelly Position Sizing", value=True)
    
    min_conf_percent = st.sidebar.slider("Minimum Confidence %", 60, 95, 70, 5)
    min_score = st.sidebar.slider("Minimum Score", 5, 10, 6, 1)
    scan_limit = st.sidebar.selectbox("Scan Limit", ["All Stocks", "Top 40", "Top 20"], index=0)
    max_scan_map = {"All Stocks": None, "Top 40": 40, "Top 20": 20}
    max_scan = max_scan_map[scan_limit]

    # Enhanced Tabs with Trade History
    tabs = st.tabs([
        "📈 Dashboard", 
        "🚦 Signals", 
        "💰 Paper Trading", 
        "📋 Trade History",
        "📊 Market Profile", 
        "📉 RSI Extreme", 
        "🔍 Backtest", 
        "⚡ Strategies",
        "🔔 Alerts",
        "🎯 High Accuracy Scanner"  # RENAMED: Now for all stocks
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
        
        # Add high accuracy strategies
        for strategy, config in HIGH_ACCURACY_STRATEGIES.items():
            if strategy in trader.strategy_performance:
                perf_data = trader.strategy_performance[strategy]
                if perf_data["trades"] > 0:
                    win_rate = perf_data["wins"] / perf_data["trades"]
                    strategy_data.append({
                        "Strategy": f"🔥 {config['name']}",
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
            generate_btn = st.button("Generate Signals", type="primary", use_container_width=True)
        with col2:
            if trader.auto_execution:
                st.success("🔴 Auto Execution: ACTIVE")
            else:
                st.info("⚪ Auto Execution: INACTIVE")
                
        if generate_btn or trader.auto_execution:
            with st.spinner(f"Scanning {universe} stocks with enhanced strategies..."):
                # CHANGED: Always use high accuracy when enabled
                signals = trader.generate_quality_signals(
                    universe, 
                    max_scan=max_scan, 
                    min_confidence=min_conf_percent/100.0, 
                    min_score=min_score,
                    use_high_accuracy=enable_high_accuracy  # Use the toggle value
                )
            
            if signals:
                # Separate BUY and SELL signals
                buy_signals = [s for s in signals if s["action"] == "BUY"]
                sell_signals = [s for s in signals if s["action"] == "SELL"]
                
                st.success(f"Found {len(buy_signals)} BUY signals and {len(sell_signals)} SELL signals")
                
                data_rows = []
                for s in signals:
                    # Check if it's a high accuracy strategy
                    is_high_acc = s["strategy"] in HIGH_ACCURACY_STRATEGIES
                    strategy_display = f"🔥 {s['strategy_name']}" if is_high_acc else s['strategy_name']
                    
                    data_rows.append({
                        "Symbol": s["symbol"].replace(".NS",""),
                        "Action": s["action"],
                        "Strategy": strategy_display,
                        "Entry Price": f"₹{s['entry']:.2f}",
                        "Current Price": f"₹{s['current_price']:.2f}",
                        "Target": f"₹{s['target']:.2f}",
                        "Stop Loss": f"₹{s['stop_loss']:.2f}",
                        "Confidence": f"{s['confidence']:.1%}",
                        "ML Confidence": f"{s.get('ml_confidence', 0.7):.1%}",
                        "Historical Win %": f"{s.get('historical_accuracy', 0.7):.1%}",
                        "Current Win %": f"{s.get('win_probability',0.7):.1%}",
                        "R:R": f"{s['risk_reward']:.2f}",
                        "Score": s['score'],
                        "RSI": f"{s['rsi']:.1f}",
                        "Market Regime": s.get('market_regime', 'NEUTRAL')
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
                        is_high_acc = s["strategy"] in HIGH_ACCURACY_STRATEGIES
                        strategy_display = f"🔥 {s['strategy_name']}" if is_high_acc else s['strategy_name']
                        st.write(f"{action_color} **{s['symbol'].replace('.NS','')}** - {s['action']} @ ₹{s['entry']:.2f} | Strategy: {strategy_display} | Historical Win: {s.get('historical_accuracy',0.7):.1%} | R:R: {s['risk_reward']:.2f}")
                    with col_b:
                        if kelly_sizing:
                            qty = trader.data_manager.calculate_optimal_position_size(
                                s["symbol"], s["win_probability"], s["risk_reward"], 
                                trader.cash, s["entry"], 
                                trader.data_manager.get_stock_data(s["symbol"], "15m")["ATR"].iloc[-1]
                            )
                        else:
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

    with tabs[9]:  # UPDATED: High Accuracy Scanner for All Stocks
        st.session_state.current_tab = "🎯 High Accuracy Scanner"
        st.subheader("🎯 High Accuracy Scanner - All Stocks")
        st.markdown(f"""
        <div class="alert-success">
            <strong>🔥 High Accuracy Strategies Enabled:</strong> 
            Scanning <strong>{universe}</strong> with enhanced high-accuracy strategies including
            Multi-Confirmation, Volume Breakouts, RSI Divergence, and MACD Trend Momentum.
            These strategies focus on volume confirmation, multi-timeframe alignment, 
            and divergence patterns for higher probability trades.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high_acc_scan_btn = st.button("🚀 Scan High Accuracy", type="primary", use_container_width=True)
        with col2:
            min_high_acc_confidence = st.slider("Min Confidence", 70, 90, 75, 5, key="high_acc_conf")
        with col3:
            min_high_acc_score = st.slider("Min Score", 6, 10, 7, 1, key="high_acc_score")
        
        if high_acc_scan_btn:
            with st.spinner(f"Scanning {universe} with high-accuracy strategies..."):
                high_acc_signals = trader.generate_quality_signals(
                    universe, 
                    max_scan=50 if universe == "All Stocks" else max_scan,  # Limit for All Stocks
                    min_confidence=min_high_acc_confidence/100.0,
                    min_score=min_high_acc_score,
                    use_high_accuracy=True  # Force high accuracy
                )
            
            if high_acc_signals:
                st.success(f"🎯 Found {len(high_acc_signals)} high-confidence signals!")
                
                # Display high accuracy signals with special styling
                for signal in high_acc_signals:
                    with st.container():
                        st.markdown(f"""
                        <div class="midcap-signal">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>{signal['symbol'].replace('.NS', '')}</strong> | 
                                    <span style="color: {'#059669' if signal['action'] == 'BUY' else '#dc2626'}">
                                        {signal['action']}
                                    </span> | 
                                    ₹{signal['entry']:.2f}
                                </div>
                                <div>
                                    <span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                                        {signal['strategy_name']}
                                    </span>
                                </div>
                            </div>
                            <div style="font-size: 12px; margin-top: 5px;">
                                Target: ₹{signal['target']:.2f} | SL: ₹{signal['stop_loss']:.2f} | 
                                R:R: {signal['risk_reward']:.2f} | Confidence: {signal['confidence']:.1%}
                            </div>
                            <div style="font-size: 11px; color: #6b7280;">
                                {signal.get('reason', 'High probability setup')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Quick execution buttons for high accuracy signals
                st.subheader("Quick Execution")
                exec_cols = st.columns(3)
                for idx, signal in enumerate(high_acc_signals[:6]):  # Show first 6
                    with exec_cols[idx % 3]:
                        if st.button(
                            f"{signal['action']} {signal['symbol'].replace('.NS', '')}", 
                            key=f"high_acc_exec_{signal['symbol']}",
                            use_container_width=True
                        ):
                            if kelly_sizing:
                                qty = trader.data_manager.calculate_optimal_position_size(
                                    signal["symbol"], signal["win_probability"], signal["risk_reward"], 
                                    trader.cash, signal["entry"], 
                                    trader.data_manager.get_stock_data(signal["symbol"], "15m")["ATR"].iloc[-1]
                                )
                            else:
                                qty = int((trader.cash * TRADE_ALLOC) / signal["entry"])
                            
                            success, msg = trader.execute_trade(
                                symbol=signal["symbol"],
                                action=signal["action"],
                                quantity=qty,
                                price=signal["entry"],
                                stop_loss=signal["stop_loss"],
                                target=signal["target"],
                                win_probability=signal.get("win_probability", 0.75),
                                strategy=signal.get("strategy")
                            )
                            if success:
                                st.success(msg)
            else:
                st.info("No high-confidence signals found with current filters.")

    # [Add other tab implementations here...]
    # Paper Trading, Trade History, Market Profile, RSI Extreme, Backtest, Strategies, Alerts

    st.markdown("---")
    st.markdown("<div style='text-align:center; color: #6b7280;'>Enhanced Intraday Terminal Pro with AI/ML & High Accuracy Strategies - All Stocks Universe Enabled</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Please refresh the page and try again")
    logger.error(f"Application crash: {e}")
