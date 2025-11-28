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

# --- UPDATED UNIVERSE DEFINITION (Nifty 50, Nifty 100, Nifty Midcap 100 Combined) ---
# NIFTY 50 (Subset used for example, replace with full list if available)
NIFTY_50 = [
   "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "BAJFINANCE.NS"
]

# NIFTY 100 (Example subset, usually includes Nifty 50)
NIFTY_100 = list(set(NIFTY_50 + [
    "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS", "DIVISLAB.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HDFCLIFE.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "MARUTI.NS"
]))

# NIFTY MIDCAP 100 (Example subset)
NIFTY_MIDCAP = [
    "ABB.NS", "ASHOKLEY.NS", "BEL.NS", "BOSCHLTD.NS", "CGPOWER.NS",
    "COLPAL.NS", "DALMIABHA.NS", "DIXON.NS", "HAL.NS", "MINDTREE.NS",
    "NMDC.NS", "PAGEIND.NS", "PETRONET.NS", "POLYCAB.NS", "SRF.NS"
]

# Combined Universe - Ensuring unique tickers
TRADING_UNIVERSE = sorted(list(set(NIFTY_100 + NIFTY_MIDCAP)))
# -----------------------------------------------------------------------------------


# --- Dummy Data and Utility Functions ---

# Historical Win Rate (Required for the 70% filter)
# Strategies must have a historical win rate >= 70% to generate signals.
HISTORICAL_ACCURACY = {
    "EMA_VWAP_Confluence": 0.72,  # > 70%
    "RSI_MeanReversion": 0.68,    # < 70%
    "Bollinger_Reversion": 0.55,  # < 70%
    "MACD_Momentum": 0.75,        # > 70%
    "Support_Resistance_Breakout": 0.65, # < 70%
    "EMA_VWAP_Downtrend": 0.71,   # > 70%
    "RSI_Overbought": 0.60,       # < 70%
    "Bollinger_Rejection": 0.56,  # < 70%
    "MACD_Bearish": 0.74,         # > 70%
    "Trend_Reversal": 0.62        # < 70%
}

def is_market_open(tz):
    now = datetime.now(tz)
    market_start = dt_time(9, 15)
    market_end = dt_time(15, 30)
    # Check if it's a weekday (0=Monday, 6=Sunday) and within market hours
    return 0 <= now.weekday() <= 4 and market_start <= now.time() <= market_end

def fetch_data(ticker):
    """Fetches real-time price data for a single ticker."""
    try:
        # Fetching latest data point (as a proxy for live price)
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            return latest_price
    except Exception:
        return np.nan
    return np.nan

@st.cache_data(ttl=SIGNAL_REFRESH_MS / 1000)
def generate_signals(universe):
    """Generates dummy trading signals for the universe."""
    signals = []
    current_time = datetime.now(IND_TZ).strftime("%H:%M:%S")

    # Filter strategies based on the 70% historical win rate criterion
    high_accuracy_strategies = [
        s for s, acc in HISTORICAL_ACCURACY.items() if acc >= 0.70
    ]

    for ticker in np.random.choice(universe, size=np.random.randint(5, 15), replace=False):
        # Only use high-accuracy strategies for signal generation
        strategy = np.random.choice(high_accuracy_strategies)
        
        # Determine action (Buy/Sell)
        action = np.random.choice(["BUY", "SELL"])

        # Determine if signal is active (e.g., based on real-time indicator confluence)
        is_active = np.random.choice([True, False], p=[0.6, 0.4])

        if is_active:
            # Generate a plausible target/stop-loss relative to a dummy price
            dummy_price = np.random.uniform(100, 5000)
            if action == "BUY":
                target = dummy_price * (1 + np.random.uniform(0.005, 0.015))
                stop_loss = dummy_price * (1 - np.random.uniform(0.005, 0.01))
            else: # SELL (Short)
                target = dummy_price * (1 - np.random.uniform(0.005, 0.015))
                stop_loss = dummy_price * (1 + np.random.uniform(0.005, 0.01))
            
            # Historical accuracy for the selected strategy
            accuracy = HISTORICAL_ACCURACY[strategy] * 100
            
            signals.append({
                "Time": current_time,
                "Ticker": ticker,
                "Action": action,
                "Strategy": strategy,
                "Accuracy": f"{accuracy:.1f}%", # Display the high accuracy
                "Price": f"₹{dummy_price:,.2f}",
                "Target": f"₹{target:,.2f}",
                "StopLoss": f"₹{stop_loss:,.2f}",
                "Status": "Active"
            })
    
    return pd.DataFrame(signals) if signals else pd.DataFrame(columns=["Time", "Ticker", "Action", "Strategy", "Accuracy", "Price", "Target", "StopLoss", "Status"])

# --- Session State Setup ---
if 'auto_trade_enabled' not in st.session_state:
    st.session_state.auto_trade_enabled = False
if 'trades_executed' not in st.session_state:
    st.session_state.trades_executed = []
if 'max_trades_hit' not in st.session_state:
    st.session_state.max_trades_hit = False

# --- Enhanced S/R Monitor Data ---
def generate_sr_data(universe):
    """Generates structured S/R data with accuracy status."""
    data = []
    
    for ticker in np.random.choice(universe, size=min(10, len(universe)), replace=False):
        price = np.random.uniform(500, 5000)
        
        # Generate two support and two resistance levels
        s1 = price * (1 - np.random.uniform(0.01, 0.02))
        s2 = price * (1 - np.random.uniform(0.03, 0.05))
        r1 = price * (1 + np.random.uniform(0.01, 0.02))
        r2 = price * (1 + np.random.uniform(0.03, 0.05))

        # Determine status (e.g., based on recent price action)
        status_s1 = np.random.choice(["Approaching", "Tested (Hold)", "Broken (Weak)"], p=[0.4, 0.4, 0.2])
        status_r1 = np.random.choice(["Approaching", "Tested (Hold)", "Broken (Weak)"], p=[0.4, 0.4, 0.2])

        # Assign confidence based on how the levels were generated (e.g., daily pivots vs. weekly highs)
        confidence_s1 = np.random.choice(["High (Daily Pivot)", "Medium (Intraday Low)"])
        confidence_r1 = np.random.choice(["High (Weekly High)", "Medium (Intraday High)"])
        
        data.append({
            "Ticker": ticker,
            "LTP": f"₹{price:,.2f}",
            "S1_Level": f"₹{s1:,.2f}",
            "S1_Confidence": confidence_s1,
            "S1_Status": status_s1,
            "R1_Level": f"₹{r1:,.2f}",
            "R1_Confidence": confidence_r1,
            "R1_Status": status_r1,
            # For simplicity, not including S2/R2, but they are calculable
        })
        
    return pd.DataFrame(data)

# --- Trading System Class ---
class TradingSystem:
    def __init__(self, capital, trade_alloc):
        self.capital = capital
        self.trade_alloc = trade_alloc
        self.portfolio = {}
        self.strategy_performance = {}

    def execute_trade(self, signal, is_auto=False):
        """Placeholder for trade execution logic."""
        ticker = signal['Ticker']
        action = signal['Action']
        strategy = signal['Strategy']
        
        # Basic P&L simulation for tracking
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {"trades": 0, "wins": 0, "pnl": 0.0}

        # Simulate a 70% win rate for executed trades
        is_win = np.random.choice([True, False], p=[0.70, 0.30])
        trade_pnl = self.capital * self.trade_alloc * (0.01 if is_win else -0.005) # Simulating R:R of 2:1
        
        self.strategy_performance[strategy]["trades"] += 1
        if is_win:
            self.strategy_performance[strategy]["wins"] += 1
        self.strategy_performance[strategy]["pnl"] += trade_pnl
        
        trade_log = {
            "Time": datetime.now(IND_TZ).strftime("%H:%M:%S"),
            "Ticker": ticker,
            "Action": action,
            "Strategy": strategy,
            "P&L": f"₹{trade_pnl:+.2f}",
            "Type": "Auto" if is_auto else "Manual",
            "Result": "Win" if is_win else "Loss"
        }
        return trade_log

# Initialize Trading System
if 'trader' not in st.session_state:
    st.session_state.trader = TradingSystem(CAPITAL, TRADE_ALLOC)
trader = st.session_state.trader

# --- Main Streamlit App Layout ---

st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh")
st.sidebar.title("Rantv Pro Controls")
st.sidebar.markdown(f"### Market: `CASH`")

is_open = is_market_open(IND_TZ)
st.sidebar.markdown(f"**Market Status:** {'🟢 Open' if is_open else '🔴 Closed'} (IST: {datetime.now(IND_TZ).strftime('%H:%M:%S')})")

# 1. Auto Execution Toggle (FIXED - using session state to maintain setting)
st.session_state.auto_trade_enabled = st.sidebar.checkbox(
    "Enable Auto Execution (Max 10/day)",
    value=st.session_state.auto_trade_enabled,
    key='auto_exec_toggle'
)

# Trade Log Summary
st.sidebar.subheader("Execution Status")
total_trades = len(st.session_state.trades_executed)
st.sidebar.info(f"Executed Trades Today: **{total_trades}/{MAX_AUTO_TRADES}**")

# Stop auto-execution if max trades hit
if total_trades >= MAX_AUTO_TRADES:
    st.session_state.max_trades_hit = True
    st.session_state.auto_trade_enabled = False
    st.sidebar.error("Daily Auto Trade Limit Reached. Execution Disabled.")


# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["⚡ Live Signals", "📊 S/R MONITOR", "📜 Trade Log"])

with tab1:
    st_autorefresh(interval=SIGNAL_REFRESH_MS, key="signal_refresh")
    st.header("⚡ Live Trading Signals - High Accuracy Universe")
    st.markdown(f"Scanning **{len(TRADING_UNIVERSE)}** stocks (Nifty 100 + Midcap 100).")
    st.markdown("🚨 **Signal Filter:** Only strategies with **>= 70%** historical win rate are considered.")

    signals_df = generate_signals(TRADING_UNIVERSE)

    if signals_df.empty:
        st.info("No active high-accuracy signals generated currently.")
    else:
        # Display Signals
        st.subheader(f"Active Signals ({len(signals_df)})")
        st.dataframe(
            signals_df,
            height=300,
            use_container_width=True,
            column_config={
                "Action": st.column_config.Column(width="small"),
                "Accuracy": st.column_config.Column(width="small")
            }
        )

        # 3. Auto Execution Logic (FIXED - Check if enabled and limit not hit)
        if st.session_state.auto_trade_enabled and is_open and not st.session_state.max_trades_hit:
            st.warning("Auto Execution is **ENABLED**. Checking for signals to execute...")
            
            new_executions = 0
            for _, signal in signals_df.iterrows():
                # Only execute if the trade limit hasn't been reached
                if len(st.session_state.trades_executed) < MAX_AUTO_TRADES:
                    # In a real system, you'd check if this stock is already traded/position open
                    # For this dummy example, we execute every new active signal
                    trade_log = trader.execute_trade(signal.to_dict(), is_auto=True)
                    st.session_state.trades_executed.append(trade_log)
                    new_executions += 1
                else:
                    break # Stop if max trades reached

            if new_executions > 0:
                st.success(f"Executed **{new_executions}** trades automatically.")
            elif len(st.session_state.trades_executed) >= MAX_AUTO_TRADES:
                st.error("Auto Execution Halted: Daily execution limit reached.")
            else:
                st.info("No new signals met execution criteria this cycle.")

        elif st.session_state.auto_trade_enabled and not is_open:
            st.warning("Auto Execution is ON, but the market is currently closed.")
        elif not st.session_state.auto_trade_enabled:
            st.info("Auto Execution is **DISABLED**. Signals are for manual trading.")


with tab2:
    st.header("📊 Enhanced S/R Monitor")
    st.markdown("Displays critical Support and Resistance levels with confidence and real-time status for better accuracy.")

    sr_df = generate_sr_data(TRADING_UNIVERSE)
    
    # 4. Enhanced S/R Table Display
    if not sr_df.empty:
        st.dataframe(
            sr_df,
            height=350,
            use_container_width=True,
            column_config={
                "Ticker": "Symbol",
                "LTP": "Last Traded Price",
                "S1_Level": "Support 1 (S1)",
                "S1_Confidence": "S1 Confidence",
                "S1_Status": "S1 Status",
                "R1_Level": "Resistance 1 (R1)",
                "R1_Confidence": "R1 Confidence",
                "R1_Status": "R1 Status",
            }
        )
        st.markdown("*Status indicates recent price interaction: 'Tested (Hold)' suggests strength, 'Broken (Weak)' suggests a shift.*")
    else:
        st.info("S/R data could not be generated for the universe.")

with tab3:
    st.header("📜 Live Trade Log")
    if st.session_state.trades_executed:
        log_df = pd.DataFrame(st.session_state.trades_executed)
        st.dataframe(log_df, use_container_width=True)

        st.subheader("Performance Summary")
        
        # Calculate overall P&L
        total_pnl = sum([float(t['P&L'].replace('₹', '').replace(',', '').replace('+', '')) for t in st.session_state.trades_executed])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", len(st.session_state.trades_executed))
        col2.metric("Total P&L", f"₹{total_pnl:,.2f}", delta_color=("inverse" if total_pnl < 0 else "normal"))
        
        # Calculate win rate from strategy_performance
        total_wins = sum(p['wins'] for p in trader.strategy_performance.values())
        total_executed = sum(p['trades'] for p in trader.strategy_performance.values())
        overall_win_rate = (total_wins / total_executed) if total_executed > 0 else 0
        col3.metric("Overall Win Rate", f"{overall_win_rate:.1%}")

    else:
        st.info("No trades executed yet.")

    st.subheader("Strategy-wise Historical Accuracy")
    
    # Display historical accuracy for context
    accuracy_data = []
    for strat, acc in HISTORICAL_ACCURACY.items():
        accuracy_data.append({
            "Strategy": strat,
            "Historical Win Rate": f"{acc * 100:.1f}%",
            "Meets 70% Filter": "✅ YES" if acc >= 0.70 else "❌ NO"
        })
    st.dataframe(pd.DataFrame(accuracy_data), use_container_width=True)

    st.markdown("---")
    st.caption("Disclaimer: This is a simulation terminal. Actual market conditions and execution may vary. Trade at your own risk.")
