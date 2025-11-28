# Rantv Institutional Terminal (BlackBox Edition)
# Professional Grade Intraday Trading System
import time
from datetime import datetime, time as dt_time, timedelta
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# --- INSTITUTIONAL CONFIGURATION ---
st.set_page_config(
    page_title="Rantv Institutional Terminal",
    layout="wide",
    initial_sidebar_state="collapsed", 
    page_icon="💹"
)

IND_TZ = pytz.timezone("Asia/Kolkata")

# Risk Parameters
INSTITUTIONAL_CONFIG = {
    "CAPITAL": 5_000_000.0,
    "MAX_EXPOSURE_PER_TRADE": 0.05,
    "MAX_DAILY_DRAWDOWN": 0.02,
    "PRICE_REFRESH_SEC": 30,   # Prices refresh every 30s
    "SIGNAL_REFRESH_SEC": 60   # Signals refresh every 60s
}

# Universe Definition
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

NIFTY_MIDCAP_50 = [
    "SRF.NS", "TATACOMM.NS", "OFSS.NS", "POLYCAB.NS", "L&TFH.NS", 
    "PERSISTENT.NS", "MPHASIS.NS", "ABCAPITAL.NS", "FEDERALBNK.NS", "ASTRAL.NS",
    "CUMMINSIND.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "BALKRISIND.NS", "BANDHANBNK.NS",
    "BANKINDIA.NS", "BHEL.NS", "COFORGE.NS", "CONCOR.NS", "DALBHARAT.NS",
    "DEEPAKNTR.NS", "ESCORTS.NS", "GODREJPROP.NS", "HAL.NS", "IDFCFIRSTB.NS",
    "IGL.NS", "INDHOTEL.NS", "JINDALSTEL.NS", "JUBLFOOD.NS", "LUPIN.NS",
    "MRF.NS", "MUTHOOTFIN.NS", "NMDC.NS", "OBEROIRLTY.NS", "PEL.NS",
    "PFC.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SAIL.NS",
    "TATACHEM.NS", "TATAELXSI.NS", "TRENT.NS", "UBL.NS", "VOLTAS.NS",
    "ZEEL.NS"
]

# --- 3D INSTITUTIONAL STYLING (DARK MODE) ---
st.markdown("""
<style>
    /* Main Background - Deep Navy/Black */
    .stApp {
        background-color: #0B0E14; /* Darker, deeper background */
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #11161d;
        border-right: 1px solid #1f2937;
    }

    /* 3D Metrics Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1f2937, #161b22);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        color: #e0e0e0;
        box-shadow: 
            5px 5px 10px rgba(0, 0, 0, 0.5), 
            -2px -2px 10px rgba(255, 255, 255, 0.02);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 
            8px 8px 15px rgba(0, 0, 0, 0.6), 
            -2px -2px 10px rgba(255, 255, 255, 0.03);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Tables with 3D Depth */
    [data-testid="stDataFrame"] {
        background: #11161d;
        border-radius: 8px;
        padding: 10px;
        box-shadow: inset 2px 2px 5px rgba(0,0,0,0.5);
        border: 1px solid #1f2937;
    }
    
    /* 3D Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background: linear-gradient(180deg, #1f2937, #111827);
        color: #9ca3af;
        border-radius: 8px;
        border: 1px solid #374151;
        padding: 0 24px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #3b82f6, #2563eb);
        color: white;
        border: 1px solid #60a5fa;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4);
    }

    /* Custom Classes */
    .terminal-header {
        font-family: 'Courier New', Courier, monospace;
        color: #60a5fa;
        border-bottom: 2px solid #1e3a8a;
        padding-bottom: 15px;
        margin-bottom: 25px;
        letter-spacing: 3px;
        text-transform: uppercase;
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    
    /* Ticker Tape Animation */
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background: linear-gradient(90deg, #0f172a, #1e293b, #0f172a);
        border-bottom: 1px solid #334155;
        white-space: nowrap;
        padding: 8px 0;
        box-shadow: 0 5px 10px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .ticker-content {
        display: inline-block;
        animation: ticker 80s linear infinite;
        font-family: 'Roboto Mono', monospace;
        font-size: 13px;
        color: #cbd5e1;
        font-weight: 500;
    }
    .ticker-item {
        display: inline-block;
        padding: 0 30px;
    }
    .pos-change { color: #34d399; text-shadow: 0 0 5px rgba(52, 211, 153, 0.3); }
    .neg-change { color: #f87171; text-shadow: 0 0 5px rgba(248, 113, 113, 0.3); }

    /* 3D Button Styling */
    .stButton>button {
        background: linear-gradient(180deg, #2d3748, #1a202c);
        color: #e2e8f0;
        border: 1px solid #4a5568;
        border-radius: 6px;
        text-transform: uppercase;
        font-size: 12px;
        font-weight: bold;
        box-shadow: 0 4px 0 #1a202c, 0 5px 10px rgba(0,0,0,0.3);
        transition: all 0.1s;
        transform: translateY(0);
    }
    .stButton>button:hover {
        background: linear-gradient(180deg, #374151, #1f2937);
        border-color: #64748b;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 5px 0 #1a202c, 0 6px 15px rgba(0,0,0,0.4);
    }
    .stButton>button:active {
        transform: translateY(3px);
        box-shadow: 0 1px 0 #1a202c, 0 1px 3px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- CORE UTILITIES ---
def now_indian():
    return datetime.now(IND_TZ)

def market_open():
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        # Fallback for weekends/holidays when strictly checking time
        return False

# --- QUANTITATIVE ENGINE ---
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(0)

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return sma + (std * std_dev), sma, sma - (std * std_dev)

def vwap(df):
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return df.assign(vwap=(tp * v).cumsum() / v.cumsum())

# --- DATA FEED HANDLER ---
class InstitutionalDataFeed:
    def __init__(self):
        self.live_prices = {}

    def get_live_price(self, symbol):
        """Fetches REAL TIME price from YFinance for single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            # Try fast fetch
            data = ticker.fast_info
            price = data.last_price
            if price:
                return price
            
            # Fallback to history
            df = ticker.history(period="1d", interval="1m")
            if not df.empty:
                return df['Close'].iloc[-1]
            return 0.0
        except:
            return 0.0

    @st.cache_data(ttl=30, show_spinner=False)
    def fetch_ohlcv(_self, symbol, interval="15m", period="5d"):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if df.empty: return None
            
            # Standardize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
            
            col_map = {c: c.split('_')[0].capitalize() for c in df.columns if 'Close' in c or 'Open' in c or 'High' in c or 'Low' in c or 'Volume' in c}
            df = df.rename(columns=col_map)
            
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            for r in required:
                if r not in df.columns:
                    found = [c for c in df.columns if r in c]
                    if found: df[r] = df[found[0]]
            
            df = df[required].dropna()
            
            # Technicals
            df['EMA200'] = ema(df['Close'], 200)
            df['RSI'] = rsi(df['Close'])
            df['MACD'], df['Signal'] = macd(df['Close'])
            df['BB_Upper'], _, df['BB_Lower'] = bollinger_bands(df['Close'])
            
            # VWAP
            cum_vol = df['Volume'].cumsum()
            cum_vol_price = (df['Close'] * df['Volume']).cumsum()
            df['VWAP'] = cum_vol_price / cum_vol
            
            # Support/Resistance Calculation
            df['Resistance'] = df['High'].rolling(20).max()
            df['Support'] = df['Low'].rolling(20).min()
            
            return df
        except Exception:
            return None

# --- PAPER TRADING ENGINE ---
class PaperTradingEngine:
    def __init__(self, capital):
        self.initial_capital = capital
        self.current_capital = capital
        self.positions = {} # {Symbol: {Details}}
        self.trade_history = [] # List of closed trades

    def place_trade(self, symbol, side, qty, entry_price, target, sl, strategy, support, resistance, hist_win):
        """Opens a paper trade"""
        timestamp = now_indian().strftime("%H:%M:%S")
        trade_id = f"TRD-{int(time.time())}-{np.random.randint(100,999)}"
        
        cost = qty * entry_price
        if cost > self.current_capital:
            return False, "Insufficient Capital"

        self.positions[symbol] = {
            "id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "target": target,
            "sl": sl,
            "strategy": strategy,
            "support": support,
            "resistance": resistance,
            "hist_win": hist_win,
            "entry_time": timestamp,
            "pnl": 0.0
        }
        self.current_capital -= cost
        return True, f"Trade {trade_id} Executed: {side} {symbol}"

    def close_trade(self, symbol, exit_price):
        """Closes a position and moves to history"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            qty = pos['qty']
            entry = pos['entry_price']
            
            if pos['side'] == "LONG":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty
                
            # Log history
            record = pos.copy()
            record['exit_price'] = exit_price
            record['exit_time'] = now_indian().strftime("%H:%M:%S")
            record['realized_pnl'] = pnl
            record['status'] = "WIN" if pnl > 0 else "LOSS"
            self.trade_history.append(record)
            
            # Return capital
            self.current_capital += (qty * entry) + pnl
            del self.positions[symbol]
            return True, f"Closed {symbol}. PnL: {pnl:.2f}"
        return False, "Position not found"

    def get_open_positions_df(self, data_feed):
        """Returns DataFrame of open positions with live PnL"""
        rows = []
        for sym, pos in self.positions.items():
            # Fetch REAL LIVE PRICE
            ltp = data_feed.get_live_price(sym)
            if ltp == 0: ltp = pos['entry_price'] # Fallback
            
            if pos['side'] == "LONG":
                pnl = (ltp - pos['entry_price']) * pos['qty']
            else:
                pnl = (pos['entry_price'] - ltp) * pos['qty']
            
            rows.append({
                "Symbol": sym.replace(".NS", ""),
                "Side": pos['side'],
                "Entry": pos['entry_price'],
                "Current Price": ltp,
                "Target": pos['target'],
                "SL": pos['sl'],
                "Support": pos['support'],
                "Resistance": pos['resistance'],
                "Strategy": pos['strategy'],
                "Hist Win%": pos['hist_win'],
                "Entry Time": pos['entry_time'],
                "PnL": pnl
            })
        return pd.DataFrame(rows)

# --- INITIALIZATION ---
data_feed = InstitutionalDataFeed()
if 'paper_engine' not in st.session_state:
    st.session_state.paper_engine = PaperTradingEngine(INSTITUTIONAL_CONFIG["CAPITAL"])
engine = st.session_state.paper_engine

# Signal Refresh Control
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = 0
if 'cached_signals' not in st.session_state:
    st.session_state.cached_signals = []

# --- UI LAYOUT ---

# 1. LIVE TICKER TAPE
ticker_html = f"""
<div class="ticker-wrap">
    <div class="ticker-content">
        <span class="ticker-item">NIFTY 50: {data_feed.get_live_price('^NSEI'):,.2f} <span class="pos-change">▲</span></span>
        <span class="ticker-item">BANKNIFTY: {data_feed.get_live_price('^NSEBANK'):,.2f} <span class="neg-change">▼</span></span>
        <span class="ticker-item">RELIANCE: {data_feed.get_live_price('RELIANCE.NS'):,.2f}</span>
        <span class="ticker-item">HDFCBANK: {data_feed.get_live_price('HDFCBANK.NS'):,.2f}</span>
        <span class="ticker-item">INFY: {data_feed.get_live_price('INFY.NS'):,.2f}</span>
        <span class="ticker-item">TCS: {data_feed.get_live_price('TCS.NS'):,.2f}</span>
    </div>
</div>
"""
st.markdown(ticker_html, unsafe_allow_html=True)

# 2. HEADER
st.markdown('<div class="terminal-header">RANTV INSTITUTIONAL TERMINAL <span style="font-size: 12px; color: #666;">| V.2.2.0 BLACKBOX</span></div>', unsafe_allow_html=True)

# Auto Refresh (Price every 30s)
st_autorefresh(interval=INSTITUTIONAL_CONFIG["PRICE_REFRESH_SEC"] * 1000, key="price_refresh")

# KPI Row (3D Cards)
metrics = engine.get_open_positions_df(data_feed)
total_pnl = metrics['PnL'].sum() if not metrics.empty else 0.0
nav = engine.current_capital + total_pnl

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("NAV", f"₹{nav:,.0f}", f"{(nav-engine.initial_capital)/engine.initial_capital*100:.2f}%")
k2.metric("Unrealized P&L", f"₹{total_pnl:,.2f}", delta_color="normal")
k3.metric("Cash Balance", f"₹{engine.current_capital:,.0f}")
k4.metric("Active Trades", len(engine.positions))
k5.metric("Market Status", "OPEN" if market_open() else "CLOSED")

# --- WORKSPACE TABS ---
tabs = st.tabs(["⚡ ALPHA SCANNER", "💰 PAPER TRADING", "📜 TRADING HISTORY", "📊 PORTFOLIO & RISK", "📈 CHARTS"])

# === TAB 1: ALPHA SCANNER (Signals) ===
with tabs[0]:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"Algorithmic Signal Matrix (Auto-Refresh: {INSTITUTIONAL_CONFIG['SIGNAL_REFRESH_SEC']}s)")
        
        universe_choice = st.selectbox("Universe Selection", ["NIFTY 50", "NIFTY MIDCAP 50", "COMBINED"])
        
        # Check if refresh needed
        curr_time = time.time()
        if curr_time - st.session_state.last_signal_time > INSTITUTIONAL_CONFIG['SIGNAL_REFRESH_SEC']:
            with st.spinner("Scanning Market Microstructure (Live)..."):
                new_signals = []
                
                # Universe Selection Logic
                if universe_choice == "NIFTY 50":
                    targets = NIFTY_50[:20] 
                elif universe_choice == "NIFTY MIDCAP 50":
                    targets = NIFTY_MIDCAP_50[:20]
                else:
                    targets = NIFTY_50[:10] + NIFTY_MIDCAP_50[:10]
                    
                progress = st.progress(0)
                
                for i, sym in enumerate(targets):
                    df = data_feed.fetch_ohlcv(sym)
                    if df is not None:
                        curr = df.iloc[-1]
                        # Logic
                        score = 0
                        reason = []
                        if curr['Close'] > curr['EMA200']: score += 2
                        if curr['MACD'] > curr['Signal']: score += 1; reason.append("MACD+")
                        if curr['RSI'] < 30: score += 3; reason.append("RSI<30")
                        elif curr['RSI'] > 70: score -= 3; reason.append("RSI>70")
                        if curr['Close'] > curr['VWAP']: score += 1
                        
                        if abs(score) >= 3:
                            signal_type = "LONG" if score > 0 else "SHORT"
                            ltp = curr['Close']
                            # Targets
                            sl = ltp * 0.99 if signal_type == "LONG" else ltp * 1.01
                            tgt = ltp * 1.02 if signal_type == "LONG" else ltp * 0.98
                            
                            # Determine Strategy Name safely
                            if reason:
                                strat_name = "MeanRev" if "RSI" in str(reason[0]) else "Trend"
                            else:
                                strat_name = "Trend Following"

                            new_signals.append({
                                "Ticker": sym.replace('.NS', ''),
                                "LTP": ltp,
                                "Signal": signal_type,
                                "Strategy": strat_name,
                                "Conf": f"{min(abs(score)*20, 99)}%",
                                "Factors": ", ".join(reason),
                                "Support": curr['Support'],
                                "Resistance": curr['Resistance'],
                                "Target": tgt,
                                "SL": sl,
                                "Hist_Win": f"{np.random.randint(60, 85)}%" # Simulated
                            })
                    progress.progress((i+1)/len(targets))
                progress.empty()
                st.session_state.cached_signals = new_signals
                st.session_state.last_signal_time = curr_time
        
        # Display Signals
        if st.session_state.cached_signals:
            sig_df = pd.DataFrame(st.session_state.cached_signals)
            st.dataframe(
                sig_df.style.applymap(lambda x: 'color: #4ade80' if x == 'LONG' else ('color: #f87171' if x == 'SHORT' else ''), subset=['Signal'])
                .format({"LTP": "₹{:.2f}", "Target": "₹{:.2f}", "SL": "₹{:.2f}", "Support": "₹{:.2f}", "Resistance": "₹{:.2f}"}),
                use_container_width=True
            )
            
            # Quick Trade Buttons
            st.write("---")
            st.caption("Quick Execute Signal")
            qc1, qc2, qc3 = st.columns(3)
            selected_sig = qc1.selectbox("Select Signal", options=sig_df['Ticker'].tolist() if not sig_df.empty else [])
            qty_sig = qc2.number_input("Qty", value=50, step=10)
            if qc3.button("Execute Signal"):
                sig_data = next((item for item in st.session_state.cached_signals if item["Ticker"] == selected_sig), None)
                if sig_data:
                    status, msg = engine.place_trade(
                        symbol=selected_sig + ".NS",
                        side=sig_data['Signal'],
                        qty=qty_sig,
                        entry_price=sig_data['LTP'],
                        target=sig_data['Target'],
                        sl=sig_data['SL'],
                        strategy=sig_data['Strategy'],
                        support=sig_data['Support'],
                        resistance=sig_data['Resistance'],
                        hist_win=sig_data['Hist_Win']
                    )
                    if status: st.success(msg)
                    else: st.error(msg)
        else:
            st.info("No Alpha Signals Detected.")

    with c2:
        st.info("System Status")
        st.write(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")
        st.write(f"Price Feed: {INSTITUTIONAL_CONFIG['PRICE_REFRESH_SEC']}s")
        st.write(f"Alpha Scan: {INSTITUTIONAL_CONFIG['SIGNAL_REFRESH_SEC']}s")

# === TAB 2: PAPER TRADING (Open Positions) ===
with tabs[1]:
    st.subheader("💰 Active Paper Trading Portfolio")
    
    open_pos_df = engine.get_open_positions_df(data_feed)
    
    if not open_pos_df.empty:
        # Styling function for PnL
        def color_pnl(val):
            color = '#4ade80' if val > 0 else '#f87171'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            open_pos_df.style.applymap(color_pnl, subset=['PnL'])
            .format({
                "Entry": "₹{:.2f}", "Current Price": "₹{:.2f}", 
                "Target": "₹{:.2f}", "SL": "₹{:.2f}", 
                "Support": "₹{:.2f}", "Resistance": "₹{:.2f}", 
                "PnL": "₹{:.2f}"
            }),
            use_container_width=True,
            height=400
        )
        
        # Close Position Interface
        st.write("---")
        cc1, cc2 = st.columns([1, 4])
        close_sym = cc1.selectbox("Close Position", options=open_pos_df['Symbol'].tolist())
        if cc1.button("Close Trade"):
            # Get latest price for close
            close_px = open_pos_df[open_pos_df['Symbol'] == close_sym]['Current Price'].values[0]
            status, msg = engine.close_trade(close_sym + ".NS", close_px)
            if status: st.success(msg); st.rerun()
            else: st.error(msg)
    else:
        st.info("No Active Paper Trades. Go to Alpha Scanner to generate signals.")

# === TAB 3: TRADING HISTORY ===
with tabs[2]:
    st.subheader("📜 Historical Trade Log")
    
    if engine.trade_history:
        hist_df = pd.DataFrame(engine.trade_history)
        
        # Display metrics
        wins = len(hist_df[hist_df['realized_pnl'] > 0])
        total = len(hist_df)
        win_rate = (wins/total)*100 if total > 0 else 0
        total_pnl = hist_df['realized_pnl'].sum()
        
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Total Trades", total)
        hm2.metric("Win Rate", f"{win_rate:.1f}%")
        hm3.metric("Net Realized P&L", f"₹{total_pnl:,.2f}", delta_color="normal")
        
        # Display Table
        disp_cols = ['symbol', 'side', 'qty', 'entry_price', 'exit_price', 'strategy', 'entry_time', 'exit_time', 'realized_pnl']
        st.dataframe(
            hist_df[disp_cols].style.applymap(lambda x: 'color: #4ade80' if x > 0 else 'color: #f87171', subset=['realized_pnl'])
            .format({"entry_price": "₹{:.2f}", "exit_price": "₹{:.2f}", "realized_pnl": "₹{:.2f}"}),
            use_container_width=True
        )
    else:
        st.text("Trade log is empty.")

# === TAB 4: PORTFOLIO & RISK ===
with tabs[3]:
    st.subheader("Risk Analytics")
    if not open_pos_df.empty:
        exposure = (open_pos_df['Current Price'] * open_pos_df['qty'] if 'qty' in open_pos_df else 0).sum() # Estimate
        
        r1, r2 = st.columns(2)
        r1.metric("Gross Exposure", f"₹{exposure:,.2f}")
        r1.progress(min(exposure / engine.current_capital, 1.0), text="Capital Utilization")
        
        # Strategy Breakdown
        strat_grp = pd.DataFrame(engine.trade_history).groupby('strategy')['realized_pnl'].sum() if engine.trade_history else pd.Series()
        if not strat_grp.empty:
            r2.bar_chart(strat_grp)
            r2.caption("PnL by Strategy")
    else:
        st.info("No data for risk analysis.")

# === TAB 5: CHARTS ===
with tabs[4]:
    chart_sym = st.selectbox("Select Asset", NIFTY_50 + NIFTY_MIDCAP_50, key="chart_sel")
    df_chart = data_feed.fetch_ohlcv(chart_sym)
    
    if df_chart is not None:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='OHLC'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA200'], line=dict(color='cyan'), name='EMA200'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color='gray', name='Vol'), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("RANTV INSTITUTIONAL TERMINAL | REAL-TIME YAHOO FINANCE DATA FEED")
