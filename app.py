
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import time

st.set_page_config(page_title="Indian Trading Scanner", layout="wide")

st.title("🇮🇳 Indian Stock Trading Scanner")
st.caption("NSE + BSE | Intraday + Swing | Technical screening")

# ---------------- SETTINGS ----------------

@st.cache_data(ttl=3600)
def get_stock_list(exchange):
    urls = {
        "NSE": "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "BSE": "https://www.bseindia.com/downloads1/List_of_Scrips.csv"
    }

    response = requests.get(
        urls[exchange],
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20
    )
    response.raise_for_status()

    data = pd.read_csv(StringIO(response.text))
    data.columns = [str(c).strip().upper() for c in data.columns]

    if exchange == "NSE":
        if "SYMBOL" not in data.columns:
            raise ValueError("NSE symbol column not found.")
        symbols = data["SYMBOL"].dropna().astype(str).tolist()
        symbols = [s.strip() + ".NS" for s in symbols]
    else:
        code_col = next(
            (c for c in data.columns if "SECURITY ID" in c or c == "SCRIP ID"),
            None
        )
        if code_col is None:
            raise ValueError("BSE security ID column not found.")
        symbols = data[code_col].dropna().astype(str).tolist()
        symbols = [s.strip() + ".BO" for s in symbols]

    return sorted(set(symbols))


def analyse_stock(symbol, mode, interval):
    try:
        if mode == "Intraday":
            period = "5d"
            data = yf.Ticker(symbol).history(
                period=period, interval=interval, auto_adjust=True
            )
            minimum = 30
        else:
            data = yf.Ticker(symbol).history(
                period="1y", interval="1d", auto_adjust=True
            )
            minimum = 60

        if data.empty or len(data) < minimum:
            return None

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        change = close.diff()
        gain = change.clip(lower=0).rolling(14).mean()
        loss = (-change.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        macd = (
            close.ewm(span=12, adjust=False).mean()
            - close.ewm(span=26, adjust=False).mean()
        )
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        latest_price = float(close.iloc[-1])
        latest_rsi = float(rsi.iloc[-1])
        score = 0

        score += 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
        score += 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1
        score += 1 if latest_rsi > 50 else -1

        if mode == "Intraday":
            typical = (high + low + close) / 3
            vwap = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
            score += 1 if latest_price > vwap.iloc[-1] else -1

        if score >= 3:
            signal = "BUY"
        elif score <= -3:
            signal = "SELL"
        else:
            signal = "WAIT"

        previous_high = float(high.iloc[-21:-1].max())
        breakout = latest_price > previous_high

        return {
            "Stock": symbol,
            "Price": round(latest_price, 2),
            "Signal": signal,
            "Score": score,
            "RSI": round(latest_rsi, 2),
            "EMA Trend": "Bullish" if ema20.iloc[-1] > ema50.iloc[-1] else "Bearish",
            "Breakout": "Yes" if breakout else "No"
        }

    except Exception:
        return None


# ---------------- USER CONTROLS ----------------

c1, c2 = st.columns(2)

with c1:
    exchange = st.selectbox("Exchange", ["NSE", "BSE"])

with c2:
    mode = st.selectbox("Trading style", ["Intraday", "Swing"])

if mode == "Intraday":
    interval = st.selectbox("Chart interval", ["5m", "15m"])
else:
    interval = "1d"
    st.info("Swing scanner uses daily price data.")

c1, c2 = st.columns(2)

with c1:
    min_price = st.number_input("Minimum stock price (₹)", 0.0, 100000.0, 200.0)

with c2:
    max_price = st.number_input("Maximum stock price (₹)", 0.0, 100000.0, 400.0)

batch_size = st.selectbox(
    "Stocks to scan in this batch",
    [25, 50, 100],
    index=1
)

start_at = st.number_input(
    "Start position in stock list",
    min_value=0,
    value=0,
    step=25
)

st.caption(
    "Change the start position to scan the next batch. "
    "For example, 0, 50, 100, 150."
)

# ---------------- SCANNER ----------------

if st.button("🚀 Start Stock Scan", type="primary"):

    if max_price <= min_price:
        st.error("Maximum price must be greater than minimum price.")
        st.stop()

    try:
        with st.spinner("Loading exchange stock list..."):
            stocks = get_stock_list(exchange)

        st.write(f"Available symbols in retrieved list: {len(stocks)}")

        selected = stocks[int(start_at):int(start_at) + int(batch_size)]

        if not selected:
            st.warning("No stocks in this batch. Try an earlier start position.")
            st.stop()

        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, symbol in enumerate(selected):
            status.write(f"Scanning {symbol} ({i + 1}/{len(selected)})")

            result = analyse_stock(symbol, mode, interval)

            if result is not None:
                if min_price <= result["Price"] <= max_price:
                    results.append(result)

            progress.progress((i + 1) / len(selected))
            time.sleep(0.15)

        status.success("Batch scan completed.")

        if results:
            table = pd.DataFrame(results)
            table = table.sort_values(
                by=["Score", "RSI"],
                ascending=[False, False]
            )

            buys = int((table["Signal"] == "BUY").sum())
            sells = int((table["Signal"] == "SELL").sum())
            waits = int((table["Signal"] == "WAIT").sum())

            a, b, c = st.columns(3)
            a.metric("BUY signals", buys)
            b.metric("SELL signals", sells)
            c.metric("WAIT signals", waits)

            st.subheader("Stock scan results")
            st.dataframe(table, use_container_width=True, hide_index=True)

            st.download_button(
                "Download scan results (CSV)",
                table.to_csv(index=False),
                file_name="stock_scan_results.csv",
                mime="text/csv"
            )

        else:
            st.warning(
                "No stocks with sufficient data were found in this batch "
                "and selected price range. Try another batch or price range."
            )

        st.info(
            "Only successfully retrieved stocks are displayed. "
            "This batch is not a complete scan of the exchange."
        )

    except Exception as e:
        st.error(f"Could not load the exchange list or scan stocks: {e}")
        st.caption(
            "The exchange website may block automated downloads. "
            "Try again later or use a supported stock-list CSV."
        )

st.divider()
st.caption(
    "Educational screening tool only. Data can be delayed, incomplete "
    "or unavailable. Signals are indicator-based, not AI predictions "
    "or guaranteed trading advice. Verify every trade with your broker."
)

