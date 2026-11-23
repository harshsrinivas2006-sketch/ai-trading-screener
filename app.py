
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="AI Trading Screener", layout="wide")

st.title("📈 AI Indian Stock Trading Screener")
st.caption("Technical analysis dashboard for NSE stocks")

symbol = st.text_input("Enter NSE stock symbol", "RELIANCE").upper().strip()
period = st.selectbox("Analysis period", ["1d", "5d", "1mo", "3mo"], index=1)
interval = st.selectbox("Chart interval", ["5m", "15m", "30m", "1h", "1d"], index=1)

if st.button("Analyse Stock"):
    with st.spinner("Fetching market data..."):
        try:
            ticker = yf.Ticker(symbol + ".NS")
            df = ticker.history(period=period, interval=interval)

            if df.empty or len(df) < 30:
                st.error("Not enough data. Try another stock or interval.")
                st.stop()

            close = df["Close"]
            volume = df["Volume"]

            df["EMA20"] = close.ewm(span=20, adjust=False).mean()
            df["EMA50"] = close.ewm(span=50, adjust=False).mean()

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df["RSI"] = 100 - (100 / (1 + rs))

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

            typical = (df["High"] + df["Low"] + close) / 3
            df["VWAP"] = (
                (typical * volume).groupby(df.index.date).cumsum()
                / volume.groupby(df.index.date).cumsum().replace(0, np.nan)
            )

            latest = df.iloc[-1]
            score = 0
            score += 1 if latest["EMA20"] > latest["EMA50"] else -1
            score += 1 if latest["MACD"] > latest["Signal"] else -1
            score += 1 if latest["RSI"] > 50 else -1
            score += 1 if latest["Close"] > latest["VWAP"] else -1

            if score >= 3:
                signal = "BUY"
            elif score <= -3:
                signal = "SELL"
            else:
                signal = "WAIT"

            c1, c2, c3 = st.columns(3)
            c1.metric("Latest Price", f"₹{latest['Close']:.2f}")
            c2.metric("Technical Signal", signal)
            c3.metric("Signal Score", f"{score}/4")

            st.caption("Signal is based on four simple indicators, not an AI prediction or guaranteed outcome.")

            st.subheader("Price chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price"
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20"))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50"))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Indicator readings")
            st.write({
                "RSI": round(float(latest["RSI"]), 2),
                "MACD": round(float(latest["MACD"]), 2),
                "MACD Signal": round(float(latest["Signal"]), 2),
                "VWAP": round(float(latest["VWAP"]), 2)
            })

            st.subheader("Recent market data")
            st.dataframe(df.tail(10), use_container_width=True)

        except Exception as e:
            st.error(f"Could not analyse this stock: {e}")

st.warning("Educational tool only. Market data may be delayed or incomplete. Verify prices with your broker before trading.")
