import os
import json
import smtplib
import datetime
import yfinance as yf
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from anthropic import Anthropic

# Always resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
FINNHUB_API_KEY   = os.environ["FINNHUB_API_KEY"]
GMAIL_PASSWORD    = os.environ["GMAIL_APP_PASSWORD"]
EMAIL_ADDRESS     = os.environ["EMAIL_ADDRESS"]

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Load tickers (portfolio vs watchlist) ─────────────────────────────────────
def load_tickers():
    portfolio, watchlist = [], []
    current = None
    with open(os.path.join(SCRIPT_DIR, "tickers.txt")) as f:
        for raw in f:
            line = raw.split("#")[0].strip()   # strip inline comments
            if not line:
                continue
            if line == "[PORTFOLIO]":
                current = portfolio
            elif line == "[WATCHLIST]":
                current = watchlist
            elif current is not None:
                current.append(line.upper())
    return portfolio, watchlist

# ── MACD calculation ──────────────────────────────────────────────────────────
def calc_macd(close_series):
    ema12  = close_series.ewm(span=12, adjust=False).mean()
    ema26  = close_series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return round(macd.iloc[-1], 3), round(signal.iloc[-1], 3), round(hist.iloc[-1], 3)

# ── Fetch price, technical & fundamental data ─────────────────────────────────
def fetch_price_data(ticker):
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        info = tk.info

        if hist.empty:
            return None

        close        = hist["Close"]
        last_close   = round(float(close.iloc[-1]), 2)
        prev_close   = round(float(close.iloc[-2]), 2)
        pct_change   = round((last_close - prev_close) / prev_close * 100, 2)
        week52_high  = round(float(close.max()), 2)
        week52_low   = round(float(close.min()), 2)
        ma50         = round(float(close.tail(50).mean()), 2)
        ma200        = round(float(close.tail(200).mean()), 2) if len(close) >= 200 else None
        vs_200ma_pct = round((last_close - ma200) / ma200 * 100, 2) if ma200 else None
        volume       = int(hist["Volume"].iloc[-1])
        avg_volume   = int(hist["Volume"].tail(20).mean())

        macd_val, macd_sig, macd_hist = calc_macd(close)

        # Forward P/E
        fwd_eps = info.get("forwardEps")
        fwd_pe  = round(last_close / fwd_eps, 1) if fwd_eps and fwd_eps > 0 else info.get("forwardPE")

        # Rolling historical P/E proxy (last 52 weeks of fwd P/E approximation)
        # We use trailing P/E from info as reference point
        ttm_pe     = info.get("trailingPE")
        pb         = info.get("priceToBook")
        rev_growth = info.get("revenueGrowth")   # YoY
        fwd_rev_growth = info.get("earningsGrowth")  # fwd earnings growth as proxy

        return {
            "ticker":         ticker,
            "last_close":     last_close,
            "pct_change":     pct_change,
            "week52_high":    week52_high,
            "week52_low":     week52_low,
            "pct_from_52h":   round((last_close - week52_high) / week52_high * 100, 2),
            "ma50":           ma50,
            "ma200":          ma200,
            "vs_200ma_pct":   vs_200ma_pct,
            "macd":           macd_val,
            "macd_signal":    macd_sig,
            "macd_hist":      macd_hist,
            "volume":         volume,
            "avg_volume":     avg_volume,
            "vol_vs_avg":     round(volume / avg_volume, 2) if avg_volume else None,
            "fwd_pe":         fwd_pe,
            "ttm_pe":         ttm_pe,
            "pb":             round(pb, 1) if pb else None,
            "rev_growth_yoy": round(rev_growth * 100, 1) if rev_growth else None,
            "fwd_rev_growth": round(fwd_rev_growth * 100, 1) if fwd_rev_growth else None,
            "mkt_cap_b":      round(info.get("marketCap", 0) / 1e9, 1),
            "sector":         info.get("sector"),
        }
    except Exception as e:
        print(f"  yfinance error for {ticker}: {e}")
        return None

# ── Fetch news via Finnhub ────────────────────────────────────────────────────
def fetch_news(ticker):
    try:
        today     = datetime.date.today()
        from_date = (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
        to_date   = today.strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={from_date}&to={to_date}"
            f"&token={FINNHUB_API_KEY}"
        )
        resp     = requests.get(url, timeout=10)
        articles = resp.json()[:5]
        return [{"headline": a["headline"], "source": a["source"]} for a in articles]
    except Exception as e:
        print(f"  Finnhub news error for {ticker}: {e}")
        return []

# ── Fetch analyst recommendations via Finnhub ─────────────────────────────────
def fetch_estimates(ticker):
    try:
        url  = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data:
            latest = data[0]
            total  = sum([
                latest.get("strongBuy", 0), latest.get("buy", 0),
                latest.get("hold", 0), latest.get("sell", 0), latest.get("strongSell", 0)
            ])
            return {
                "strong_buy":  latest.get("strongBuy"),
                "buy":         latest.get("buy"),
                "hold":        latest.get("hold"),
                "sell":        latest.get("sell"),
                "strong_sell": latest.get("strongSell"),
                "total":       total,
                "period":      latest.get("period"),
                "bull_pct":    round((latest.get("strongBuy", 0) + latest.get("buy", 0)) / total * 100, 0) if total else None,
            }
        return None
    except Exception as e:
        print(f"  Finnhub estimates error for {ticker}: {e}")
        return None

# ── Build full data payload ───────────────────────────────────────────────────
def build_payload(tickers):
    payload = []
    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        price = fetch_price_data(ticker)
        if price:
            price["news"]      = fetch_news(ticker)
            price["estimates"] = fetch_estimates(ticker)
            payload.append(price)
    return payload

# ── Call Claude API with web search for macro context + external picks ─────────
def generate_brief(portfolio_data, watchlist_data):
    today = datetime.date.today().strftime("%A, %B %d, %Y")

    prompt = f"""
You are a sharp equity analyst writing a personal daily morning brief. Today is {today}.

You have two sets of holdings:

PORTFOLIO (currently owned — monitor for hold/sell signals):
{json.dumps(portfolio_data, indent=2)}

WATCHLIST (not yet owned — monitor for buy entry signals):
{json.dumps(watchlist_data, indent=2)}

First, use your web search tool to find:
1. Today's key macro context: S&P 500 and Nasdaq moves, 10-year yield, VIX, oil prices, dollar index, any Fed commentary
2. The current S&P 500 median forward P/E (for valuation context)
3. Any major sector rotation or thematic signals today (e.g. energy, semis, AI names, defensives)
4. 3-4 tickers NOT on the lists above that look interesting right now — could be undervalued, breaking out technically, benefiting from macro regime, or showing strong sentiment shift. Consider current macro environment signals (e.g. oil supply, rate environment, AI capex cycle, etc.)

Then write the daily brief in clean HTML with these exact sections:

---

<h2>📊 Market Pulse</h2>
6-8 sentences. Cover: equity index performance, bond market, VIX, commodities (especially oil), dollar, any Fed/macro narrative driving the day. End with one sentence on what it means for the portfolio broadly.

---

<h2>📋 Portfolio Snapshot</h2>
HTML table with these columns for PORTFOLIO names only:
Ticker | Price | 1D % | vs 52W High | vs 200MA | MACD | Fwd P/E | P/E vs SPX Median | Rev Growth | Vol/Avg | Signal

For Signal use: 🟢 Buy More | 🔴 Trim/Sell | 🟡 Watch | ⚪ Hold
Base signal equally on: (1) technicals — price vs MAs, MACD direction, volume, (2) valuation — Fwd P/E vs SPX median and vs own history, (3) news/catalysts

Then a second table for WATCHLIST names with same columns but Signal options: 🟢 Buy Now | 🟡 Getting Interesting | ⚪ Not Yet

---

<h2>📝 Name-by-Name Commentary</h2>
For each ticker (portfolio first, then watchlist), write 3-5 sentences:
- What happened today (price action, volume, news)
- What valuation says vs SPX median P/E and its own history
- MACD signal interpretation (bullish/bearish crossover, divergence, momentum)
- The signal and why
- One specific thing to watch tomorrow
Skip to one sentence only if nothing notable.

---

<h2>🎯 Top Opportunities</h2>
2-3 names from the WATCHLIST that look most actionable as buys right now. One paragraph each with full rationale covering all three signal dimensions.

---

<h2>💡 Claude's Picks</h2>
3-4 tickers NOT on either list that you identified via web search. For each:
- Ticker + company name
- Why it's interesting right now (technicals, valuation, macro tailwind, sentiment)
- Key risk to the thesis
- Suggested entry approach (e.g. "wait for pullback to 200MA" or "break above $X confirms momentum")

---

<h2>⚠️ Risk Flags</h2>
2-3 names (from either list) showing the most concerning signals — technically breaking down, valuation stretched, negative catalysts, or deteriorating analyst sentiment. Be direct.

---

<h2>👀 Tomorrow's Watch List</h2>
2-3 names with specific price levels or catalysts to monitor tomorrow.

---

Formatting rules:
- Clean HTML, white background, readable sans-serif font
- Table borders: 1px solid #e0e0e0, alternating row shading
- Section headers with subtle color (use #1a1a2e for dark navy)
- Keep commentary tight — no fluff, no disclaimers
- Dollar signs and % symbols throughout
- Do not add any preamble or closing remarks outside the HTML
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract text from response (may include tool use blocks)
    html_output = ""
    for block in response.content:
        if hasattr(block, "text"):
            html_output += block.text

    return html_output

# ── Send email ────────────────────────────────────────────────────────────────
def send_email(html_body):
    today   = datetime.date.today().strftime("%B %d, %Y")
    subject = f"📈 Daily Market Brief — {today}"

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = EMAIL_ADDRESS
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, GMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())

    print("  Email sent.")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading tickers...")
    portfolio_tickers, watchlist_tickers = load_tickers()
    print(f"  Portfolio ({len(portfolio_tickers)}): {', '.join(portfolio_tickers)}")
    print(f"  Watchlist ({len(watchlist_tickers)}): {', '.join(watchlist_tickers)}")

    print("Fetching portfolio data...")
    portfolio_data = build_payload(portfolio_tickers)

    print("Fetching watchlist data...")
    watchlist_data = build_payload(watchlist_tickers)

    print("Generating brief with Claude...")
    brief = generate_brief(portfolio_data, watchlist_data)

    print("Sending email...")
    send_email(brief)

    print("Done.")
