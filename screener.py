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

Then write the daily brief as a complete, self-contained HTML document. Use this EXACT structure and styling:

<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; background: #ffffff; color: #1a1a1a; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 8px; }}
  h2 {{ color: #1a1a2e; margin-top: 36px; margin-bottom: 12px; font-size: 1.2em; border-left: 4px solid #1a1a2e; padding-left: 10px; }}
  .subtitle {{ color: #666; font-size: 0.9em; margin-top: -8px; margin-bottom: 20px; }}
  .pulse-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 16px; }}
  .pulse-item {{ background: #f5f7fa; border-radius: 6px; padding: 10px 14px; }}
  .pulse-label {{ font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }}
  .pulse-value {{ font-size: 1.05em; font-weight: 600; color: #1a1a2e; }}
  .pulse-change.up {{ color: #16a34a; }} .pulse-change.down {{ color: #dc2626; }}
  ul.pulse-bullets {{ margin: 12px 0; padding-left: 20px; line-height: 1.8; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; margin-top: 8px; }}
  th {{ background: #1a1a2e; color: white; padding: 8px 10px; text-align: left; font-weight: 600; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #e8e8e8; }}
  tr:nth-child(even) {{ background: #f9fafb; }}
  .up {{ color: #16a34a; font-weight: 600; }} .down {{ color: #dc2626; font-weight: 600; }}
  .signal-buy {{ background: #dcfce7; color: #15803d; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; white-space: nowrap; }}
  .signal-sell {{ background: #fee2e2; color: #dc2626; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; white-space: nowrap; }}
  .signal-watch {{ background: #fef9c3; color: #854d0e; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; white-space: nowrap; }}
  .signal-hold {{ background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; white-space: nowrap; }}
  .commentary-block {{ margin-bottom: 18px; padding: 14px 16px; border: 1px solid #e8e8e8; border-radius: 8px; }}
  .commentary-block h3 {{ margin: 0 0 6px 0; font-size: 1em; color: #1a1a2e; }}
  .pick-block {{ margin-bottom: 16px; padding: 14px 16px; background: #f0f9ff; border-left: 4px solid #0ea5e9; border-radius: 0 8px 8px 0; }}
  .pick-block h3 {{ margin: 0 0 6px 0; color: #0369a1; }}
  .risk-block {{ margin-bottom: 16px; padding: 14px 16px; background: #fff7ed; border-left: 4px solid #f97316; border-radius: 0 8px 8px 0; }}
  .risk-block h3 {{ margin: 0 0 6px 0; color: #c2410c; }}
  .section-label {{ font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
  p {{ line-height: 1.65; margin: 6px 0; }}
  .tag {{ display: inline-block; font-size: 0.75em; padding: 1px 7px; border-radius: 10px; margin-left: 6px; vertical-align: middle; }}
  .tag-portfolio {{ background: #ede9fe; color: #6d28d9; }}
  .tag-watchlist {{ background: #dbeafe; color: #1d4ed8; }}
</style>
</head>
<body>

SECTION 1 — HEADER:
<h1>📈 Daily Morning Brief</h1>
<p class="subtitle">[Day, Date] &nbsp;|&nbsp; Personal Portfolio Intelligence &nbsp;|&nbsp; SPX Fwd P/E: [value]x &nbsp;|&nbsp; 10Y: [value]%</p>

SECTION 2 — MARKET PULSE:
<h2>📊 Market Pulse</h2>
First show a pulse-grid with 6 tiles: S&P 500, Nasdaq, Dow, 10Y Yield, VIX, WTI Crude — each with label, value, and colored change.
Then write 6-8 bullet points (ul class="pulse-bullets") covering: index performance narrative, bond market, VIX reading, oil/commodities, dollar, Fed/macro theme of the day, sector rotation signals, and what it means for the portfolio broadly.

SECTION 3 — PORTFOLIO SNAPSHOT:
<h2>📋 Portfolio Snapshot</h2>
<p class="section-label">Portfolio Holdings — SPX Fwd P/E Reference: [X]x</p>
Full HTML table. INCLUDE EVERY SINGLE PORTFOLIO TICKER — do not skip any. Columns:
Ticker | Price | 1D % | vs 52W High | vs 200MA | MACD Hist | Fwd P/E | P/E vs SPX | Rev Growth | Vol/Avg | Signal
- Color 1D%, vs 52W High, vs 200MA green/red using class="up" or class="down"
- Signal uses styled spans: <span class="signal-buy">🟢 Buy More</span> or signal-sell, signal-watch, signal-hold
- Base signal equally on technicals + valuation + news

Then same table for WATCHLIST tickers (if any) with header "Watchlist — Entry Signals"
- Signal options: <span class="signal-buy">🟢 Buy Now</span>, <span class="signal-watch">🟡 Getting Interesting</span>, <span class="signal-hold">⚪ Not Yet</span>

SECTION 4 — NAME-BY-NAME COMMENTARY:
<h2>📝 Name-by-Name Commentary</h2>
For EVERY ticker (portfolio first, then watchlist), use a commentary-block div:
<div class="commentary-block">
  <h3>TICKER — Company Name <span class="tag tag-portfolio">Portfolio</span></h3>
  <p>3-5 sentences: price action today + news | valuation vs SPX median | MACD interpretation | signal rationale | one thing to watch</p>
</div>
If nothing notable, one sentence is fine.

SECTION 5 — TOP OPPORTUNITIES (only if watchlist has names):
<h2>🎯 Top Opportunities</h2>
2-3 most actionable watchlist buys. Use pick-block divs. One paragraph each covering all three signal dimensions.

SECTION 6 — CLAUDE'S PICKS:
<h2>💡 Claude's Picks</h2>
<p class="section-label">3-4 names not on either list — identified from macro signals, technicals, and sentiment</p>
For each, use a pick-block div with: ticker + company name as h3, then bullets for: Why interesting now | Key risk | Entry approach

SECTION 7 — RISK FLAGS:
<h2>⚠️ Risk Flags</h2>
2-3 most concerning names. Use risk-block divs. Be direct — technically breaking down, valuation stretched, negative catalysts.

SECTION 8 — TOMORROW'S WATCH LIST:
<h2>👀 Tomorrow's Watch List</h2>
2-3 names with specific price levels or catalysts. Use commentary-block divs.

</body></html>

CRITICAL RULES:
- Output ONLY the complete HTML document — no markdown, no backticks, no preamble
- Include EVERY ticker in both the table and commentary — do not truncate
- All sections must be present and clearly labeled
- Keep commentary tight and direct — no disclaimers or fluff
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
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
