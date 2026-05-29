import os
import re
import json
import time
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

    h_now  = hist.iloc[-1]
    h_prev = hist.iloc[-2]

    # Detect crossover (sign flip in last 2 bars)
    if h_prev < 0 and h_now >= 0:
        momentum_label = "⟳ Crossing Up"
    elif h_prev > 0 and h_now <= 0:
        momentum_label = "⟳ Crossing Down"
    elif h_now > 0:
        momentum_label = "▲ Building" if h_now > h_prev else "▲ Weakening"
    else:
        momentum_label = "▼ Building" if h_now < h_prev else "▼ Weakening"

    return round(macd.iloc[-1], 3), round(signal.iloc[-1], 3), round(float(h_now), 3), momentum_label

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

        macd_val, macd_sig, macd_hist, macd_momentum = calc_macd(close)

        fwd_eps = info.get("forwardEps")
        raw_fwd_pe = info.get("forwardPE")
        if fwd_eps and isinstance(fwd_eps, (int, float)) and fwd_eps > 0:
            fwd_pe = round(last_close / fwd_eps, 1)
        elif raw_fwd_pe and isinstance(raw_fwd_pe, (int, float)):
            fwd_pe = round(raw_fwd_pe, 1)
        else:
            fwd_pe = None

        ttm_pe         = info.get("trailingPE")
        pb             = info.get("priceToBook")
        rev_growth     = info.get("revenueGrowth")
        fwd_rev_growth = info.get("earningsGrowth")

        def safe_pct(val):
            return round(val * 100, 1) if val and isinstance(val, (int, float)) else None

        def safe_round(val, n=1):
            return round(val, n) if val and isinstance(val, (int, float)) else None

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
            "macd_momentum":  macd_momentum,
            "volume":         volume,
            "avg_volume":     avg_volume,
            "vol_vs_avg":     round(volume / avg_volume, 2) if avg_volume else None,
            "fwd_pe":         fwd_pe,
            "ttm_pe":         ttm_pe,
            "pb":             safe_round(pb),
            "rev_growth_yoy": safe_pct(rev_growth),
            "fwd_rev_growth": safe_pct(fwd_rev_growth),
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
        return [{"headline": a.get("headline", ""), "source": a.get("source", "")} for a in articles if isinstance(a, dict)]
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

# ── Shared CSS ────────────────────────────────────────────────────────────────
CSS = """
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; background: #ffffff; color: #1a1a1a; }
  h1 { color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 8px; }
  h2 { color: #1a1a2e; margin-top: 36px; margin-bottom: 12px; font-size: 1.2em; border-left: 4px solid #1a1a2e; padding-left: 10px; }
  .subtitle { color: #666; font-size: 0.9em; margin-top: -8px; margin-bottom: 20px; }
  .pulse-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 16px; }
  .pulse-item { background: #1a1a2e; border-radius: 6px; padding: 10px 14px; }
  .pulse-label { font-size: 0.75em; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.05em; }
  .pulse-value { font-size: 1.15em; font-weight: 700; color: #ffffff; }
  .pulse-change { font-size: 0.9em; font-weight: 600; }
  .pulse-change.up { color: #4ade80; }
  .pulse-change.down { color: #f87171; }
  ul.pulse-bullets { margin: 12px 0; padding-left: 20px; line-height: 1.9; color: #1a1a1a; }
  ul.pulse-bullets li { color: #1a1a1a; margin-bottom: 4px; }
  ul.pulse-bullets li strong { color: #1a1a2e; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88em; margin-top: 8px; }
  th { background: #1a1a2e; color: white; padding: 8px 10px; text-align: left; font-weight: 600; }
  td { padding: 7px 10px; border-bottom: 1px solid #e8e8e8; color: #1a1a1a; }
  tr:nth-child(even) { background: #f9fafb; }
  .up { color: #16a34a; font-weight: 600; } .down { color: #dc2626; font-weight: 600; }
  .signal-buy { background: #dcfce7; color: #15803d; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; white-space: nowrap; }
  .signal-sell { background: #fee2e2; color: #dc2626; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; white-space: nowrap; }
  .signal-watch { background: #fef9c3; color: #854d0e; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; white-space: nowrap; }
  .signal-hold { background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; white-space: nowrap; }
  .commentary-block { margin-bottom: 18px; padding: 14px 16px; border: 1px solid #e8e8e8; border-radius: 8px; }
  .commentary-block h3 { margin: 0 0 6px 0; font-size: 1em; color: #1a1a2e; }
  .commentary-block p { color: #1a1a1a; }
  .pick-block { margin-bottom: 16px; padding: 14px 16px; background: #f0f9ff; border-left: 4px solid #0ea5e9; border-radius: 0 8px 8px 0; }
  .pick-block h3 { margin: 0 0 6px 0; color: #0369a1; }
  .pick-block p { color: #1a1a1a; }
  .risk-block { margin-bottom: 16px; padding: 14px 16px; background: #fff7ed; border-left: 4px solid #f97316; border-radius: 0 8px 8px 0; }
  .risk-block h3 { margin: 0 0 6px 0; color: #c2410c; }
  .risk-block p { color: #1a1a1a; }
  .section-label { font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
  p { line-height: 1.65; margin: 6px 0; color: #1a1a1a; }
  .tag { display: inline-block; font-size: 0.75em; padding: 1px 7px; border-radius: 10px; margin-left: 6px; vertical-align: middle; }
  .tag-portfolio { background: #ede9fe; color: #6d28d9; }
  .tag-watchlist { background: #dbeafe; color: #1d4ed8; }
"""

# ── Slim payload helpers ──────────────────────────────────────────────────────
def slim_for_table(data):
    """Only the fields needed to build the snapshot table."""
    return [
        {
            "ticker":        d["ticker"],
            "last_close":    d.get("last_close"),
            "pct_change":    d.get("pct_change"),
            "pct_from_52h":  d.get("pct_from_52h"),
            "vs_200ma_pct":  d.get("vs_200ma_pct"),
            "macd_momentum": d.get("macd_momentum"),
            "fwd_pe":        d.get("fwd_pe"),
            "rev_growth_yoy":d.get("rev_growth_yoy"),
            "vol_vs_avg":    d.get("vol_vs_avg"),
        }
        for d in data
    ]

def slim_for_news(data):
    """Only the fields needed for news/events analysis."""
    return [
        {
            "ticker":     d["ticker"],
            "pct_change": d.get("pct_change"),
            "news":       d.get("news", []),
            "estimates":  d.get("estimates"),
        }
        for d in data
    ]

def extract_text(response):
    return "".join(b.text for b in response.content if hasattr(b, "text"))

# ── CALL 1: Market Pulse + Portfolio Snapshot ─────────────────────────────────
def generate_part1(portfolio_data, watchlist_data, today):
    port_table = json.dumps(slim_for_table(portfolio_data))
    watch_table = json.dumps(slim_for_table(watchlist_data))

    prompt = f"""
You are a sharp equity analyst. Today is {today}.

PORTFOLIO TABLE DATA: {port_table}
WATCHLIST TABLE DATA: {watch_table}

Use web search to find: today's S&P 500, Nasdaq, Dow levels and % changes, 10Y yield, VIX, WTI/Brent crude, dollar index (DXY), any Fed commentary, S&P 500 forward P/E, and major sector rotation themes.

Output ONLY this HTML fragment. Follow this EXACT order — do not deviate:

FIRST: Header
<h1>📈 Daily Morning Brief</h1>
<p class="subtitle">[Day, Date] &nbsp;|&nbsp; Personal Portfolio Intelligence &nbsp;|&nbsp; SPX Fwd P/E: [X]x &nbsp;|&nbsp; 10Y: [X]%</p>

SECOND: Market Pulse section
<h2>📊 Market Pulse</h2>
- pulse-grid div with exactly 6 pulse-item tiles: S&P 500, Nasdaq, Dow, 10Y Yield, VIX, WTI Crude
- Each tile: pulse-label, pulse-value, pulse-change (class="up" or "down")
- Then ul class="pulse-bullets" with 6-8 li items: index narrative, bonds, VIX, oil, dollar, Fed theme, sector rotation, portfolio read
- All bullet text must be plain dark color — do NOT use gray or muted colors on bullet text

THIRD: Portfolio Snapshot — THIS MUST COME IMMEDIATELY AFTER MARKET PULSE, NO EXCEPTIONS
<h2>📋 Portfolio Snapshot</h2>
<p class="section-label">Portfolio Holdings — SPX Fwd P/E Reference: [X]x</p>
Build a complete HTML table with ALL {len(portfolio_data)} portfolio tickers. Do not skip or truncate any row.
Columns: Ticker | Price | 1D % | vs 52W High | vs 200MA | MACD Momentum | Fwd P/E | P/E vs SPX | Rev Growth | Vol/Avg | Signal
- 1D %, vs 52W High, vs 200MA: use class="up" for positive, class="down" for negative
- MACD Momentum: show macd_momentum field value; wrap in <span class="up"> if ▲ or Crossing Up, <span class="down"> if ▼ or Crossing Down
- P/E vs SPX: fwd_pe divided by SPX fwd P/E as Xx ratio; green (class="up") if under 1.0x, red (class="down") if over 1.5x
- Rev Growth: show rev_growth_yoy as +X% with class="up"
- Vol/Avg: show vol_vs_avg as Xx
- Signal: <span class="signal-buy">🟢 Buy More</span> or signal-sell, signal-watch, signal-hold

If watchlist is non-empty, add a second table immediately after:
<p class="section-label" style="margin-top:20px">Watchlist — Entry Signals</p>
Same columns. Signal options: signal-buy (Buy Now), signal-watch (Getting Interesting), signal-hold (Not Yet)

NO OTHER CONTENT after the tables. Stop here.

RULES: Raw HTML only. No markdown, no backticks, no preamble, no content after the watchlist table.
"""
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}]
    )
    raw = extract_text(resp)
    match = re.search(r'<h1', raw)
    return raw[match.start():] if match else raw

# ── CALL 2: News, Picks, Risk Flags, Watch List ───────────────────────────────
def generate_part2(portfolio_data, watchlist_data, today):
    portfolio_tickers = [d["ticker"] for d in portfolio_data]
    watchlist_tickers = [d["ticker"] for d in watchlist_data]
    news_json = json.dumps(slim_for_news(portfolio_data + watchlist_data))

    prompt = f"""
You are a sharp equity analyst. Today is {today}.
Portfolio: {portfolio_tickers}
Watchlist: {watchlist_tickers}
News/estimates data: {news_json}

Use web search to find:
1. Any material news in last 48h for the above tickers (analyst actions, earnings, fund disclosures, CEO commentary, deals, major price moves)
2. Any WATCHLIST tickers showing strong buy signals right now (technicals + catalysts + valuation all aligned)
3. 2-3 stocks NOT on either list that have a compelling opportunity TODAY based on macro trends, breaking news, fund activity, clinical results, geopolitical signals, or any other catalyst — be specific and opportunistic, think like a hedge fund analyst scanning the tape
4. 2-3 names from the portfolio showing the clearest signs of deterioration or structural weakness

Output ONLY this HTML fragment — no DOCTYPE, no html/head/body tags, no CSS:

<!-- PART2_START -->
<h2>📰 News & Events</h2>
<p class="section-label">Material events only — last 24-48 hours</p>
[For each ticker with a trigger event, one commentary-block div:]
[Triggers: price ±5% 🔥 | earnings 📊 | analyst action 🎯 | fund disclosure 🏦 | CEO/conference 🎤 | deal/contract 📋 | major news 📰]
[Format: <div class="commentary-block"><h3>[EMOJI] TICKER — Name <span class="tag tag-portfolio">Portfolio</span></h3><p>2-3 sentences: what happened, why it matters, what to watch.</p></div>]
[If nothing material: <div class="commentary-block"><p>No material events today. The table tells the story.</p></div>]

<h2>🎯 Watchlist Opportunities</h2>
<p class="section-label">Only highlight watchlist names where buy signals are clearly aligned</p>
[Use pick-block divs. Only include names where technicals + valuation + catalyst all point to a near-term entry. If no watchlist names qualify, write: <div class="pick-block"><p>No watchlist names with strong enough buy alignment today.</p></div>]

<h2>⚠️ Risk Flags</h2>
<p class="section-label">Portfolio names showing clear deterioration — be direct</p>
[2-3 risk-block divs. Flag names with: technical breakdown (below 200MA + negative MACD building), valuation stretched with no growth support, negative catalysts, or sector headwinds that are worsening. One short paragraph each.]

<h2>💡 Claude's Picks & Watch List</h2>
<p class="section-label">Opportunistic calls + key levels to monitor tomorrow — no overlap between entries</p>
[Use pick-block divs. Include 3-4 total entries combining both outside picks AND key levels/catalysts to watch tomorrow. Each entry must be distinct — do not repeat the same ticker or thesis. Format each as: ticker + company name as h3, then: Catalyst | Why it creates upside | Key risk | Entry/watch level. Prioritize names with a specific catalyst from TODAY.]
<!-- PART2_END -->

RULES: Output only the HTML fragment. No markdown, no backticks, no preamble.
"""
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_text(resp)

# ── Stitch both parts into one complete HTML email ────────────────────────────
def generate_brief(portfolio_data, watchlist_data):
    today = datetime.date.today().strftime("%A, %B %d, %Y")

    print("  Generating Part 1 (Market Pulse + Snapshot)...")
    part1 = generate_part1(portfolio_data, watchlist_data, today)

    print("  Waiting 120s between API calls to avoid rate limit...")
    time.sleep(120)

    print("  Generating Part 2 (News, Picks, Risk Flags)...")
    # Retry once if rate limited
    for attempt in range(2):
        try:
            part2 = generate_part2(portfolio_data, watchlist_data, today)
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt == 0:
                print("  Rate limit hit on Part 2 — waiting 90s and retrying...")
                time.sleep(90)
            else:
                raise

    # Strip comment markers if present
    for marker in ["<!-- PART1_START -->", "<!-- PART1_END -->", "<!-- PART2_START -->", "<!-- PART2_END -->"]:
        part1 = part1.replace(marker, "")
        part2 = part2.replace(marker, "")

    # Assemble full HTML document
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{CSS}
</style>
</head>
<body>
{part1.strip()}
{part2.strip()}
</body>
</html>"""

    return html

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
