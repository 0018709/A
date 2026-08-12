#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone, time as dtime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
import exchange_calendars as xcals
from bs4 import BeautifulSoup

TICKER = "NBIS"
EU_TICKERS = ["YDX.DE", "YDX.F"]   # Xetra first, Frankfurt fallback
FX_TICKER = "EURUSD=X"

NY = ZoneInfo("America/New_York")
BERLIN = ZoneInfo("Europe/Berlin")
UTC = timezone.utc

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"

# 0 = ALL expirations returned by Yahoo.
MAX_EXPIRATIONS = int(os.getenv("MAX_EXPIRATIONS", "0"))

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NBIS-VEX-personal-dashboard/1.0)",
    "Accept": "text/plain,text/html;q=0.9,*/*;q=0.8",
}

def safe_num(v, default=0.0):
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_gamma_vanna(S, K, T, sigma, r=0.04, q=0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or not math.isfinite(sigma):
        return 0.0, 0.0
    sigma = min(max(sigma, 0.01), 5.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    phi = norm_pdf(d1)
    disc_q = math.exp(-q * T)
    gamma = disc_q * phi / (S * sigma * sqrtT)
    vanna = -disc_q * phi * d2 / sigma
    return gamma, vanna

def calendar_info(name, local_tz, now_utc):
    now_local = now_utc.astimezone(local_tz)
    d = pd.Timestamp(now_local.date())
    start = d - pd.Timedelta(days=14)
    end = d + pd.Timedelta(days=14)
    cal = xcals.get_calendar(name, start=start, end=end)

    is_session = cal.is_session(d)
    if not is_session:
        return {
            "calendar": name,
            "holiday": True,
            "status": "holiday",
            "date": str(now_local.date()),
            "open": None,
            "close": None,
        }

    sched = cal.schedule.loc[str(now_local.date())]
    op = sched["open"].to_pydatetime()
    cl = sched["close"].to_pydatetime()
    if op.tzinfo is None:
        op = op.replace(tzinfo=UTC)
    if cl.tzinfo is None:
        cl = cl.replace(tzinfo=UTC)

    if now_utc < op:
        status = "pre"
    elif now_utc < cl:
        status = "open"
    else:
        status = "closed"

    return {
        "calendar": name,
        "holiday": False,
        "status": status,
        "date": str(now_local.date()),
        "open": op.isoformat(),
        "close": cl.isoformat(),
    }

def previous_us_session(now_utc):
    now_et = now_utc.astimezone(NY)
    today = pd.Timestamp(now_et.date())
    cal = xcals.get_calendar(
        "XNYS",
        start=today - pd.Timedelta(days=20),
        end=today + pd.Timedelta(days=3),
    )
    sessions = cal.sessions_in_range(today - pd.Timedelta(days=20), today)
    prior = [s for s in sessions if s.date() < today.date()]
    if not prior:
        raise RuntimeError("Could not determine previous US trading session")
    return prior[-1].date()

def latest_history_point(symbol, period="5d", interval="1m", prepost=False):
    t = yf.Ticker(symbol)
    df = t.history(
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=prepost,
    )
    if df is None or df.empty or "Close" not in df:
        return None
    s = df["Close"].dropna()
    if s.empty:
        return None
    ts = s.index[-1]
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return {
        "price": float(s.iloc[-1]),
        "time": ts.isoformat(),
    }

def regular_close_for_date(symbol, target_date):
    t = yf.Ticker(symbol)
    df = t.history(period="15d", interval="1d", auto_adjust=False, prepost=False)
    if df is None or df.empty:
        raise RuntimeError(f"No daily history for {symbol}")
    for idx, row in df.iloc[::-1].iterrows():
        dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if dt.date() == target_date:
            return float(row["Close"]), dt
    # fallback: most recent daily close strictly before target date+1
    candidates = []
    for idx, row in df.iterrows():
        dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if dt.date() <= target_date:
            candidates.append((dt, float(row["Close"])))
    if not candidates:
        raise RuntimeError(f"No previous close for {symbol} on {target_date}")
    dt, px = candidates[-1]
    return px, dt

def us_price_bundle(now_utc):
    market = calendar_info("XNYS", NY, now_utc)
    t = yf.Ticker(TICKER)

    reg = latest_history_point(TICKER, "5d", "1m", False)
    if reg is None:
        reg = latest_history_point(TICKER, "10d", "1d", False)

    # Premarket = today's 04:00–09:30 ET observations.
    pre = None
    try:
        df = t.history(period="2d", interval="1m", auto_adjust=False, prepost=True)
        if df is not None and not df.empty:
            today_et = now_utc.astimezone(NY).date()
            pts = []
            for idx, row in df.iterrows():
                ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=NY)
                ts_et = ts.astimezone(NY)
                if (
                    ts_et.date() == today_et
                    and dtime(4, 0) <= ts_et.time() < dtime(9, 30)
                    and math.isfinite(safe_num(row.get("Close"), float("nan")))
                ):
                    pts.append((ts, float(row["Close"])))
            if pts:
                ts, px = pts[-1]
                pre = {"price": px, "time": ts.isoformat()}
    except Exception as e:
        print("Premarket fetch warning:", e)

    return {
        "exchange": "NASDAQ",
        "symbol": TICKER,
        "currency": "USD",
        "market": market,
        "regular": reg,
        "premarket": pre,
    }

def eu_price_bundle(now_utc):
    picked = None
    quote = None
    for sym in EU_TICKERS:
        try:
            quote = latest_history_point(sym, "5d", "1m", False)
            if quote:
                picked = sym
                break
        except Exception as e:
            print(f"EU quote warning {sym}:", e)

    if picked == "YDX.DE":
        cal_name, exch = "XETR", "Xetra"
    else:
        cal_name, exch = "XFRA", "Frankfurt"

    market = calendar_info(cal_name, BERLIN, now_utc)
    fx = latest_history_point(FX_TICKER, "5d", "1m", False)
    if fx is None:
        fx = latest_history_point(FX_TICKER, "10d", "1d", False)

    usd = None
    if quote and fx:
        usd = quote["price"] * fx["price"]

    return {
        "exchange": exch,
        "symbol": picked or "YDX",
        "currency": "EUR",
        "market": market,
        "regular": quote,
        "usd_price": usd,
        "fx": fx,
    }

def parse_occ_plain(text):
    """
    OCC series-search plain text layout historically looks like:
    ProductSymbol year Month Day Integer Dec C P CallOI PutOI PositionLimit
    Example:
    AA 2022 11 18 17 500 C P 0 5950 25000000
    """
    out = {}
    for raw in text.splitlines():
        line = " ".join(raw.replace(",", "").split())
        parts = line.split()
        if len(parts) < 11 or parts[0].upper() != TICKER:
            continue
        try:
            y, m, d = map(int, parts[1:4])
            strike = int(parts[4]) + int(parts[5]) / (10 ** len(parts[5]))
            call_oi = int(float(parts[-3]))
            put_oi = int(float(parts[-2]))
        except Exception:
            continue
        exp = f"{y:04d}-{m:02d}-{d:02d}"
        out[(round(strike, 6), exp)] = (call_oi, put_oi)
    return out

def parse_occ_html(text):
    out = {}
    soup = BeautifulSoup(text, "html.parser")
    for tr in soup.find_all("tr"):
        cells = [" ".join(c.get_text(" ", strip=True).replace(",", "").split())
                 for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        flat = " ".join(cells).split()
        if not flat or flat[0].upper() != TICKER:
            continue
        # Try the same fixed OCC field ordering after flattening.
        if len(flat) >= 11:
            try:
                y, m, d = map(int, flat[1:4])
                strike = int(flat[4]) + int(flat[5]) / (10 ** len(flat[5]))
                call_oi = int(float(flat[-3]))
                put_oi = int(float(flat[-2]))
                exp = f"{y:04d}-{m:02d}-{d:02d}"
                out[(round(strike, 6), exp)] = (call_oi, put_oi)
            except Exception:
                pass
    if not out:
        # OCC sometimes returns data in a <pre> or text-like response.
        out = parse_occ_plain(soup.get_text("\n"))
    return out

def fetch_occ_open_interest():
    """
    Official OCC Series Search.
    OCC states these OI values are derived from the previous day's settlement.
    """
    url = "https://marketdata.theocc.com/series-search"
    r = requests.get(
        url,
        params={"symbolType": "U", "symbol": TICKER},
        headers=HTTP_HEADERS,
        timeout=40,
    )
    r.raise_for_status()
    text = r.text
    oi = parse_occ_plain(text)
    if not oi:
        oi = parse_occ_html(text)
    if not oi:
        sample = re.sub(r"\s+", " ", text[:1000])
        raise RuntimeError("OCC OI parse returned zero rows. Response sample: " + sample)
    print(f"OCC: parsed {len(oi)} strike/expiration OI rows")
    return oi

def option_map(df):
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        strike = safe_num(row.get("strike"))
        if strike <= 0:
            continue
        out[round(strike, 6)] = {
            "yahoo_oi": max(0, int(safe_num(row.get("openInterest")))),
            "iv": max(0.0, safe_num(row.get("impliedVolatility"))),
            "volume": max(0, int(safe_num(row.get("volume")))),
            "bid": max(0.0, safe_num(row.get("bid"))),
            "ask": max(0.0, safe_num(row.get("ask"))),
            "last": max(0.0, safe_num(row.get("lastPrice"))),
        }
    return out

def build_snapshot(mode):
    now_utc = datetime.now(UTC)
    now_et = now_utc.astimezone(NY)
    previous_session = previous_us_session(now_utc)

    prices = {
        "us": us_price_bundle(now_utc),
        "eu": eu_price_bundle(now_utc),
    }

    if mode == "previous_close":
        spot, spot_dt = regular_close_for_date(TICKER, previous_session)
        calc_time = datetime.combine(previous_session, dtime(16, 0), tzinfo=NY)
        snapshot_label = f"Previous US close {previous_session.isoformat()}"
    else:
        reg = prices["us"]["regular"]
        if not reg:
            spot, spot_dt = regular_close_for_date(TICKER, previous_session)
        else:
            spot = reg["price"]
            spot_dt = datetime.fromisoformat(reg["time"])
        calc_time = now_et
        snapshot_label = "Current/latest"

    occ_oi = fetch_occ_open_interest()

    ticker = yf.Ticker(TICKER)
    expirations = list(ticker.options or [])
    if MAX_EXPIRATIONS > 0:
        expirations = expirations[:MAX_EXPIRATIONS]
    if not expirations:
        raise RuntimeError("Yahoo returned no NBIS option expirations")

    raw = {}
    strike_set = set()
    yahoo_oi_nonzero = 0
    occ_oi_nonzero = 0

    for n, exp in enumerate(expirations, start=1):
        print(f"Yahoo option chain {n}/{len(expirations)}: {exp}")
        chain = ticker.option_chain(exp)
        calls = option_map(chain.calls)
        puts = option_map(chain.puts)

        strikes = sorted(
            set(calls) |
            set(puts) |
            {k for (k, e) in occ_oi.keys() if e == exp}
        )

        exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(
            hour=16, minute=0, second=0, tzinfo=NY
        )
        T = max(
            (exp_dt - calc_time).total_seconds() / (365.0 * 24 * 3600),
            1.0 / (365.0 * 24),
        )

        for K in strikes:
            c = calls.get(K, {
                "yahoo_oi":0, "iv":0, "volume":0, "bid":0, "ask":0, "last":0
            })
            p = puts.get(K, {
                "yahoo_oi":0, "iv":0, "volume":0, "bid":0, "ask":0, "last":0
            })

            occ_call, occ_put = occ_oi.get((round(K, 6), exp), (None, None))
            call_oi = c["yahoo_oi"] if occ_call is None else int(occ_call)
            put_oi = p["yahoo_oi"] if occ_put is None else int(occ_put)

            yahoo_oi_nonzero += int(c["yahoo_oi"] > 0) + int(p["yahoo_oi"] > 0)
            occ_oi_nonzero += int(call_oi > 0) + int(put_oi > 0)

            cg, cv = bs_gamma_vanna(spot, K, T, c["iv"]) if c["iv"] > 0 else (0.0, 0.0)
            pg, pv = bs_gamma_vanna(spot, K, T, p["iv"]) if p["iv"] > 0 else (0.0, 0.0)

            call_gex = cg * call_oi * 100.0 * spot * spot * 0.01
            put_gex  = pg * put_oi  * 100.0 * spot * spot * 0.01
            call_vex = cv * call_oi * 100.0 * spot * 0.01
            put_vex  = pv * put_oi  * 100.0 * spot * 0.01

            total_oi = call_oi + put_oi
            weighted_iv = (
                (c["iv"] * call_oi + p["iv"] * put_oi) / total_oi
                if total_oi > 0
                else max(c["iv"], p["iv"])
            )

            raw[(K, exp)] = {
                "vex": call_vex - put_vex,
                "gex": call_gex - put_gex,
                "oi": total_oi,
                "iv": weighted_iv,
                "call": {
                    "oi": call_oi,
                    "oi_yahoo": c["yahoo_oi"],
                    "iv": c["iv"],
                    "volume": c["volume"],
                    "bid": c["bid"],
                    "ask": c["ask"],
                    "last": c["last"],
                    "vex": call_vex,
                    "gex": call_gex,
                },
                "put": {
                    "oi": put_oi,
                    "oi_yahoo": p["yahoo_oi"],
                    "iv": p["iv"],
                    "volume": p["volume"],
                    "bid": p["bid"],
                    "ask": p["ask"],
                    "last": p["last"],
                    "vex": put_vex,
                    "gex": put_gex,
                },
            }
            strike_set.add(K)

        # Be gentle with Yahoo when requesting every expiration.
        time.sleep(0.15)

    strikes = sorted(strike_set, reverse=True)
    rows = []
    for K in strikes:
        rows.append({
            "strike": K,
            "values": {exp: raw.get((K, exp)) for exp in expirations},
        })

    if occ_oi_nonzero == 0:
        raise RuntimeError("OCC returned no non-zero NBIS open interest; refusing to publish zero exposure map")

    return {
        "symbol": TICKER,
        "mode": mode,
        "snapshot_label": snapshot_label,
        "spot": spot,
        "spot_basis_time": (
            spot_dt.isoformat() if hasattr(spot_dt, "isoformat") else str(spot_dt)
        ),
        "previous_us_session": previous_session.isoformat(),
        "oi_settlement_date": previous_session.isoformat(),
        "updated_utc": now_utc.isoformat(),
        "updated_et": now_et.isoformat(),
        "prices": prices,
        "expirations": expirations,
        "strikes": strikes,
        "rows": rows,
        "diagnostics": {
            "occ_contract_sides_nonzero": occ_oi_nonzero,
            "yahoo_contract_sides_nonzero": yahoo_oi_nonzero,
            "strike_count": len(strikes),
            "expiration_count": len(expirations),
            "min_strike": min(strikes) if strikes else None,
            "max_strike": max(strikes) if strikes else None,
        },
        "method": {
            "oi_source": "OCC Series Search — previous-day settlement",
            "iv_source": "Yahoo Finance via yfinance",
            "europe_source": "Yahoo Finance YDX.DE (Xetra), YDX.F fallback",
            "dealer_note": "NET sign is calls minus puts heuristic, not known dealer positioning.",
            "vex": "vanna * OI * 100 * spot * 0.01",
            "gex": "gamma * OI * 100 * spot^2 * 0.01",
        },
    }

def write_snapshot(snapshot):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    mode = snapshot["mode"]
    if mode == "previous_close":
        primary = DATA_DIR / "previous_close.json"
    else:
        primary = DATA_DIR / "latest.json"

    with primary.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    ts = datetime.fromisoformat(snapshot["updated_et"])
    day_dir = HISTORY_DIR / ts.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    hist = day_dir / f'{ts.strftime("%H-%M")}-{mode}.json'
    with hist.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print("Wrote", primary)
    print("Wrote", hist)
    print("Diagnostics:", snapshot["diagnostics"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["current", "previous_close"],
        default=os.getenv("SNAPSHOT_MODE", "current"),
    )
    args = parser.parse_args()
    snapshot = build_snapshot(args.mode)
    write_snapshot(snapshot)

if __name__ == "__main__":
    main()
