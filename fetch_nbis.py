#!/usr/bin/env python3
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "NBIS"
NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LATEST = DATA_DIR / "latest.json"
HISTORY_DIR = DATA_DIR / "history"

# Keep the heatmap readable on a phone.
MAX_EXPIRATIONS = int(os.getenv("MAX_EXPIRATIONS", "8"))
STRIKE_RANGE = float(os.getenv("STRIKE_RANGE", "0.45"))  # +/- 45% around spot

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_gamma_vanna(S: float, K: float, T: float, sigma: float, r: float = 0.04, q: float = 0.0):
    """
    Black-Scholes gamma and vanna.
    vanna = d(delta)/d(sigma), same sign formula for calls and puts.
    """
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

def safe_num(v, default=0.0):
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default

def current_spot(t: yf.Ticker) -> float:
    # Try fast_info first, then history.
    try:
        v = t.fast_info.get("last_price")
        if v:
            return float(v)
    except Exception:
        pass
    hist = t.history(period="1d", interval="1m", auto_adjust=False, prepost=False)
    if not hist.empty:
        return float(hist["Close"].dropna().iloc[-1])
    hist = t.history(period="5d", interval="1d", auto_adjust=False)
    if not hist.empty:
        return float(hist["Close"].dropna().iloc[-1])
    raise RuntimeError("Could not determine NBIS spot price")

def option_map(df: pd.DataFrame):
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        strike = safe_num(row.get("strike"))
        if strike <= 0:
            continue
        out[round(strike, 4)] = {
            "oi": max(0.0, safe_num(row.get("openInterest"))),
            "iv": max(0.0, safe_num(row.get("impliedVolatility"))),
            "volume": max(0.0, safe_num(row.get("volume"))),
            "bid": max(0.0, safe_num(row.get("bid"))),
            "ask": max(0.0, safe_num(row.get("ask"))),
            "last": max(0.0, safe_num(row.get("lastPrice"))),
        }
    return out

def build_snapshot():
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY)

    ticker = yf.Ticker(TICKER)
    spot = current_spot(ticker)

    expirations = list(ticker.options or [])[:MAX_EXPIRATIONS]
    if not expirations:
        raise RuntimeError("Yahoo returned no option expirations for NBIS")

    low = spot * (1.0 - STRIKE_RANGE)
    high = spot * (1.0 + STRIKE_RANGE)

    raw = {}
    strike_set = set()

    for exp in expirations:
        chain = ticker.option_chain(exp)
        calls = option_map(chain.calls)
        puts = option_map(chain.puts)

        strikes = sorted(set(calls) | set(puts))
        strikes = [k for k in strikes if low <= k <= high]

        exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(
            hour=16, minute=0, second=0, tzinfo=NY
        )
        T = max((exp_dt - now_ny).total_seconds() / (365.0 * 24 * 3600), 1.0 / (365.0 * 24))

        for K in strikes:
            c = calls.get(K, {"oi":0,"iv":0,"volume":0,"bid":0,"ask":0,"last":0})
            p = puts.get(K, {"oi":0,"iv":0,"volume":0,"bid":0,"ask":0,"last":0})

            cg, cv = bs_gamma_vanna(spot, K, T, c["iv"]) if c["iv"] > 0 else (0.0, 0.0)
            pg, pv = bs_gamma_vanna(spot, K, T, p["iv"]) if p["iv"] > 0 else (0.0, 0.0)

            # Heuristic exposure conventions:
            # GEX: dollar gamma for a 1% spot move, calls positive / puts negative.
            # VEX: dollar-equivalent delta sensitivity to a 1 vol-point move,
            #      calls positive / puts negative.
            call_gex = cg * c["oi"] * 100.0 * spot * spot * 0.01
            put_gex  = pg * p["oi"] * 100.0 * spot * spot * 0.01
            call_vex = cv * c["oi"] * 100.0 * spot * 0.01
            put_vex  = pv * p["oi"] * 100.0 * spot * 0.01

            net_gex = call_gex - put_gex
            net_vex = call_vex - put_vex

            cell = {
                "vex": net_vex,
                "gex": net_gex,
                "oi": c["oi"] + p["oi"],
                "iv": (
                    (c["iv"] * c["oi"] + p["iv"] * p["oi"]) / (c["oi"] + p["oi"])
                    if (c["oi"] + p["oi"]) > 0 else max(c["iv"], p["iv"])
                ),
                "call": {
                    "oi": c["oi"], "iv": c["iv"], "volume": c["volume"],
                    "bid": c["bid"], "ask": c["ask"], "last": c["last"],
                    "vex": call_vex, "gex": call_gex
                },
                "put": {
                    "oi": p["oi"], "iv": p["iv"], "volume": p["volume"],
                    "bid": p["bid"], "ask": p["ask"], "last": p["last"],
                    "vex": put_vex, "gex": put_gex
                },
            }
            raw[(K, exp)] = cell
            strike_set.add(K)

    strikes = sorted(strike_set, reverse=True)
    cells = []
    for K in strikes:
        row = {"strike": K, "values": {}}
        for exp in expirations:
            row["values"][exp] = raw.get((K, exp))
        cells.append(row)

    return {
        "symbol": TICKER,
        "spot": spot,
        "updated_utc": now_utc.isoformat(),
        "updated_et": now_ny.isoformat(),
        "expirations": expirations,
        "strikes": strikes,
        "rows": cells,
        "method": {
            "source": "Yahoo Finance via yfinance",
            "note": "NET VEX/GEX use a simple calls-minus-puts heuristic; this is not known dealer positioning.",
            "vex": "vanna * OI * 100 * spot * 0.01",
            "gex": "gamma * OI * 100 * spot^2 * 0.01"
        }
    }

def write_snapshot(snapshot):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    with LATEST.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    ts = datetime.fromisoformat(snapshot["updated_et"])
    day_dir = HISTORY_DIR / ts.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f'{ts.strftime("%H-%M")}.json'
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"Wrote {LATEST}")
    print(f"Wrote {path}")

if __name__ == "__main__":
    snapshot = build_snapshot()
    write_snapshot(snapshot)
