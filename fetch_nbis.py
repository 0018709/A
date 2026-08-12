#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
import exchange_calendars as xcals
from bs4 import BeautifulSoup

TICKER = "NBIS"
EU_TICKERS = ["YDX.DE", "YDX.F"]
FX_TICKER = "EURUSD=X"

NY = ZoneInfo("America/New_York")
BERLIN = ZoneInfo("Europe/Berlin")
MSK = ZoneInfo("Europe/Moscow")
UTC = timezone.utc

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LATEST = DATA_DIR / "latest.json"
PREV_CLOSE = DATA_DIR / "previous_close.json"
MARKET = DATA_DIR / "market.json"
EVENTS = DATA_DIR / "events.json"
PREV_METRICS = DATA_DIR / "prev_metrics.json"

MAX_EXPIRATIONS = int(os.getenv("MAX_EXPIRATIONS", "0"))  # 0 = all
EVENT_RETENTION_DAYS = 7
MAX_EVENTS = 120
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.04"))

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NBIS-VEX-personal-dashboard/3.0)",
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

def iso_dt(v):
    if not v:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "to_pydatetime"):
        return v.to_pydatetime().isoformat()
    return str(v)

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_gamma_vanna(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
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
    cal = xcals.get_calendar(
        name,
        start=d - pd.Timedelta(days=15),
        end=d + pd.Timedelta(days=15),
    )
    if not cal.is_session(d):
        return {
            "calendar": name, "holiday": True, "status": "holiday",
            "date": str(now_local.date()), "open": None, "close": None,
        }
    sched = cal.schedule.loc[str(now_local.date())]
    op = sched["open"].to_pydatetime()
    cl = sched["close"].to_pydatetime()
    if op.tzinfo is None: op = op.replace(tzinfo=UTC)
    if cl.tzinfo is None: cl = cl.replace(tzinfo=UTC)
    status = "pre" if now_utc < op else ("open" if now_utc < cl else "closed")
    return {
        "calendar": name, "holiday": False, "status": status,
        "date": str(now_local.date()), "open": op.isoformat(), "close": cl.isoformat(),
    }

def previous_us_session(now_utc):
    today = pd.Timestamp(now_utc.astimezone(NY).date())
    cal = xcals.get_calendar(
        "XNYS",
        start=today - pd.Timedelta(days=20),
        end=today + pd.Timedelta(days=3),
    )
    sessions = cal.sessions_in_range(today - pd.Timedelta(days=20), today)
    prior = [s for s in sessions if s.date() < today.date()]
    if not prior:
        raise RuntimeError("Could not determine previous US session")
    return prior[-1].date()

def history(symbol, period, interval, prepost=False):
    return yf.Ticker(symbol).history(
        period=period, interval=interval, auto_adjust=False,
        prepost=prepost, repair=True, raise_errors=False,
    )

def session_points_us():
    """Latest PRE / REGULAR / POST prices over the most recent 5 days."""
    df = history(TICKER, "5d", "1m", True)
    out = {"premarket": None, "regular": None, "postmarket": None}
    if df is None or df.empty:
        return out

    for idx, row in df.iterrows():
        px = safe_num(row.get("Close"), float("nan"))
        if not math.isfinite(px):
            continue
        ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=NY)
        et = ts.astimezone(NY)
        tm = et.time()

        if dtime(4, 0) <= tm < dtime(9, 30):
            key = "premarket"
        elif dtime(9, 30) <= tm < dtime(16, 0):
            key = "regular"
        elif dtime(16, 0) <= tm < dtime(20, 0):
            key = "postmarket"
        else:
            continue

        out[key] = {"price": px, "time": ts.isoformat(), "date_et": str(et.date())}
    return out

def latest_regular(symbol, period="5d", interval="1m"):
    df = history(symbol, period, interval, False)
    if df is None or df.empty:
        return None
    s = df["Close"].dropna()
    if s.empty:
        return None
    idx = s.index[-1]
    ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return {"price": float(s.iloc[-1]), "time": ts.isoformat()}

def regular_close_for_date(symbol, target_date):
    df = history(symbol, "15d", "1d", False)
    if df is None or df.empty:
        raise RuntimeError(f"No daily history for {symbol}")
    candidates = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if ts.date() <= target_date:
            candidates.append((ts, float(row["Close"])))
    if not candidates:
        raise RuntimeError(f"No close for {symbol} on/before {target_date}")
    return candidates[-1][1], candidates[-1][0]

def current_us_session(now_utc, market, sessions):
    et = now_utc.astimezone(NY)
    if market["holiday"]:
        return "holiday"

    tm = et.time()
    if dtime(4,0) <= tm < dtime(9,30):
        return "premarket"
    if dtime(9,30) <= tm < dtime(16,0):
        return "regular"
    if dtime(16,0) <= tm < dtime(20,0):
        return "postmarket"
    return "closed"

def active_us_quote(now_utc, market, sessions):
    cur = current_us_session(now_utc, market, sessions)
    if cur in sessions and sessions.get(cur):
        return cur, sessions[cur]

    # Outside active hours choose freshest available quote.
    available = [(k, v) for k, v in sessions.items() if v and v.get("time")]
    if not available:
        return "none", None
    available.sort(key=lambda kv: datetime.fromisoformat(kv[1]["time"]))
    return available[-1]

def eu_bundle(now_utc):
    quote = None
    symbol = None
    exchange = None
    cal_name = None

    for sym, exch, cal in [
        ("YDX.DE", "Xetra", "XETR"),
        ("YDX.F", "Frankfurt", "XFRA"),
    ]:
        try:
            q = latest_regular(sym)
            if q:
                quote, symbol, exchange, cal_name = q, sym, exch, cal
                break
        except Exception as e:
            print("EU quote warning:", sym, e)

    if not cal_name:
        cal_name, exchange = "XETR", "Xetra"

    market = calendar_info(cal_name, BERLIN, now_utc)
    fx = latest_regular(FX_TICKER)
    usd = quote["price"] * fx["price"] if quote and fx else None

    return {
        "exchange": exchange,
        "symbol": symbol or "YDX",
        "currency": "EUR",
        "market": market,
        "regular": quote,
        "usd_price": usd,
        "fx": fx,
        "source": "Yahoo Finance via yfinance",
    }

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def build_market_tape():
    now_utc = datetime.now(UTC)
    us_market = calendar_info("XNYS", NY, now_utc)
    sessions = session_points_us()
    active_name, active_quote = active_us_quote(now_utc, us_market, sessions)
    eu = eu_bundle(now_utc)

    previous = read_json(MARKET, {})
    prev_active = previous.get("active_us", {}) if previous else {}
    move = None
    if active_quote and prev_active and prev_active.get("price"):
        old = safe_num(prev_active.get("price"))
        if old > 0:
            move = {
                "absolute": active_quote["price"] - old,
                "percent": (active_quote["price"] / old - 1.0) * 100.0,
                "from_time": prev_active.get("time"),
            }

    tape = {
        "symbol": TICKER,
        "updated_utc": now_utc.isoformat(),
        "updated_et": now_utc.astimezone(NY).isoformat(),
        "updated_msk": now_utc.astimezone(MSK).isoformat(),
        "us": {
            "exchange": "NASDAQ",
            "currency": "USD",
            "market": us_market,
            "current_session": current_us_session(now_utc, us_market, sessions),
            "premarket": sessions.get("premarket"),
            "regular": sessions.get("regular"),
            "postmarket": sessions.get("postmarket"),
            "source": "Yahoo Finance via yfinance",
        },
        "active_us": {
            "session": active_name,
            **(active_quote or {}),
            "move_from_last_update": move,
            "source": "Yahoo Finance via yfinance",
        },
        "eu": eu,
    }
    write_json(MARKET, tape)
    print("Wrote", MARKET)
    return tape

def parse_occ_plain(text):
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
    soup = BeautifulSoup(text, "html.parser")
    out = {}
    for tr in soup.find_all("tr"):
        cells = [
            " ".join(c.get_text(" ", strip=True).replace(",", "").split())
            for c in tr.find_all(["td", "th"])
        ]
        flat = " ".join(cells).split()
        if len(flat) < 11 or flat[0].upper() != TICKER:
            continue
        try:
            y, m, d = map(int, flat[1:4])
            strike = int(flat[4]) + int(flat[5]) / (10 ** len(flat[5]))
            call_oi = int(float(flat[-3]))
            put_oi = int(float(flat[-2]))
            exp = f"{y:04d}-{m:02d}-{d:02d}"
            out[(round(strike, 6), exp)] = (call_oi, put_oi)
        except Exception:
            pass
    return out or parse_occ_plain(soup.get_text("\n"))

def fetch_occ_open_interest():
    r = requests.get(
        "https://marketdata.theocc.com/series-search",
        params={"symbolType": "U", "symbol": TICKER},
        headers=HTTP_HEADERS,
        timeout=40,
    )
    r.raise_for_status()
    oi = parse_occ_plain(r.text) or parse_occ_html(r.text)
    if not oi:
        raise RuntimeError("OCC OI parse returned zero rows")
    return oi

def option_map(df):
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        K = safe_num(row.get("strike"))
        if K <= 0:
            continue
        out[round(K, 6)] = {
            "yahoo_oi": max(0, int(safe_num(row.get("openInterest")))),
            "iv": max(0.0, safe_num(row.get("impliedVolatility"))),
            "volume": max(0, int(safe_num(row.get("volume")))),
            "bid": max(0.0, safe_num(row.get("bid"))),
            "ask": max(0.0, safe_num(row.get("ask"))),
            "last": max(0.0, safe_num(row.get("lastPrice"))),
        }
    return out

def percentile(vals, p):
    arr = sorted(abs(float(x)) for x in vals if x is not None and math.isfinite(float(x)))
    if not arr:
        return 0.0
    i = min(len(arr)-1, max(0, int((len(arr)-1)*p)))
    return arr[i]

def flatten_metrics(snapshot):
    out = {}
    for row in snapshot.get("rows", []):
        K = row["strike"]
        for exp, cell in row.get("values", {}).items():
            if not cell:
                continue
            out[f"{K}|{exp}"] = [
                safe_num(cell.get("vex")),
                safe_num(cell.get("gex")),
                safe_num(cell.get("oi")),
                safe_num(cell.get("iv")),
            ]
    return out

def add_weekly_events(snapshot):
    now = datetime.fromisoformat(snapshot["updated_utc"])
    prev = read_json(PREV_METRICS, {})
    prev_cells = prev.get("cells", {}) if prev else {}
    prev_spot = safe_num(prev.get("spot")) if prev else 0.0
    cells = flatten_metrics(snapshot)

    dv = []
    dg = []
    candidates = []

    for key, cur in cells.items():
        old = prev_cells.get(key)
        if not old:
            continue
        d_vex = cur[0] - old[0]
        d_gex = cur[1] - old[1]
        d_oi = cur[2] - old[2]
        dv.append(d_vex)
        dg.append(d_gex)
        K, exp = key.split("|")
        candidates.append((key, float(K), exp, cur, old, d_vex, d_gex, d_oi))

    vex_thr = percentile(dv, 0.985)
    gex_thr = percentile(dg, 0.985)

    new_events = []
    for key, K, exp, cur, old, d_vex, d_gex, d_oi in candidates:
        strength = 0.0
        kind = None
        title = None
        text = None

        sign_flip = old[0] != 0 and cur[0] != 0 and (old[0] > 0) != (cur[0] > 0)
        if sign_flip and abs(cur[0]) >= percentile([v[0] for v in cells.values()], 0.90):
            strength = 96
            kind = "flip"
            title = f"⚡ VEX сменил знак · ${K:g}"
            text = (
                f"На страйке ${K:g} для {exp} VEX сменил знак: "
                f"{old[0]:,.0f} → {cur[0]:,.0f}. Это резкая перестройка модели."
            )
        elif vex_thr > 0 and abs(d_vex) >= vex_thr:
            strength = min(95, 65 + 30 * abs(d_vex) / max(vex_thr, 1))
            kind = "vex_jump"
            title = f"⚡ Резкий ΔVEX · ${K:g}"
            text = (
                f"VEX на ${K:g}, {exp} изменился на {d_vex:,.0f}: "
                f"{old[0]:,.0f} → {cur[0]:,.0f}."
            )
        elif gex_thr > 0 and abs(d_gex) >= gex_thr:
            strength = min(92, 62 + 28 * abs(d_gex) / max(gex_thr, 1))
            kind = "gex_jump"
            title = f"⚡ Резкий ΔGEX · ${K:g}"
            text = (
                f"GEX на ${K:g}, {exp} изменился на {d_gex:,.0f}: "
                f"{old[1]:,.0f} → {cur[1]:,.0f}."
            )
        elif old[2] > 0 and abs(d_oi) >= max(500, old[2] * 0.35):
            strength = min(88, 55 + 25 * abs(d_oi) / max(old[2], 1))
            kind = "oi_jump"
            title = f"⚡ Резкий ΔOI · ${K:g}"
            text = (
                f"OI на ${K:g}, {exp}: {old[2]:,.0f} → {cur[2]:,.0f} "
                f"(изменение {d_oi:+,.0f})."
            )

        if kind:
            new_events.append({
                "id": f"{int(now.timestamp())}-{kind}-{K}-{exp}",
                "time": now.isoformat(),
                "kind": kind,
                "title": title,
                "text": text,
                "strength": round(strength, 1),
                "focus": {
                    "type": "cell",
                    "strike_min": K,
                    "strike_max": K,
                    "exp_min": exp,
                    "exp_max": exp,
                },
            })

    if prev_spot > 0:
        move_pct = (snapshot["spot"] / prev_spot - 1.0) * 100
        if abs(move_pct) >= 1.2:
            new_events.append({
                "id": f"{int(now.timestamp())}-spot",
                "time": now.isoformat(),
                "kind": "spot_move",
                "title": f"⚡ Цена {move_pct:+.1f}%",
                "text": (
                    f"Цена в расчётном срезе изменилась с ${prev_spot:.2f} "
                    f"до ${snapshot['spot']:.2f} ({move_pct:+.2f}%)."
                ),
                "strength": min(100, 60 + abs(move_pct) * 8),
                "focus": {
                    "type": "row_band",
                    "strike_min": snapshot["spot"] * 0.97,
                    "strike_max": snapshot["spot"] * 1.03,
                    "exp_min": snapshot["expirations"][0] if snapshot["expirations"] else None,
                    "exp_max": snapshot["expirations"][-1] if snapshot["expirations"] else None,
                },
            })

    new_events.sort(key=lambda e: e["strength"], reverse=True)
    new_events = new_events[:12]

    old_events_doc = read_json(EVENTS, {"events": []}) or {"events": []}
    cutoff = now.timestamp() - EVENT_RETENTION_DAYS * 86400
    kept = []
    seen = set()

    for event in new_events + old_events_doc.get("events", []):
        try:
            ts = datetime.fromisoformat(event["time"]).timestamp()
        except Exception:
            continue
        if ts < cutoff or event["id"] in seen:
            continue
        seen.add(event["id"])
        kept.append(event)

    kept.sort(key=lambda e: (e.get("strength", 0), e.get("time", "")), reverse=True)
    kept = kept[:MAX_EVENTS]

    doc = {
        "updated_utc": now.isoformat(),
        "retention_days": EVENT_RETENTION_DAYS,
        "events": kept,
    }
    write_json(EVENTS, doc)
    write_json(PREV_METRICS, {
        "updated_utc": now.isoformat(),
        "spot": snapshot["spot"],
        "cells": cells,
    })
    return kept

def build_structural_insights(snapshot, events):
    spot = snapshot["spot"]
    cells = []

    for row in snapshot["rows"]:
        K = row["strike"]
        for exp, c in row["values"].items():
            if not c:
                continue
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=NY)
                dte = (exp_dt.date() - datetime.fromisoformat(snapshot["updated_et"]).date()).days
            except Exception:
                dte = 999
            cells.append({
                "strike": K, "exp": exp, "dte": dte,
                "vex": safe_num(c["vex"]), "gex": safe_num(c["gex"]),
                "oi": safe_num(c["oi"]), "iv": safe_num(c["iv"]),
            })

    if not cells:
        return [], {
            "bias": "Нет данных", "confidence": 0,
            "primary": None, "alternative": None,
            "text": "Недостаточно данных."
        }

    vex_scale = percentile([x["vex"] for x in cells], .97) or 1
    gex_scale = percentile([x["gex"] for x in cells], .97) or 1
    oi_scale = percentile([x["oi"] for x in cells], .97) or 1

    def relevance(x, metric):
        val = abs(x[metric]) / max(
            vex_scale if metric=="vex" else gex_scale if metric=="gex" else oi_scale, 1
        )
        prox = 1 / (1 + abs(x["strike"] - spot) / max(spot * .08, 1))
        tenor = 1 if x["dte"] <= 90 else .65
        return 100 * min(1.5, val) * (0.55 + 0.45*prox) * tenor

    insights = []

    for metric, label in [("vex","VEX"),("gex","GEX")]:
        pos = sorted([x for x in cells if x[metric] > 0],
                     key=lambda x: relevance(x, metric), reverse=True)[:3]
        neg = sorted([x for x in cells if x[metric] < 0],
                     key=lambda x: relevance(x, metric), reverse=True)[:3]
        for x in pos + neg:
            side = "положительный" if x[metric] > 0 else "отрицательный"
            where = "выше" if x["strike"] > spot else "ниже" if x["strike"] < spot else "у цены"
            insights.append({
                "id": f"{metric}-{x['strike']}-{x['exp']}",
                "kind": metric,
                "title": f"{label} {side} · ${x['strike']:g}",
                "strength": round(relevance(x, metric), 1),
                "text": (
                    f"{label} {side} на страйке ${x['strike']:g} ({x['exp']}), "
                    f"{where} текущей цены ${spot:.2f}. Значение: {x[metric]:,.0f}. "
                    f"Смысл зависит от режима IV и положения узла относительно spot; "
                    f"это модельный уровень, а не гарантированная цель."
                ),
                "focus": {
                    "type": "cell",
                    "strike_min": x["strike"], "strike_max": x["strike"],
                    "exp_min": x["exp"], "exp_max": x["exp"],
                },
            })

    above = [x for x in cells if x["strike"] > spot]
    below = [x for x in cells if x["strike"] < spot]

    for arr, direction, title in [
        (above, "выше цены", "Крупный OI сверху"),
        (below, "ниже цены", "Крупный OI снизу")
    ]:
        if arr:
            x = max(arr, key=lambda z: relevance(z, "oi"))
            insights.append({
                "id": f"oi-{direction}-{x['strike']}-{x['exp']}",
                "kind": "oi",
                "title": f"{title} · ${x['strike']:g}",
                "strength": round(relevance(x, "oi"), 1),
                "text": (
                    f"Крупный открытый интерес {direction}: {x['oi']:,.0f} контрактов "
                    f"на ${x['strike']:g}, экспирация {x['exp']}. OI сам по себе "
                    f"не сообщает, кто именно long/short."
                ),
                "focus": {
                    "type": "cell",
                    "strike_min": x["strike"], "strike_max": x["strike"],
                    "exp_min": x["exp"], "exp_max": x["exp"],
                },
            })

    # Bring recent sharp events into the list as high-priority insights.
    for e in events[:15]:
        insights.append({
            "id": "event-" + e["id"],
            "kind": "event",
            "title": e["title"],
            "strength": e["strength"],
            "text": e["text"] + " Событие хранится в ленте 7 дней.",
            "focus": e["focus"],
        })

    insights.sort(key=lambda x: x["strength"], reverse=True)

    # Scenario: proximity-weighted near/mid-dated exposure alignment.
    active = [x for x in cells if x["dte"] <= 90 and abs(x["strike"]-spot) <= spot*.25]
    score_v = sum(
        math.copysign(min(abs(x["vex"])/vex_scale, 2), x["vex"])
        * 1/(1+abs(x["strike"]-spot)/(spot*.08))
        for x in active
    )
    score_g = sum(
        math.copysign(min(abs(x["gex"])/gex_scale, 2), x["gex"])
        * 1/(1+abs(x["strike"]-spot)/(spot*.08))
        for x in active
    )

    # We deliberately use weak language: dealer-side sign is heuristic.
    combo = 0.55*score_v + 0.45*score_g
    if combo > 1.2:
        bias = "Умеренно бычье"
    elif combo < -1.2:
        bias = "Умеренно медвежье"
    else:
        bias = "Смешанное / нейтральное"

    agreement = 1.0 if score_v == 0 or score_g == 0 else (
        1.0 if (score_v > 0) == (score_g > 0) else 0.4
    )
    confidence = int(max(25, min(72, 35 + abs(combo)*4 + agreement*12)))

    target_above = max(above, key=lambda x: relevance(x,"vex"), default=None)
    target_below = max(below, key=lambda x: relevance(x,"vex"), default=None)
    primary = target_above["strike"] if combo >= 0 and target_above else (
        target_below["strike"] if target_below else None
    )
    alternative = target_below["strike"] if combo >= 0 and target_below else (
        target_above["strike"] if target_above else None
    )

    scenario = {
        "bias": bias,
        "confidence": confidence,
        "primary": primary,
        "alternative": alternative,
        "text": (
            "Оценка строится на согласовании VEX/GEX, близости узлов к spot и сроке "
            "экспирации. Это сценарий структуры опционов, а не прогноз с гарантией."
        ),
    }
    return insights[:40], scenario

def build_snapshot(mode, market_tape):
    now_utc = datetime.now(UTC)
    now_et = now_utc.astimezone(NY)
    previous_session = previous_us_session(now_utc)

    if mode == "previous_close":
        spot, spot_dt = regular_close_for_date(TICKER, previous_session)
        calc_time = datetime.combine(previous_session, dtime(16, 0), tzinfo=NY)
        label = f"Предыдущий срез · закрытие США {previous_session.isoformat()}"
    else:
        q = market_tape.get("active_us") or {}
        if q.get("price"):
            spot = q["price"]
            spot_dt = datetime.fromisoformat(q["time"])
        else:
            spot, spot_dt = regular_close_for_date(TICKER, previous_session)
        calc_time = now_et
        label = "Текущий срез"

    occ_oi = fetch_occ_open_interest()
    ticker = yf.Ticker(TICKER)
    expirations = list(ticker.options or [])
    if MAX_EXPIRATIONS > 0:
        expirations = expirations[:MAX_EXPIRATIONS]
    if not expirations:
        raise RuntimeError("Yahoo returned no NBIS expirations")

    raw = {}
    strike_set = set()
    occ_nonzero = 0

    for n, exp in enumerate(expirations, start=1):
        print(f"Option chain {n}/{len(expirations)} {exp}")
        chain = ticker.option_chain(exp)
        calls, puts = option_map(chain.calls), option_map(chain.puts)

        strikes = sorted(
            set(calls) | set(puts) |
            {K for (K, e) in occ_oi if e == exp}
        )

        exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(
            hour=16, tzinfo=NY
        )
        T = max(
            (exp_dt - calc_time).total_seconds() / (365*24*3600),
            1/(365*24)
        )

        for K in strikes:
            c = calls.get(K, {"yahoo_oi":0,"iv":0,"volume":0,"bid":0,"ask":0,"last":0})
            p = puts.get(K, {"yahoo_oi":0,"iv":0,"volume":0,"bid":0,"ask":0,"last":0})

            occ_call, occ_put = occ_oi.get((round(K,6), exp), (None,None))
            call_oi = c["yahoo_oi"] if occ_call is None else int(occ_call)
            put_oi = p["yahoo_oi"] if occ_put is None else int(occ_put)
            occ_nonzero += int(call_oi > 0) + int(put_oi > 0)

            cg, cv = bs_gamma_vanna(spot, K, T, c["iv"]) if c["iv"] > 0 else (0,0)
            pg, pv = bs_gamma_vanna(spot, K, T, p["iv"]) if p["iv"] > 0 else (0,0)

            # Screenshot-like VEX scale: per UNIT IV change (1.00).
            # Also store a 1-vol-point version for interpretation.
            call_vex = cv * call_oi * 100.0 * spot
            put_vex = pv * put_oi * 100.0 * spot
            call_vex_1pt = call_vex * 0.01
            put_vex_1pt = put_vex * 0.01

            # GEX as dollar gamma for a 1% spot move.
            call_gex = cg * call_oi * 100.0 * spot * spot * 0.01
            put_gex = pg * put_oi * 100.0 * spot * spot * 0.01

            total_oi = call_oi + put_oi
            iv = (
                (c["iv"]*call_oi + p["iv"]*put_oi)/total_oi
                if total_oi > 0 else max(c["iv"], p["iv"])
            )

            raw[(K,exp)] = {
                "vex": call_vex - put_vex,
                "vex_1volpt": call_vex_1pt - put_vex_1pt,
                "gex": call_gex - put_gex,
                "oi": total_oi,
                "iv": iv,
                "call": {
                    "oi": call_oi, "iv": c["iv"], "volume": c["volume"],
                    "bid": c["bid"], "ask": c["ask"], "last": c["last"],
                    "vex": call_vex, "vex_1volpt": call_vex_1pt,
                    "gex": call_gex,
                },
                "put": {
                    "oi": put_oi, "iv": p["iv"], "volume": p["volume"],
                    "bid": p["bid"], "ask": p["ask"], "last": p["last"],
                    "vex": put_vex, "vex_1volpt": put_vex_1pt,
                    "gex": put_gex,
                },
            }
            strike_set.add(K)
        time.sleep(.12)

    if occ_nonzero == 0:
        raise RuntimeError("OCC OI has no non-zero rows; refusing zero exposure snapshot")

    strikes = sorted(strike_set, reverse=True)
    rows = [{
        "strike": K,
        "values": {exp: raw.get((K,exp)) for exp in expirations}
    } for K in strikes]

    snapshot = {
        "symbol": TICKER,
        "mode": mode,
        "snapshot_label": label,
        "spot": spot,
        "spot_basis_time": iso_dt(spot_dt),
        "previous_us_session": previous_session.isoformat(),
        "oi_settlement_date": previous_session.isoformat(),
        "updated_utc": now_utc.isoformat(),
        "updated_et": now_et.isoformat(),
        "expirations": expirations,
        "strikes": strikes,
        "rows": rows,
        "diagnostics": {
            "occ_contract_sides_nonzero": occ_nonzero,
            "strike_count": len(strikes),
            "expiration_count": len(expirations),
            "min_strike": min(strikes) if strikes else None,
            "max_strike": max(strikes) if strikes else None,
        },
        "method": {
            "oi_source": "OCC Series Search — previous-day settlement",
            "iv_source": "Yahoo Finance via yfinance",
            "quotes_source": "Yahoo Finance via yfinance",
            "vex_scale": "vanna * OI * 100 * spot (per unit IV change)",
            "vex_1volpt": "vex * 0.01",
            "gex_scale": "gamma * OI * 100 * spot^2 * 0.01 (1% spot move)",
            "dealer_note": "NET sign = calls minus puts heuristic, not observed dealer positioning.",
        },
    }

    if mode == "current":
        events = add_weekly_events(snapshot)
    else:
        events_doc = read_json(EVENTS, {"events":[]}) or {"events":[]}
        events = events_doc.get("events", [])

    insights, scenario = build_structural_insights(snapshot, events)
    snapshot["insights"] = insights
    snapshot["scenario"] = scenario
    snapshot["recent_events"] = events[:30]
    return snapshot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["market_only","current","previous_close"],
        default="market_only",
    )
    args = parser.parse_args()

    tape = build_market_tape()
    if args.mode == "market_only":
        return

    snapshot = build_snapshot(args.mode, tape)
    target = PREV_CLOSE if args.mode == "previous_close" else LATEST
    write_json(target, snapshot)
    print("Wrote", target)

if __name__ == "__main__":
    main()
