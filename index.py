#!/usr/bin/env python3
"""
bot_bitunix_reco_sentiment.py
Bot Telegram: /recommend, /sentiment, /status y búsqueda rápida #coin para BTC/USDT usando Bitunix (futuros).
No ejecuta órdenes — solo sugiere.
"""

import os
import time
import math
import re
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ---------- CONFIG ----------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
BITUNIX_API_KEY = os.environ.get("BITUNIX_API_KEY", "")
BITUNIX_API_SECRET = os.environ.get("BITUNIX_API_SECRET", "")

# Endpoints
KLINE_ENDPOINT = os.environ.get("KLINE_ENDPOINT", "https://fapi.bitunix.com/api/v1/futures/market/kline")
TICKERS_ENDPOINT = os.environ.get("TICKERS_ENDPOINT", "https://fapi.bitunix.com/api/v1/futures/market/tickers")

PAIR = os.environ.get("PAIR", "BTCUSDT")   # default pair
_raw_interval = os.environ.get("INTERVAL", "1")
if _raw_interval.isdigit():
    INTERVAL = f"{_raw_interval}m"
else:
    INTERVAL = _raw_interval
KLINE_LIMIT = int(os.environ.get("KLINE_LIMIT", 200))
DEFAULT_CAPITAL = float(os.environ.get("DEFAULT_CAPITAL", 20.0))
RISK_PERCENT = float(os.environ.get("RISK_PERCENT", 1.0))
MAX_LEVERAGE = int(os.environ.get("MAX_LEVERAGE", 20))
VOLATILITY_LEVERAGE_THRESHOLD = float(os.environ.get("VOLATILITY_LEVERAGE_THRESHOLD", 0.008))

# Webhook/polling config
USE_WEBHOOK = os.environ.get(
    "USE_WEBHOOK",
    # Auto: si Render expone RENDER_EXTERNAL_URL, usa webhook
    "1" if os.environ.get("RENDER_EXTERNAL_URL") else "0",
).lower() in ("1", "true", "yes")
PORT = int(os.environ.get("PORT", "10000"))
BASE_URL = (os.environ.get("WEBHOOK_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "").rstrip("/")
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", f"/telegram/{(TELEGRAM_TOKEN or 'token')[:10]}")
if not WEBHOOK_PATH.startswith("/"):
    WEBHOOK_PATH = "/" + WEBHOOK_PATH
WEBHOOK_SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET_TOKEN")  # opcional, recomendado

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Configured INTERVAL=%s (derived from raw=%s)", INTERVAL, _raw_interval)
logger.info("Mode: %s", "WEBHOOK" if USE_WEBHOOK else "POLLING")

# ---------- RUN STATS ----------
START_TIME = time.time()
LAST_KLINE_INFO = {"ts": None, "price": None, "duration": None, "ok": False, "fetched_at": None}

# ---------- UTIL: INDICADORES ----------
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(prices: pd.Series, period: int = 14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(prices: pd.Series, a=12, b=26, c=9):
    ema_a = ema(prices, a)
    ema_b = ema(prices, b)
    macd_line = ema_a - ema_b
    signal = macd_line.ewm(span=c, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def atr(df: pd.DataFrame, period: int = 14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# ---------- FETCH KLINES (robusto) ----------
def _normalize_timestamp(raw_ts):
    try:
        raw = int(raw_ts)
    except Exception:
        return int(time.time())
    if raw >= 10**12:
        return int(raw // 1000)
    if raw >= 10**9:
        return int(raw)
    return int(time.time())

def _parse_data_items(data_items):
    rows = []
    for item in data_items:
        try:
            if isinstance(item, (list, tuple)):
                if len(item) >= 6:
                    first = int(item[0])
                    if first > 10**6:
                        ts = _normalize_timestamp(item[0])
                        o, h, l, c = float(item[1]), float(item[2]), float(item[3]), float(item[4])
                        v = float(item[5])
                    else:
                        o, h, c, l = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                        ts = _normalize_timestamp(item[4])
                        v = float(item[5])
                else:
                    continue
            elif isinstance(item, dict):
                raw_ts = item.get("time") or item.get("ts") or item.get("id")
                ts = _normalize_timestamp(raw_ts)
                o = float(item.get("open", item.get("o", 0)))
                h = float(item.get("high", item.get("h", 0)))
                l = float(item.get("low", item.get("l", 0)))
                c = float(item.get("close", item.get("c", 0)))
                v = 0.0
                if item.get("baseVol") is not None:
                    try: v = float(item.get("baseVol"))
                    except: v = 0.0
                elif item.get("quoteVol") is not None:
                    try: v = float(item.get("quoteVol"))
                    except: v = 0.0
                else:
                    try: v = float(item.get("volume", 0.0))
                    except: v = 0.0
            else:
                continue

            rows.append({"ts": int(ts), "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v)})
        except Exception:
            continue
    return rows

def fetch_klines(symbol: str = PAIR, interval: str = INTERVAL, limit: int = KLINE_LIMIT):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    start = time.time()
    try:
        resp = requests.get(KLINE_ENDPOINT, params=params, timeout=10)
        duration = time.time() - start
        resp.raise_for_status()
        j = resp.json()
        data = None
        if isinstance(j, dict) and "data" in j:
            data = j.get("data")
        elif isinstance(j, dict) and "result" in j:
            data = j.get("result")
        else:
            data = j if isinstance(j, (list, tuple)) else None

        if isinstance(data, dict):
            data = [data]

        if not data:
            # fallback limit=1
            try:
                fallback = requests.get(KLINE_ENDPOINT, params={"symbol": symbol, "interval": interval, "limit": 1}, timeout=6)
                fallback.raise_for_status()
                jf = fallback.json()
                data_f = jf.get("data") if isinstance(jf, dict) else jf
                if isinstance(data_f, dict):
                    data_f = [data_f]
                data = data_f
                duration = round(duration + (time.time() - start), 4)
            except Exception as e:
                logger.exception("fetch_klines fallback failed: %s", e)
                LAST_KLINE_INFO.update({"ok": False, "duration": round(duration,4), "fetched_at": datetime.now(timezone.utc).isoformat()})
                return None

        if not data:
            LAST_KLINE_INFO.update({"ok": False, "duration": round(duration,4), "fetched_at": datetime.now(timezone.utc).isoformat()})
            return None

        rows = _parse_data_items(data)
        if not rows:
            logger.warning("fetch_klines: no rows parsed; sample JSON: %s", str(j)[:300])
            LAST_KLINE_INFO.update({"ok": False, "duration": round(duration,4), "fetched_at": datetime.now(timezone.utc).isoformat()})
            return None

        df = pd.DataFrame(rows)
        df['dt'] = pd.to_datetime(df['ts'], unit='s')
        df = df.sort_values('ts').reset_index(drop=True)
        last_ts = int(df['ts'].iloc[-1])
        last_price = float(df['close'].iloc[-1])
        LAST_KLINE_INFO.update({"ts": last_ts, "price": last_price, "duration": round(duration, 4), "ok": True, "fetched_at": datetime.now(timezone.utc).isoformat()})
        return df
    except Exception as e:
        duration = time.time() - start
        logger.exception("Error fetching klines: %s", e)
        LAST_KLINE_INFO.update({"ok": False, "duration": round(duration,4), "fetched_at": datetime.now(timezone.utc).isoformat()})
        return None

# ---------- ANALYZE & RECOMMEND ----------
def analyze_and_recommend(df: pd.DataFrame, capital_usdt: float = DEFAULT_CAPITAL, risk_pct: float = RISK_PERCENT):
    close = df['close']
    ema_fast = ema(close, span=8).iloc[-1]
    ema_slow = ema(close, span=21).iloc[-1]
    rsi_val = rsi(close, 14).iloc[-1]
    macd_line, macd_signal, macd_hist = macd(close)
    macd_hist_val = macd_hist.iloc[-1]
    atr_val = atr(df, 14).iloc[-1]
    last_price = float(close.iloc[-1])

    long_score = 0
    short_score = 0
    if ema_fast > ema_slow:
        long_score += 1
    else:
        short_score += 1
    if macd_hist_val > 0:
        long_score += 1
    else:
        short_score += 1
    if rsi_val < 30:
        long_score += 1
    elif rsi_val > 70:
        short_score += 1

    side = "LONG" if long_score > short_score else "SHORT" if short_score > long_score else "NEUTRAL"
    entry = last_price

    sl_distance = 1.5 * atr_val
    if side == "LONG":
        stop = max(0.0001, entry - sl_distance)
        tp = entry + 2 * sl_distance
    elif side == "SHORT":
        stop = min(entry + sl_distance, entry * 1.5)
        tp = entry - 2 * sl_distance
    else:
        stop, tp = None, None

    vol_ratio = atr_val / entry
    suggested_leverage = min(10, MAX_LEVERAGE) if vol_ratio > VOLATILITY_LEVERAGE_THRESHOLD else min(20, MAX_LEVERAGE)

    risk_amount = capital_usdt * (risk_pct / 100.0)
    if stop and entry:
        per_contract_risk = (entry - stop) if side == "LONG" else (stop - entry)
        per_contract_risk = max(per_contract_risk, 1e-8)
        position_notional = (risk_amount * (entry / per_contract_risk))
        position_notional = max(0.0, position_notional)
    else:
        position_notional = 0.0

    rationale = f"EMA8={ema_fast:.1f} EMA21={ema_slow:.1f} MACD_hist={macd_hist_val:.6f} RSI={rsi_val:.1f} ATR={atr_val:.3f} vol_ratio={vol_ratio:.6f}"

    return {
        "pair": PAIR,
        "side": side,
        "entry": entry,
        "stop": stop,
        "take_profit": tp,
        "leverage": int(suggested_leverage),
        "position_notional_usdt": round(position_notional, 4),
        "risk_amount_usdt": round(risk_amount, 4),
        "rationale": rationale,
        "indicator_values": {"ema_fast": ema_fast, "ema_slow": ema_slow, "macd_hist": macd_hist_val, "rsi": rsi_val, "atr": atr_val}
    }

# ---------- SENTIMENT (sin noticias) ----------
analyzer = SentimentIntensityAnalyzer()

def compute_sentiment_from_market(df: pd.DataFrame):
    close = df['close']
    last = float(close.iloc[-1])
    close_5 = float(close.iloc[-5]) if len(close) >= 5 else last
    pct_5 = (last - close_5) / close_5 if close_5 else 0.0

    ema_fast = ema(close, span=8).iloc[-1]
    ema_slow = ema(close, span=21).iloc[-1]
    rsi_val = rsi(close, 14).iloc[-1]
    macd_line, macd_signal, macd_hist = macd(close)
    macd_hist_val = macd_hist.iloc[-1]

    score = 0.0
    if ema_fast > ema_slow: score += 0.4
    else: score -= 0.4
    if macd_hist_val > 0: score += 0.3
    else: score -= 0.3
    if rsi_val < 30: score += 0.2
    elif rsi_val > 70: score -= 0.2
    if pct_5 > 0.002: score += 0.15
    elif pct_5 < -0.002: score -= 0.15

    score = max(-1.0, min(1.0, score))
    summary = f"Precio actual {last:.2f}, cambio corto {pct_5*100:+.2f}%, RSI {rsi_val:.1f}, MACD_hist {macd_hist_val:.6f}"
    v = analyzer.polarity_scores(summary)['compound']
    combined = 0.7 * score + 0.3 * v
    label = "Positivo" if combined > 0.1 else ("Negativo" if combined < -0.1 else "Neutro")
    return {"label": label, "score": round(combined, 4), "summary": summary, "heuristic": round(score,4), "vader_compound": round(v,4)}

# ---------- UTILIDADES para #coin ----------
def format_money(value):
    try:
        return f"{value:,.4f}"
    except Exception:
        return str(value)

def format_vol_m(value):
    try:
        m = float(value) / 1_000_000.0
        return f"{m:,.2f}M"
    except Exception:
        return str(value)

def fetch_ticker_for_symbol(symbol: str):
    """
    Llama al endpoint /tickers?symbols=SYMBOL y devuelve dict con last, high, low, open, quoteVol, baseVol
    """
    try:
        resp = requests.get(TICKERS_ENDPOINT, params={"symbols": symbol}, timeout=6)
        resp.raise_for_status()
        j = resp.json()
        data = j.get("data") if isinstance(j, dict) else j
        if isinstance(data, list):
            # buscar objeto con 'symbol' igual
            for item in data:
                if item.get("symbol", "").upper() == symbol.upper():
                    return item
            return data[0] if data else None
        elif isinstance(data, dict):
            return data
        return None
    except Exception as e:
        logger.debug("fetch_ticker_for_symbol error: %s", e)
        return None

def fetch_24h_summary_from_ticker(symbol: str):
    item = fetch_ticker_for_symbol(symbol)
    if not item:
        return None
    try:
        last = float(item.get("lastPrice") or item.get("last") or item.get("lastPrice") or item.get("last") or 0.0)
    except:
        last = None
    try:
        high = float(item.get("high") or item.get("highPrice") or 0.0)
    except:
        high = None
    try:
        low = float(item.get("low") or item.get("lowPrice") or 0.0)
    except:
        low = None
    try:
        open_p = float(item.get("open") or item.get("openPrice") or 0.0)
    except:
        open_p = None
    # volumes
    quoteVol = item.get("quoteVol")
    baseVol = item.get("baseVol")
    try:
        qv = float(quoteVol) if quoteVol is not None else None
    except:
        qv = None
    try:
        bv = float(baseVol) if baseVol is not None else None
    except:
        bv = None

    # compute change pct from open if possible
    change_pct = None
    if open_p and last:
        try:
            change_pct = ((last - open_p) / open_p) * 100.0
        except:
            change_pct = None

    # if quoteVol missing but have baseVol and last -> estimate quote vol
    if qv is None and bv is not None and last is not None:
        qv = bv * last

    return {"last": last, "high": high, "low": low, "change_pct": change_pct, "volume_quote": qv}

def fetch_24h_summary(symbol: str):
    # 1) Try ticker endpoint
    t = fetch_24h_summary_from_ticker(symbol)
    if t:
        return t
    # 2) fallback: compute from klines (last 24h)
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 24 * 3600 * 1000
        params = {"symbol": symbol, "interval": INTERVAL, "startTime": start_ms, "endTime": now_ms, "limit": 1000}
        resp = requests.get(KLINE_ENDPOINT, params=params, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        data = j.get("data") if isinstance(j, dict) else j
        if not data:
            df = fetch_klines(symbol=symbol, interval=INTERVAL, limit=KLINE_LIMIT)
        else:
            rows = _parse_data_items(data)
            df = pd.DataFrame(rows)
            if df.empty:
                df = fetch_klines(symbol=symbol, interval=INTERVAL, limit=KLINE_LIMIT)
        if df is None or df.empty:
            return None
        high = float(df['high'].max())
        low = float(df['low'].min())
        last = float(df['close'].iloc[-1])
        first = float(df['close'].iloc[0])
        change_pct = ((last - first) / first) * 100.0 if first != 0 else None
        volume_quote = float(df['volume'].sum()) * last if 'volume' in df.columns else None
        return {"last": last, "high": high, "low": low, "change_pct": change_pct, "volume_quote": volume_quote}
    except Exception:
        return None

# ---------- HELPERS ----------
def format_uptime(seconds: float) -> str:
    sec = int(seconds)
    days, sec = divmod(sec, 86400)
    hours, sec = divmod(sec, 3600)
    minutes, sec = divmod(sec, 60)
    parts = []
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    parts.append(f"{sec}s")
    return " ".join(parts)

def format_coin_summary_message(coin: str, summary: dict) -> str:
    """Formatea el mensaje del resumen de la moneda"""
    last = summary.get("last")
    high = summary.get("high")
    low = summary.get("low")
    change = summary.get("change_pct")
    vol = summary.get("volume_quote") or summary.get("volume") or summary.get("volume_quote")

    change_sign = "🔴" if change is not None and float(change) < 0 else ("🟢" if change is not None and float(change) > 0 else "➖")
    change_str = f"{change:+.2f}%" if change is not None else "N/A"
    vol_str = format_vol_m(vol) + " USDT" if vol is not None else "N/A"

    msg_lines = [
        f"Coin: {coin}",
        f"💰Price:  {format_money(last)} USDT" if last is not None else "💰Price: N/A",
        f"⬆️High（24H）:  {format_money(high)}" if high is not None else "⬆️High（24H）: N/A",
        f"⬇️Low（24H）:  {format_money(low)}" if low is not None else "⬇️Low（24H）: N/A",
        f"🔃Change（24H）:  {change_str} {change_sign}",
        f"📊Vol（24H）:  {vol_str}"
    ]
    return "\n".join(msg_lines)

# ---------- TELEGRAM HANDLERS ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = ("Hola 👋\nBot de recomendaciones BTC/USDT (solo SUGERENCIAS).\n"
           "Comandos:\n"
           "/recommend [capital] [risk%]  - Obtener recomendación de trading\n"
           "/sentiment                    - Obtener sentimiento de mercado (sin noticias)\n"
           "/status                       - Estado del bot y conexión a Bitunix\n"
           "También puedes enviar: #btc  o #eth  para obtener resumen 24H.\n\n"
           "Ejemplo: /recommend 50 1")
    await update.message.reply_text(txt)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

async def recommend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Obteniendo datos y analizando... (unos segundos)")
    args = context.args
    try:
        capital = float(args[0]) if len(args) >= 1 else DEFAULT_CAPITAL
    except:
        capital = DEFAULT_CAPITAL
    try:
        risk = float(args[1]) if len(args) >= 2 else RISK_PERCENT
    except:
        risk = RISK_PERCENT

    df = fetch_klines(symbol=PAIR, interval=INTERVAL, limit=KLINE_LIMIT)
    if df is None or df.empty:
        await update.message.reply_text("Error: no pude leer velas desde Bitunix. Revisa KLINE_ENDPOINT y conectividad.")
        return

    rec = analyze_and_recommend(df, capital_usdt=capital, risk_pct=risk)
    lines = [
        f"📌 *Recomendación para* `{rec['pair']}`",
        f"*Lado:* {rec['side']}",
        f"*Entrada:* `{rec['entry']:.2f}` USDT",
        f"*Stop Loss:* `{rec['stop']:.2f}` USDT" if rec['stop'] else "Stop Loss: N/A",
        f"*Take Profit:* `{rec['take_profit']:.2f}` USDT" if rec['take_profit'] else "Take Profit: N/A",
        f"*Apalancamiento sugerido:* `{rec['leverage']}x`",
        f"*Tamaño notional sugerido:* `{rec['position_notional_usdt']}` USDT (riesgo ≈ `{rec['risk_amount_usdt']}` USDT)",
        f"*Razonamiento:* `{rec['rationale']}`",
        "",
        "⚠️ Esto es solo una SUGERENCIA. Revisa todo antes de operar. El bot NO ejecuta órdenes."
    ]
    await update.message.reply_text("\n".join(lines), parse_mode='Markdown')

async def sentiment_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Calculando sentimiento a partir de datos de mercado (sin titulares)...")
    df = fetch_klines(symbol=PAIR, interval=INTERVAL, limit=KLINE_LIMIT)
    if df is None or df.empty:
        await update.message.reply_text("Error: no pude leer velas desde Bitunix.")
        return
    sent = compute_sentiment_from_market(df)
    emoji = "📈" if sent['label']=="Positivo" else ("📉" if sent['label']=="Negativo" else "➖")
    text = (f"{emoji} *Sentimiento BTC/USDT:* *{sent['label']}*\n"
            f"*Score:* `{sent['score']}`  (heur:{sent['heuristic']}, vader:{sent['vader_compound']})\n"
            f"*Resumen:* `{sent['summary']}`\n\n"
            "⚠️ Esto es una estimación automática basada en indicadores técnicos y heurística. No son titulares ni análisis humano.")
    await update.message.reply_text(text, parse_mode='Markdown')

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime = format_uptime(time.time() - START_TIME)
    if LAST_KLINE_INFO.get("ts"):
        last_dt = datetime.utcfromtimestamp(LAST_KLINE_INFO["ts"]).strftime("%Y-%m-%d %H:%M:%S UTC")
        last_price = LAST_KLINE_INFO.get("price", "N/A")
        last_duration = LAST_KLINE_INFO.get("duration", "N/A")
        fetched_at = LAST_KLINE_INFO.get("fetched_at", "N/A")
        last_ok = LAST_KLINE_INFO.get("ok", False)
    else:
        last_dt, last_price, last_duration, fetched_at, last_ok = "N/A", "N/A", "N/A", "N/A", False

    ping_result = "N/A"
    try:
        ping_start = time.time()
        r = requests.get(KLINE_ENDPOINT, params={"symbol": PAIR, "interval": INTERVAL, "limit": 1}, timeout=5)
        ping_duration = round(time.time() - ping_start, 4)
        status_code = r.status_code
        ping_result = f"{ping_duration}s (HTTP {status_code})"
    except Exception as e:
        ping_result = f"error: {str(e)}"

    text = (
        f"🛰 *Estado del bot*\n"
        f"*Uptime:* `{uptime}`\n\n"
        f"🧾 *Última vela conocida:*\n"
        f"- Fecha (última vela): `{last_dt}`\n"
        f"- Precio (close): `{last_price}`\n"
        f"- Duración última fetch: `{last_duration}` s\n"
        f"- Última fetch exitosa: `{last_ok}` (fetched_at: `{fetched_at}`)\n\n"
        f"🔗 *Ping al endpoint KLINE:* `{ping_result}`\n\n"
        f"🔧 *Configuración:* pair=`{PAIR}` interval=`{INTERVAL}` limit=`{KLINE_LIMIT}`\n"
        f"⚠️ Nota: /recommend y /sentiment usan los mismos datos. Si ves errores, revisa KLINE_ENDPOINT y las variables de entorno."
    )
    await update.message.reply_text(text, parse_mode='Markdown')

# ---------- HANDLER para mensajes que empiezan con #coin ----------
COIN_RE = re.compile(r'(?i)^#([a-z]{2,9})\b')  # captura 2-6 letras después de '#'

async def coin_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    m = COIN_RE.match(txt)
    if not m:
        return
    coin = m.group(1).upper()
    symbol = f"{coin}USDT"
    await update.message.reply_text(f"Obteniendo resumen 24H para {symbol}...")

    summary = fetch_24h_summary(symbol)
    if not summary:
        await update.message.reply_text("Lo siento, no pude obtener el resumen 24H ahora. Revisa KLINE/TICKER endpoints.")
        return

    # Crear el botón de actualizar
    keyboard = [[InlineKeyboardButton("🔄 Actualizar", callback_data=f"update_coin_{coin}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Formatear y enviar el mensaje con el botón
    message_text = format_coin_summary_message(coin, summary)
    await update.message.reply_text(message_text, reply_markup=reply_markup)

# ---------- CALLBACK HANDLER para el botón de actualizar ----------
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Confirmar que se recibió el callback
    
    # Extraer el coin del callback_data
    if query.data.startswith("update_coin_"):
        coin = query.data.replace("update_coin_", "")
        symbol = f"{coin}USDT"
        
        # Mostrar mensaje de carga (editando el mensaje original)
        await query.edit_message_text(f"🔄 Actualizando resumen 24H para {symbol}...")
        
        # Obtener nuevos datos
        summary = fetch_24h_summary(symbol)
        if not summary:
            await query.edit_message_text("❌ Error: no pude obtener el resumen 24H actualizado. Inténtalo más tarde.")
            return
        
        # Crear el botón de actualizar nuevamente
        keyboard = [[InlineKeyboardButton("🔄 Actualizar", callback_data=f"update_coin_{coin}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Formatear y actualizar el mensaje
        message_text = format_coin_summary_message(coin, summary)
        # Agregar timestamp de actualización
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        message_text += f"\n\n🕐 Actualizado: {now}"
        
        await query.edit_message_text(message_text, reply_markup=reply_markup)

# ---------- MAIN ----------
def build_app():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN is required")
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("recommend", recommend))
    app.add_handler(CommandHandler("sentiment", sentiment_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.Regex(COIN_RE), coin_message_handler))
    # Agregar el handler para los botones inline
    app.add_handler(CallbackQueryHandler(button_callback))
    return app

async def set_webhook(app: Application):
    """Set up webhook for the application"""
    try:
        full_url = f"{BASE_URL}{WEBHOOK_PATH}"
        logger.info(f"Setting webhook URL: {full_url}")
        await app.bot.set_webhook(
            url=full_url,
            secret_token=WEBHOOK_SECRET_TOKEN,
            drop_pending_updates=True
        )
        logger.info("Webhook set successfully")
    except Exception as e:
        logger.error(f"Failed to set webhook: {e}")
        raise

def main():
    if TELEGRAM_TOKEN is None:
        print("ERROR: define TELEGRAM_TOKEN en las variables de entorno o .env")
        return

    app = build_app()

    if USE_WEBHOOK:
        if not BASE_URL:
            print("ERROR: establece WEBHOOK_BASE_URL o deja que Render ponga RENDER_EXTERNAL_URL")
            return
        full_url = f"{BASE_URL}{WEBHOOK_PATH}"
        logger.info("Iniciando webhook. listen=0.0.0.0 port=%s path=%s url=%s", PORT, WEBHOOK_PATH, full_url)
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=WEBHOOK_PATH.lstrip("/"),
            webhook_url=full_url,
            secret_token=WEBHOOK_SECRET_TOKEN,
            drop_pending_updates=True,
        )
    else:
        logger.info("Iniciando en modo polling (desarrollo local)")
        app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
