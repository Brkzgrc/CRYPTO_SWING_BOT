import os
import sys
import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import customtkinter as ctk
from tkinter import messagebox
import mplfinance as mpf
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from fpdf import FPDF
import screeninfo
import threading
import customtkinter as ctk
from tkinter import messagebox
import ccxt

# Terminal başlığına dosya adını yaz (sadece Windows'ta çalışır)
if os.name == 'nt':
    os.system(f"title {os.path.basename(sys.argv[0])}")

# Terminal ekranında ilk satıra dosya adını yaz
print("="*60)
print(f" ÇALIŞTIRILAN DOSYA: {os.path.basename(sys.argv[0])}")
print("="*60)
print()

def get_binance_usdt_symbols():
    """
    Binance Global'de aktif SPOT USDT çiftlerini döndürür.
    Stablecoin ve fiat coinlerini hariç tutar.
    """
    STABLES = {'USDT', 'BUSD', 'USDC', 'TUSD', 'DAI', 'PAX', 'HUSD', 'USDP', 'EURI', 'FDUSD', 'XUSD'}
    FIATS   = {'EUR', 'USD', 'TRY', 'GBP', 'AUD', 'BRL', 'RUB', 'NGN', 'UAH', 'CHF', 'JPY', 'KRW', 'IDR', 'ZAR'}

    # Zorunlu olarak spot piyasayı kullanıyoruz
    exchange = ccxt.binance({
        'options': {'defaultType': 'spot'}
    })

    # fetch_markets(), her pazar yeri için active/type/info.status bilgisini getirir
    markets = exchange.fetch_markets()
    symbols = []

    for m in markets:
        # Sadece:
        #  - quote == 'USDT'
        #  - aktif (m['active'] == True)
        #  - spot (m.get('contract', False) == False)
        #  - raw bilgi statüsü 'TRADING'
        if (m['quote'] == 'USDT'
                and m.get('active', False)
                and not m.get('contract', False)
                and m.get('info', {}).get('status') == 'TRADING'):
            base = m['base']
            if base not in STABLES and base not in FIATS:
                symbols.append(base)

    return sorted(set(symbols))

ANALYSIS_THRESHOLDS = {
    "POC_DIST": 0.005,
    "LVN_HVN_DIST": 0.005,
    "ADX": 20,
    "RSI_LONG_MAX": 60,
    "RSI_SHORT_MIN": 30,
}

class GuiLogger:
    def __init__(self, gui_instance):
        self.gui = gui_instance

    def write(self, msg):
        if msg.strip() != "":
            try:
                if hasattr(self.gui, "log_message"):
                    self.gui.log_message(msg.strip())
            except Exception as e:
                # Sessizce geç veya debug için aç: print(f"Logger error: {e}")
                pass
            try:
                sys.__stdout__.write(msg)
            except Exception:
                pass

    def flush(self):
        pass

# ------------------------
# Configuration
# ------------------------
THRESHOLDS = {
    "15m": {"RSI_LONG_MAX": 70, "RSI_SHORT_MIN": 30, "ADX": 20},
    "1h":  {"RSI_LONG_MAX": 65, "RSI_SHORT_MIN": 35, "ADX": 20},
    "4h":  {"RSI_LONG_MAX": 60, "RSI_SHORT_MIN": 35, "ADX": 25},
    "1d":  {"RSI_LONG_MAX": 60, "RSI_SHORT_MIN": 35, "ADX": 25},
}

# ------------------------
# ▶️ Sistem Ayarları
# ------------------------

ACTIVE_TIMEFRAMES = ["15m", "1h", "4h", "1d"]  # Kullanmak istemediklerini çıkarabilirsin

ENABLE_RISK_MANAGEMENT = False
ENABLE_REPORT = False
ENABLE_TELEGRAM = False
ENABLE_DISCORD = False

# ------------------------
# 💰 Risk Yönetimi
# ------------------------

ACCOUNT_BALANCE = 1000
RISK_PERCENTAGE = 0.01

# ------------------------
# 📣 Telegram Ayarları
# ------------------------

TELEGRAM_BOT_TOKEN = "your-bot-token"
TELEGRAM_CHAT_ID = "your-chat-id"

# ------------------------
# 📣 Discord Ayarları
# ------------------------

DISCORD_WEBHOOK_URL = "https://your-discord-webhook-url"


# ------------------------
# Data Fetching
# ------------------------

def convert_to_dataframe(ohlcv):
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def fetch_ohlcv(symbol="AVAX/USDT", timeframes=["15m", "1h", "4h", "1d"], limit=500):
    exchange = ccxt.binance()
    market_data = {}

    for tf in timeframes:
        print(f"📊 Veriler çekiliyor: {symbol} [{tf}]...")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = convert_to_dataframe(ohlcv)
            market_data[tf] = df
        except Exception as e:
            print(f"❌ Hata [{tf}]:", e)

    return market_data


# ------------------------
# Market Structure Detection
# ------------------------

def identify_swings(df, window=3):
    highs = df['high']
    lows = df['low']
    df['swing_high'] = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    df['swing_low'] = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
    return df


def detect_trend(df):
    swings = identify_swings(df.copy())
    swings = swings.dropna(subset=['swing_high', 'swing_low'])

    if swings['swing_high'].dropna().empty or swings['swing_low'].dropna().empty:
        return "Unknown", swings

    last_highs = swings['swing_high'].dropna().tail(2).values
    last_lows = swings['swing_low'].dropna().tail(2).values

    if len(last_highs) == 2 and last_highs[1] > last_highs[0]:
        if len(last_lows) == 2 and last_lows[1] > last_lows[0]:
            return "Uptrend", swings

    if len(last_highs) == 2 and last_highs[1] < last_highs[0]:
        if len(last_lows) == 2 and last_lows[1] < last_lows[0]:
            return "Downtrend", swings

    return "Range", swings


# ------------------------
# Liquidity
# ------------------------

def find_equal_highs_lows(df, precision=0.2, window=5):
    df = df.copy()
    highs = df['high'].round(2)
    lows = df['low'].round(2)

    df['equal_high'] = False
    df['equal_low'] = False

    for i in range(window, len(df)):
        high_window = highs.iloc[i-window:i]
        low_window = lows.iloc[i-window:i]
        if any(abs(high_window - highs.iloc[i]) <= precision):
            df.loc[df.index[i], 'equal_high'] = True
        if any(abs(low_window - lows.iloc[i]) <= precision):
            df.loc[df.index[i], 'equal_low'] = True

    return df


def detect_liquidity_sweep(df):
    df = df.copy()
    df['sweep_high'] = False
    df['sweep_low'] = False

    for i in range(2, len(df)):
        if df['high'].iloc[i] > df['high'].iloc[i-1] and df['equal_high'].iloc[i-1]:
            df.loc[df.index[i], 'sweep_high'] = True

        if df['low'].iloc[i] < df['low'].iloc[i-1] and df['equal_low'].iloc[i-1]:
            df.loc[df.index[i], 'sweep_low'] = True

    return df


# ------------------------
# Technical Indicators
# ------------------------

# ADX ve Supertrend hesaplama, calculate_indicators fonksiyonuna eklenmeli

def calculate_indicators(df):
    df = df.copy()
    df['OBV'] = ta.obv(close=df['close'], volume=df['volume'])
    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['ATR'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df = calculate_vsa(df)
    df = calculate_cvd(df)
    # ADX hesapla
    adx = ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14)
    if adx is not None and isinstance(adx, pd.DataFrame) and 'ADX_14' in adx.columns:
        df['ADX'] = adx['ADX_14']
    else:
        df['ADX'] = 0  # Ya da np.nan
    # Supertrend hesapla
    try:
        supertrend = ta.supertrend(high=df['high'], low=df['low'], close=df['close'], length=10, multiplier=3.0)
        if supertrend is not None and isinstance(supertrend, pd.DataFrame):
            df['supertrend'] = supertrend['SUPERT_10_3.0']
            df['supertrend_direction'] = supertrend['SUPERTd_10_3.0']
        else:
            df['supertrend'] = 0
            df['supertrend_direction'] = 0
    except Exception:
        df['supertrend'] = 0
        df['supertrend_direction'] = 0
    return df

# ------------------------
# Volume Profile (POC, HVN, LVN) Calculation
# ------------------------

def calculate_volume_profile(df, bins=30):
    """
    Volume profile: Hangi fiyat aralığında ne kadar hacim birikmiş?
    bins: Kaç aralıkta böleceği (varsayılan: 30)
    """
    df = df.copy()
    price_bins = pd.cut(df['close'], bins=bins)
    volume_profile = df.groupby(price_bins)['volume'].sum()
    poc_bin = volume_profile.idxmax()
    poc_price = poc_bin.mid
    # HVN: POC çevresinde yüksek hacimli bölgeler (ilk 3 en yüksek hacim)
    hvn_bins = volume_profile.sort_values(ascending=False).head(3).index
    hvn_prices = [b.mid for b in hvn_bins]
    # LVN: En düşük 3 hacimli bölge
    lvn_bins = volume_profile.sort_values(ascending=True).head(3).index
    lvn_prices = [b.mid for b in lvn_bins]
    return {
        "POC": poc_price,
        "HVN": hvn_prices,
        "LVN": lvn_prices,
        "volume_profile": volume_profile
    }

# ------------------------
# Divergence Detection
# ------------------------

def detect_divergence(df, indicator='OBV', lookback=10, kind='bullish'):
    """
    Pozitif (bullish) veya negatif (bearish) diverjansı otomatik tespit eder.
    Fiyat ve seçili indikatör için:
      - Pozitif diverjans: Fiyat lower low, indikatör higher low (dipte boğa uyumsuzluğu)
      - Negatif diverjans: Fiyat higher high, indikatör lower high (tepede ayı uyumsuzluğu)
    """
    df = df.copy()
    recent = df.iloc[-lookback:]

    if kind == 'bullish':
        # Fiyat yeni düşük yapıyor mu?
        price_low = recent['low'].values
        indi_low = recent[indicator].values
        if price_low[-1] < price_low[0] and indi_low[-1] > indi_low[0]:
            return True
    elif kind == 'bearish':
        price_high = recent['high'].values
        indi_high = recent[indicator].values
        if price_high[-1] > price_high[0] and indi_high[-1] < indi_high[0]:
            return True
    return False

# ------------------------
# Volume Spread Analysis (VSA)
# ------------------------

def calculate_vsa(df):
    """
    VSA barlarının tipini tespit eder:
      - 'no_demand': Düşük hacim + küçük gövdeli yeşil bar (alıcı ilgisi zayıf)
      - 'no_supply': Düşük hacim + küçük gövdeli kırmızı bar (satıcı ilgisi zayıf)
      - 'buying_climax': Çok yüksek hacim + uzun fitilli, gövdesi küçük yeşil bar (zirve riskli)
      - 'selling_climax': Çok yüksek hacim + uzun fitilli, gövdesi küçük kırmızı bar (dip riskli)
      - 'normal': Diğer barlar
    """
    df = df.copy()
    vsa_types = []
    volumes = df['volume']
    closes = df['close']
    opens = df['open']
    highs = df['high']
    lows = df['low']
    median_vol = volumes.rolling(20).median()
    vol_threshold = volumes.rolling(20).quantile(0.8)
    body = (closes - opens).abs()
    bar_range = highs - lows

    for i in range(len(df)):
        # Yetersiz geçmiş varsa 'normal'
        if i < 21:
            vsa_types.append('normal')
            continue

        # Temel ölçümler
        v = volumes.iloc[i]
        med = median_vol.iloc[i]
        thresh = vol_threshold.iloc[i]
        b = body.iloc[i]
        r = bar_range.iloc[i]

        # Yüksek hacimli bar mı?
        is_high_vol = v > thresh
        # Düşük hacimli bar mı?
        is_low_vol = v < med * 0.7

        # Küçük gövde barı mı? (doygun olmayan hareket)
        is_small_body = b < r * 0.33

        # Fitil uzunluğu (doji tipi)
        upper_wick = highs.iloc[i] - max(closes.iloc[i], opens.iloc[i])
        lower_wick = min(closes.iloc[i], opens.iloc[i]) - lows.iloc[i]
        is_long_upper_wick = upper_wick > r * 0.4
        is_long_lower_wick = lower_wick > r * 0.4

        # Bar tipi (yeşil/kırmızı)
        is_bull = closes.iloc[i] > opens.iloc[i]
        is_bear = closes.iloc[i] < opens.iloc[i]

        # VSA Sinyalleri
        if is_low_vol and is_small_body and is_bull:
            vsa_types.append('no_demand')
        elif is_low_vol and is_small_body and is_bear:
            vsa_types.append('no_supply')
        elif is_high_vol and is_small_body and is_long_upper_wick and is_bull:
            vsa_types.append('buying_climax')
        elif is_high_vol and is_small_body and is_long_lower_wick and is_bear:
            vsa_types.append('selling_climax')
        else:
            vsa_types.append('normal')
    df['vsa_type'] = vsa_types
    return df

# ------------------------
# Aggressive Volume & CVD (Delta) Calculation
# ------------------------

def calculate_cvd(df):
    """
    CVD (Cumulative Volume Delta) ve bar bazlı delta hesaplar.
    Up_volume: Close > Open olan barlarda hacim.
    Down_volume: Close < Open olan barlarda hacim.
    Delta: Up_volume - Down_volume
    CVD: Kümülatif Delta
    """
    df = df.copy()
    df['up_volume'] = df.apply(lambda row: row['volume'] if row['close'] > row['open'] else 0, axis=1)
    df['down_volume'] = df.apply(lambda row: row['volume'] if row['close'] < row['open'] else 0, axis=1)
    df['delta'] = df['up_volume'] - df['down_volume']
    df['CVD'] = df['delta'].cumsum()
    return df

# ------------------------
# Setup Detection
# ------------------------

def detect_long_setup(df, trend, tf):
    thresholds = THRESHOLDS[tf]
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    # --- VSA filtresi
    if last_row.get("vsa_type", "normal") in ["buying_climax", "no_demand"]:
        return None
    # --- CVD filtresi
    if last_row.get("CVD", 0) <= prev_row.get("CVD", 0):
        return None
    # --- ADX ve RSI EŞİKLERİ ---
    if last_row.get("ADX", 0) < thresholds["ADX"]:
        return None
    # LONG için RSI çok yüksekse (ör. 70 üzeri) girmeyeceğiz!
    if last_row.get("RSI", 0) >= thresholds["RSI_LONG_MAX"]:
        return None
    # --- Supertrend filtresi
    if last_row.get("supertrend_direction", 0) != 1:
        return None
    setup_conditions = [
        trend == "Uptrend",
        last_row.get("sweep_low", False) == True,
        last_row.get("OBV", 0) > prev_row['OBV']
    ]
    if all(setup_conditions):
        reason = "Uptrend + Liquidity Sweep + OBV + RSI + VSA + CVD + ADX + Supertrend Uyumlu"
        if detect_divergence(df, indicator='OBV', lookback=10, kind='bullish'):
            reason += " + Pozitif Diverjans"
        if detect_divergence(df, indicator='OBV', lookback=10, kind='bearish'):
            reason += " ⚠️ Negatif Diverjans (Uyumsuzluk)"
        return {
            "signal": "LONG",
            "reason": reason,
            "entry": last_row['close'],
            "sl": df['low'].iloc[-2],
            "tp": last_row['close'] + (last_row['close'] - df['low'].iloc[-2]) * 2
        }
    return None

def detect_short_setup(df, trend, tf):
    thresholds = THRESHOLDS[tf]
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    # --- VSA filtresi
    if last_row.get("vsa_type", "normal") in ["selling_climax", "no_supply"]:
        return None
    # --- CVD filtresi
    if last_row.get("CVD", 0) >= prev_row.get("CVD", 0):
        return None
    # --- ADX ve RSI EŞİKLERİ ---
    if last_row.get("ADX", 0) < thresholds["ADX"]:
        return None
    # SHORT için RSI düşükse (örn. 30 ve altı) short'a girmemeli!
    if last_row.get("RSI", 0) <= thresholds["RSI_SHORT_MIN"]:
        return None
    # --- Supertrend filtresi
    if last_row.get("supertrend_direction", 0) != -1:
        return None
    setup_conditions = [
        trend == "Downtrend",
        last_row.get("sweep_high", False) == True,
        last_row.get("OBV", 0) < prev_row['OBV']
    ]
    if all(setup_conditions):
        reason = "Downtrend + Liquidity Sweep + OBV + RSI + VSA + CVD + ADX + Supertrend Uyumlu"
        if detect_divergence(df, indicator='OBV', lookback=10, kind='bearish'):
            reason += " + Negatif Diverjans"
        if detect_divergence(df, indicator='OBV', lookback=10, kind='bullish'):
            reason += " ⚠️ Pozitif Diverjans (Uyumsuzluk)"
        return {
            "signal": "SHORT",
            "reason": reason,
            "entry": last_row['close'],
            "sl": df['high'].iloc[-2],
            "tp": last_row['close'] - (df['high'].iloc[-2] - last_row['close']) * 2
        }
    return None

# ------------------------
# Risk Management
# ------------------------

def calculate_risk_position(signal: dict, account_balance=1000, risk_pct=0.01):
    entry = signal['entry']
    stop = signal['sl']
    target = signal['tp']
    risk_per_unit = abs(entry - stop)
    reward = abs(target - entry)
    if risk_per_unit == 0:
        return None
    rr = round(reward / risk_per_unit, 2)
    risk_amount = account_balance * risk_pct
    position_size = round(risk_amount / risk_per_unit, 2)
    return {
        "RR_ratio": rr,
        "risk_amount_usd": round(risk_amount, 2),
        "position_size_units": position_size,
        "potential_profit": round(position_size * reward, 2)
    }


# ------------------------
# Notification
# ------------------------

def send_telegram_message(message: str):
    if not ENABLE_TELEGRAM:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Hatası: {e}")

def send_discord_message(message: str):
    if not ENABLE_DISCORD:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception as e:
        print(f"Discord Hatası: {e}")

def notify_all(message: str):
    send_telegram_message(message)
    send_discord_message(message)


# ------------------------
# Reporting
# ------------------------

def generate_text_report(setups: list, filename="daily_report.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"📅 Günlük Swing Trade Raporu - {datetime.now().strftime('%Y-%m-%d')}\n\n")
        for s in setups:
            f.write(f"🕒 Timeframe: {s['timeframe']}\n")
            f.write(f"💵 Coin: {s['symbol']}\n")
            f.write(f"🎯 Sinyal: {s['signal']}\n")
            f.write(f"📍 Entry: {s['entry']}, SL: {s['sl']}, TP: {s['tp']}\n")
            f.write(f"📊 RR: {s['rr']}, Pot. Kâr: {s['profit']} USD\n")
            f.write(f"🔍 Neden: {s['reason']}\n")
            # --- Volume Profile ekle (varsa) ---
            if "POC" in s:
                f.write(f"📊 POC: {s['POC']:.2f}\n")
            if "HVN" in s:
                f.write(f"🏦 HVN: {', '.join([f'{x:.2f}' for x in s['HVN']])}\n")
            if "LVN" in s:
                f.write(f"🕳️ LVN: {', '.join([f'{x:.2f}' for x in s['LVN']])}\n")
            f.write("-" * 40 + "\n")

def generate_pdf_report(setups: list, filename="daily_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"📅 Günlük Swing Trade Raporu - {datetime.now().strftime('%Y-%m-%d')}", ln=1)
    for s in setups:
        text = (
            f"🕒 Timeframe: {s['timeframe']}\n"
            f"💵 Coin: {s['symbol']}\n"
            f"🎯 Sinyal: {s['signal']}\n"
            f"📍 Entry: {s['entry']}, SL: {s['sl']}, TP: {s['tp']}\n"
            f"📊 RR: {s['rr']}, Pot. Kâr: {s['profit']} USD\n"
            f"🔍 Neden: {s['reason']}\n"
        )
        if "POC" in s:
            text += f"📊 POC: {s['POC']:.2f}\n"
        if "HVN" in s:
            text += f"🏦 HVN: {', '.join([f'{x:.2f}' for x in s['HVN']])}\n"
        if "LVN" in s:
            text += f"🕳️ LVN: {', '.join([f'{x:.2f}' for x in s['LVN']])}\n"
        text += "-" * 30
        pdf.multi_cell(0, 8, txt=text)
        pdf.ln(5)
    pdf.output(filename)

# ------------------------
# Order Blocks & FVG
# ------------------------

def detect_bullish_order_blocks(df):
    ob_list = []
    for i in range(2, len(df)-1):
        if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i+1] > df['open'].iloc[i+1]:
            ob = {
                'timestamp': df.index[i], 'open': df['open'].iloc[i], 'high': df['high'].iloc[i],
                'low': df['low'].iloc[i], 'close': df['close'].iloc[i]
            }
            ob_list.append(ob)
    return pd.DataFrame(ob_list)

def detect_fvg(df):
    fvg_list = []
    for i in range(2, len(df)-1):
        prev_low = df['low'].iloc[i-2]
        current_high = df['high'].iloc[i]
        mid_low = df['low'].iloc[i-1]
        mid_high = df['high'].iloc[i-1]
        if mid_low > prev_low and mid_high < current_high:
            fvg = {'timestamp': df.index[i], 'fvg_low': prev_low, 'fvg_high': current_high}
            fvg_list.append(fvg)
    return pd.DataFrame(fvg_list)


# ------------------------
# Backend Analysis Engine
# ------------------------

def run_analysis(symbol: str, timeframes: list, telegram: bool = False, discord: bool = False) -> str:
    result_log = ""
    if "/" not in symbol:
        symbol += "/USDT"
    data = fetch_ohlcv(symbol)
    HIGHER_TF_MAP = {
        "15m": "1h",
        "1h": "4h",
        "4h": "1d",
        "1d": "1w",
    }
    collected_setups = []
    for tf, df in data.items():
        if tf not in timeframes:
            continue
        result_log += f"\n📍 Timeframe: {tf}\n"
        trend, _ = detect_trend(df)
        result_log += f"➡️ Trend: {trend}\n"
        # ---- Multi-Timeframe Alignment ----
        higher_tf = HIGHER_TF_MAP.get(tf, None)
        higher_tf_trend = None
        if higher_tf and higher_tf in data:
            higher_trend, _ = detect_trend(data[higher_tf])
            result_log += f"⬆️ Higher TF ({higher_tf}) Trend: {higher_trend}\n"
            higher_tf_trend = higher_trend
        else:
            result_log += f"⬆️ Higher TF ({higher_tf}) Trend: Bilgi yok\n"
        if higher_tf_trend and trend != higher_tf_trend:
            result_log += "❌ Trend yönü üst zaman dilimiyle uyumlu değil. Setup aranmıyor.\n"
            continue
        # ----- Volume Profile ve POC kontrolü -----
        volume_profile = calculate_volume_profile(df)
        poc = volume_profile['POC']
        hvn = volume_profile['HVN']
        lvn = volume_profile['LVN']
        close_price = df['close'].iloc[-1]
        poc_distance = abs(close_price - poc) / close_price
        # POC'a çok yakınsa işlem arama!
        if poc_distance < ANALYSIS_THRESHOLDS["POC_DIST"]:
            result_log += f"⚠️ Fiyat, POC ({poc:.2f}) seviyesine çok yakın ({close_price:.2f}). Setup aranmıyor.\n"
            continue
        # ----- Setup işlemleri -----
        df = find_equal_highs_lows(df)
        df = detect_liquidity_sweep(df)
        df = calculate_indicators(df)

        # ---- Diverjans sadece uyarı olarak kontrol edilecek ----
        long_bullish_div = detect_divergence(df, indicator='OBV', lookback=10, kind='bullish')
        long_bearish_div = detect_divergence(df, indicator='OBV', lookback=10, kind='bearish')
        short_bullish_div = detect_divergence(df, indicator='OBV', lookback=10, kind='bullish')
        short_bearish_div = detect_divergence(df, indicator='OBV', lookback=10, kind='bearish')

        long_signal = detect_long_setup(df, trend, tf)
        short_signal = detect_short_setup(df, trend, tf)

        # --- Sinyal sonrası uyarı ve filtreler ---
        for sig_type, signal in (("LONG", long_signal), ("SHORT", short_signal)):
            if signal:
                signal_price = signal["entry"]
                poc_distance = abs(signal_price - poc) / signal_price
                lvn_distances = [abs(signal_price - lv) / signal_price for lv in lvn]
                hvn_distances = [abs(signal_price - hv) / signal_price for hv in hvn]

                # --- OB & FVG tespiti ---
                bullish_obs = detect_bullish_order_blocks(df)
                fvgs = detect_fvg(df)
                in_bullish_ob = False
                if not bullish_obs.empty:
                    for idx, ob in bullish_obs.iterrows():
                        if ob['low'] <= signal_price <= ob['high']:
                            in_bullish_ob = True
                            break
                in_fvg = False
                if not fvgs.empty:
                    for idx, fvg in fvgs.iterrows():
                        if fvg['fvg_low'] <= signal_price <= fvg['fvg_high']:
                            in_fvg = True
                            break

                warnings = []
                if poc_distance < ANALYSIS_THRESHOLDS["POC_DIST"]:
                    warnings.append(f"⚠️ Sinyal fiyatı POC ({poc:.2f}) seviyesine çok yakın!")
                    result_log += f"⚠️ Sinyal fiyatı POC ({poc:.2f}) seviyesine çok yakın ({signal_price:.2f}). İşlem açılmadı.\n"
                    break  # Hem long hem short olamayacağı için devam etmeye gerek yok
                if any(ld < ANALYSIS_THRESHOLDS["LVN_HVN_DIST"] for ld in lvn_distances):
                    warnings.append(f"⚠️ Sinyal fiyatı LVN (düşük hacimli bölgeye) çok yakın!")
                if any(hd < ANALYSIS_THRESHOLDS["LVN_HVN_DIST"] for hd in hvn_distances):
                    warnings.append(f"⚠️ Sinyal fiyatı HVN (yüksek hacimli bölgeye) çok yakın!")
                if in_bullish_ob:
                    warnings.append("⚠️ Sinyal Order Block (OB) bölgesinde.")
                if in_fvg:
                    warnings.append("⚠️ Sinyal FVG (likidite boşluğu) bölgesinde.")

                # --- Diverjans uyarısı reason'a ekleniyor (artık filtrelemiyor!) ---
                reason = signal["reason"]
                if sig_type == "LONG":
                    if long_bullish_div:
                        reason += " + Pozitif Diverjans"
                    if long_bearish_div:
                        reason += " ⚠️ Negatif Diverjans (Uyumsuzluk!)"
                elif sig_type == "SHORT":
                    if short_bearish_div:
                        reason += " + Negatif Diverjans"
                    if short_bullish_div:
                        reason += " ⚠️ Pozitif Diverjans (Uyumsuzluk!)"

                if sig_type == "LONG":
                    result_log += "📈 LONG SETUP BULUNDU!\n"
                else:
                    result_log += "📉 SHORT SETUP BULUNDU!\n"
                risk = calculate_risk_position(signal, ACCOUNT_BALANCE, RISK_PERCENTAGE) if ENABLE_RISK_MANAGEMENT else None
                result_log += f"Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']}\n"
                result_log += f"RR: {risk['RR_ratio'] if risk else 'N/A'} | Neden: {reason}\n"
                if warnings:
                    result_log += "\n".join(warnings) + "\n"
                collected_setups.append({
                    "timeframe": tf,
                    "symbol": symbol,
                    "signal": sig_type,
                    "entry": signal["entry"],
                    "sl": signal["sl"],
                    "tp": signal["tp"],
                    "rr": risk["RR_ratio"] if risk else "N/A",
                    "profit": risk["potential_profit"] if risk else "N/A",
                    "reason": reason + ("\n" + "\n".join(warnings) if warnings else ""),
                    "POC": poc,
                    "HVN": hvn,
                    "LVN": lvn,
                })
                notify_all(f"{sig_type} sinyali tespit edildi: {symbol}/{tf}\n" + ("\n".join(warnings) if warnings else ""))
                break  # Birden fazla sinyal için gerek yok
        else:
            result_log += "❌ Setup bulunamadı.\n"

    if ENABLE_REPORT and collected_setups:
        generate_text_report(collected_setups)
        generate_pdf_report(collected_setups)
        result_log += "\n📄 Rapor dosyaları oluşturuldu.\n"
    return result_log

# ------------------------
# Chart Tab UI Component
# ------------------------

class ChartTab(ctk.CTkTabview):
    def __init__(self, master):
        super().__init__(master)
        self.pack(padx=20, pady=20, fill="both", expand=True)
        self.add("Grafik")
        self.symbol_entry = ctk.CTkEntry(self.tab("Grafik"), placeholder_text="Coin (örn: AVAX)")
        self.symbol_entry.pack(pady=10)
        self.tf_option = ctk.CTkOptionMenu(self.tab("Grafik"), values=["15m", "1h", "4h", "1d"])
        self.tf_option.set("1h")
        self.tf_option.pack(pady=5)
        self.load_btn = ctk.CTkButton(self.tab("Grafik"), text="📈 Grafik Yükle", command=self.load_chart)
        self.load_btn.pack(pady=10)
        self.chart_frame = ctk.CTkFrame(self.tab("Grafik"))
        self.chart_frame.pack(pady=10, padx=10, fill="both", expand=True)

    def load_chart(self):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        symbol = self.symbol_entry.get().strip().upper()
        timeframe = self.tf_option.get()
        if not symbol:
            messagebox.showerror("Hata", "Lütfen coin adı girin.")
            return

        # Binance ile OHLCV çek
        ccxt_timeframes = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        binance_symbol = f"{symbol}/USDT"
        try:
            exchange = ccxt.binance()
            data = exchange.fetch_ohlcv(binance_symbol, timeframe=ccxt_timeframes[timeframe], limit=200)
            if not data or len(data) < 10:
                messagebox.showerror("Veri Yok", f"{symbol} için yeterli veri yok.")
                return
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(inplace=True)
            fig, axlist = mpf.plot(
                df,
                type="candle",
                volume=True,
                returnfig=True,
                style="charles",
                title=f"{symbol} - {timeframe} Grafik"
            )
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            messagebox.showerror("Hata", f"{type(e).__name__}: {str(e)}")

# ------------------------
# GUI Application
# ------------------------

class CryptoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.timeframes = ["15m", "1h", "4h", "1d"]   # <-- EKLE!

        # Monitör ve pencere ayarı
        monitors = screeninfo.get_monitors()
        if len(monitors) > 1:
            right_monitor = sorted(monitors, key=lambda m: m.x)[-1]
            screen_x = right_monitor.x
            screen_y = right_monitor.y
            screen_width = right_monitor.width
            screen_height = right_monitor.height
        else:
            screen_x = monitors[0].x
            screen_y = monitors[0].y
            screen_width = monitors[0].width
            screen_height = monitors[0].height

        window_width = 1100
        window_height = 900
        x = screen_x + (screen_width // 2) - (window_width // 2)
        y = screen_y + (screen_height // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.title("💹 Crypto Swing Trade Analyzer")
        self.resizable(True, True)

        # 1) TabView ve sekmeleri oluştur
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # 2) Her sekmeyi sadece bir kere ekle
        self.analysis_tab = self.tabview.add("Analiz")
        self.chart_tab = self.tabview.add("Grafik")
        self.settings_tab = self.tabview.add("Ayarlar")   # <-- ayarlar sekmesi

        # 3) Ayarlar sekmesine ayar panelini ekle
        self.thresholds_panel = ThresholdsPanel(self.settings_tab, THRESHOLDS)
        self.thresholds_panel.pack(fill="both", expand=True, padx=20, pady=20)

        # 4) Tüm Coinler sekmesine paneli ekle
        self.allcoins_tab = self.tabview.add("Tüm Coinler")
        self.allcoins_panel = AllCoinsTab(
            self.allcoins_tab,
            self.timeframes,
            run_analysis
        )
        self.allcoins_panel.pack(fill="both", expand=True, padx=20, pady=20)

        # --- LOG KUTUSU (mini terminal gibi, 4 satır, sadece analiz sekmesinde) ---
        self.log_box = ctk.CTkTextbox(self.analysis_tab, height=80, width=650, wrap="word")
        self.log_box.pack(padx=10, pady=(5, 0), fill="x")
        self.log_box.configure(state="disabled")

        # --- ANALİZ PANELİ ---
        self.coin_label = ctk.CTkLabel(self.analysis_tab, text="Coin Adı (örn: AVAX):", font=("Arial", 16))
        self.coin_label.pack(pady=10)
        self.coin_entry = ctk.CTkEntry(self.analysis_tab, placeholder_text="örn: AVAX")
        self.coin_entry.pack(pady=5)
        self.coin_entry.bind("<Return>", lambda event: self.start_analysis())

        self.tf_label = ctk.CTkLabel(self.analysis_tab, text="Zaman Dilimleri:", font=("Arial", 14))
        self.tf_label.pack(pady=10)
        self.tf_frame = ctk.CTkFrame(self.analysis_tab)
        self.tf_frame.pack(pady=5)
        self.timeframes = ["15m", "1h", "4h", "1d"]
        self.tf_vars = {}
        for tf in self.timeframes:
            var = ctk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.tf_frame, text=tf, variable=var)
            cb.pack(side="left", padx=10)
            self.tf_vars[tf] = var
        self.telegram_toggle = ctk.CTkSwitch(self.analysis_tab, text="Telegram Bildirimi", onvalue=True, offvalue=False)
        self.telegram_toggle.pack(pady=10)
        self.discord_toggle = ctk.CTkSwitch(self.analysis_tab, text="Discord Bildirimi", onvalue=True, offvalue=False)
        self.discord_toggle.pack(pady=5)
        self.analyze_btn = ctk.CTkButton(self.analysis_tab, text="🚀 Analizi Başlat", command=self.start_analysis, fg_color="green")
        self.analyze_btn.pack(pady=20)
        self.result_box = ctk.CTkTextbox(self.analysis_tab, height=300, wrap="word")
        self.result_box.pack(padx=10, pady=10, fill="both", expand=True)
        self.result_box.insert("1.0", "🔎 Hazır...\n")

        # --- GRAFİK SEKME ARAYÜZÜ ---
        self.chart_frame = ChartTab(self.chart_tab)

        # sys.stdout yönlendirmesini after ile (GUI tamamen oturduktan sonra) yap:
        self.after(300, lambda: setattr(sys, 'stdout', GuiLogger(self)))

    def on_close(self):
        # 1) Tarama işini durdur
        try:
            self.allcoins_panel.stop_scanning = True
            # 2) Thread canlıysa 0.5 sn bekle; kritik değil, pencereyi kilitlemez
            if (self.allcoins_panel.scan_thread
                    and self.allcoins_panel.scan_thread.is_alive()):
                self.allcoins_panel.scan_thread.join(timeout=0.5)
        except Exception:
            pass
        # 3) Pencereyi kapat
        self.destroy()

    def log_message(self, msg):
        self.log_box.configure(state="normal")
        current = self.log_box.get("1.0", "end").strip().split("\n")
        current.append(msg)
        last_lines = current[-4:]
        self.log_box.delete("1.0", "end")
        self.log_box.insert("end", "\n".join(last_lines) + "\n")
        self.log_box.configure(state="disabled")
        self.log_box.update_idletasks()

    def start_analysis(self):
        coin = self.coin_entry.get().upper()
        selected_tfs = [tf for tf, var in self.tf_vars.items() if var.get()]
        telegram = self.telegram_toggle.get()
        discord = self.discord_toggle.get()
        if not coin:
            self.log_message("Hata: Coin adı girilmelidir.")
            messagebox.showerror("Hata", "Coin adı girilmelidir.")
            return
        if not selected_tfs:
            self.log_message("Hata: En az bir zaman dilimi seçmelisiniz.")
            messagebox.showerror("Hata", "En az bir zaman dilimi seçmelisiniz.")
            return
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", f"⏳ Analiz başlatılıyor...\n\n")
        self.log_message("Analiz başlatılıyor...")

        # --- Analiz işlemini thread'e at! ---
        t = threading.Thread(
            target=self.analysis_thread,
            args=(coin, selected_tfs, telegram, discord),
            daemon=True
        )
        t.start()

    def analysis_thread(self, coin, selected_tfs, telegram, discord):
        try:
            log = run_analysis(symbol=coin, timeframes=selected_tfs, telegram=telegram, discord=discord)
            # Sonuçlar thread'den GUI'ye ekleniyor!
            self.result_box.insert("end", log)
            self.log_message("Analiz tamamlandı!")
        except Exception as e:
            self.result_box.insert("end", f"\n❌ Hata oluştu:\n{str(e)}")
            self.log_message(f"Hata: {str(e)}")

class ThresholdsPanel(ctk.CTkFrame):
    def __init__(self, master, thresholds):
        super().__init__(master)
        self.thresholds = thresholds
        self.entries = {}
        row = 0
        for tf, params in thresholds.items():
            tf_label = ctk.CTkLabel(self, text=f"{tf} Zaman Dilimi", font=("Arial", 14, "bold"))
            tf_label.grid(row=row, column=0, columnspan=2, pady=(10, 2), sticky="w")
            row += 1
            for key, value in params.items():
                label = ctk.CTkLabel(self, text=key)
                label.grid(row=row, column=0, padx=10, pady=3, sticky="w")
                entry = ctk.CTkEntry(self)
                entry.insert(0, str(value))
                entry.grid(row=row, column=1, padx=10, pady=3)
                self.entries[(tf, key)] = entry
                row += 1
        self.save_btn = ctk.CTkButton(self, text="Kaydet", command=self.save_thresholds)
        self.save_btn.grid(row=row, column=0, columnspan=2, pady=10)

        desc_big = (
            "RSI_LONG_MAX: Long sinyali için RSI üst limiti. \n"
            "RSI_SHORT_MIN: Short sinyali için RSI alt limiti.\n"
            "ADX: Trend gücü eşiği, daha düşükse işlem açılmaz.\n"
        )
        desc_label = ctk.CTkLabel(self, text=desc_big, font=("Arial", 9), text_color="#888888", justify="left")
        desc_label.grid(row=row+1, column=0, columnspan=2, padx=10, pady=10, sticky="w")


    def save_thresholds(self):
        for (tf, key), entry in self.entries.items():
            try:
                val = float(entry.get())
            except Exception:
                val = entry.get()
            self.thresholds[tf][key] = val
        messagebox.showinfo("Ayarlar", "Tüm analiz eşikleri güncellendi.")

import threading

class AllCoinsTab(ctk.CTkFrame):
    def __init__(self, master, timeframes, analysis_func):
        super().__init__(master)
        self.timeframes = timeframes
        self.analysis_func = analysis_func
        self.coin_buttons = {}
        self.coin_reports = {}
        self.coin_list = []
        self.scan_thread = None
        self.stop_scanning = False

        # --- pencere yeniden boyut kontrolü için ---
        self._last_resize_width = 0
        self._grid_redraw_after_id = None

        # --- LOG ALANI (üstte) ---
        self.log_box = ctk.CTkTextbox(self, height=80, width=650, wrap="word")
        self.log_box.grid(row=0, column=0, columnspan=10, pady=10, padx=10, sticky="ew")
        self.log_box.configure(state="disabled")

        # --- BUTONLAR için üst kontrol çubuğu ---
        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=1, column=0, columnspan=10, pady=5, padx=10, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self.scan_btn = ctk.CTkButton(btn_frame, text="TÜM COİNLERİ TARA",
                                      command=self.start_scan, fg_color="blue")
        self.scan_btn.grid(row=0, column=0, padx=15, pady=5, sticky="e")

        self.stop_btn = ctk.CTkButton(btn_frame, text="DURDUR",
                                      command=self.stop_scan, fg_color="red")
        self.stop_btn.grid(row=0, column=1, padx=15, pady=5, sticky="w")

        # --- SCROLLABLE ALAN (coin butonları) ---
        # width/height sabitleme -> min. değer olarak küçük tut; grid ile genişlesin.
        self.scroll_frame = ctk.CTkScrollableFrame(self, width=0, height=600)
        self.scroll_frame.grid(row=2, column=0, columnspan=10,
                               sticky="nsew", padx=10, pady=10)

        # AllCoinsTab grid büyüyebilsin
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Artık scroll_frame'e değil; dış frame'e resize bind ---
        # Scroll sırasında tetiklenmesin diye self'e bağlıyoruz
        self.bind("<Configure>", self._on_resize)

    # ------------------ Resize / Debounce ------------------
    def _on_resize(self, event):
        # Sadece genişlik değiştiğinde reflow
        if event.width == self._last_resize_width:
            return
        self._last_resize_width = event.width
        self._schedule_create_grid()

    def _schedule_create_grid(self):
        if self._grid_redraw_after_id is not None:
            self.after_cancel(self._grid_redraw_after_id)
        self._grid_redraw_after_id = self.after(150, self.create_grid)

    # ------------------ Logging ------------------
    def log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ------------------ Data ------------------
    def fetch_and_draw_coins(self):
        self.coin_list = get_binance_usdt_symbols()

    # ------------------ Grid Oluşturma / Yeniden Yerleşim ------------------
    def create_grid(self):
        if not self.coin_list:
            return

        # Kullanılabilir panel genişliği: dış frame genişliği - yatay padding tahmini
        panel_width = max(self.winfo_width() - 40, 200)

        btn_width = 120
        btn_padx = 10
        max_cols = max(1, panel_width // (btn_width + btn_padx * 2))
        max_cols = min(max_cols, 14)

        # Eğer buton sayısı aynıysa sadece konum güncelle (destroy etme -> performans)
        if self.coin_buttons and len(self.coin_buttons) == len(self.coin_list):
            for idx, coin in enumerate(self.coin_list):
                row = idx // max_cols
                col = idx % max_cols
                self.coin_buttons[coin].grid_configure(row=row, column=col)
            return

        # Tam yeniden çizim
        for btn in self.coin_buttons.values():
            btn.destroy()
        self.coin_buttons.clear()

        for idx, coin in enumerate(self.coin_list):
            row = idx // max_cols
            col = idx % max_cols
            btn = ctk.CTkButton(self.scroll_frame, text=coin, width=btn_width,
                                command=lambda c=coin: self.show_report(c))
            btn.grid(row=row, column=col, padx=btn_padx, pady=6)
            self.coin_buttons[coin] = btn

    # ------------------ Tarama ------------------
    def start_scan(self):
        if self.scan_thread and self.scan_thread.is_alive():
            self.log("Zaten tarama sürüyor...")
            return

        # COİN LİSTESİ BURADA ÇEK
        self.coin_list = get_binance_usdt_symbols()
        self.create_grid()  # ilk dizilim

        # log’u temizle
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.stop_scanning = False
        self.log("Tüm coinlerde tarama başlıyor...")
        self.scan_thread = threading.Thread(target=self.scan_all_coins, daemon=True)
        self.scan_thread.start()

    def stop_scan(self):
        self.stop_scanning = True
        self.log("Tarama durduruluyor...")

    def scan_all_coins(self):
        for idx, coin in enumerate(self.coin_list):
            if self.stop_scanning:
                break

            report = self.analysis_func(coin, self.timeframes, telegram=False, discord=False)
            self.coin_reports[coin] = report

            def set_btn_color(c=coin, color="gray"):
                try:
                    self.coin_buttons[c].configure(fg_color=color)
                except Exception:
                    pass

            def log_msg(msg):
                try:
                    self.log(msg)
                except Exception:
                    pass

            if "LONG SETUP" in report or "SHORT SETUP" in report:
                self.after(0, set_btn_color, coin, "green")
                status = "Sinyal VAR ✅"
            else:
                self.after(0, set_btn_color, coin, "gray")
                status = "YOK ❌"

            self.after(0, log_msg, f"[{idx+1}/{len(self.coin_list)}] {coin} tarandı. {status}")

            # *** BURADA break KESİNLİKLE OLMAMALI! ***

    # ------------------ Rapor Popup ------------------
    def show_report(self, coin):
        report = self.coin_reports.get(coin, "Henüz analiz yapılmadı.")
        popup = ctk.CTkToplevel(self)
        popup.title(f"{coin} Analiz Raporu")
        popup.geometry("800x500+200+100")

        textbox = ctk.CTkTextbox(popup, width=780, height=460, wrap="word")
        textbox.pack(padx=10, pady=10, fill="both", expand=True)
        textbox.insert("end", report)
        textbox.configure(state="disabled")
        popup.focus()

if __name__ == "__main__":
    app = CryptoApp()
    app.mainloop()
