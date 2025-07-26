import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime

from CRYPTO_SWING_BOT import (
    run_analysis,
    get_binance_usdt_symbols,
    THRESHOLDS
)

st.set_page_config(page_title="📈 Crypto Swing Trade Analyzer", layout="wide")
st.title("📈 Crypto Swing Trade Analyzer")

symbol_list = get_binance_usdt_symbols()
selected_symbol = st.sidebar.selectbox(
    "Coin seç",
    symbol_list,
    index=symbol_list.index("AVAX") if "AVAX" in symbol_list else 0
)

timeframes = ["15m", "1h", "4h", "1d"]
selected_tfs = st.sidebar.multiselect("Zaman Dilimleri", timeframes, default=["1h"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Eşik Ayarları")
for tf in selected_tfs:
    st.sidebar.markdown(f"**⏱️ {tf}**")
    for key in THRESHOLDS[tf]:
        new_val = st.sidebar.number_input(
            f"{tf} - {key}",
            value=float(THRESHOLDS[tf][key]),
            step=1.0,
            format="%.2f",
            key=f"{tf}_{key}"
        )
        THRESHOLDS[tf][key] = new_val

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Analizi Başlat")

if run_button:
    with st.spinner("Analiz çalıştırılıyor..."):
        result = run_analysis(symbol=selected_symbol, timeframes=selected_tfs)
    st.success("✅ Analiz tamamlandı!")
    st.text_area("📋 Analiz Sonucu", result, height=500)
    st.download_button(
        label="📄 Sonucu .txt olarak indir",
        data=result,
        file_name=f"{selected_symbol}_analiz_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )
else:
    st.info("🔍 Analiz başlatmak için ayarları yapın ve butona tıklayın.")
