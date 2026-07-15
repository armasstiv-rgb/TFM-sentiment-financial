"""
price_fetcher.py
================
Script de recolección de datos de precios financieros.

Fuentes:
  - Yahoo Finance (yfinance): Acciones, ETFs, índices
  - CoinGecko API: Criptomonedas (BTC, ETH, ADA, etc.)
  - Banco Central del Ecuador (BCE): Tasas de interés macro

Activos del TFM:
  - Criptomonedas: BTC-USD, ETH-USD, ADA-USD
  - ETFs globales: SPY, EEM (mercados emergentes)
  - Acciones latinoamericanas: disponibles en yfinance

Uso:
    python src/ingestion/price_fetcher.py

Autores: Armas Silva Stiv, Armas Silva Jonathan, Requelme Adrian
TFM — UNIR — Master Visual Analytics and Big Data — 2026
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ── Configuración de logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("price_fetcher")

# ── Rutas del proyecto ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: Yahoo Finance — Acciones y ETFs
# ════════════════════════════════════════════════════════════════════════════════

def fetch_yfinance_data(
    tickers: list[str],
    start_date: str = "2022-01-01",
    end_date: str | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Descarga precios OHLCV desde Yahoo Finance.

    Args:
        tickers:    Lista de tickers (ej: ['BTC-USD', 'SPY'])
        start_date: Fecha inicio en formato 'YYYY-MM-DD'
        end_date:   Fecha fin (None = hoy)
        interval:   Intervalo temporal ('1d', '1h', '1wk')

    Returns:
        Diccionario {ticker: DataFrame con columnas OHLCV}
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance no instalado. Ejecuta: pip install yfinance")
        return {}

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    results = {}
    logger.info(f"Descargando {len(tickers)} tickers desde Yahoo Finance...")

    for ticker in tickers:
        try:
            logger.info(f"  ⬇️  {ticker}...")
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                logger.warning(f"  ⚠️  Sin datos para {ticker}")
                continue

            # Limpiar columnas multi-nivel si existen
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.index = pd.to_datetime(data.index)
            data["ticker"] = ticker
            data["source"] = "yahoo_finance"
            results[ticker] = data

            # Guardar en CSV
            output_path = DATA_DIR / f"{ticker.replace('-', '_')}_{interval}.csv"
            data.to_csv(output_path)
            logger.info(f"  ✅  {ticker}: {len(data)} registros → {output_path.name}")

            time.sleep(0.5)  # Rate limiting gentil

        except Exception as e:
            logger.error(f"  ❌  Error con {ticker}: {e}")
            continue

    return results


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: CoinGecko — Criptomonedas
# ════════════════════════════════════════════════════════════════════════════════

COINGECKO_IDS = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "ADA-USD": "cardano",
    "BNB-USD": "binancecoin",
    "SOL-USD": "solana",
    "USDT-USD": "tether",
}


def fetch_coingecko_history(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 730,  # 2 años
) -> pd.DataFrame:
    """
    Descarga precios históricos diarios desde CoinGecko API (gratuita).

    Args:
        coin_id:     ID de CoinGecko (ej: 'bitcoin', 'ethereum')
        vs_currency: Moneda de referencia
        days:        Número de días históricos (max 365 en plan free para granularidad diaria)

    Returns:
        DataFrame con columnas: timestamp, price, market_cap, total_volume
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily",
        "precision": "6",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        raw = response.json()

        # Parsear respuesta
        prices = pd.DataFrame(raw["prices"], columns=["timestamp_ms", "price"])
        volumes = pd.DataFrame(raw["total_volumes"], columns=["timestamp_ms", "volume"])
        market_caps = pd.DataFrame(raw["market_caps"], columns=["timestamp_ms", "market_cap"])

        df = prices.merge(volumes, on="timestamp_ms").merge(market_caps, on="timestamp_ms")
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.normalize()
        df = df.drop(columns=["timestamp_ms"]).set_index("date")
        df["coin_id"] = coin_id
        df["vs_currency"] = vs_currency
        df["source"] = "coingecko"

        return df

    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logger.warning(f"Rate limit CoinGecko para {coin_id}. Esperando 60s...")
            time.sleep(60)
            return fetch_coingecko_history(coin_id, vs_currency, days)
        logger.error(f"HTTP Error {coin_id}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error descargando {coin_id}: {e}")
        return pd.DataFrame()


def fetch_all_crypto(days: int = 730) -> dict[str, pd.DataFrame]:
    """Descarga todos los activos cripto definidos en COINGECKO_IDS."""
    results = {}
    logger.info(f"Descargando {len(COINGECKO_IDS)} criptomonedas desde CoinGecko...")

    for ticker, coin_id in COINGECKO_IDS.items():
        logger.info(f"  ⬇️  {ticker} ({coin_id})...")
        df = fetch_coingecko_history(coin_id=coin_id, days=days)

        if not df.empty:
            output_path = DATA_DIR / f"{ticker.replace('-', '_')}_coingecko_daily.csv"
            df.to_csv(output_path)
            results[ticker] = df
            logger.info(f"  ✅  {ticker}: {len(df)} registros → {output_path.name}")
        else:
            logger.warning(f"  ⚠️  Sin datos para {ticker}")

        time.sleep(12)  # CoinGecko free tier: 5-10 req/min

    return results


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: Banco Central del Ecuador — Datos Macro
# ════════════════════════════════════════════════════════════════════════════════

def fetch_bce_macro_indicators() -> pd.DataFrame:
    """
    Descarga indicadores macroeconómicos del BCE de Ecuador.
    Nota: El BCE no tiene API REST oficial, se usan endpoints conocidos.
    """
    logger.info("Descargando indicadores macro BCE Ecuador...")

    # Tasas de interés referenciales BCE Ecuador
    # Fuente: https://www.bce.fin.ec/index.php/component/k2/item/320
    bce_endpoints = {
        "tasas_activas": "https://contenido.bce.fin.ec/documentos/Estadisticas/SectorMonFin/TasasInteres/Indice.htm",
        # Se complementa con datos del FRED para Ecuador
    }

    # Alternativa: usar FRED para datos macroeconómicos de Ecuador disponibles
    records = []

    try:
        # Intentar descargar desde BCE formato CSV si disponible
        url = "https://contenido.bce.fin.ec/documentos/Estadisticas/SectorMonFin/TasasInteres/tasas_activas.csv"
        resp = requests.get(url, timeout=15, verify=False)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            output_path = DATA_DIR / "bce_tasas_activas.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"  ✅  BCE tasas activas: {len(df)} registros")
            return df
    except Exception as e:
        logger.warning(f"BCE directo no disponible: {e}. Usando FRED como alternativa.")

    # Fallback: FRED — indicadores de Ecuador
    fred_series = {
        "ECUPCPIPCH": "Inflacion_Ecuador_Anual",      # CPI Ecuador YoY
        "ECUNFINS": "Desempleo_Ecuador",              # Unemployment Ecuador
        "ECUPGDPQDSNAQ": "PIB_Ecuador_Trimestral",   # GDP Ecuador
    }

    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        logger.warning("FRED_API_KEY no configurada. Saltando datos FRED.")
        return pd.DataFrame()

    all_series = []
    for series_id, label in fred_series.items():
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": fred_key,
                "file_type": "json",
                "observation_start": "2020-01-01",
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            df_s = pd.DataFrame(obs)
            df_s["series"] = label
            all_series.append(df_s)
            logger.info(f"  ✅  FRED {series_id}: {len(df_s)} observaciones")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"  ⚠️  FRED {series_id}: {e}")

    if all_series:
        df_macro = pd.concat(all_series, ignore_index=True)
        output_path = DATA_DIR / "macro_ecuador_fred.csv"
        df_macro.to_csv(output_path, index=False)
        return df_macro

    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: Función principal — Ejecutar todo el pipeline
# ════════════════════════════════════════════════════════════════════════════════

def run_price_collection(
    start_date: str = "2022-01-01",
    crypto_days: int = 730,
    include_macro: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo de recolección de precios.

    Returns:
        Diccionario con resumen de datos recolectados
    """
    logger.info("=" * 60)
    logger.info("  TFM — Pipeline de Recolección de Precios")
    logger.info(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    summary = {
        "run_date": datetime.now().isoformat(),
        "start_date": start_date,
        "sources": {},
    }

    # ── 1. Acciones y ETFs desde Yahoo Finance ─────────────────────────────
    yfinance_tickers = [
        # ETFs globales con alta liquidez
        "SPY",    # S&P 500
        "EEM",    # MSCI Emerging Markets
        "ILF",    # iShares Latin America 40 ETF
        "EWZ",    # Brasil (referencia regional)
        # Commodities relevantes para Ecuador
        "GLD",    # Oro (correlación con BTC y economía emergente)
        "USO",    # Petróleo (Ecuador es exportador)
        # Criptomonedas también disponibles en yfinance
        "BTC-USD",
        "ETH-USD",
    ]

    yf_data = fetch_yfinance_data(
        tickers=yfinance_tickers,
        start_date=start_date,
        interval="1d",
    )
    summary["sources"]["yahoo_finance"] = {
        "tickers_requested": len(yfinance_tickers),
        "tickers_downloaded": len(yf_data),
        "tickers": list(yf_data.keys()),
    }

    # ── 2. Criptomonedas desde CoinGecko ──────────────────────────────────
    logger.info("\n" + "─" * 40)
    crypto_data = fetch_all_crypto(days=crypto_days)
    summary["sources"]["coingecko"] = {
        "coins_downloaded": len(crypto_data),
        "days": crypto_days,
        "coins": list(crypto_data.keys()),
    }

    # ── 3. Macro Ecuador ───────────────────────────────────────────────────
    if include_macro:
        logger.info("\n" + "─" * 40)
        macro_data = fetch_bce_macro_indicators()
        summary["sources"]["macro"] = {
            "records": len(macro_data) if not macro_data.empty else 0,
        }

    # ── 4. Guardar resumen ─────────────────────────────────────────────────
    summary_path = DATA_DIR / "collection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Recolección de precios completada")
    logger.info(f"  📁 Datos guardados en: {DATA_DIR}")
    logger.info(f"  📊 Yahoo Finance: {summary['sources']['yahoo_finance']['tickers_downloaded']} tickers")
    logger.info(f"  🪙 CoinGecko: {summary['sources']['coingecko']['coins_downloaded']} criptomonedas")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    run_price_collection(
        start_date="2022-01-01",
        crypto_days=730,
        include_macro=True,
    )
