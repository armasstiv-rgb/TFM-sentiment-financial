"""
sentiment_index.py
==================
Módulo para calcular el Índice de Sentimiento Financiero (ISF).

El ISF es la métrica central del TFM:
  - Agrega scores de sentimiento de múltiples fuentes (noticias, tweets)
  - Genera una serie temporal diaria normalizada [-1, +1]
  - Permite correlacionar el sentimiento con movimientos de precios

Fórmula del ISF:
  ISF_t = Σ(w_i * score_i) / Σ(w_i)
  
  Donde w_i son pesos por:
    - Fuente (noticias > tweets en confiabilidad)
    - Popularidad (likes, retweets, etc.)
    - Recencia (textos más recientes pesan más)

Autores: Armas Silva Stiv, Armas Silva Jonathan, Requelme Adrian
TFM — UNIR — 2026
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("sentiment_index")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Pesos por fuente de datos
SOURCE_WEIGHTS = {
    "El Comercio Ecuador": 1.5,
    "Primicias Ecuador": 1.5,
    "El Universo Ecuador": 1.4,
    "NewsAPI:Reuters":     2.0,
    "NewsAPI:Bloomberg":   2.0,
    "NewsAPI:Financial Times": 2.0,
    "snscrape":            0.8,   # Tweets menor peso
    "twitter_api_v2":      0.8,
    "default":             1.0,
}


# ════════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════════

def load_sentiment_data(
    asset_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carga datos de sentimiento desde las bases de datos SQLite.

    Args:
        asset_filter: Filtrar por activo (ej: 'BTC', 'GENERAL')
        start_date:   Fecha inicio 'YYYY-MM-DD'
        end_date:     Fecha fin 'YYYY-MM-DD'

    Returns:
        DataFrame con columnas: date, text, sentiment_score, source, asset
    """
    dfs = []

    # ── Noticias ───────────────────────────────────────────────────────────
    news_db = DATA_DIR / "raw" / "news" / "news.db"
    if news_db.exists():
        conn = sqlite3.connect(news_db)
        query = """
            SELECT 
                DATE(published_at) as date,
                title as text,
                source,
                asset_related as asset,
                sentiment_score,
                sentiment_label
            FROM news_articles
            WHERE sentiment_score IS NOT NULL
        """
        if asset_filter:
            query += f" AND (asset_related = '{asset_filter}' OR asset_related = 'GENERAL')"
        if start_date:
            query += f" AND DATE(published_at) >= '{start_date}'"
        if end_date:
            query += f" AND DATE(published_at) <= '{end_date}'"

        try:
            df_news = pd.read_sql_query(query, conn)
            df_news["data_type"] = "news"
            dfs.append(df_news)
            logger.info(f"Noticias con sentimiento: {len(df_news)}")
        except Exception as e:
            logger.warning(f"Error cargando noticias: {e}")
        finally:
            conn.close()

    # ── Tweets ─────────────────────────────────────────────────────────────
    tweets_db = DATA_DIR / "raw" / "tweets" / "tweets.db"
    if tweets_db.exists():
        conn = sqlite3.connect(tweets_db)
        query = """
            SELECT 
                DATE(created_at) as date,
                text,
                source_method as source,
                asset_related as asset,
                sentiment_score,
                sentiment_label,
                like_count,
                retweet_count
            FROM tweets
            WHERE sentiment_score IS NOT NULL
        """
        if asset_filter:
            query += f" AND (asset_related = '{asset_filter}' OR asset_related = 'GENERAL')"

        try:
            df_tweets = pd.read_sql_query(query, conn)
            df_tweets["data_type"] = "tweet"
            # Calcular peso por popularidad (likes + retweets)
            df_tweets["popularity"] = (
                df_tweets.get("like_count", 0).fillna(0) +
                df_tweets.get("retweet_count", 0).fillna(0) * 2
            )
            dfs.append(df_tweets)
            logger.info(f"Tweets con sentimiento: {len(df_tweets)}")
        except Exception as e:
            logger.warning(f"Error cargando tweets: {e}")
        finally:
            conn.close()

    if not dfs:
        logger.warning("No hay datos de sentimiento disponibles aún.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "sentiment_score"])

    return df


# ════════════════════════════════════════════════════════════════════════════════
# CÁLCULO DEL ISF
# ════════════════════════════════════════════════════════════════════════════════

def compute_source_weight(source: str) -> float:
    """Obtiene el peso de una fuente de datos."""
    for key, weight in SOURCE_WEIGHTS.items():
        if key.lower() in source.lower():
            return weight
    return SOURCE_WEIGHTS["default"]


def compute_isf(
    df: pd.DataFrame,
    frequency: str = "D",
    min_texts: int = 1,
    use_popularity_weight: bool = True,
    smoothing_window: int = 3,
) -> pd.DataFrame:
    """
    Calcula el Índice de Sentimiento Financiero (ISF).

    Args:
        df:                    DataFrame con datos de sentimiento
        frequency:             Frecuencia de agregación ('D'=diario, 'W'=semanal)
        min_texts:             Mínimo de textos para considerar el período
        use_popularity_weight: Ponderar por popularidad del tweet/noticia
        smoothing_window:      Ventana de suavizado (días)

    Returns:
        DataFrame con ISF por período:
          - isf_raw:     ISF sin suavizado
          - isf_smooth:  ISF con media móvil
          - isf_std:     Desviación estándar
          - n_texts:     Número de textos procesados
          - bullish_pct: % textos positivos
          - bearish_pct: % textos negativos
    """
    if df.empty:
        logger.warning("DataFrame vacío — no se puede calcular ISF")
        return pd.DataFrame()

    # Calcular pesos combinados
    df = df.copy()
    df["source_weight"] = df["source"].apply(compute_source_weight)

    if use_popularity_weight and "popularity" in df.columns:
        # Normalizar popularidad entre 1 y 3
        pop = df["popularity"].fillna(0)
        if pop.max() > 0:
            pop_normalized = 1 + 2 * (pop - pop.min()) / (pop.max() - pop.min())
        else:
            pop_normalized = pd.Series(1.0, index=df.index)
        df["total_weight"] = df["source_weight"] * pop_normalized
    else:
        df["total_weight"] = df["source_weight"]

    # Sentimiento ponderado
    df["weighted_score"] = df["sentiment_score"] * df["total_weight"]

    # Agregar por período
    df = df.set_index("date").sort_index()

    grouped = df.resample(frequency).agg(
        weighted_score_sum=("weighted_score", "sum"),
        total_weight_sum=("total_weight", "sum"),
        sentiment_std=("sentiment_score", "std"),
        n_texts=("sentiment_score", "count"),
        n_positive=("sentiment_label", lambda x: (x == "positive").sum()),
        n_negative=("sentiment_label", lambda x: (x == "negative").sum()),
        n_neutral=("sentiment_label", lambda x: (x == "neutral").sum()),
    )

    # Filtrar períodos con mínimo de textos
    grouped = grouped[grouped["n_texts"] >= min_texts]

    # ISF = media ponderada de scores
    grouped["isf_raw"] = grouped["weighted_score_sum"] / grouped["total_weight_sum"].replace(0, np.nan)

    # Suavizado con media móvil
    grouped["isf_smooth"] = grouped["isf_raw"].rolling(
        window=smoothing_window, min_periods=1, center=False
    ).mean()

    # Percentajes
    grouped["bullish_pct"] = grouped["n_positive"] / grouped["n_texts"] * 100
    grouped["bearish_pct"] = grouped["n_negative"] / grouped["n_texts"] * 100
    grouped["neutral_pct"] = grouped["n_neutral"] / grouped["n_texts"] * 100

    # Señal de trading basada en ISF
    grouped["signal"] = grouped["isf_raw"].apply(
        lambda x: "BUY" if x > 0.15 else "SELL" if x < -0.15 else "HOLD"
    )

    # Renombrar para claridad
    grouped = grouped.rename(columns={"sentiment_std": "isf_std"})
    grouped = grouped.drop(columns=["weighted_score_sum", "total_weight_sum"])

    logger.info(f"ISF calculado: {len(grouped)} períodos, freq={frequency}")
    logger.info(f"  Rango: [{grouped['isf_raw'].min():.3f}, {grouped['isf_raw'].max():.3f}]")
    logger.info(f"  Media: {grouped['isf_raw'].mean():.3f}")

    return grouped


def merge_isf_with_prices(
    isf: pd.DataFrame,
    prices_path: str,
    price_col: str = "Close",
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Combina el ISF con datos de precios para análisis de correlación.

    Args:
        isf:         DataFrame del ISF
        prices_path: Ruta al CSV de precios
        price_col:   Columna de precio a usar
        lag_days:    Días de desfase del ISF respecto al precio

    Returns:
        DataFrame combinado con ISF y precios
    """
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    if price_col not in prices.columns:
        available = prices.columns.tolist()
        logger.warning(f"Columna '{price_col}' no encontrada. Disponibles: {available}")
        price_col = available[0]

    prices_daily = prices[price_col].resample("D").last().ffill()

    # Retornos logarítmicos
    prices_returns = np.log(prices_daily / prices_daily.shift(1))
    prices_returns.name = "log_return"

    # Volatilidad realizada (ventana 5 días)
    prices_vol = prices_returns.rolling(5).std()
    prices_vol.name = "volatility_5d"

    # Combinar
    combined = pd.concat([
        isf[["isf_raw", "isf_smooth", "n_texts", "bullish_pct", "bearish_pct", "signal"]],
        prices_daily.rename("price"),
        prices_returns,
        prices_vol,
    ], axis=1).dropna(subset=["price"])

    # Añadir ISF con lag (para análisis predictivo)
    combined[f"isf_lag{lag_days}d"] = combined["isf_raw"].shift(lag_days)

    logger.info(f"ISF + Precios combinados: {len(combined)} registros")
    return combined


def save_isf(isf: pd.DataFrame, asset: str = "BTC", frequency: str = "D"):
    """Guarda el ISF calculado en disco."""
    output_path = PROCESSED_DIR / f"isf_{asset}_{frequency}.csv"
    isf.to_csv(output_path)
    logger.info(f"ISF guardado: {output_path}")
    return output_path


# ════════════════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ════════════════════════════════════════════════════════════════════════════════

def demo_with_synthetic_data():
    """
    Demuestra el cálculo del ISF con datos sintéticos.
    Útil para validar el pipeline antes de tener datos reales.
    """
    import numpy as np

    logger.info("Generando datos sintéticos para demo del ISF...")

    # Simular 90 días de datos de sentimiento
    dates = pd.date_range(start="2025-01-01", periods=90, freq="D")
    np.random.seed(42)

    n_texts_per_day = 20
    records = []

    for date in dates:
        for _ in range(n_texts_per_day):
            score = np.random.normal(0.1, 0.4)  # Ligero sesgo positivo
            score = np.clip(score, -1, 1)
            label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
            records.append({
                "date": date,
                "text": f"Texto sintético para {date.date()}",
                "sentiment_score": score,
                "sentiment_label": label,
                "source": np.random.choice([
                    "El Comercio Ecuador", "snscrape", "NewsAPI:Reuters"
                ]),
                "asset": "BTC",
                "popularity": np.random.randint(0, 1000),
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Calcular ISF
    isf = compute_isf(df, frequency="D", smoothing_window=7)

    print("\n=== ISF DEMO (primeros 10 días) ===")
    print(isf[["isf_raw", "isf_smooth", "n_texts", "bullish_pct", "bearish_pct", "signal"]].head(10).to_string())
    print(f"\nISF promedio: {isf['isf_raw'].mean():.4f}")
    print(f"Señales BUY: {(isf['signal']=='BUY').sum()}")
    print(f"Señales SELL: {(isf['signal']=='SELL').sum()}")
    print(f"Señales HOLD: {(isf['signal']=='HOLD').sum()}")

    # Guardar
    output_path = PROCESSED_DIR / "isf_demo_synthetic.csv"
    isf.to_csv(output_path)
    logger.info(f"Demo ISF guardado: {output_path}")

    return isf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    demo_with_synthetic_data()
