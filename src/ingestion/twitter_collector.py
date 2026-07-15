"""
twitter_collector.py
====================
Recolector de tweets financieros en español para el TFM.

Estrategias (por orden de preferencia y costo):
  1. snscrape — scraping sin API (gratuito, sin límites)
  2. Twitter/X API v2 — si tienes Bearer Token ($100/mes o Academic)

Queries de búsqueda enfocadas en Ecuador y finanzas:
  - Mercado financiero ecuatoriano
  - Criptomonedas (alta comunidad hispanohablante)
  - Economía Ecuador, dólar, inflación

Uso:
    python src/ingestion/twitter_collector.py --method snscrape --days 30
    python src/ingestion/twitter_collector.py --method api --days 7

Autores: Armas Silva Stiv, Armas Silva Jonathan, Requelme Adrian
TFM — UNIR — 2026
"""

import os
import sys
import json
import time
import sqlite3
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterator

from dotenv import load_dotenv

# ── Configuración ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("twitter_collector")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "tweets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "tweets.db"

load_dotenv(PROJECT_ROOT / ".env")


# ── Queries de búsqueda temáticas ─────────────────────────────────────────────
SEARCH_QUERIES = {
    "ecuador_finance": [
        "economía Ecuador lang:es",
        "mercado financiero Ecuador lang:es",
        "finanzas Ecuador 2024 lang:es",
        "inflación Ecuador lang:es",
        "BCE Ecuador banco central lang:es",
        "dólar Ecuador lang:es",
        "BVQ bolsa Quito lang:es",
    ],
    "crypto_latam": [
        "bitcoin Ecuador lang:es",
        "ethereum Ecuador lang:es",
        "criptomonedas Ecuador lang:es",
        "crypto inversión Ecuador lang:es",
        "BTC precio lang:es",
        "ETH precio lang:es",
        "#bitcoin OR #btc lang:es",
        "#ethereum OR #eth lang:es",
    ],
    "commodities_ecuador": [
        "petróleo Ecuador Petroecuador lang:es",
        "exportación banano Ecuador lang:es",
        "exportación camarón Ecuador lang:es",
        "precio petróleo barril lang:es",
    ],
    "sentiment_finance_global_es": [
        "mercado baja crash lang:es -is:retweet",
        "mercado sube rally lang:es -is:retweet",
        "crisis financiera lang:es",
        "inversión acciones rendimiento lang:es",
    ],
}


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: Base de datos SQLite para tweets
# ════════════════════════════════════════════════════════════════════════════════

def init_tweet_database() -> sqlite3.Connection:
    """Inicializa la base de datos para tweets."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id        TEXT UNIQUE,
            text            TEXT NOT NULL,
            created_at      TEXT,
            username        TEXT,
            user_followers  INTEGER,
            like_count      INTEGER DEFAULT 0,
            retweet_count   INTEGER DEFAULT 0,
            reply_count     INTEGER DEFAULT 0,
            quote_count     INTEGER DEFAULT 0,
            language        TEXT DEFAULT 'es',
            query_used      TEXT,
            source_method   TEXT,       -- 'snscrape' o 'api'
            scraped_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            sentiment_label TEXT,       -- null hasta procesar
            sentiment_score REAL,       -- null hasta procesar
            asset_related   TEXT,
            is_retweet      INTEGER DEFAULT 0,
            has_url         INTEGER DEFAULT 0,
            hashtags        TEXT        -- JSON array
        )
    """)
    conn.commit()
    logger.info(f"✅ BD Tweets: {DB_PATH}")
    return conn


def save_tweet(conn: sqlite3.Connection, tweet: dict) -> bool:
    """Guarda un tweet. Retorna True si es nuevo."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO tweets
            (tweet_id, text, created_at, username, user_followers,
             like_count, retweet_count, reply_count, language,
             query_used, source_method, asset_related, is_retweet,
             has_url, hashtags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tweet.get("tweet_id", ""),
            tweet.get("text", ""),
            tweet.get("created_at", ""),
            tweet.get("username", ""),
            tweet.get("user_followers", 0),
            tweet.get("like_count", 0),
            tweet.get("retweet_count", 0),
            tweet.get("reply_count", 0),
            tweet.get("language", "es"),
            tweet.get("query_used", ""),
            tweet.get("source_method", "snscrape"),
            tweet.get("asset_related", "GENERAL"),
            tweet.get("is_retweet", 0),
            tweet.get("has_url", 0),
            json.dumps(tweet.get("hashtags", []), ensure_ascii=False),
        ))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.debug(f"Error guardando tweet: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: Recolección con snscrape (sin API, gratuito)
# ════════════════════════════════════════════════════════════════════════════════

def collect_with_snscrape(
    conn: sqlite3.Connection,
    query: str,
    max_tweets: int = 500,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
) -> int:
    """
    Recolecta tweets usando snscrape (sin API key requerida).

    NOTA: snscrape puede fallar si Twitter cambia su HTML.
    En ese caso, usar el método de API o datasets pre-construidos.
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        logger.error("snscrape no instalado. Ejecuta: pip install snscrape")
        return 0

    # Construir query con filtros de fecha
    full_query = query
    if since_date:
        full_query += f" since:{since_date}"
    if until_date:
        full_query += f" until:{until_date}"

    logger.info(f"  🐦 snscrape query: '{full_query[:80]}...'")
    saved = 0
    errors = 0

    try:
        scraper = sntwitter.TwitterSearchScraper(full_query)
        for i, tweet in enumerate(scraper.get_items()):
            if i >= max_tweets:
                break

            # Detectar hashtags
            hashtags = [tag.strip("#") for tag in tweet.content.split()
                        if tag.startswith("#")]

            tweet_data = {
                "tweet_id": str(tweet.id),
                "text": tweet.content,
                "created_at": tweet.date.isoformat() if tweet.date else "",
                "username": tweet.user.username if tweet.user else "",
                "user_followers": tweet.user.followersCount if tweet.user else 0,
                "like_count": tweet.likeCount or 0,
                "retweet_count": tweet.retweetCount or 0,
                "reply_count": tweet.replyCount or 0,
                "language": tweet.lang or "es",
                "query_used": query,
                "source_method": "snscrape",
                "asset_related": detect_tweet_asset(tweet.content),
                "is_retweet": 1 if tweet.retweetedTweet else 0,
                "has_url": 1 if tweet.links else 0,
                "hashtags": hashtags[:10],
            }

            if save_tweet(conn, tweet_data):
                saved += 1

            if i % 100 == 0 and i > 0:
                logger.debug(f"    {i} tweets procesados, {saved} nuevos...")

    except Exception as e:
        logger.warning(f"  Error snscrape: {e}")
        errors += 1

    return saved


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: Recolección con Twitter API v2
# ════════════════════════════════════════════════════════════════════════════════

def collect_with_twitter_api(
    conn: sqlite3.Connection,
    query: str,
    max_results: int = 100,
    start_time: Optional[str] = None,
) -> int:
    """
    Recolecta tweets usando la Twitter/X API v2.
    Requiere TWITTER_BEARER_TOKEN en .env
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
    if not bearer_token:
        logger.warning("⚠️  TWITTER_BEARER_TOKEN no configurado.")
        return 0

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": f"{query} lang:es",
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,public_metrics,lang,entities",
        "user.fields": "public_metrics,username",
        "expansions": "author_id",
    }
    if start_time:
        params["start_time"] = start_time

    saved = 0
    next_token = None
    pages = 0
    max_pages = max_results // 100 + 1

    while pages < max_pages:
        if next_token:
            params["next_token"] = next_token

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)

            if resp.status_code == 429:
                logger.warning("Rate limit Twitter API. Esperando 15 min...")
                time.sleep(15 * 60)
                continue

            resp.raise_for_status()
            data = resp.json()
            tweets_batch = data.get("data", [])
            users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}

            for tw in tweets_batch:
                metrics = tw.get("public_metrics", {})
                author = users.get(tw.get("author_id", ""), {})
                author_metrics = author.get("public_metrics", {})

                hashtags = [
                    ht["tag"] for ht in
                    tw.get("entities", {}).get("hashtags", [])
                ]

                tweet_data = {
                    "tweet_id": tw["id"],
                    "text": tw.get("text", ""),
                    "created_at": tw.get("created_at", ""),
                    "username": author.get("username", ""),
                    "user_followers": author_metrics.get("followers_count", 0),
                    "like_count": metrics.get("like_count", 0),
                    "retweet_count": metrics.get("retweet_count", 0),
                    "reply_count": metrics.get("reply_count", 0),
                    "language": tw.get("lang", "es"),
                    "query_used": query,
                    "source_method": "twitter_api_v2",
                    "asset_related": detect_tweet_asset(tw.get("text", "")),
                    "is_retweet": 0,
                    "has_url": 1 if tw.get("entities", {}).get("urls") else 0,
                    "hashtags": hashtags[:10],
                }

                if save_tweet(conn, tweet_data):
                    saved += 1

            next_token = data.get("meta", {}).get("next_token")
            pages += 1
            if not next_token:
                break

            time.sleep(1)

        except Exception as e:
            logger.error(f"  Error Twitter API: {e}")
            break

    return saved


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: Utilidades
# ════════════════════════════════════════════════════════════════════════════════

ASSET_MAP = {
    "BTC": ["bitcoin", "btc", "#bitcoin", "#btc"],
    "ETH": ["ethereum", "eth", "#ethereum", "#eth"],
    "ADA": ["cardano", "ada", "#cardano"],
    "SOL": ["solana", "#solana"],
    "PETRÓLEO": ["petróleo", "petroleo", "oil", "crude", "petroecuador"],
    "ORO": ["oro", "gold", "#gold"],
    "BANANO": ["banano", "banana"],
    "CAMARÓN": ["camarón", "camaron", "shrimp"],
}


def detect_tweet_asset(text: str) -> str:
    """Detecta el activo financiero mencionado en el tweet."""
    text_lower = text.lower()
    for asset, keywords in ASSET_MAP.items():
        if any(kw.lower() in text_lower for kw in keywords):
            return asset
    return "GENERAL"


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: Pipeline principal
# ════════════════════════════════════════════════════════════════════════════════

def run_twitter_collection(
    method: str = "snscrape",
    days_back: int = 30,
    max_per_query: int = 200,
    query_categories: list = None,
) -> dict:
    """
    Ejecuta el pipeline completo de recolección de tweets.

    Args:
        method:           'snscrape' (gratuito) o 'api' (requiere token)
        days_back:        Días hacia atrás
        max_per_query:    Máximo tweets por query
        query_categories: Lista de categorías (None = todas)

    Returns:
        Resumen de resultados
    """
    logger.info("=" * 60)
    logger.info("  TFM — Pipeline de Recolección de Tweets")
    logger.info(f"  Método: {method} | Días: {days_back}")
    logger.info("=" * 60)

    conn = init_tweet_database()
    since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    until_date = datetime.now().strftime("%Y-%m-%d")

    total_saved = 0
    results = {"method": method, "sources": {}}

    categories = query_categories or list(SEARCH_QUERIES.keys())

    for category in categories:
        queries = SEARCH_QUERIES.get(category, [])
        logger.info(f"\n📂 Categoría: {category} ({len(queries)} queries)")
        cat_total = 0

        for query in queries:
            logger.info(f"  🔍 Query: {query[:60]}")
            if method == "snscrape":
                n = collect_with_snscrape(
                    conn, query,
                    max_tweets=max_per_query,
                    since_date=since_date,
                    until_date=until_date,
                )
            else:  # api
                start_iso = (datetime.now() - timedelta(days=days_back)).strftime(
                    "%Y-%m-%dT00:00:00Z"
                )
                n = collect_with_twitter_api(
                    conn, query,
                    max_results=max_per_query,
                    start_time=start_iso,
                )

            cat_total += n
            logger.info(f"    → {n} tweets nuevos")
            time.sleep(2)

        results["sources"][category] = cat_total
        total_saved += cat_total

    # Estadísticas finales
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tweets")
    total_db = cursor.fetchone()[0]
    conn.close()

    results["session_total"] = total_saved
    results["total_in_db"] = total_db

    summary_path = DATA_DIR / "tweets_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info(f"  ✅ Tweets recolección completada")
    logger.info(f"  🐦 Tweets esta sesión: {total_saved}")
    logger.info(f"  🗄️  Total en BD: {total_db}")
    logger.info(f"  📁 BD: {DB_PATH}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    # Necesitamos requests importado en este scope también
    import requests

    parser = argparse.ArgumentParser(description="TFM Twitter Collector")
    parser.add_argument("--method", default="snscrape",
                        choices=["snscrape", "api"],
                        help="Método de recolección")
    parser.add_argument("--days", type=int, default=30,
                        help="Días hacia atrás")
    parser.add_argument("--max", type=int, default=200,
                        help="Máximo tweets por query")
    parser.add_argument("--categories", nargs="+",
                        choices=list(SEARCH_QUERIES.keys()),
                        help="Categorías de queries a usar")
    args = parser.parse_args()

    run_twitter_collection(
        method=args.method,
        days_back=args.days,
        max_per_query=args.max,
        query_categories=args.categories,
    )
