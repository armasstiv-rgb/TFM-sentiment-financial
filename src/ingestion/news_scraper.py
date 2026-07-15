"""
news_scraper.py
===============
Scraper de noticias financieras para el TFM.

Fuentes cubiertas:
  1. El Comercio (Ecuador) — sección economía/finanzas
  2. Primicias.ec (Ecuador) — sección economía
  3. El Universo (Ecuador) — sección economía
  4. NewsAPI — noticias internacionales en español
  5. GNews API — noticias con geolocalización Ecuador

Estrategia:
  - Web scraping con BeautifulSoup para medios ecuatorianos
  - APIs REST para noticias internacionales
  - Almacenamiento en SQLite + JSON

Uso:
    python src/ingestion/news_scraper.py --source all --days 30

Autores: Armas Silva Stiv, Armas Silva Jonathan, Requelme Adrian
TFM — UNIR — 2026
"""

import os
import sys
import time
import json
import sqlite3
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ── Configuración ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("news_scraper")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "news"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "news.db"

load_dotenv(PROJECT_ROOT / ".env")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "es-EC,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: Base de datos SQLite
# ════════════════════════════════════════════════════════════════════════════════

def init_database() -> sqlite3.Connection:
    """Inicializa la base de datos SQLite para almacenamiento de noticias."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT NOT NULL,
            content         TEXT,
            summary         TEXT,
            url             TEXT UNIQUE,
            source          TEXT NOT NULL,
            source_country  TEXT DEFAULT 'EC',
            language        TEXT DEFAULT 'es',
            published_at    TEXT,
            scraped_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            category        TEXT DEFAULT 'finance',
            sentiment_label TEXT,       -- null hasta que se procese
            sentiment_score REAL,       -- null hasta que se procese
            keywords        TEXT,       -- JSON array
            asset_related   TEXT        -- BTC, ETH, etc. si aplica
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraping_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT,
            run_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            articles_scraped INTEGER,
            status      TEXT
        )
    """)

    conn.commit()
    logger.info(f"✅ Base de datos inicializada: {DB_PATH}")
    return conn


def save_article(conn: sqlite3.Connection, article: dict) -> bool:
    """
    Guarda un artículo en la base de datos.
    Retorna True si se insertó, False si ya existía (URL duplicada).
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO news_articles
            (title, content, summary, url, source, language, published_at, category, keywords, asset_related)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article.get("title", ""),
            article.get("content", ""),
            article.get("summary", ""),
            article.get("url", ""),
            article.get("source", ""),
            article.get("language", "es"),
            article.get("published_at", ""),
            article.get("category", "finance"),
            json.dumps(article.get("keywords", []), ensure_ascii=False),
            article.get("asset_related", ""),
        ))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error guardando artículo: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: Scrapers de medios ecuatorianos
# ════════════════════════════════════════════════════════════════════════════════

class ElComercioScraper:
    """
    Scraper para El Comercio Ecuador — sección Economía.
    URL base: https://www.elcomercio.com/seccion/economia/
    """

    BASE_URL = "https://www.elcomercio.com"
    SECTION_URL = "https://www.elcomercio.com/seccion/economia/"
    SOURCE_NAME = "El Comercio Ecuador"

    def __init__(self, session: requests.Session):
        self.session = session
        self.scraped = 0
        self.errors = 0

    def get_article_links(self, pages: int = 3) -> list[str]:
        """Obtiene URLs de artículos de la sección economía."""
        links = []
        for page in range(1, pages + 1):
            url = f"{self.SECTION_URL}?page={page}" if page > 1 else self.SECTION_URL
            try:
                resp = self.session.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # El Comercio usa diferentes selectores según la versión del sitio
                # Intentar múltiples selectores
                article_tags = (
                    soup.find_all("article", class_=lambda x: x and "post" in x.lower())
                    or soup.find_all("div", class_=lambda x: x and "article" in x.lower())
                    or soup.find_all("h2", class_=lambda x: x and "title" in x.lower())
                )

                for tag in article_tags:
                    a_tag = tag.find("a", href=True)
                    if a_tag:
                        href = a_tag["href"]
                        if not href.startswith("http"):
                            href = self.BASE_URL + href
                        if href not in links and "/economia/" in href:
                            links.append(href)

                logger.info(f"  El Comercio página {page}: {len(links)} links acumulados")
                time.sleep(2)

            except Exception as e:
                logger.warning(f"  Error El Comercio página {page}: {e}")
                self.errors += 1

        return links[:50]  # Limitar por sesión

    def scrape_article(self, url: str) -> Optional[dict]:
        """Extrae el contenido de un artículo individual."""
        try:
            resp = self.session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Título
            title = (
                soup.find("h1")
                or soup.find("h1", class_=lambda x: x and "title" in str(x).lower())
            )
            title_text = title.get_text(strip=True) if title else ""

            # Fecha de publicación
            date_tag = (
                soup.find("time")
                or soup.find(attrs={"itemprop": "datePublished"})
                or soup.find("span", class_=lambda x: x and "date" in str(x).lower())
            )
            pub_date = ""
            if date_tag:
                pub_date = date_tag.get("datetime", date_tag.get_text(strip=True))

            # Contenido del artículo
            content_div = (
                soup.find("div", class_=lambda x: x and "article-body" in str(x).lower())
                or soup.find("div", class_=lambda x: x and "content" in str(x).lower())
                or soup.find("article")
            )
            content = ""
            if content_div:
                paragraphs = content_div.find_all("p")
                content = " ".join(p.get_text(strip=True) for p in paragraphs)

            if not title_text or not content:
                return None

            # Palabras clave financieras relevantes para Ecuador
            keywords = extract_financial_keywords(title_text + " " + content)

            return {
                "title": title_text,
                "content": content[:5000],  # Limitar a 5000 chars
                "summary": content[:500],
                "url": url,
                "source": self.SOURCE_NAME,
                "language": "es",
                "published_at": pub_date,
                "category": "finance",
                "keywords": keywords,
                "asset_related": detect_asset(title_text + " " + content),
            }

        except Exception as e:
            logger.debug(f"  Error scraping {url}: {e}")
            self.errors += 1
            return None

    def run(self, conn: sqlite3.Connection, pages: int = 5) -> int:
        """Ejecuta el scraping completo de El Comercio."""
        logger.info("📰 Iniciando scraper: El Comercio Ecuador")
        links = self.get_article_links(pages=pages)
        saved = 0

        for i, url in enumerate(links, 1):
            article = self.scrape_article(url)
            if article and save_article(conn, article):
                saved += 1
                logger.debug(f"  [{i}/{len(links)}] ✅ {article['title'][:60]}...")
            time.sleep(1.5)  # Respetar el servidor

        self.scraped = saved
        logger.info(f"  El Comercio: {saved} artículos nuevos guardados")
        return saved


class PrimiciasScraper:
    """
    Scraper para Primicias.ec — sección Economía.
    URL: https://www.primicias.ec/noticias/economia/
    """

    BASE_URL = "https://www.primicias.ec"
    SECTION_URL = "https://www.primicias.ec/noticias/economia/"
    SOURCE_NAME = "Primicias Ecuador"

    def __init__(self, session: requests.Session):
        self.session = session

    def run(self, conn: sqlite3.Connection, pages: int = 5) -> int:
        """Ejecuta el scraping de Primicias."""
        logger.info("📰 Iniciando scraper: Primicias.ec")
        saved = 0

        for page in range(1, pages + 1):
            url = f"{self.SECTION_URL}?page={page}" if page > 1 else self.SECTION_URL
            try:
                resp = self.session.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                articles = soup.find_all("article") or soup.find_all(
                    "div", class_=lambda x: x and ("card" in str(x) or "item" in str(x))
                )

                for art in articles:
                    a_tag = art.find("a", href=True)
                    h_tag = art.find(["h2", "h3", "h4"])
                    if not a_tag or not h_tag:
                        continue

                    href = a_tag["href"]
                    if not href.startswith("http"):
                        href = self.BASE_URL + href

                    title_text = h_tag.get_text(strip=True)

                    # Intentar obtener resumen/lead
                    p_tag = art.find("p")
                    summary = p_tag.get_text(strip=True) if p_tag else ""

                    # Fecha
                    time_tag = art.find("time")
                    pub_date = time_tag.get("datetime", "") if time_tag else ""

                    if title_text:
                        article_data = {
                            "title": title_text,
                            "content": summary,
                            "summary": summary[:300],
                            "url": href,
                            "source": self.SOURCE_NAME,
                            "language": "es",
                            "published_at": pub_date,
                            "category": "finance",
                            "keywords": extract_financial_keywords(title_text + " " + summary),
                            "asset_related": detect_asset(title_text + " " + summary),
                        }
                        if save_article(conn, article_data):
                            saved += 1

                time.sleep(2)

            except Exception as e:
                logger.warning(f"  Primicias página {page}: {e}")

        logger.info(f"  Primicias: {saved} artículos nuevos")
        return saved


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: NewsAPI
# ════════════════════════════════════════════════════════════════════════════════

def fetch_newsapi(
    conn: sqlite3.Connection,
    query: str = "mercado financiero Ecuador economía",
    days_back: int = 30,
    language: str = "es",
    page_size: int = 100,
) -> int:
    """
    Descarga noticias desde NewsAPI.org.
    Requiere NEWSAPI_KEY en .env

    Args:
        query:     Términos de búsqueda
        days_back: Días hacia atrás desde hoy
        language:  Idioma ('es' para español, 'en' para inglés)

    Returns:
        Número de artículos guardados
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        logger.warning("⚠️  NEWSAPI_KEY no configurada. Saltando NewsAPI.")
        return 0

    logger.info(f"📡 Descargando noticias desde NewsAPI (query: '{query}')...")

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": min(page_size, 100),
        "apiKey": api_key,
    }

    saved = 0
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        logger.info(f"  NewsAPI: {data.get('totalResults', 0)} resultados, procesando {len(articles)}")

        for art in articles:
            content = art.get("content", "") or art.get("description", "")
            article_data = {
                "title": art.get("title", ""),
                "content": content,
                "summary": art.get("description", "")[:500],
                "url": art.get("url", ""),
                "source": f"NewsAPI:{art.get('source', {}).get('name', 'Unknown')}",
                "language": language,
                "published_at": art.get("publishedAt", ""),
                "category": "finance",
                "keywords": extract_financial_keywords(
                    (art.get("title", "") or "") + " " + (content or "")
                ),
                "asset_related": detect_asset(
                    (art.get("title", "") or "") + " " + (content or "")
                ),
            }
            if save_article(conn, article_data):
                saved += 1

    except Exception as e:
        logger.error(f"Error NewsAPI: {e}")

    logger.info(f"  NewsAPI: {saved} artículos nuevos")
    return saved


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: Utilidades NLP básicas
# ════════════════════════════════════════════════════════════════════════════════

FINANCIAL_KEYWORDS_ES = [
    "bolsa", "acciones", "mercado", "inversión", "economía", "dólar", "precio",
    "finanzas", "banco", "crédito", "inflación", "pib", "exportación", "importación",
    "petróleo", "banano", "camarón", "crypto", "bitcoin", "ethereum", "blockchain",
    "trading", "rendimiento", "dividendo", "deuda", "déficit", "superávit",
    "BCE", "Banco Central", "SBS", "Superintendencia", "BIESS",
    "BVQ", "Bolsa de Quito", "BVG", "Bolsa de Guayaquil",
    "riesgo país", "EMBI", "deuda externa", "reservas internacionales",
    "liquidez", "solvencia", "cartera", "morosidad",
]

ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "satoshi", "crypto bitcoin"],
    "ETH": ["ethereum", "ether", "eth"],
    "ADA": ["cardano", "ada"],
    "BNB": ["binance coin", "bnb"],
    "SOL": ["solana", "sol"],
    "PETRÓLEO": ["petróleo", "oil", "crude", "WTI", "Brent", "petroecuador"],
    "ORO": ["oro", "gold", "XAU"],
    "BANANO": ["banano", "banana", "exportación banano"],
    "CAMARÓN": ["camarón", "shrimp", "acuacultura"],
    "SPY": ["S&P 500", "S&P500", "sp500"],
}


def extract_financial_keywords(text: str) -> list[str]:
    """Extrae palabras clave financieras del texto."""
    text_lower = text.lower()
    found = [kw for kw in FINANCIAL_KEYWORDS_ES if kw.lower() in text_lower]
    return found[:10]  # Máximo 10 keywords


def detect_asset(text: str) -> str:
    """Detecta el activo financiero mencionado en el texto."""
    text_lower = text.lower()
    for asset, keywords in ASSET_KEYWORDS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            return asset
    return "GENERAL"


# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: Pipeline principal
# ════════════════════════════════════════════════════════════════════════════════

def run_news_scraping(
    sources: str = "all",
    days_back: int = 30,
    pages_per_source: int = 5,
) -> dict:
    """
    Ejecuta el pipeline completo de scraping de noticias.

    Args:
        sources:          'all', 'elcomercio', 'primicias', 'newsapi'
        days_back:        Días hacia atrás para NewsAPI
        pages_per_source: Páginas a scrapear por medio local

    Returns:
        Resumen de resultados
    """
    logger.info("=" * 60)
    logger.info("  TFM — Pipeline de Recolección de Noticias")
    logger.info(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    conn = init_database()
    session = requests.Session()
    session.headers.update(HEADERS)

    results = {
        "run_date": datetime.now().isoformat(),
        "sources": {},
    }

    if sources in ("all", "elcomercio"):
        scraper = ElComercioScraper(session)
        n = scraper.run(conn, pages=pages_per_source)
        results["sources"]["elcomercio"] = n

    if sources in ("all", "primicias"):
        scraper = PrimiciasScraper(session)
        n = scraper.run(conn, pages=pages_per_source)
        results["sources"]["primicias"] = n

    if sources in ("all", "newsapi"):
        # Múltiples queries para cubrir distintos ángulos
        queries = [
            "mercado financiero Ecuador",
            "economía Ecuador 2024 2025",
            "bitcoin crypto Ecuador",
            "bolsa valores Ecuador inversión",
            "banco central Ecuador inflación",
        ]
        total_newsapi = 0
        for q in queries:
            total_newsapi += fetch_newsapi(conn, query=q, days_back=days_back)
            time.sleep(1)
        results["sources"]["newsapi"] = total_newsapi

    # Estadísticas finales
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM news_articles")
    total = cursor.fetchone()[0]
    conn.close()

    results["total_in_db"] = total
    results["session_total"] = sum(results["sources"].values())

    summary_path = DATA_DIR / "scraping_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info(f"  ✅ Scraping completado")
    logger.info(f"  📰 Artículos esta sesión: {results['session_total']}")
    logger.info(f"  🗄️  Total en BD: {total}")
    logger.info(f"  📁 BD: {DB_PATH}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFM News Scraper")
    parser.add_argument("--source", default="all",
                        choices=["all", "elcomercio", "primicias", "newsapi"],
                        help="Fuente de noticias a scrapear")
    parser.add_argument("--days", type=int, default=30,
                        help="Días hacia atrás para NewsAPI")
    parser.add_argument("--pages", type=int, default=5,
                        help="Páginas por fuente local")
    args = parser.parse_args()

    run_news_scraping(
        sources=args.source,
        days_back=args.days,
        pages_per_source=args.pages,
    )
