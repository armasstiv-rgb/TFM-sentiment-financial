"""
Dataset Downloader — Fine-Tuning Datasets
==========================================
Descarga los datasets pre-etiquetados necesarios para el fine-tuning
de los modelos de sentimiento financiero.

Datasets:
  1. Financial PhraseBank (HuggingFace)
  2. SemEval-2017 Task 5 (HuggingFace)
  3. FinancialPhraseBank traducido al español (generado con Gemini)

Uso:
    python src/ingestion/dataset_downloader.py

Autores: Armas Silva Stiv, Armas Silva Jonathan, Requelme Adrian
TFM — UNIR — 2026
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dataset_downloader")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


def download_financial_phrasebank():
    """
    Descarga el dataset Financial PhraseBank desde HuggingFace.
    Es el dataset de referencia para análisis de sentimiento financiero en inglés.
    4,845 frases etiquetadas: positive / negative / neutral
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets no instalado. Ejecuta: pip install datasets")
        return None

    logger.info("⬇️  Descargando Financial PhraseBank desde HuggingFace...")

    try:
        # Versión con acuerdo unánime entre anotadores (más confiable)
        dataset = load_dataset(
            "financial_phrasebank",
            "sentences_allagree",
            trust_remote_code=True,
        )

        # Guardar en CSV
        import pandas as pd
        df = pd.DataFrame(dataset["train"])
        df.columns = ["text", "label"]

        # Mapear etiquetas numéricas a texto
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        df["label_text"] = df["label"].map(label_map)

        output_path = DATASETS_DIR / "financial_phrasebank_en.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Financial PhraseBank: {len(df)} registros → {output_path.name}")
        logger.info(f"   Distribución: {df['label_text'].value_counts().to_dict()}")

        return df

    except Exception as e:
        logger.error(f"Error descargando Financial PhraseBank: {e}")
        return None


def download_semeval_2017():
    """
    Descarga SemEval-2017 Task 5 — Sentiment en headlines financieras.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets no instalado.")
        return None

    logger.info("⬇️  Descargando SemEval-2017 Task 5...")

    try:
        dataset = load_dataset(
            "zeroshot/twitter-financial-news-sentiment",
            trust_remote_code=True,
        )
        import pandas as pd
        df_train = pd.DataFrame(dataset["train"])
        df_valid = pd.DataFrame(dataset["validation"])
        df = pd.concat([df_train, df_valid], ignore_index=True)

        output_path = DATASETS_DIR / "twitter_financial_news_sentiment_en.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Twitter Financial News Sentiment: {len(df)} registros")
        return df

    except Exception as e:
        logger.warning(f"SemEval/Twitter dataset: {e}. Intentando alternativa...")

    # Alternativa: cargar datos de sentimiento financiero en inglés
    try:
        dataset = load_dataset(
            "ProsusAI/finqa",
            trust_remote_code=True,
        )
        logger.info("✅ FinQA alternativo descargado")
    except Exception as e2:
        logger.error(f"Error alternativa: {e2}")

    return None


def create_spanish_finance_dataset_sample():
    """
    Crea un dataset de muestra manual en español para bootstrapping.
    
    Este es el punto de partida para el fine-tuning en español.
    Se amplía después con pseudo-labels de Gemini.
    """
    logger.info("📝 Creando dataset de muestra en español...")

    samples = [
        # POSITIVO
        {"text": "Bitcoin alcanza nuevo máximo histórico y supera los $70,000 dólares", "label": "positive", "label_text": "positive"},
        {"text": "La bolsa de Quito cierra con ganancias del 2.3% impulsada por acciones bancarias", "label": "positive", "label_text": "positive"},
        {"text": "Ecuador registra superávit comercial por tercer mes consecutivo", "label": "positive", "label_text": "positive"},
        {"text": "Ethereum sube 8% tras anuncio de actualización de red exitosa", "label": "positive", "label_text": "positive"},
        {"text": "El Banco Pichincha reporta utilidades récord en el primer trimestre", "label": "positive", "label_text": "positive"},
        {"text": "Las exportaciones de camarón ecuatoriano crecen 15% en 2024", "label": "positive", "label_text": "positive"},
        {"text": "Inversores recuperan confianza en mercados emergentes de Latinoamérica", "label": "positive", "label_text": "positive"},
        {"text": "El PIB de Ecuador crece 2.1% superando las expectativas del mercado", "label": "positive", "label_text": "positive"},
        {"text": "Bitcoin se recupera con fuerza tras corrección temporal del mercado", "label": "positive", "label_text": "positive"},
        {"text": "Las criptomonedas registran su mejor semana del año con ganancias generalizadas", "label": "positive", "label_text": "positive"},

        # NEGATIVO
        {"text": "Bitcoin colapsa un 15% en las últimas 24 horas generando pánico en el mercado", "label": "negative", "label_text": "negative"},
        {"text": "Ecuador enfrenta déficit fiscal creciente que preocupa a inversores internacionales", "label": "negative", "label_text": "negative"},
        {"text": "La inflación en Ecuador sube al 4.2% afectando el poder adquisitivo", "label": "negative", "label_text": "negative"},
        {"text": "Crisis bancaria en Ecuador: tres entidades reportan problemas de liquidez", "label": "negative", "label_text": "negative"},
        {"text": "Ethereum cae 12% tras noticias regulatorias en Estados Unidos", "label": "negative", "label_text": "negative"},
        {"text": "Las exportaciones de petróleo ecuatoriano disminuyen ante caída del precio del barril", "label": "negative", "label_text": "negative"},
        {"text": "El desempleo en Ecuador sube al 4.8% en el tercer trimestre del año", "label": "negative", "label_text": "negative"},
        {"text": "Mercados de Latinoamérica caen arrastrados por incertidumbre económica global", "label": "negative", "label_text": "negative"},
        {"text": "Bitcoin en mínimos de seis meses tras intensificación de regulación cripto", "label": "negative", "label_text": "negative"},
        {"text": "La deuda externa de Ecuador alcanza niveles preocupantes según el FMI", "label": "negative", "label_text": "negative"},

        # NEUTRAL
        {"text": "El Banco Central del Ecuador publica su informe trimestral de política monetaria", "label": "neutral", "label_text": "neutral"},
        {"text": "Bitcoin cotiza alrededor de los $45,000 dólares en las últimas horas", "label": "neutral", "label_text": "neutral"},
        {"text": "La Superintendencia de Bancos presenta nuevas normativas de capital para 2025", "label": "neutral", "label_text": "neutral"},
        {"text": "El índice IECX de la Bolsa de Valores de Quito cierra sin cambios significativos", "label": "neutral", "label_text": "neutral"},
        {"text": "La Reserva Federal mantiene las tasas de interés sin modificaciones", "label": "neutral", "label_text": "neutral"},
        {"text": "Ethereum actualiza su protocolo según lo planificado por los desarrolladores", "label": "neutral", "label_text": "neutral"},
        {"text": "El volumen de transacciones en la bolsa de Guayaquil se mantiene estable", "label": "neutral", "label_text": "neutral"},
        {"text": "Ecuador firma acuerdo comercial de rutina con Colombia para el sector agrícola", "label": "neutral", "label_text": "neutral"},
        {"text": "Las tasas activas referenciales del BCE se mantienen en los niveles anteriores", "label": "neutral", "label_text": "neutral"},
        {"text": "Bitcoin registra movimientos laterales en un rango de precio definido", "label": "neutral", "label_text": "neutral"},
    ]

    import pandas as pd
    df = pd.DataFrame(samples)
    output_path = DATASETS_DIR / "finance_sentiment_es_seed.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"✅ Dataset semilla español: {len(df)} ejemplos → {output_path.name}")
    logger.info(f"   Distribución: {df['label_text'].value_counts().to_dict()}")

    return df


def auto_label_with_gemini(
    texts: list[str],
    batch_size: int = 10,
    max_texts: int = 500,
) -> list[dict]:
    """
    Usa la API de Gemini para auto-etiquetar textos financieros en español.
    Esta es la componente de IA Generativa del TFM.

    El prompt está diseñado para obtener etiquetas consistentes:
    positive / negative / neutral con justificación.

    Args:
        texts:      Lista de textos a etiquetar
        batch_size: Textos por llamada a la API
        max_texts:  Límite total de textos

    Returns:
        Lista de dicts con {text, label, confidence, reasoning}
    """
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("⚠️  GEMINI_API_KEY no configurada. Saltando auto-etiquetado.")
        return []

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    SENTIMENT_PROMPT = """Eres un experto en análisis de sentimiento de textos financieros en español latinoamericano.

Para cada texto, clasifica el sentimiento como: POSITIVE, NEGATIVE, o NEUTRAL.

Criterios:
- POSITIVE: El texto indica buenas noticias, subidas de precio, crecimiento, oportunidades de inversión, o perspectivas optimistas para los mercados o activos.
- NEGATIVE: El texto indica malas noticias, caídas de precio, crisis, pérdidas, riesgos elevados, o perspectivas pesimistas.
- NEUTRAL: El texto es informativo/descriptivo sin dirección clara de sentimiento, anuncia eventos sin valoración positiva ni negativa, o habla de cambios mínimos.

IMPORTANTE: Responde ÚNICAMENTE en formato JSON válido como array.

Textos a clasificar:
{texts}

Responde con un JSON array en este formato exacto:
[
  {{"text_index": 0, "label": "POSITIVE", "confidence": 0.95, "reasoning": "breve justificación"}},
  {{"text_index": 1, "label": "NEGATIVE", "confidence": 0.87, "reasoning": "breve justificación"}},
  ...
]"""

    results = []
    texts_to_process = texts[:max_texts]

    import time
    for i in range(0, len(texts_to_process), batch_size):
        batch = texts_to_process[i:i + batch_size]
        texts_formatted = "\n".join(f"[{j}] {t}" for j, t in enumerate(batch))

        try:
            prompt = SENTIMENT_PROMPT.format(texts=texts_formatted)
            response = model.generate_content(prompt)
            response_text = response.text.strip()

            # Extraer JSON de la respuesta
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            batch_results = json.loads(response_text)
            for result in batch_results:
                idx = result.get("text_index", 0)
                if idx < len(batch):
                    results.append({
                        "text": batch[idx],
                        "label": result.get("label", "NEUTRAL").lower(),
                        "confidence": result.get("confidence", 0.5),
                        "reasoning": result.get("reasoning", ""),
                        "source": "gemini_auto_label",
                    })

            logger.info(f"  Batch {i//batch_size + 1}: {len(batch_results)} textos etiquetados")
            time.sleep(2)  # Rate limiting Gemini

        except Exception as e:
            logger.warning(f"  Error batch {i//batch_size + 1}: {e}")

    return results


def run_dataset_download():
    """Ejecuta la descarga completa de datasets."""
    logger.info("=" * 60)
    logger.info("  TFM — Descarga de Datasets de Fine-Tuning")
    logger.info("=" * 60)

    # 1. Financial PhraseBank (inglés, para FinBERT)
    fpb = download_financial_phrasebank()

    # 2. Twitter Financial News Sentiment
    semeval = download_semeval_2017()

    # 3. Dataset semilla en español (manual)
    seed_es = create_spanish_finance_dataset_sample()

    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Datasets listos")
    logger.info(f"  📁 Directorio: {DATASETS_DIR}")
    logger.info("=" * 60)

    return {
        "financial_phrasebank": len(fpb) if fpb is not None else 0,
        "twitter_financial": len(semeval) if semeval is not None else 0,
        "seed_spanish": len(seed_es) if seed_es is not None else 0,
    }


if __name__ == "__main__":
    run_dataset_download()
