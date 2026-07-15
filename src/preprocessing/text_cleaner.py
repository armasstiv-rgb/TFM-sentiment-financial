"""
text_cleaner.py
===============
Pipeline de preprocesamiento NLP para datos textuales del TFM.

Funciones principales:
  - Limpieza básica (URLs, emojis, caracteres especiales)
  - Normalización de texto en español
  - Tokenización (con HuggingFace Tokenizer)
  - Detección de idioma
  - Segmentación de oraciones
  - Generación de features de texto

Autores: Armas Silva Stiv, Armas Silva Jonathan, Requelme Adrian
TFM — UNIR — 2026
"""

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger("text_cleaner")

# ── Patrones de limpieza ──────────────────────────────────────────────────────
PATTERNS = {
    "urls":          re.compile(r"https?://\S+|www\.\S+"),
    "emails":        re.compile(r"\S+@\S+\.\S+"),
    "mentions":      re.compile(r"@\w+"),
    "hashtags_keep": re.compile(r"#(\w+)"),           # Mantener texto del hashtag
    "hashtags_rm":   re.compile(r"#\w+"),             # Eliminar hashtags completos
    "numbers_rm":    re.compile(r"\b\d+([.,]\d+)*\b"),
    "special_chars": re.compile(r"[^\w\s\u00C0-\u024F.,;:!?¡¿\-]"),  # Mantiene acentos ES
    "whitespace":    re.compile(r"\s+"),
    "repeated_chars":re.compile(r"(.)\1{3,}"),        # caracteres repetidos 3+ veces
    "cashtags":      re.compile(r"\$[A-Z]{1,5}"),     # $BTC $ETH etc.
}

STOP_WORDS_ES_FINANCE = {
    # Stop words españolas básicas (complementar con NLTK si disponible)
    "de", "la", "el", "en", "y", "a", "que", "se", "del", "los", "las",
    "un", "una", "por", "con", "para", "es", "al", "lo", "su", "una",
    "como", "más", "pero", "sus", "le", "ya", "o", "fue", "este", "ha",
    "si", "porque", "muy", "sin", "sobre", "también", "ser", "hay",
    "nos", "me", "mi", "te", "tu", "él", "ella", "esto", "esta",
    "son", "han", "era", "fue", "van", "hoy", "ayer", "haber",
    # Palabras muy comunes en tweets que no aportan valor semántico
    "via", "rt", "cc", "vía", "hilo", "thread",
}


# ════════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE LIMPIEZA
# ════════════════════════════════════════════════════════════════════════════════

def remove_urls(text: str) -> str:
    """Elimina URLs del texto."""
    return PATTERNS["urls"].sub(" ", text)


def remove_emails(text: str) -> str:
    """Elimina emails."""
    return PATTERNS["emails"].sub(" ", text)


def normalize_mentions(text: str, replace_with: str = "@USER") -> str:
    """
    Normaliza menciones de Twitter (@usuario).
    Por defecto las reemplaza con @USER para preservar estructura del tweet.
    """
    return PATTERNS["mentions"].sub(replace_with, text)


def process_hashtags(text: str, keep_text: bool = True) -> str:
    """
    Procesa hashtags.
    Si keep_text=True, mantiene el texto (#bitcoin → bitcoin)
    Si keep_text=False, elimina el hashtag completo
    """
    if keep_text:
        return PATTERNS["hashtags_keep"].sub(r"\1", text)
    return PATTERNS["hashtags_rm"].sub(" ", text)


def process_cashtags(text: str) -> str:
    """Normaliza cashtags financieros ($BTC → BTC)."""
    return PATTERNS["cashtags"].sub(
        lambda m: m.group(0)[1:],  # Elimina el $
        text
    )


def normalize_accents(text: str) -> str:
    """
    Normaliza caracteres Unicode manteniendo acentos españoles.
    Elimina caracteres de control y caracteres extraños.
    """
    # Normalizar a NFC (mantiene letras con acento como caracteres únicos)
    text = unicodedata.normalize("NFC", text)
    # Eliminar caracteres de control
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return text


def fix_repeated_chars(text: str) -> str:
    """
    Elimina repeticiones excesivas de caracteres.
    'loooooco' → 'looco', 'jajajaja' → 'jaja'
    """
    return PATTERNS["repeated_chars"].sub(r"\1\1", text)


def expand_abbreviations(text: str) -> str:
    """Expande abreviaciones financieras comunes en español."""
    abbreviations = {
        r"\bBCE\b": "Banco Central Ecuador",
        r"\bBVQ\b": "Bolsa de Valores de Quito",
        r"\bBVG\b": "Bolsa de Valores de Guayaquil",
        r"\bSBS\b": "Superintendencia de Bancos",
        r"\bSECCO\b": "Superintendencia de Compañías",
        r"\bBIESS\b": "Banco del Instituto Ecuatoriano de Seguridad Social",
        r"\bFMI\b": "Fondo Monetario Internacional",
        r"\bBM\b": "Banco Mundial",
        r"\bBID\b": "Banco Interamericano de Desarrollo",
        r"\bCAF\b": "Corporación Andina de Fomento",
        r"\bTIIE\b": "Tasa de Interés Interbancaria de Equilibrio",
        r"\bBTC\b": "bitcoin",
        r"\bETH\b": "ethereum",
        r"\bDeFi\b": "finanzas descentralizadas",
        r"\bNFT\b": "token no fungible",
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_whitespace(text: str) -> str:
    """Normaliza espacios en blanco."""
    text = text.strip()
    text = PATTERNS["whitespace"].sub(" ", text)
    return text


def to_lowercase(text: str) -> str:
    """Convierte a minúsculas."""
    return text.lower()


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

def clean_text(
    text: str,
    remove_url: bool = True,
    remove_email: bool = True,
    normalize_mention: bool = True,
    keep_hashtag_text: bool = True,
    expand_abbrev: bool = True,
    fix_repeated: bool = True,
    lowercase: bool = True,
    min_length: int = 10,
) -> Optional[str]:
    """
    Pipeline de limpieza de texto completo.

    Args:
        text:              Texto de entrada
        remove_url:        Eliminar URLs
        remove_email:      Eliminar emails
        normalize_mention: Normalizar menciones @user → @USER
        keep_hashtag_text: Mantener texto de hashtags sin el #
        expand_abbrev:     Expandir abreviaciones financieras
        fix_repeated:      Corregir caracteres repetidos
        lowercase:         Convertir a minúsculas
        min_length:        Longitud mínima del texto limpio

    Returns:
        Texto limpio o None si es demasiado corto
    """
    if not text or not isinstance(text, str):
        return None

    # 1. Normalizar Unicode
    text = normalize_accents(text)

    # 2. Procesar cashtags ($BTC → BTC)
    text = process_cashtags(text)

    # 3. Eliminar URLs
    if remove_url:
        text = remove_urls(text)

    # 4. Eliminar emails
    if remove_email:
        text = remove_emails(text)

    # 5. Normalizar menciones
    if normalize_mention:
        text = normalize_mentions(text, replace_with="")

    # 6. Procesar hashtags
    text = process_hashtags(text, keep_text=keep_hashtag_text)

    # 7. Expandir abreviaciones
    if expand_abbrev:
        text = expand_abbreviations(text)

    # 8. Corregir repeticiones
    if fix_repeated:
        text = fix_repeated_chars(text)

    # 9. Eliminar caracteres especiales no deseados
    # Mantener letras (con acentos), números, puntuación básica
    text = re.sub(r"[^\w\s\u00C0-\u024F.,;:!?¡¿\-]", " ", text)

    # 10. Normalizar espacios
    text = clean_whitespace(text)

    # 11. Lowercase (opcional — los modelos BERT prefieren case original)
    if lowercase:
        text = to_lowercase(text)

    # Validar longitud mínima
    if len(text.split()) < min_length // 4:  # Aproximado: mínimo N/4 palabras
        return None

    return text


def clean_for_bert(text: str) -> Optional[str]:
    """
    Limpieza específica para modelos BERT/FinBERT/BETO.
    BERT es case-sensitive en algunas variantes, por lo que NO
    se convierte todo a minúsculas y se preservan más caracteres.
    """
    return clean_text(
        text,
        remove_url=True,
        remove_email=True,
        normalize_mention=True,
        keep_hashtag_text=True,
        expand_abbrev=True,
        fix_repeated=True,
        lowercase=False,  # BERT maneja case
        min_length=5,
    )


def clean_for_lstm(text: str) -> Optional[str]:
    """
    Limpieza para modelos LSTM — más agresiva, todo a minúsculas.
    """
    return clean_text(
        text,
        lowercase=True,
        min_length=10,
    )


# ════════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE ANÁLISIS DE TEXTO
# ════════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """Detecta el idioma del texto. Retorna código ISO 639-1."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42  # Reproducibilidad
        return detect(text)
    except Exception:
        return "unknown"


def count_financial_terms(text: str, lexicon: Optional[list] = None) -> dict:
    """
    Cuenta términos financieros en el texto.
    Útil para filtrar textos relevantes.

    Returns:
        {'positive': N, 'negative': N, 'neutral': N, 'total': N}
    """
    if lexicon is None:
        # Léxico financiero básico español
        positive_terms = [
            "sube", "alza", "gana", "beneficio", "crecimiento", "rentable",
            "positivo", "oportunidad", "record", "máximo", "ganancia",
            "bull", "alcista", "recuperación", "rebote", "compra", "inversión",
            "aumento", "mejora", "supera", "histórico",
        ]
        negative_terms = [
            "baja", "cae", "pérdida", "riesgo", "negativo", "crisis",
            "colapso", "quiebra", "default", "caída", "mínimo", "sell",
            "bearish", "bajista", "pánico", "recesión", "inflación alta",
            "deuda", "déficit", "retrocede", "desplome", "venta",
        ]
    else:
        positive_terms, negative_terms = lexicon[0], lexicon[1]

    text_lower = text.lower()
    pos_count = sum(1 for t in positive_terms if t in text_lower)
    neg_count = sum(1 for t in negative_terms if t in text_lower)

    return {
        "positive_terms": pos_count,
        "negative_terms": neg_count,
        "net_sentiment": pos_count - neg_count,
        "total_financial_terms": pos_count + neg_count,
    }


def extract_entities(text: str) -> dict:
    """
    Extrae entidades financieras del texto.
    Retorna diccionario con activos, organizaciones y métricas detectadas.
    """
    entities = {
        "assets": [],
        "percentages": [],
        "currencies": [],
    }

    # Porcentajes
    entities["percentages"] = re.findall(r"\d+(?:[.,]\d+)?%", text)

    # Valores monetarios
    entities["currencies"] = re.findall(
        r"(?:\$|USD|EUR|€)\s*[\d.,]+(?:\s*(?:millones?|miles|bn|B|M|K))?",
        text, re.IGNORECASE
    )

    # Activos financieros
    asset_patterns = [
        r"\b(?:BTC|ETH|ADA|SOL|BNB|XRP|USDT)\b",  # Cripto
        r"\b(?:SPY|EEM|ILF|EWZ|GLD|USO)\b",         # ETFs
        r"\b(?:bitcoin|ethereum|cardano|solana)\b",  # Cripto por nombre
    ]
    for pattern in asset_patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        entities["assets"].extend(found)

    entities["assets"] = list(set(entities["assets"]))
    return entities


# ════════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ════════════════════════════════════════════════════════════════════════════════

def batch_clean(texts: list[str], mode: str = "bert") -> list[Optional[str]]:
    """
    Limpieza por lotes de textos.

    Args:
        texts: Lista de textos a limpiar
        mode:  'bert' (preserva case) o 'lstm' (lowercase agresivo)

    Returns:
        Lista de textos limpios (None donde el texto no pasó el filtro)
    """
    cleaner = clean_for_bert if mode == "bert" else clean_for_lstm
    cleaned = []
    for text in texts:
        try:
            cleaned.append(cleaner(text))
        except Exception:
            cleaned.append(None)
    return cleaned


if __name__ == "__main__":
    # Prueba del pipeline
    test_texts = [
        "¡El #Bitcoin $BTC sube 5% hoy! 🚀🚀🚀 https://t.co/example @CryptoEcuador",
        "El BCE Ecuador anuncia nuevas medidas para controlar la inflación... #economía",
        "BREAKING: mercado de acciones COLAPSA -3% tras noticias del FMI sobre Ecuador",
        "Primicias: La BVQ registra su mayor alza del año con +2.3% en el IECX",
        "jajajajaja esto es un tweet irrelevante sin contenido financiero lol",
    ]

    print("=" * 60)
    print("  TEST: Pipeline de Limpieza de Texto")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        cleaned = clean_for_bert(text)
        entities = extract_entities(text) if cleaned else {}
        finance_score = count_financial_terms(text) if cleaned else {}

        print(f"\n[{i}] Original:  {text[:80]}")
        print(f"    Limpio:    {str(cleaned)[:80]}")
        if entities.get("assets"):
            print(f"    Activos:   {entities['assets']}")
        if finance_score.get("total_financial_terms", 0) > 0:
            print(f"    Score:     {finance_score}")
