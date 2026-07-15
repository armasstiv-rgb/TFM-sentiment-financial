"""
finbert_baseline.py
===================
Modelo FinBERT — Baseline para análisis de sentimiento financiero en inglés.

FinBERT es el modelo de referencia para NLP financiero.
Publicado por ProsusAI, fine-tuneado sobre BERT con datos financieros.

Capacidades:
  - Clasificación: positive / negative / neutral
  - Inference sobre noticias y textos financieros en inglés
  - Evaluación sobre Financial PhraseBank

Uso:
    from src.models.finbert_baseline import FinBERTClassifier
    clf = FinBERTClassifier()
    results = clf.predict(["Apple stock rises 5% after earnings beat"])

Autores: Armas Silva Stiv, Armas Silva Jonathan, Requelme Adrian
TFM — UNIR — 2026
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Union

import torch

logger = logging.getLogger("finbert_baseline")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# CLASIFICADOR FINBERT
# ════════════════════════════════════════════════════════════════════════════════

class FinBERTClassifier:
    """
    Wrapper para el modelo FinBERT de ProsusAI.
    
    El modelo está pre-entrenado en datos financieros y puede clasificar
    textos en inglés como: positive, negative, neutral.
    
    Modelo base: ProsusAI/finbert
    Referencia: Huang et al. (2023)
    """

    MODEL_NAME = "ProsusAI/finbert"
    LABELS = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self, device: str = "auto"):
        """
        Inicializa el clasificador FinBERT.

        Args:
            device: 'auto' detecta GPU, 'cpu', o 'cuda'
        """
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _setup_device(self, device: str) -> str:
        """Configura el dispositivo de cómputo."""
        if device == "auto":
            if torch.cuda.is_available():
                logger.info("🖥️  GPU detectada — usando CUDA")
                return "cuda"
            else:
                logger.info("💻 Sin GPU — usando CPU")
                return "cpu"
        return device

    def _load_model(self):
        """Carga el tokenizer y modelo FinBERT desde HuggingFace."""
        try:
            from transformers import BertTokenizer, BertForSequenceClassification

            logger.info(f"⬇️  Cargando {self.MODEL_NAME}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ FinBERT cargado en {self.device}")

        except ImportError:
            logger.error("transformers no instalado. Ejecuta: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Error cargando FinBERT: {e}")
            raise

    def predict_single(self, text: str, max_length: int = 512) -> dict:
        """
        Predice el sentimiento de un texto individual.

        Args:
            text:       Texto financiero en inglés
            max_length: Longitud máxima de tokens

        Returns:
            {label: str, score: float, scores: {positive, negative, neutral}}
        """
        import torch.nn.functional as F

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze()

        label_idx = torch.argmax(probs).item()
        label = self.LABELS[label_idx]
        score = probs[label_idx].item()

        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "label": label,
            "score": round(score, 4),
            "scores": {
                "positive": round(probs[0].item(), 4),
                "negative": round(probs[1].item(), 4),
                "neutral": round(probs[2].item(), 4),
            },
            "sentiment_score": round(probs[0].item() - probs[1].item(), 4),
        }

    def predict(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Predice el sentimiento de uno o múltiples textos.

        Args:
            texts:         Texto o lista de textos
            batch_size:    Tamaño del lote para procesamiento
            show_progress: Mostrar barra de progreso

        Returns:
            Lista de dicts con predicciones
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch = texts[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            if show_progress:
                logger.info(f"  Procesando lote {batch_idx + 1}/{total_batches} ({len(batch)} textos)")

            for text in batch:
                try:
                    result = self.predict_single(text)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"  Error prediciendo: {e}")
                    results.append({
                        "text": text[:100],
                        "label": "neutral",
                        "score": 0.0,
                        "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                        "sentiment_score": 0.0,
                        "error": str(e),
                    })

        return results

    def evaluate(
        self,
        texts: list[str],
        true_labels: list[str],
        output_report: bool = True,
    ) -> dict:
        """
        Evalúa el modelo sobre un conjunto de datos etiquetados.

        Args:
            texts:         Lista de textos
            true_labels:   Etiquetas verdaderas (positive/negative/neutral)
            output_report: Guardar reporte en disco

        Returns:
            Diccionario con métricas: accuracy, f1, mcc, etc.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report,
            matthews_corrcoef, confusion_matrix,
        )
        import numpy as np

        logger.info(f"📊 Evaluando FinBERT sobre {len(texts)} ejemplos...")
        predictions = self.predict(texts, show_progress=True)
        pred_labels = [p["label"] for p in predictions]
        pred_scores = [p["sentiment_score"] for p in predictions]

        # Métricas
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(true_labels, pred_labels)

        label_order = ["positive", "negative", "neutral"]
        clf_report = classification_report(
            true_labels, pred_labels,
            labels=label_order,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "model": "FinBERT (ProsusAI/finbert)",
            "dataset_size": len(texts),
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "mcc": round(mcc, 4),
            "classification_report": clf_report,
            "confusion_matrix": confusion_matrix(
                true_labels, pred_labels, labels=label_order
            ).tolist(),
        }

        logger.info(f"  Accuracy:    {accuracy:.4f}")
        logger.info(f"  F1-Macro:    {f1_macro:.4f}")
        logger.info(f"  F1-Weighted: {f1_weighted:.4f}")
        logger.info(f"  MCC:         {mcc:.4f}")

        if output_report:
            report_path = REPORTS_DIR / "finbert_evaluation.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"  📁 Reporte guardado: {report_path}")

        return metrics

    def compute_sentiment_index(
        self,
        texts_with_dates: list[dict],
        aggregation: str = "daily",
    ):
        """
        Computa el Índice de Sentimiento Financiero (ISF) a partir de textos.

        Args:
            texts_with_dates: Lista de {text, date, source, asset}
            aggregation:      'daily', 'weekly', 'monthly'

        Returns:
            DataFrame con ISF por período
        """
        import pandas as pd

        logger.info(f"📈 Computando ISF para {len(texts_with_dates)} textos...")

        texts = [t["text"] for t in texts_with_dates]
        predictions = self.predict(texts)

        # Combinar predicciones con metadatos
        results = []
        for meta, pred in zip(texts_with_dates, predictions):
            results.append({
                **meta,
                "sentiment_score": pred["sentiment_score"],
                "label": pred["label"],
                "confidence": pred["score"],
            })

        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Agregar por período
        if aggregation == "daily":
            freq = "D"
        elif aggregation == "weekly":
            freq = "W"
        else:
            freq = "ME"

        isf = df["sentiment_score"].resample(freq).agg(["mean", "std", "count"])
        isf.columns = ["isf_mean", "isf_std", "n_texts"]
        isf["isf_7d_ma"] = isf["isf_mean"].rolling(7).mean()

        return isf


# ════════════════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ════════════════════════════════════════════════════════════════════════════════

def demo():
    """Demostración del clasificador FinBERT."""
    test_texts = [
        "Apple stock soared 8% after reporting record quarterly earnings.",
        "Bitcoin crashes 20% amid regulatory fears, investors panic selling.",
        "The Federal Reserve maintained interest rates unchanged at its meeting.",
        "Ecuador's economy shows resilience with GDP growing 2.1% this quarter.",
        "Cryptocurrency market suffers massive sell-off as China bans crypto mining.",
        "S&P 500 reaches new all-time high driven by tech sector gains.",
    ]

    print("\n" + "=" * 60)
    print("  FinBERT Demo — Análisis de Sentimiento Financiero")
    print("=" * 60)

    clf = FinBERTClassifier()

    for text in test_texts:
        result = clf.predict_single(text)
        emoji = "🟢" if result["label"] == "positive" else "🔴" if result["label"] == "negative" else "⚪"
        print(f"\n{emoji} [{result['label'].upper():8}] ({result['score']:.2f}) {text[:70]}...")
        print(f"   Scores → pos:{result['scores']['positive']:.3f} | neg:{result['scores']['negative']:.3f} | neu:{result['scores']['neutral']:.3f}")
        print(f"   ISF Score: {result['sentiment_score']:+.3f}")


if __name__ == "__main__":
    demo()
