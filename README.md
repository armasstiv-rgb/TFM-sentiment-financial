# TFM — Integración de Algoritmos de Aprendizaje Profundo para el Análisis de Sentimiento en Mercados Financieros

**Máster Universitario en Análisis y Visualización de Datos Masivos / Visual Analytics and Big Data**  
**Universidad Internacional de La Rioja (UNIR)**  
**Director:** Cigales Canga Jesús  
**Autores:** Armas Silva Stiv Alexis | Armas Silva Jonathan Fernando | Requelme Campoverde Adrian Alexander

---

## 🎯 Descripción del Proyecto

Sistema avanzado de análisis de sentimiento financiero que integra algoritmos de Deep Learning (FinBERT, BETO, RoBERTa-es) con modelos predictivos de precios (LSTM, TFT) para cuantificar el impacto del sentimiento del mercado en la dinámica de activos financieros, con enfoque en el mercado ecuatoriano.

---

## 📁 Estructura del Proyecto

```
tfm-sentiment-financial/
├── data/
│   ├── raw/
│   │   ├── news/          # Noticias crudas + SQLite DB
│   │   ├── tweets/        # Tweets + SQLite DB
│   │   └── prices/        # Precios OHLCV CSV
│   ├── processed/         # Datos procesados y etiquetados
│   └── datasets/          # Datasets externos (FinancialPhraseBank, etc.)
│
├── notebooks/
│   ├── 01_data_collection.ipynb      # Recolección y exploración inicial
│   ├── 02_eda_preprocessing.ipynb    # EDA + Preprocesamiento
│   ├── 03_sentiment_models.ipynb     # Entrenamiento modelos NLP
│   ├── 04_sentiment_index.ipynb      # ISF + Análisis causalidad
│   ├── 05_price_prediction.ipynb     # LSTM + TFT con sentimiento
│   └── 06_backtesting.ipynb          # Backtesting estrategia trading
│
├── src/
│   ├── ingestion/
│   │   ├── price_fetcher.py          # Precios: Yahoo Finance + CoinGecko
│   │   ├── news_scraper.py           # Noticias: El Comercio, Primicias, NewsAPI
│   │   └── twitter_collector.py      # Tweets: snscrape / Twitter API v2
│   ├── preprocessing/
│   │   └── text_cleaner.py           # Pipeline NLP: limpieza y normalización
│   ├── models/
│   │   ├── finbert_baseline.py       # FinBERT (baseline inglés)
│   │   ├── beto_finetuned.py         # BETO fine-tuning español financiero
│   │   └── lstm_predictor.py         # LSTM + TFT predicción precios
│   ├── analysis/
│   │   ├── sentiment_index.py        # Índice de Sentimiento Financiero (ISF)
│   │   └── correlation_analysis.py   # Granger causality + correlaciones
│   └── dashboard/
│       └── app.py                    # Dashboard Streamlit
│
├── models/                # Modelos entrenados guardados
├── reports/
│   ├── figures/           # Gráficos y visualizaciones
│   └── metrics/           # Métricas de evaluación
├── .env.example           # Template de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/[tu_usuario]/tfm-sentiment-financial.git
cd tfm-sentiment-financial
```

### 2. Crear entorno virtual
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus API keys
```

### 5. Configurar spaCy (modelo español)
```bash
python -m spacy download es_core_news_sm
```

---

## 📊 Fuentes de Datos

### Datos de Precios
| Fuente | Tickers | Período |
|---|---|---|
| Yahoo Finance | SPY, EEM, ILF, BTC-USD, ETH-USD, GLD, USO | 2022–2025 |
| CoinGecko | BTC, ETH, ADA, BNB, SOL | 2022–2025 |
| BCE Ecuador (FRED) | Inflación, PIB, Desempleo | 2020–2025 |

### Datos de Texto
| Fuente | Tipo | Idioma |
|---|---|---|
| El Comercio Ecuador | Noticias financieras | Español |
| Primicias.ec | Noticias económicas | Español |
| El Universo | Sección economía | Español |
| NewsAPI | Noticias internacionales | Español/Inglés |
| Twitter/X (snscrape) | Tweets financieros | Español |

### Datasets de Fine-Tuning
| Dataset | Tamaño | Fuente |
|---|---|---|
| Financial PhraseBank | 4,845 frases | HuggingFace |
| SemEval-2017 Task 5 | ~1,500 items | HuggingFace |
| TASS 2020 | ~7,000 tweets | tass.sepln.org |

---

## 🤖 Modelos Implementados

| Modelo | Tipo | Idioma | Uso |
|---|---|---|---|
| **FinBERT** | Transformer | Inglés | Baseline NLP |
| **BETO fine-tuned** | Transformer | Español | Modelo principal |
| **RoBERTa-es** | Transformer | Español | Comparación |
| **Gemini API** | LLM | Multi | IA Generativa + auto-etiquetado |
| **LSTM** | RNN | — | Predicción precios |
| **TFT** | Transformer temporal | — | Predicción multivariante |
| **XGBoost** | Gradient Boosting | — | Baseline predictivo |

---

## 🏃 Uso Rápido

### Recolección de Datos
```bash
# Precios históricos
python src/ingestion/price_fetcher.py

# Noticias ecuatorianas
python src/ingestion/news_scraper.py --source all --days 30

# Tweets (sin API, gratis)
python src/ingestion/twitter_collector.py --method snscrape --days 30
```

### Preprocesamiento
```bash
# Prueba del pipeline de limpieza
python src/preprocessing/text_cleaner.py
```

### Notebooks (orden recomendado)
```bash
jupyter lab notebooks/
# Ejecutar en orden: 01 → 02 → 03 → 04 → 05 → 06
```

### Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 📈 Métricas de Evaluación

### Módulo NLP
- Accuracy, F1-Macro, MCC, Cohen's Kappa

### Módulo Predictivo
- MAE, RMSE, Directional Accuracy
- Sharpe Ratio, Max Drawdown (backtesting)

---

## 📚 Referencias Clave

- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
- Huang et al. (2023). FinBERT: A Pretrained Language Model for Financial Communications
- Bollen et al. (2011). Twitter Mood Predicts the Stock Market
- Tetlock (2007). Giving Content to Investor Sentiment
- Loughran & McDonald (2011). When is a Liability Not a Liability?

---

## 📝 Licencia

Proyecto académico — TFM UNIR 2026. Uso educativo.
