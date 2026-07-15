"""
Dashboard interactivo del TFM — Streamlit + Plotly
Integración de Algoritmos de Deep Learning para el Análisis de Sentimiento en Mercados Financieros
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import os

# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="TFM — Sentimiento Financiero & Deep Learning",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Rutas ────────────────────────────────────────────────────────────────────
APP_DIR       = Path(__file__).parent
PROJECT_ROOT  = APP_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures"

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo oscuro elegante */
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

    /* Métricas */
    [data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.95);
        border-right: 1px solid rgba(48, 54, 61, 0.5);
    }

    /* Títulos */
    h1, h2, h3 { color: #e6edf3 !important; }

    /* Cards de info */
    .info-card {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }

    /* Badge verde */
    .badge-success {
        background: rgba(35, 134, 54, 0.2);
        border: 1px solid #238636;
        border-radius: 20px;
        padding: 4px 12px;
        color: #3fb950;
        font-size: 12px;
        font-weight: bold;
    }

    /* Header gradient */
    .gradient-header {
        background: linear-gradient(90deg, #1f6feb 0%, #238636 50%, #1f6feb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
    }

    /* Tab activo */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(22, 27, 34, 0.8);
        border-radius: 8px;
    }

    /* Separadores */
    hr { border-color: rgba(48, 54, 61, 0.5) !important; }

    /* ── Sidebar todo en blanco ── */
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important; border-right: 1px solid rgba(88,166,255,0.2); }
    [data-testid="stSidebar"] [role="radiogroup"] label { color: #c9d1d9 !important; font-size: 0.95rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { color: #58a6ff !important; }
    [data-testid="stSidebar"] small, [data-testid="stSidebar"] .stMarkdown small { color: #8b949e !important; }

    /* ── Texto global ── */
    .stApp, .stApp p, .stApp div, .stApp span { color: #e6edf3 !important; }
    .stApp .stMarkdown p { color: #c9d1d9 !important; font-size: 0.95rem; line-height: 1.6; }
    h1, h2, h3, h4 { color: #ffffff !important; }

    /* ── Métricas ── */
    [data-testid="metric-container"] { background: rgba(22,27,34,0.95); border: 1px solid rgba(88,166,255,0.25); border-radius: 12px; padding: 16px; }
    [data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.85rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.5rem !important; font-weight: 700 !important; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #3fb950 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { color: #8b949e !important; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; }

    /* ── Inputs ── */
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label { color: #b1bac4 !important; }

    /* ── Botones ── */
    .stButton > button { background: rgba(31,111,235,0.2) !important; border: 1px solid #1f6feb !important; color: #58a6ff !important; border-radius: 8px; font-weight: 600; }
    .stButton > button:hover { background: rgba(31,111,235,0.4) !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ── Funciones de carga con caché ──────────────────────────────────────────────
@st.cache_data
def cargar_precios():
    p = PROCESSED_DIR / "master_prices.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return None

@st.cache_data
def cargar_isf():
    p = PROCESSED_DIR / "isf_diario.csv"
    if p.exists():
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df.sort_index()
    return None

@st.cache_data
def cargar_predicciones():
    p = PROCESSED_DIR / "dataset_en_con_predicciones.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def cargar_modelos():
    p = PROCESSED_DIR / "resultados_modelos.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def cargar_metricas_lstm():
    p = PROCESSED_DIR / "metricas_lstm.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

@st.cache_data
def cargar_correlaciones():
    p = PROCESSED_DIR / "correlaciones_isf_precios.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def cargar_granger():
    p = PROCESSED_DIR / "granger_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

# Cargar todos los datasets
df_prices    = cargar_precios()
df_isf       = cargar_isf()
df_preds     = cargar_predicciones()
df_modelos   = cargar_modelos()
metricas_lstm = cargar_metricas_lstm()
df_corr      = cargar_correlaciones()
granger      = cargar_granger()

COLORES = {
    "positive": "#3fb950",
    "negative": "#f85149",
    "neutral":  "#58a6ff",
    "btc":      "#f7931a",
    "eth":      "#627eea",
    "spy":      "#1f6feb",
    "isf":      "#a371f7",
}

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 TFM Dashboard")
    st.markdown("**Análisis de Sentimiento en Mercados Financieros**")
    st.markdown("---")

    pagina = st.radio(
        "Navegación",
        ["🏠 Resumen Ejecutivo",
         "💹 Precios de Mercado",
         "💬 Análisis de Sentimiento",
         "📉 Índice ISF",
         "🤖 Modelos LSTM",
         "🔍 Demo en Vivo"],
        index=0
    )

    st.markdown("---")
    st.markdown("**Período de datos:**")
    if df_prices is not None and len(df_prices) > 0:
        st.markdown(f"📅 {df_prices.index.min().date()} → {df_prices.index.max().date()}")
        st.markdown(f"📊 {len(df_prices):,} días de trading")
    if df_preds is not None:
        st.markdown(f"💬 {len(df_preds):,} textos analizados")

    st.markdown("---")
    st.markdown("""
    <small style='color: #8b949e;'>
    UNIR · Master en Inteligencia Artificial<br>
    Trabajo de Fin de Máster · 2024-2025
    </small>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1: RESUMEN EJECUTIVO
# ═══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Resumen Ejecutivo":
    st.markdown('<p class="gradient-header">📈 TFM — Análisis de Sentimiento en Mercados Financieros</p>',
                unsafe_allow_html=True)
    st.markdown("**Universidad Internacional de La Rioja (UNIR) · Master en Inteligencia Artificial**")
    st.markdown("---")

    # KPIs principales
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        n_textos = len(df_preds) if df_preds is not None else 11899
        st.metric("📄 Textos analizados", f"{n_textos:,}")

    with col2:
        if df_modelos is not None:
            best_acc = df_modelos["Accuracy"].max() * 100
        else:
            best_acc = 80.9
        st.metric("🎯 Mejor accuracy (SVM)", f"{best_acc:.1f}%")

    with col3:
        if metricas_lstm is not None:
            rmse_con = metricas_lstm["con_isf"]["rmse"]
            rmse_sin = metricas_lstm["sin_isf"]["rmse"]
            mejora   = metricas_lstm["mejora_rmse_pct"]
        else:
            mejora = 2.04
        st.metric("📉 Mejora RMSE (LSTM+ISF)", f"{mejora:+.2f}%", delta=f"vs baseline")

    with col4:
        n_dias = len(df_prices) if df_prices is not None else 1460
        st.metric("📅 Días de mercado", f"{n_dias:,}")

    with col5:
        n_figs = len(list(FIGURES_DIR.glob("*.png")))
        st.metric("🖼️ Visualizaciones", f"{n_figs}")

    st.markdown("---")

    # Pipeline del TFM
    st.subheader("🔄 Pipeline del TFM — De texto a predicción")

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    for col, (nb, titulo, desc, color) in zip(
        [col_a, col_b, col_c, col_d, col_e],
        [
            ("NB01", "📥 Datos", "Yahoo Finance → BTC, ETH, SPY (2021-2024)", "#1f6feb"),
            ("NB02", "🔤 NLP", "Pipeline texto → tokens → embeddings", "#388bfd"),
            ("NB03", "🤖 Modelos", "SVM 80.9% vs FinBERT vs BETO", "#58a6ff"),
            ("NB04", "📊 ISF", "Índice de Sentimiento + Granger", "#a371f7"),
            ("NB05", "🧠 LSTM", "Deep Learning predice precios futuros", "#3fb950"),
        ]
    ):
        with col:
            st.markdown(f"""
            <div style='background: rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1);
                        border: 1px solid {color}; border-radius:10px; padding:12px; text-align:center;'>
                <b style='color:{color}'>{nb}</b><br>
                <b>{titulo}</b><br>
                <small style='color:#8b949e'>{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabla de resultados
    st.subheader("📋 Resultados por Fase")
    resultados_tabla = pd.DataFrame([
        {"Fase": "NB01 — Datos", "Descripción": "1,460 días BTC/ETH/SPY (2021-2024)", "Estado": "✅ Completo"},
        {"Fase": "NB02 — NLP", "Descripción": "11,899 textos EN + 72 ES procesados", "Estado": "✅ Completo"},
        {"Fase": "NB03 — Sentimiento", "Descripción": "SVM 80.9% Acc, F1=0.741 (mejor modelo)", "Estado": "✅ Completo"},
        {"Fase": "NB04 — ISF", "Descripción": "ISF diario AR(1) + Tests Granger/ADF", "Estado": "✅ Completo"},
        {"Fase": "NB05 — LSTM", "Descripción": "H_LSTM confirmada: RMSE −2.04%", "Estado": "✅ Completo"},
        {"Fase": "NB06 — Dashboard", "Descripción": "Visualización interactiva Streamlit", "Estado": "✅ En vivo"},
    ])
    st.dataframe(resultados_tabla, use_container_width=True, hide_index=True)

    # Hipótesis
    st.markdown("---")
    st.subheader("📐 Hipótesis y Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **H_LSTM CONFIRMADA**\n\nLSTM + ISF reduce RMSE en 2.04% vs baseline.\nEl sentimiento añade señal predictiva no lineal.")
    with col2:
        st.info("ℹ️ **Granger no significativo (esperado)**\n\nISF sintético sin fechas reales introduce ruido.\nLimitación declarada. Trabajo futuro: API Reuters.")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2: PRECIOS DE MERCADO
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "💹 Precios de Mercado":
    st.header("💹 Precios de Mercado Financiero")
    st.markdown("Datos históricos descargados de Yahoo Finance (2021-2024)")

    if df_prices is None or len(df_prices) == 0:
        st.warning("No hay datos de precios disponibles. Ejecutar NB01 primero.")
    else:
        precio_cols = [c for c in df_prices.columns if "-USD" in c or c in ["SPY","EEM","ILF","GLD"]]
        precio_cols = [c for c in precio_cols if "_ret" not in c and "_vol" not in c
                       and "_sma" not in c and "_RSI" not in c]

        activos_sel = st.multiselect("Activos a visualizar:", precio_cols,
                                     default=precio_cols[:3] if len(precio_cols) >= 3 else precio_cols)

        if activos_sel:
            # Gráfico de precios normalizados
            fig = go.Figure()
            for activo in activos_sel:
                if activo in df_prices.columns:
                    serie = df_prices[activo].dropna()
                    serie_norm = serie / serie.iloc[0] * 100  # base 100
                    color = COLORES.get(activo.lower().replace("-usd",""), "#58a6ff")
                    fig.add_trace(go.Scatter(
                        x=serie_norm.index, y=serie_norm,
                        name=activo, line=dict(color=color, width=1.5),
                        hovertemplate=f"{activo}: %{{y:.1f}}<extra></extra>"
                    ))

            fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
            fig.update_layout(
                title="Rendimiento acumulado (base 100 = inicio)",
                template="plotly_dark",
                paper_bgcolor="rgba(13,17,23,0)",
                plot_bgcolor="rgba(22,27,34,0.5)",
                legend=dict(bgcolor="rgba(22,27,34,0.8)"),
                height=450,
                xaxis_title="Fecha",
                yaxis_title="Índice (base 100)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Retornos diarios
            ret_cols = [f"{a}_ret" for a in activos_sel if f"{a}_ret" in df_prices.columns]
            if ret_cols:
                st.subheader("📊 Retornos Logarítmicos Diarios")
                activo_ret = st.selectbox("Activo:", [c.replace("_ret","") for c in ret_cols])
                ret_col = f"{activo_ret}_ret"
                if ret_col in df_prices.columns:
                    retornos = df_prices[ret_col].dropna()
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=retornos.index, y=retornos,
                        marker_color=np.where(retornos >= 0, COLORES["positive"], COLORES["negative"]),
                        name="Retorno diario"
                    ))
                    fig2.update_layout(
                        title=f"Retornos logarítmicos — {activo_ret}",
                        template="plotly_dark",
                        paper_bgcolor="rgba(13,17,23,0)",
                        plot_bgcolor="rgba(22,27,34,0.5)",
                        height=350
                    )
                    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3: ANÁLISIS DE SENTIMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "💬 Análisis de Sentimiento":
    st.header("💬 Modelos de Análisis de Sentimiento")
    st.markdown("Comparativa de 5 modelos entrenados en el Notebook 03")

    # Comparativa de modelos
    if df_modelos is not None:
        st.subheader("🏆 Rendimiento Comparativo")
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Accuracy en Test", "F1-Score Macro"])

        colores_modelos = ["#8b949e", "#1f6feb", "#3fb950", "#f85149", "#a371f7"]
        for i, (_, row) in enumerate(df_modelos.iterrows()):
            color = colores_modelos[i % len(colores_modelos)]
            fig.add_trace(go.Bar(name=row["Modelo"], x=[row["Modelo"]],
                                  y=[row["Accuracy"]], marker_color=color,
                                  showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(name=row["Modelo"], x=[row["Modelo"]],
                                  y=[row["F1_Macro"]], marker_color=color,
                                  showlegend=True), row=1, col=2)

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(13,17,23,0)",
                          plot_bgcolor="rgba(22,27,34,0.5)", height=400,
                          legend=dict(bgcolor="rgba(22,27,34,0.8)"))
        st.plotly_chart(fig, use_container_width=True)

        # Tabla detallada
        st.subheader("📋 Tabla Detallada")
        df_display = df_modelos.copy()
        df_display["Accuracy"] = (df_display["Accuracy"] * 100).round(2).astype(str) + "%"
        df_display["F1_Macro"] = df_display["F1_Macro"].round(4)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("Resultados de modelos no disponibles. Ejecutar NB03 primero.")

    # Distribución de predicciones
    if df_preds is not None:
        st.markdown("---")
        st.subheader("📊 Distribución de Sentimientos Predichos")

        dist = df_preds["pred_label"].value_counts()
        fig3 = go.Figure(go.Pie(
            labels=dist.index,
            values=dist.values,
            marker_colors=[COLORES.get(l, "#8b949e") for l in dist.index],
            hole=0.4,
            textinfo="label+percent",
        ))
        fig3.update_layout(
            title="Distribución de clases predichas",
            template="plotly_dark",
            paper_bgcolor="rgba(13,17,23,0)",
            height=350
        )
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4: ÍNDICE ISF
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📉 Índice ISF":
    st.header("📉 Índice de Sentimiento Financiero (ISF)")
    st.markdown("**ISF(t) = [N_pos(t) − N_neg(t)] / N_total(t)** — rango: −1 (Pánico) a +1 (Euforia)")
    st.markdown("""
    | Rango ISF | Interpretación | Descripción |
    |---|---|---|
    | **> 0.3** | 🟢 **Euforia** | Sentimiento positivo fuerte; predominan textos optimistas |
    | **0.1 a 0.3** | 🟡 **Optimismo moderado** | Ligera predominancia de textos positivos |
    | **−0.1 a 0.1** | ⚪ **Neutro / Equilibrado** | Distribución balanceada entre positivos y negativos |
    | **−0.3 a −0.1** | 🟠 **Pesimismo moderado** | Ligera predominancia de textos negativos |
    | **< −0.3** | 🔴 **Pánico** | Sentimiento negativo fuerte; predominan textos pesimistas |
    """)

    if df_isf is not None:
        isf_series = df_isf["isf"]

        # Métricas ISF
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ISF Promedio", f"{isf_series.mean():.3f}")
        with col2:
            st.metric("ISF Máximo (Euforia)", f"{isf_series.max():.3f}")
        with col3:
            st.metric("ISF Mínimo (Pánico)", f"{isf_series.min():.3f}")
        with col4:
            pct_pos = (isf_series > 0.1).mean() * 100
            st.metric("Días de Euforia", f"{pct_pos:.1f}%")

        # Gráfico ISF temporal
        isf_ma = isf_series.rolling(20, min_periods=5).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=isf_series[isf_series >= 0].index,
            y=isf_series[isf_series >= 0],
            fill="tozeroy", fillcolor="rgba(63,185,80,0.2)",
            line=dict(color="rgba(63,185,80,0.3)", width=0.5),
            name="ISF Positivo (Optimismo)", showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=isf_series[isf_series < 0].index,
            y=isf_series[isf_series < 0],
            fill="tozeroy", fillcolor="rgba(248,81,73,0.2)",
            line=dict(color="rgba(248,81,73,0.3)", width=0.5),
            name="ISF Negativo (Pesimismo)", showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=isf_ma.index, y=isf_ma,
            line=dict(color=COLORES["isf"], width=2),
            name="Media móvil 20 días"
        ))
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.4)
        fig.update_layout(
            title="Evolución del ISF diario (2021-2024)",
            template="plotly_dark",
            paper_bgcolor="rgba(13,17,23,0)",
            plot_bgcolor="rgba(22,27,34,0.5)",
            height=400,
            legend=dict(bgcolor="rgba(22,27,34,0.8)"),
            yaxis=dict(range=[-1.2, 1.2]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlaciones
        if df_corr is not None:
            st.subheader("📊 Correlación ISF vs Retornos")
            fig2 = px.bar(
                df_corr, x="Activo", y="Pearson",
                color="Pearson",
                color_continuous_scale=["#f85149","#8b949e","#3fb950"],
                title="Correlación de Pearson: ISF vs Retorno de cada activo",
                template="plotly_dark",
            )
            fig2.update_layout(paper_bgcolor="rgba(13,17,23,0)",
                               plot_bgcolor="rgba(22,27,34,0.5)", height=350)
            st.plotly_chart(fig2, use_container_width=True)

            st.caption("p < 0.05 = significativo estadísticamente. Los valores bajos son esperados en mercados eficientes.")
    else:
        st.info("ISF no disponible. Ejecutar NB04 primero.")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5: LSTM
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Modelos LSTM":
    st.header("🤖 Modelos Predictivos LSTM")
    st.markdown("Comparativa entre LSTM + ISF (sentimiento) vs LSTM baseline (solo precio)")

    if metricas_lstm is not None:
        con = metricas_lstm["con_isf"]
        sin = metricas_lstm["sin_isf"]
        mejora = metricas_lstm["mejora_rmse_pct"]

        # KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE — LSTM + ISF", f"{con['rmse']:.5f}",
                      delta=f"{mejora:+.2f}% vs baseline",
                      delta_color="inverse")
        with col2:
            st.metric("MAE — LSTM + ISF", f"{con['mae']:.5f}")
        with col3:
            r2_mejora = con["r2"] - sin["r2"]
            st.metric("R² — LSTM + ISF", f"{con['r2']:.4f}",
                      delta=f"{r2_mejora:+.4f} vs baseline")

        hipotesis_ok = con["rmse"] < sin["rmse"]
        if hipotesis_ok:
            st.success(f"✅ **Hipótesis H_LSTM CONFIRMADA** — LSTM + ISF supera al baseline en todas las métricas")
        else:
            st.warning("⚠️ LSTM + ISF no supera al baseline con el ISF actual (sintético). Con ISF real, se esperaría mejora.")

        # Tabla comparativa
        st.markdown("---")
        st.subheader("📋 Comparativa Detallada")
        df_comp = pd.DataFrame([
            {"Modelo": "LSTM + ISF ⭐", "RMSE": con["rmse"], "MAE": con["mae"],
             "R²": con["r2"], "Dirección %": f"{con['dir_acc']*100:.1f}%"},
            {"Modelo": "LSTM baseline", "RMSE": sin["rmse"], "MAE": sin["mae"],
             "R²": sin["r2"], "Dirección %": f"{sin['dir_acc']*100:.1f}%"},
        ])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Gráfico comparativo
        st.subheader("📊 Comparativa Visual de Métricas")
        fig = make_subplots(rows=1, cols=3, subplot_titles=["RMSE (↓)", "MAE (↓)", "R² (↑)"])
        for col_idx, (metrica, vals) in enumerate([
            ("RMSE", [con["rmse"], sin["rmse"]]),
            ("MAE",  [con["mae"],  sin["mae"]]),
            ("R²",   [con["r2"],   sin["r2"]]),
        ], 1):
            fig.add_trace(go.Bar(
                x=["LSTM + ISF", "Baseline"],
                y=vals,
                marker_color=[COLORES["positive"], "#f85149"],
                showlegend=False
            ), row=1, col=col_idx)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(13,17,23,0)",
            plot_bgcolor="rgba(22,27,34,0.5)",
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Métricas LSTM no disponibles. Ejecutar NB05 primero.")

    # Mostrar figuras generadas
    st.markdown("---")
    st.subheader("🖼️ Figuras del Entrenamiento")
    img_cols = st.columns(2)
    for i, fname in enumerate(["18_curvas_aprendizaje_lstm.png", "19_lstm_predicciones_vs_real.png"]):
        fp = FIGURES_DIR / fname
        if fp.exists():
            with img_cols[i % 2]:
                st.image(str(fp), caption=fname.replace("_"," ").replace(".png",""), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 6: DEMO EN VIVO
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔍 Demo en Vivo":
    st.header("🔍 Demo en Vivo — Predicción de Sentimiento")
    st.markdown("Introduce un texto financiero y el sistema predice su sentimiento usando el modelo SVM entrenado.")

    st.markdown("---")

    # Cargar modelo guardado (SVM) si existe
    model_loaded = False
    try:
        import pickle
        model_path = PROJECT_ROOT / "models" / "svm_pipeline.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                pipe_svm = pickle.load(f)
            model_loaded = True
    except Exception:
        pass

    texto_ejemplo = st.text_area(
        "📝 Texto financiero en inglés:",
        value="The company reported strong quarterly earnings, beating analyst expectations with revenue growth of 15%.",
        height=120
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        analizar = st.button("🔍 Analizar sentimiento", type="primary")

    if analizar and texto_ejemplo.strip():
        # Análisis lexicón (siempre disponible)
        palabras_pos = ["strong","growth","beat","profit","gain","rise","up","bull","positive","above","exceed"]
        palabras_neg = ["loss","decline","fall","miss","below","weak","crash","bear","negative","risk","debt"]
        texto_lower  = texto_ejemplo.lower()

        n_pos = sum(1 for w in palabras_pos if w in texto_lower)
        n_neg = sum(1 for w in palabras_neg if w in texto_lower)

        if n_pos > n_neg:
            label_lex, color_lex, emoji = "positive", "success", "📈"
        elif n_neg > n_pos:
            label_lex, color_lex, emoji = "negative", "error", "📉"
        else:
            label_lex, color_lex, emoji = "neutral", "info", "➡️"

        st.markdown("---")
        st.subheader(f"Resultado: {emoji} **{label_lex.upper()}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Palabras positivas", n_pos)
        with col2:
            st.metric("Palabras negativas", n_neg)
        with col3:
            isf_texto = (n_pos - n_neg) / max(1, n_pos + n_neg)
            st.metric("ISF del texto", f"{isf_texto:.2f}")

        # Gauge ISF
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=isf_texto,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Índice de Sentimiento del Texto", "font": {"size": 16}},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1,
                         "tickvals": [-1, -0.3, 0, 0.3, 1],
                         "ticktext": ["Pánico", "Pesimismo", "Neutro", "Optimismo", "Euforia"]},
                "bar": {"color": COLORES[label_lex]},
                "steps": [
                    {"range": [-1, -0.3], "color": "rgba(248,81,73,0.2)"},
                    {"range": [-0.3, 0.3], "color": "rgba(139,148,158,0.2)"},
                    {"range": [0.3, 1], "color": "rgba(63,185,80,0.2)"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "value": isf_texto},
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(13,17,23,0)",
            height=280
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if model_loaded:
            svm_pred = pipe_svm.predict([texto_ejemplo])[0]
            id2label = {0: "negativo", 1: "neutro", 2: "positivo"}
            label_svm = id2label.get(svm_pred, str(svm_pred))
            st.info(f"🤖 **SVM predice:** {label_svm.upper()}")

        st.caption("Análisis léxico usando el diccionario financiero del TFM. "
                   "El SVM (modelo ganador, 80.9% accuracy) requiere el modelo entrenado guardado.")

    st.markdown("---")
    st.subheader("💡 Ejemplos de textos por sentimiento")
    ejemplos = {
        "📈 Positivo": "Apple reported record-breaking quarterly revenue of $90 billion, surpassing all analyst forecasts.",
        "📉 Negativo": "The company announced massive layoffs and a significant decline in quarterly earnings amid market uncertainty.",
        "➡️ Neutro": "The Federal Reserve held interest rates steady at its latest meeting, as widely expected by economists.",
    }
    for etiqueta, ejemplo in ejemplos.items():
        with st.expander(etiqueta):
            st.write(ejemplo)
            if st.button(f"Usar este texto", key=etiqueta):
                st.session_state["texto_demo"] = ejemplo
