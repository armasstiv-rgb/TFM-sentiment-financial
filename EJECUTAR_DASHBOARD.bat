@echo off
title TFM Dashboard - Analisis de Sentimiento Financiero
cd /d "c:\Users\StivArmas\OneDrive - ROBALINO\Documentos\MASTER\TFM\tfm-sentiment-financial"
echo Iniciando dashboard...
echo Abre tu navegador en: http://localhost:8501
echo.
python -m streamlit run app/dashboard.py --server.port 8501
pause
