# 📊 ForecastIQ — AI-Powered Sales Analytics Platform

A production-grade SaaS application for restaurant/product sales forecasting using advanced deep learning models (LSTM, Attention-LSTM, Transformer) with a highly interactive Streamlit dashboard.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python src/data_generation.py
```
This creates `data/sales_data.csv` (~100K+ rows of realistic multi-city restaurant sales data).

### 3. (Optional) Train Models
```bash
python run_pipeline.py
```
This runs the full ML pipeline: preprocessing → feature engineering → training (LSTM, Attention-LSTM, Transformer) → evaluation. Results are saved to `data/models/` and `data/model_comparison.json`.

### 4. Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```
Access at http://localhost:8501

## 📂 Project Structure
```
├── data/                    # Generated datasets & model artifacts
├── src/
│   ├── data_generation.py   # Realistic sales data generator
│   ├── preprocessing.py     # Cleaning, imputation, encoding
│   ├── feature_engineering.py# Lags, rolling stats, Fourier terms
│   ├── models/
│   │   ├── lstm.py          # Stacked LSTM
│   │   ├── attention.py     # Attention-based LSTM
│   │   ├── transformer.py   # Transformer encoder
│   ├── training/train.py    # Training pipeline
│   ├── evaluation/evaluate.py
├── app/
│   ├── streamlit_app.py     # Main dashboard
│   ├── pages/
│       ├── 1_📊_Overview.py
│       ├── 2_🔬_Deep_Analytics.py
│       ├── 3_🔮_Forecasting.py
│       ├── 4_🍕_Product_Intelligence.py
│       ├── 5_📋_Data_Explorer.py
├── utils/helpers.py         # Metrics, config, logging
├── run_pipeline.py          # Full ML pipeline runner
├── requirements.txt
```

## 🧠 Models
| Model | Architecture | Key Feature |
|-------|-------------|-------------|
| LSTM | Stacked 2-layer LSTM | Baseline sequential model |
| Attention LSTM | LSTM + Bahdanau Attention | Learns to prioritize timesteps |
| Transformer | Encoder-only + Positional Encoding | Multi-head self-attention |

## 📊 Dashboard Pages
1. **Overview** — KPIs, revenue trends, category & city breakdown
2. **Deep Analytics** — Drill-down (City→Restaurant→Product), heatmaps, weather impact, cohort analysis
3. **Forecasting** — Model comparison, confidence bands, scenario simulation, anomaly detection
4. **Product Intelligence** — Top/bottom products, demand patterns, treemap, AI recommendations
5. **Data Explorer** — Full-text search, column filters, CSV export

## ⚙️ Tech Stack
- **ML**: PyTorch, scikit-learn
- **Visualization**: Plotly
- **Dashboard**: Streamlit
- **Data**: Pandas, NumPy
