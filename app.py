"""
Bankruptcy Prediction Website — Flask Backend
Run with: python app.py
Then open: http://localhost:5000
"""

import io
import base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap

from flask import Flask, render_template, request

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
print("Loading models...")

scaler     = joblib.load("models/scaler.pkl")
rf_model   = joblib.load("models/rf_model.pkl")
xgb_model  = joblib.load("models/xgb_model.pkl")
meta_model = joblib.load("models/meta_model.pkl")

yearly_emb = pd.read_csv("data/yearly_market_embeddings.csv")
macro_df   = pd.read_csv("data/macro_features.csv")
train_df   = pd.read_csv("data/final_master.csv")

print("✅ All models loaded")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FINANCIAL_COLS = [f'X{i}' for i in range(1, 19)]

MACRO_COLS = [
    'gdp_growth',
    'inflation',
    'unemployment'
]

EMB_DIM = 32

LSTM_COLS = [f'lstm_emb_{i}' for i in range(EMB_DIM)]

ALL_FEATURES = (
    FINANCIAL_COLS +
    MACRO_COLS +
    LSTM_COLS
)

# ─────────────────────────────────────────────
# Feature Labels
# ─────────────────────────────────────────────
FEATURE_LABELS = {
    'X1':  'Working Capital / Total Assets',
    'X2':  'Retained Earnings / Total Assets',
    'X3':  'EBIT / Total Assets',
    'X4':  'Market Value of Equity / Total Liabilities',
    'X5':  'Revenue / Total Assets',
    'X6':  'Net Income',
    'X7':  'Total Liabilities / Total Assets',
    'X8':  'Current Assets / Current Liabilities',
    'X9':  'Total Assets',
    'X10': 'Revenue Growth Rate',
    'X11': 'Operating Cash Flow',
    'X12': 'Return on Assets (ROA)',
    'X13': 'Return on Equity (ROE)',
    'X14': 'Interest Coverage Ratio',
    'X15': 'Book Value per Share',
    'X16': 'Cash & Short-term Investments',
    'X17': 'Long-term Debt',
    'X18': 'Gross Profit Margin',
}

# ─────────────────────────────────────────────
# Clip bounds
# ─────────────────────────────────────────────
clip_bounds = {}

for col in FINANCIAL_COLS:
    vals = train_df[col].dropna()

    clip_bounds[col] = (
        float(vals.min()),
        float(vals.max())
    )

print("✅ Clip bounds computed")

# ─────────────────────────────────────────────
# Percentiles
# ─────────────────────────────────────────────
train_pct = {}

for col in FINANCIAL_COLS + MACRO_COLS:
    train_pct[col] = sorted(
        train_df[col].dropna().values
    )

# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────
shap_explainer = None

try:
    shap_explainer = shap.TreeExplainer(xgb_model)
    print("✅ SHAP initialized")

except Exception as e:
    print(f"⚠️ SHAP disabled: {e}")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def clip_financial_inputs(financial_ratios):

    clipped = {}

    for col in FINANCIAL_COLS:

        val = financial_ratios.get(col, 0.0)

        lo, hi = clip_bounds[col]

        clipped[col] = float(
            np.clip(val, lo, hi)
        )

    return clipped


def get_percentile(feat, val):

    arr = train_pct.get(feat, [])

    if len(arr) == 0:
        return 50

    return int(
        np.searchsorted(arr, val) / len(arr) * 100
    )


def risk_info(prob):

    if prob < 0.08:
        return {
            "label": "LOW RISK",
            "color": "#22c55e",
            "emoji": "🟢",
            "summary": "The company appears financially healthy.",
            "css": "low"
        }

    elif prob < 0.20:
        return {
            "label": "MODERATE RISK",
            "color": "#f59e0b",
            "emoji": "🟡",
            "summary": "Some financial weaknesses detected.",
            "css": "moderate"
        }

    elif prob < 0.30:
        return {
            "label": "HIGH RISK",
            "color": "#f97316",
            "emoji": "🟠",
            "summary": "Significant financial stress indicators present.",
            "css": "high"
        }

    elif prob < 0.50:
        return {
            "label": "VERY HIGH RISK",
            "color": "#ef4444",
            "emoji": "🔴",
            "summary": "Strong bankruptcy warning signals detected.",
            "css": "very-high"
        }

    else:
        return {
            "label": "CRITICAL RISK",
            "color": "#7f1d1d",
            "emoji": "💀",
            "summary": "Extreme bankruptcy risk.",
            "css": "critical"
        }

# ─────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────
def run_prediction(financial_ratios, year, company_name):

    # Step 0 — clipping
    financial_ratios_clipped = clip_financial_inputs(
        financial_ratios
    )

    # Step 1 — macro
    macro_row = macro_df[
        macro_df['year'] == year
    ]

    if macro_row.empty:
        macro_row = (
            macro_df
            .sort_values('year')
            .iloc[[-1]]
        )

    macro_vals = {
        'gdp_growth':
            float(macro_row['gdp_growth'].values[0]),

        'inflation':
            float(macro_row['inflation'].values[0]),

        'unemployment':
            float(macro_row['unemployment'].values[0]),
    }

    # Step 2 — embeddings
    emb_row = yearly_emb[
        yearly_emb['year'] == year
    ]

    if emb_row.empty:
        emb_row = (
            yearly_emb
            .sort_values('year')
            .iloc[[-1]]
        )

    lstm_vals = {
        f'lstm_emb_{i}':
            float(
                emb_row[f'lstm_emb_{i}'].values[0]
            )

        for i in range(EMB_DIM)
    }

    # Step 3 — feature vector
    row = {
        **financial_ratios_clipped,
        **macro_vals,
        **lstm_vals
    }

    X_input = pd.DataFrame([row])[ALL_FEATURES]

    X_scaled = scaler.transform(X_input)

    # Step 4 — base models
    prob_rf = float(
        rf_model.predict_proba(X_scaled)[0, 1]
    )

    prob_xgb = float(
        xgb_model.predict_proba(X_scaled)[0, 1]
    )

    # Step 5 — meta model
    lstm_emb_single = np.array([
        list(lstm_vals.values())
    ])

    X_meta = np.hstack([
        np.array([[prob_rf, prob_xgb]]),
        lstm_emb_single
    ])

    final_prob = float(
        meta_model.predict_proba(X_meta)[0, 1]
    )

    # DEBUG
    print("\n========== MODEL DEBUG ==========")
    print("RF:", prob_rf)
    print("XGB:", prob_xgb)
    print("FINAL:", final_prob)
    print("=================================\n")

    # Step 6 — SHAP
    shap_vals = np.zeros(X_scaled.shape[1])

    if shap_explainer is not None:

        try:
            raw = shap_explainer.shap_values(X_scaled)

            if isinstance(raw, list):
                shap_vals = raw[1][0]
            else:
                shap_vals = raw[0]

        except Exception as e:
            print(f"⚠️ SHAP failed: {e}")

    shap_series = pd.Series(
        shap_vals,
        index=ALL_FEATURES
    )

    shap_fin = (
        shap_series[
            FINANCIAL_COLS + MACRO_COLS
        ]
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    explanations = []

    for feat, shap_val in shap_fin.head(8).items():

        val = financial_ratios.get(
            feat,
            macro_vals.get(feat, 0.0)
        )

        pct = get_percentile(feat, val)

        impact = abs(shap_val)

        severity = (
            "strongly"
            if impact > 0.15 else
            "moderately"
            if impact > 0.07 else
            "slightly"
        )

        increases = shap_val > 0

        if pct <= 20:
            context = "very low"

        elif pct <= 40:
            context = "below average"

        elif pct <= 60:
            context = "around average"

        elif pct <= 80:
            context = "above average"

        else:
            context = "very high"

        explanations.append({
            "feature":
                FEATURE_LABELS.get(feat, feat),

            "value":
                round(val, 3),

            "context":
                context,

            "severity":
                severity,

            "increases":
                increases,

            "shap":
                round(float(shap_val), 4),
        })

    fin_shap = float(
        np.abs(shap_vals[:18]).sum()
    )

    mac_shap = float(
        np.abs(shap_vals[18:21]).sum()
    )

    lstm_shap = float(
        np.abs(shap_vals[21:]).sum()
    )

    total_s = (
        fin_shap +
        mac_shap +
        lstm_shap +
        1e-8
    )

    top_risk = [
        e for e in explanations
        if e["increases"]
    ][:3]

    if final_prob < 0.08:

        recommendation = (
            "No immediate concerns."
        )

        rec_level = "safe"

    elif final_prob < 0.18:

        recommendation = (
            "Watch: " +
            ", ".join([
                e["feature"]
                for e in top_risk
            ])
        )

        rec_level = "watch"

    elif final_prob < 0.30:

        recommendation = (
            "Action recommended for: " +
            ", ".join([
                e["feature"]
                for e in top_risk
            ])
        )

        rec_level = "action"

    else:

        recommendation = (
            "URGENT — Immediate restructuring advised."
        )

        rec_level = "urgent"

    return {

        "company":
            company_name,

        "year":
            year,

        "prob_rf":
            round(prob_rf * 100, 1),

        "prob_xgb":
            round(prob_xgb * 100, 1),

        "final_prob":
            round(final_prob * 100, 1),

        "final_prob_raw":
            final_prob,

        "risk":
            risk_info(final_prob),

        "explanations":
            explanations,

        "macro":
            macro_vals,

        "recommendation":
            recommendation,

        "rec_level":
            rec_level,

        "shap_groups": {
            "financial":
                round(fin_shap / total_s * 100, 1),

            "macro":
                round(mac_shap / total_s * 100, 1),

            "lstm":
                round(lstm_shap / total_s * 100, 1),
        },

        "_shap_fin":
            shap_fin,
    }

# ─────────────────────────────────────────────
# SHAP Chart
# ─────────────────────────────────────────────
def make_shap_chart(shap_fin):

    top = shap_fin.head(10)

    labels = [
        FEATURE_LABELS.get(f, f)
        .split('/')[0]
        .strip()[:30]

        for f in top.index
    ]

    colors = [
        '#ef4444' if v > 0
        else '#22c55e'

        for v in top.values
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    fig.patch.set_facecolor('#0f172a')

    ax.set_facecolor('#0f172a')

    bars = ax.barh(
        range(len(top)),
        top.values,
        color=colors,
        edgecolor='#1e293b',
        height=0.6
    )

    ax.set_yticks(range(len(top)))

    ax.set_yticklabels(
        labels,
        fontsize=9,
        color='#cbd5e1'
    )

    ax.axvline(
        0,
        color='#475569',
        linewidth=1
    )

    plt.tight_layout()

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format='png',
        dpi=130,
        facecolor='#0f172a',
        bbox_inches='tight'
    )

    plt.close()

    buf.seek(0)

    return base64.b64encode(
        buf.read()
    ).decode('utf-8')

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():

    return render_template(
        'index.html',
        feature_labels=FEATURE_LABELS
    )

@app.route('/predict', methods=['POST'])
def predict():

    try:

        company_name = (
            request.form
            .get(
                'company_name',
                'Unknown Company'
            )
            .strip()
        )

        if company_name == '':
            company_name = 'Unknown Company'

        year = int(
            request.form.get('year', 2008)
        )

        financial_ratios = {}

        for col in FINANCIAL_COLS:

            raw = request.form.get(
                col,
                ''
            ).strip()

            financial_ratios[col] = (
                float(raw)
                if raw != ''
                else 0.0
            )

        result = run_prediction(
            financial_ratios,
            year,
            company_name
        )

        shap_chart = make_shap_chart(
            result.pop('_shap_fin')
        )

        result['shap_chart'] = shap_chart

        return render_template(
            'result.html',
            r=result,
            feature_labels=FEATURE_LABELS
        )

    except Exception as e:

        import traceback
        traceback.print_exc()

        return render_template(
            'index.html',
            feature_labels=FEATURE_LABELS,
            error=f"Prediction failed: {str(e)}"
        )

if __name__ == '__main__':
    app.run(
        debug=False,
        port=5000
    )