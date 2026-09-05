"""
Customer Churn Intelligence Platform — Premium SaaS Edition
=============================================================
Dark glassmorphism dashboard, fixed non-collapsible sidebar, AI advice cards,
explainability, segmentation, an estimated forecast, and PDF/CSV reports —
built on a Random Forest trained on the Telco Customer Churn dataset.

Run with:  streamlit run app.py
"""

import pickle
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

PRIMARY = "#6366F1"
SECONDARY = "#06B6D4"
BG = "#0F172A"
BG2 = "#111827"
CARD = "rgba(255,255,255,0.05)"
CARD_BORDER = "rgba(255,255,255,0.10)"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
DANGER = "#EF4444"
WARNING = "#F59E0B"
SUCCESS = "#22C55E"

NAV_ITEMS = [
    ("Dashboard", "📊"),
    ("Predictions", "🔮"),
    ("AI Assistant", "🤖"),
    ("What-If Simulator", "🎛️"),
    ("Reports", "📄"),
]


# ----------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        h1, h2, h3, .app-title, .section-title {{ font-family: 'Poppins', sans-serif; }}

        /* ---- Hide Streamlit chrome, but keep header container (no arrow needed
               anymore since we remove the collapse control entirely below) ---- */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"] {{visibility: hidden;}}

        /* ---- Sidebar base styling (applies at all sizes) ---- */
        section[data-testid="stSidebar"] {{
            background: {BG2};
            border-right: 1px solid {CARD_BORDER};
        }}

        /* ---- DESKTOP (>900px): sidebar is fixed, always visible, no
               collapse arrow. This is intentionally scoped to desktop only —
               see the mobile block below for why. ---- */
        @media (min-width: 901px) {{
            [data-testid="collapsedControl"] {{ display: none !important; }}
            section[data-testid="stSidebar"] {{
                min-width: 290px !important;
                max-width: 290px !important;
                transform: none !important;
                visibility: visible !important;
            }}
            section[data-testid="stSidebar"] > div {{
                transform: none !important;
            }}
        }}

        /* ---- MOBILE / SMALL SCREENS (<=900px): restore Streamlit's native
               collapsible sidebar so it doesn't crowd the content. The
               collapse arrow is shown, and no fixed width is forced. ---- */
        @media (max-width: 900px) {{
            [data-testid="collapsedControl"] {{
                display: flex !important;
                visibility: visible !important;
            }}
            section[data-testid="stSidebar"] {{
                min-width: unset !important;
                max-width: 85vw !important;
                width: 85vw !important;
            }}
            .app-title {{ font-size: 26px !important; }}
            .app-header {{ padding: 18px 20px !important; }}
            .kpi-value {{ font-size: 22px !important; }}
        }}

        .stApp {{
            background: radial-gradient(circle at top left, {BG2} 0%, {BG} 60%);
            color: {TEXT};
        }}

        .block-container {{
            padding-top: 1.5rem;
            animation: fadeIn 0.4s ease;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        /* ---- Header banner ---- */
        .app-header {{
            padding: 26px 30px;
            border-radius: 20px;
            background: linear-gradient(135deg, {PRIMARY}22, {SECONDARY}22);
            border: 1px solid {CARD_BORDER};
            margin-bottom: 26px;
        }}
        .app-title {{
            font-size: 36px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0;
            background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .app-subtitle {{
            font-size: 15px;
            color: {MUTED};
            margin-top: 4px;
        }}
        .section-title {{
            font-size: 24px;
            font-weight: 600;
            margin: 26px 0 14px 0;
            color: {TEXT};
        }}

        /* ---- KPI cards ---- */
        .kpi-card {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 18px;
            padding: 20px 22px;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 24px rgba(0,0,0,0.18);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 30px rgba(99,102,241,0.25);
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {MUTED};
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            margin-top: 6px;
            color: {TEXT};
        }}

        /* ---- Badges ---- */
        .badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 13px;
            letter-spacing: 0.4px;
        }}
        .badge-high {{ background: {DANGER}22; color: {DANGER}; border: 1px solid {DANGER}55; }}
        .badge-medium {{ background: {WARNING}22; color: {WARNING}; border: 1px solid {WARNING}55; }}
        .badge-low {{ background: {SUCCESS}22; color: {SUCCESS}; border: 1px solid {SUCCESS}55; }}

        /* ---- Probability bar ---- */
        .prob-track {{
            width: 100%;
            height: 14px;
            border-radius: 999px;
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            overflow: hidden;
            margin: 10px 0;
        }}
        .prob-fill {{
            height: 100%;
            border-radius: 999px;
            transition: width 1s ease-in-out;
        }}
        .prob-percent {{
            font-size: 40px;
            font-weight: 800;
            font-family: 'Poppins', sans-serif;
        }}

        /* ---- AI advice cards ---- */
        .ai-card {{
            background: {CARD};
            border: 1px solid {CARD_BORDER};
            border-left: 5px solid var(--accent, {PRIMARY});
            border-radius: 16px;
            padding: 16px 20px;
            margin-bottom: 14px;
            animation: fadeInUp 0.5s ease both;
            box-shadow: 0 4px 18px rgba(0,0,0,0.15);
        }}
        .ai-card:hover {{ box-shadow: 0 8px 26px rgba(0,0,0,0.25); }}
        .ai-card-title {{ font-weight: 700; font-size: 15px; margin-bottom: 4px; }}
        .ai-card-body {{ font-size: 14px; color: {MUTED}; line-height: 1.5; }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* ---- Buttons ---- */
        div.stButton > button, div.stDownloadButton > button {{
            background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 22px;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99,102,241,0.4);
        }}

        /* ---- Sidebar nav buttons ---- */
        .nav-btn-wrap div.stButton > button {{
            background: transparent;
            color: {MUTED};
            border: 1px solid transparent;
            border-radius: 12px;
            text-align: left;
            width: 100%;
            box-shadow: none;
            font-weight: 500;
            padding: 10px 14px;
        }}
        .nav-btn-wrap div.stButton > button:hover {{
            background: {CARD};
            color: {TEXT};
            transform: none;
            box-shadow: none;
        }}
        .nav-btn-active div.stButton > button {{
            background: linear-gradient(90deg, {PRIMARY}33, {SECONDARY}33) !important;
            color: {TEXT} !important;
            border: 1px solid {PRIMARY}88 !important;
            font-weight: 700 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
CATEGORICAL_DISPLAY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

SUGGESTION_RULES = {
    "MonthlyCharges": {"increases": "Offer a discount or loyalty pricing plan to ease bill sensitivity."},
    "TotalCharges": {"increases": "Review billing history; consider a bundled discount."},
    "tenure": {"decreases": "Provide onboarding support and early check-ins for new customers."},
    "Contract": {"decreases": "Encourage switching to a longer-term contract with incentives."},
    "InternetService": {"increases": "Investigate service quality complaints (e.g. fiber optic issues)."},
    "TechSupport": {"decreases": "Promote or bundle tech support services."},
    "OnlineSecurity": {"decreases": "Offer online security add-on at a reduced rate."},
    "PaymentMethod": {"increases": "Encourage automatic/credit card payment methods over manual ones."},
    "SeniorCitizen": {"increases": "Offer simplified plans or dedicated support for senior customers."},
    "Dependents": {"decreases": "Offer family/multi-line bundle discounts."},
    "Partner": {"decreases": "Offer household bundle plans."},
    "StreamingTV": {"increases": "Bundle streaming with a discounted long-term plan."},
    "StreamingMovies": {"increases": "Bundle streaming with a discounted long-term plan."},
}
DEFAULT_SUGGESTION = "Flag for manual review — no predefined action mapped yet."

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter"),
)


# ----------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_and_encoders():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


@st.cache_data
def load_default_dataset():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df


model, label_encoders = load_model_and_encoders()
FEATURES = list(model.feature_names_in_)


# ----------------------------------------------------------------------
# CORE HELPERS
# ----------------------------------------------------------------------
def encode_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else 0
            )

    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0

    return df.fillna(0)[FEATURES]


def score(df_encoded: pd.DataFrame):
    proba = model.predict_proba(df_encoded)[:, 1]
    pred = model.predict(df_encoded)
    return pred, proba


def risk_level(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def risk_hex(level: str) -> str:
    return {"High": DANGER, "Medium": WARNING, "Low": SUCCESS}[level]


def retention_score(p: float) -> int:
    return round((1 - p) * 100)


def explain_single(row_encoded: pd.DataFrame, top_n: int = 5):
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    top = importances.sort_values(ascending=False).head(top_n)
    return [(f, "influences", v) for f, v in top.items()]


def suggestions_for(reasons):
    out = []
    for feature, direction, _ in reasons:
        rule = SUGGESTION_RULES.get(feature, {})
        out.append(rule.get(direction, DEFAULT_SUGGESTION))
    return out


# ----------------------------------------------------------------------
# UI COMPONENT HELPERS
# ----------------------------------------------------------------------
def kpi_card(label: str, value: str, accent: str = PRIMARY):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {accent};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[level]
    return f'<span class="badge {cls}">{level} Risk</span>'


def probability_bar(p: float):
    color = risk_hex(risk_level(p))
    st.markdown(
        f"""
        <div class="prob-percent" style="color:{color};">{p * 100:.1f}%</div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{p * 100:.1f}%; background:linear-gradient(90deg,{PRIMARY},{color});"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ai_advice_card(icon: str, title: str, body: str, accent: str):
    st.markdown(
        f"""
        <div class="ai-card" style="--accent:{accent};">
            <div class="ai-card-title">{icon} {title}</div>
            <div class="ai-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def typing_effect(text: str, speed_ms: int = 18, color: str = "#E2E8F0"):
    safe_text = text.replace("`", "'").replace("\n", " ")
    components.html(
        f"""
        <div style="font-family:'Inter',sans-serif; font-size:16px; font-weight:600;
                     color:{color}; min-height: 28px;">
            <span id="typing-output"></span><span style="opacity:0.6;">▌</span>
        </div>
        <script>
        const text = {safe_text!r};
        let i = 0;
        function type() {{
            if (i < text.length) {{
                document.getElementById("typing-output").innerHTML += text.charAt(i);
                i++;
                setTimeout(type, {speed_ms});
            }}
        }}
        type();
        </script>
        """,
        height=40,
    )


def render_ai_advice_panel(p: float, reasons, suggestions):
    level = risk_level(p)
    accent = risk_hex(level)
    icon = {"High": "⚠️", "Medium": "🟡", "Low": "✅"}[level]

    typing_effect(
        f"AI Verdict: {level} churn risk detected ({p*100:.1f}% probability)."
        if level != "Low" else f"AI Verdict: customer looks stable ({p*100:.1f}% churn probability).",
        color=accent,
    )
    st.write("")

    ai_advice_card(
        icon, f"Risk Alert — {level}",
        f"This customer has a <b>{p*100:.1f}%</b> estimated probability of churning.",
        accent,
    )
    for feature, direction, _ in reasons:
        verb = "pushing risk up" if direction == "increases" else (
            "helping retention" if direction == "decreases" else "a key factor"
        )
        ai_advice_card("📊", f"Reason: {feature}", f"<b>{feature}</b> is {verb} for this customer.", SECONDARY)
    for s in suggestions:
        ai_advice_card("💡", "Action Suggestion", s, SUCCESS)


# ----------------------------------------------------------------------
# SIDEBAR — fixed, non-collapsible, icon nav with active highlight
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 8px 4px 20px 4px;">
            <div style="font-family:'Poppins',sans-serif; font-weight:800; font-size:20px;
                        background: linear-gradient(90deg,{PRIMARY},{SECONDARY});
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                📊 Churn Intelligence
            </div>
            <div style="font-size:12px; color:{MUTED};">Premium Analytics Suite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for name, icon in NAV_ITEMS:
        wrap_class = "nav-btn-active" if st.session_state.page == name else "nav-btn-wrap"
        st.markdown(f'<div class="{wrap_class}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
            st.session_state.page = name
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    fast_mode = st.toggle(
        "⚡ Fast Mode",
        value=st.session_state.get("fast_mode", False),
        key="fast_mode",
        help="Skips the extra chart image in PDF reports for a quicker export.",
    )

    st.markdown("---")
    st.caption("Model: Random Forest\nTrained on Telco Customer Churn dataset")

page = st.session_state.page


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">Customer Churn Intelligence</div>
        <div class="app-subtitle">Real-time churn prediction, explainability, and retention insights</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# CACHED HELPERS — AI Assistant page (predictions)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_predictions(df: pd.DataFrame):
    encoded = encode_dataframe(df)
    _, proba = score(encoded)
    return encoded, proba


# ----------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------
if page == "Dashboard":
    df = load_default_dataset()
    with st.spinner("Scoring customer base..."):
        encoded_all = encode_dataframe(df)
        _, proba_all = score(encoded_all)

    total = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    active = total - churned
    churn_rate = churned / total * 100
    high_risk = int((proba_all >= 0.7).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Customers", f"{total:,}", PRIMARY)
    with c2: kpi_card("Churn Rate", f"{churn_rate:.1f}%", DANGER)
    with c3: kpi_card("Active Users", f"{active:,}", SUCCESS)
    with c4: kpi_card("High Risk Customers", f"{high_risk:,}", WARNING)

    st.markdown('<div class="section-title">Churn Breakdown</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Contract", color="Churn", barmode="group",
                            color_discrete_map={"Yes": DANGER, "No": SUCCESS},
                            title="Churn by Contract Type")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(df, names="Churn", hole=0.55,
                      color="Churn", color_discrete_map={"Yes": DANGER, "No": SUCCESS},
                      title="Overall Churn Distribution")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Churn Rate by Tenure (trend proxy)</div>', unsafe_allow_html=True)
    tenure_churn = (
        df.assign(is_churn=(df["Churn"] == "Yes").astype(int))
        .groupby("tenure")["is_churn"].mean().mul(100).rolling(3, min_periods=1).mean()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(tenure_churn.index), y=tenure_churn.values, mode="lines",
        line=dict(color=PRIMARY, width=3, shape="spline"),
        fill="tozeroy", fillcolor=f"{PRIMARY}22",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Tenure (months)", yaxis_title="Churn rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig = px.bar(x=importances.values, y=importances.index, orientation="h",
                 color=importances.values, color_continuous_scale=[SECONDARY, PRIMARY])
    fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# PAGE: PREDICTIONS
# ----------------------------------------------------------------------
elif page == "Predictions":
    st.markdown('<div class="section-title">Predict Churn for a Customer</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", CATEGORICAL_DISPLAY_OPTIONS["gender"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", CATEGORICAL_DISPLAY_OPTIONS["Partner"])
            dependents = st.selectbox("Dependents", CATEGORICAL_DISPLAY_OPTIONS["Dependents"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        with c2:
            phone = st.selectbox("Phone Service", CATEGORICAL_DISPLAY_OPTIONS["PhoneService"])
            lines = st.selectbox("Multiple Lines", CATEGORICAL_DISPLAY_OPTIONS["MultipleLines"])
            internet = st.selectbox("Internet Service", CATEGORICAL_DISPLAY_OPTIONS["InternetService"])
            security = st.selectbox("Online Security", CATEGORICAL_DISPLAY_OPTIONS["OnlineSecurity"])
            backup = st.selectbox("Online Backup", CATEGORICAL_DISPLAY_OPTIONS["OnlineBackup"])
        with c3:
            protection = st.selectbox("Device Protection", CATEGORICAL_DISPLAY_OPTIONS["DeviceProtection"])
            support = st.selectbox("Tech Support", CATEGORICAL_DISPLAY_OPTIONS["TechSupport"])
            tv = st.selectbox("Streaming TV", CATEGORICAL_DISPLAY_OPTIONS["StreamingTV"])
            movies = st.selectbox("Streaming Movies", CATEGORICAL_DISPLAY_OPTIONS["StreamingMovies"])
            contract = st.selectbox("Contract", CATEGORICAL_DISPLAY_OPTIONS["Contract"])

        c4, c5, c6 = st.columns(3)
        with c4:
            paperless = st.selectbox("Paperless Billing", CATEGORICAL_DISPLAY_OPTIONS["PaperlessBilling"])
        with c5:
            payment = st.selectbox("Payment Method", CATEGORICAL_DISPLAY_OPTIONS["PaymentMethod"])
        with c6:
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.35, step=0.5)

        total_charges = st.number_input("Total Charges", min_value=0.0, value=350.5, step=1.0)
        submitted = st.form_submit_button("🔮 Predict Churn")

    if submitted:
        with st.spinner("Running prediction..."):
            raw = pd.DataFrame([{
                "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner, "Dependents": dependents, "tenure": tenure,
                "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
                "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
                "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
                "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
                "MonthlyCharges": monthly, "TotalCharges": total_charges,
            }])
            encoded = encode_dataframe(raw)
            pred, proba = score(encoded)
            p = float(proba[0])
            level = risk_level(p)
            reasons = explain_single(encoded, top_n=4)
            suggestions = suggestions_for(reasons)

        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            probability_bar(p)
            st.markdown(risk_badge(level), unsafe_allow_html=True)
        with col_b:
            kpi_card("Retention Score", f"{retention_score(p)}/100", SUCCESS if level == "Low" else WARNING)

        st.markdown('<div class="section-title">🤖 AI Advice</div>', unsafe_allow_html=True)
        render_ai_advice_panel(p, reasons, suggestions)


# ----------------------------------------------------------------------
# PAGE: AI INSIGHTS  (Segmentation + AI Advice + Explainability)
# ----------------------------------------------------------------------
elif page == "AI Assistant":
    df = load_default_dataset()
    with st.spinner("Scoring customer base..."):
        encoded_all, proba_all = get_cached_predictions(df)

    st.markdown('<div class="section-title">🤖 AI Advice — Smart Retention Strategy</div>', unsafe_allow_html=True)

    with st.container():
        avg_risk = float(np.mean(proba_all))
        overall_level = risk_level(avg_risk)
        overall_color = risk_hex(overall_level)
        total_customers = len(df)
        at_risk_count = int((proba_all >= 0.7).sum())

        # ---- 1. Portfolio Summary ----
        ai_advice_card(
            "🧠", "Portfolio Summary",
            (
                f"Across <b>{total_customers:,}</b> customers, the average churn risk score is "
                f"<b>{avg_risk*100:.1f}%</b>, placing the overall portfolio in the "
                f"<b>{overall_level}</b> risk band. <b>{at_risk_count:,}</b> customers are "
                f"currently in the High-risk group and warrant closer attention."
            ),
            overall_color,
        )

        # ---- 2. Risk-Based Action ----
        risk_actions = {
            "High": (
                "🚨", DANGER,
                "With portfolio risk running High, prioritize immediate retention offers: "
                "targeted discounts, proactive retention calls, and personalized win-back "
                "offers for the highest-risk accounts before they churn."
            ),
            "Medium": (
                "⚠️", WARNING,
                "With risk in the Medium band, focus on engagement over discounts: email "
                "nudges, usage check-ins, and satisfaction surveys can catch issues before "
                "they turn into cancellations."
            ),
            "Low": (
                "✅", SUCCESS,
                "With risk running Low, this is a good time to invest in growth: loyalty "
                "programs, referral incentives, and upselling premium add-ons to your most "
                "stable customers."
            ),
        }
        r_icon, r_color, r_text = risk_actions[overall_level]
        ai_advice_card(r_icon, f"Risk-Based Action — {overall_level} Risk", r_text, r_color)

        # ---- 3. Feature-Based Advice (data-driven, not static sentences) ----
        st.markdown(
            '<div class="section-title" style="font-size:20px; margin-top:20px;">📊 Feature-Based Advice</div>',
            unsafe_allow_html=True,
        )

        high_mask = proba_all > 0.7  # same threshold used for at-risk count below

        def _numeric_advice(feature, icon, phrasing_fn):
            high_avg = df.loc[high_mask, feature].mean()
            overall_avg = df[feature].mean()
            diff_pct = ((high_avg - overall_avg) / overall_avg * 100) if overall_avg else 0.0
            return icon, phrasing_fn(high_avg, overall_avg, diff_pct)

        def _monthly_charges_text(high_avg, overall_avg, diff_pct):
            if diff_pct > 5:
                return (f"High-risk customers pay <b>${high_avg:.2f}/mo</b> on average vs "
                        f"<b>${overall_avg:.2f}/mo</b> overall (<b>{diff_pct:+.0f}%</b>) — consider "
                        f"pricing optimization, tiered plans, or targeted discounts for higher-paying "
                        f"customers showing risk signals.")
            if diff_pct < -5:
                return (f"High-risk customers actually pay <b>less</b> on average (${high_avg:.2f} vs "
                        f"${overall_avg:.2f} overall) — price isn't the main driver here; focus on "
                        f"perceived value and service quality instead of discounting.")
            return (f"Monthly charges are similar between high-risk and overall customers "
                    f"(${high_avg:.2f} vs ${overall_avg:.2f}) — price alone isn't the driver; "
                    f"look at service-quality factors instead.")

        def _tenure_text(high_avg, overall_avg, diff_pct):
            if high_avg < overall_avg:
                return (f"High-risk customers average <b>{high_avg:.1f} months</b> of tenure vs "
                        f"<b>{overall_avg:.1f} months</b> overall — newer customers are clearly more "
                        f"likely to churn, so strengthen onboarding and early engagement in the "
                        f"first 90 days.")
            return (f"High-risk customers average <b>{high_avg:.1f} months</b> of tenure, in line "
                    f"with or above the overall <b>{overall_avg:.1f} months</b> — churn here isn't "
                    f"purely an onboarding problem; look at long-term engagement instead.")

        def _total_charges_text(high_avg, overall_avg, diff_pct):
            direction = ("these tend to be higher-value customers, so prioritize white-glove "
                         "retention outreach for this group."
                         if diff_pct > 0 else
                         "these tend to be newer or lower-value accounts, so pair retention offers "
                         "with onboarding improvements.")
            return (f"High-risk customers have <b>${high_avg:,.0f}</b> in total charges on average "
                    f"vs <b>${overall_avg:,.0f}</b> overall (<b>{diff_pct:+.0f}%</b>) — {direction}")

        def _contract_advice():
            m2m_high = (df.loc[high_mask, "Contract"] == "Month-to-month").mean() * 100
            m2m_overall = (df["Contract"] == "Month-to-month").mean() * 100
            return ("📄",
                    f"<b>{m2m_high:.0f}%</b> of high-risk customers are on month-to-month contracts "
                    f"vs <b>{m2m_overall:.0f}%</b> overall — promoting longer-term contracts with "
                    f"incentives should meaningfully reduce churn exposure.")

        def _categorical_advice(feature, icon):
            overall_dist = df[feature].value_counts(normalize=True)
            high_dist = df.loc[high_mask, feature].value_counts(normalize=True)
            diffs = (high_dist - overall_dist).dropna()
            if diffs.empty:
                return icon, f"No strong pattern found for <b>{feature}</b> among high-risk customers."
            top_cat = diffs.idxmax()
            high_share = high_dist.get(top_cat, 0) * 100
            base_share = overall_dist.get(top_cat, 0) * 100
            return (icon,
                    f"'<b>{top_cat}</b>' makes up <b>{high_share:.0f}%</b> of high-risk customers vs "
                    f"<b>{base_share:.0f}%</b> overall for <b>{feature}</b> — investigate and address "
                    f"pain points specific to this segment.")

        CATEGORY_ICONS = {
            "InternetService": "🌐", "PaymentMethod": "💰", "TechSupport": "🛠️",
            "OnlineSecurity": "🔒", "OnlineBackup": "💾", "DeviceProtection": "🛡️",
            "PaperlessBilling": "📩", "StreamingTV": "📺", "StreamingMovies": "🎬",
            "SeniorCitizen": "👴", "Partner": "👪", "Dependents": "👨‍👩‍👧",
            "MultipleLines": "☎️", "PhoneService": "📞", "gender": "🧑",
        }

        top_features = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(3)

        for feature, importance in top_features.items():
            if feature == "MonthlyCharges":
                icon, text = _numeric_advice(feature, "💳", _monthly_charges_text)
            elif feature == "tenure":
                icon, text = _numeric_advice(feature, "🌱", _tenure_text)
            elif feature == "TotalCharges":
                icon, text = _numeric_advice(feature, "🧾", _total_charges_text)
            elif feature == "Contract":
                icon, text = _contract_advice()
            elif feature in df.columns:
                icon, text = _categorical_advice(feature, CATEGORY_ICONS.get(feature, "💡"))
            else:
                icon, text = "💡", f"<b>{feature}</b> is a top predictor (importance {importance:.2f}) — review it manually for retention opportunities."

            ai_advice_card(icon, f"{feature} (importance: {importance:.2f})", text, PRIMARY)

        # ---- 4. Future Strategy ----
        top_two_names = ", ".join(top_features.index[:2])
        ai_advice_card(
            "🚀", "Future Strategy — If No Action Is Taken",
            (
                f"If current patterns continue unaddressed, expect the <b>{at_risk_count:,}</b> "
                f"High-risk customers identified today to churn at a materially higher rate than "
                f"the base, compounding revenue loss over the next few billing cycles. Acting now — "
                f"targeted retention on the highest-risk segment plus preventive fixes around "
                f"<b>{top_two_names}</b> — is meaningfully cheaper than replacing lost customers later."
            ),
            WARNING,
        )

    # ---- B) AI Chat (rule-based, no external API) ----
    st.markdown('<div class="section-title">💬 AI Chat</div>', unsafe_allow_html=True)
    st.caption("Ask about churn drivers, risky customers, or how to reduce churn. Rule-based — no external API calls.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def generate_chat_response(query: str) -> str:
        q = query.lower()
        top3 = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(3)
        top3_names = ", ".join(top3.index)

        if "why" in q and ("churn" in q or "risk" in q):
            return (f"Overall churn risk is currently **{overall_level}** (average {avg_risk*100:.1f}%). "
                    f"The strongest drivers are **{top3_names}** — customers with shorter tenure, higher "
                    f"monthly charges, or month-to-month contracts tend to be the ones most likely to leave.")
        if "reduce" in q or "lower" in q or "prevent" in q or "improve" in q:
            return ("To reduce churn: promote longer-term contracts with incentives, strengthen onboarding "
                    "for customers in their first 90 days, and review pricing or offer discounts for "
                    "high-bill customers who show elevated risk.")
        if "risky" in q or ("risk" in q and "customer" in q) or "at-risk" in q or "how many" in q:
            return (f"There are currently **{at_risk_count:,}** customers flagged as High risk "
                    f"(churn probability above 70%) out of {total_customers:,} total.")
        if "driver" in q or "feature" in q or "important" in q or "cause" in q:
            return "The top churn drivers, ranked by model importance, are: " + ", ".join(
                f"**{f}** ({v:.2f})" for f, v in top3.items()
            ) + "."
        if "tenure" in q:
            avg_tenure_high = df.loc[proba_all > 0.7, "tenure"].mean()
            avg_tenure_all = df["tenure"].mean()
            return (f"High-risk customers average **{avg_tenure_high:.1f} months** of tenure, vs "
                    f"**{avg_tenure_all:.1f} months** across all customers.")
        if "price" in q or "charge" in q or "bill" in q:
            avg_charge_high = df.loc[proba_all > 0.7, "MonthlyCharges"].mean()
            avg_charge_all = df["MonthlyCharges"].mean()
            return (f"High-risk customers pay **${avg_charge_high:.2f}/mo** on average, vs "
                    f"**${avg_charge_all:.2f}/mo** across all customers.")
        return ("I can help with questions like: *\"Why is churn high?\"*, *\"How can we reduce churn?\"*, "
                "*\"How many risky customers are there?\"*, or *\"What are the top churn drivers?\"* — try "
                "asking one of those.")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    user_query = st.chat_input("Ask the AI Assistant about churn...")
    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        response = generate_chat_response(user_query)
        st.session_state.chat_history.append(("assistant", response))
        st.rerun()


# ----------------------------------------------------------------------
# PAGE: WHAT-IF SIMULATOR
# ----------------------------------------------------------------------
elif page == "What-If Simulator":
    st.markdown('<div class="section-title">🔮 What-If Simulator</div>', unsafe_allow_html=True)
    st.caption(
        "Adjust tenure and monthly charges to see how churn risk changes instantly. "
        "All other customer attributes are held at the dataset's most common values "
        "so you can isolate the effect of these factors."
    )

    df = load_default_dataset()

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with col2:
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)

    auto_total = st.checkbox("Auto-calculate Total Charges (tenure × monthly)", value=True)
    if auto_total:
        total = tenure * monthly
        st.caption(f"Total Charges (auto-calculated): **${total:,.2f}**")
    else:
        total = st.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly), step=10.0)

    # Baseline customer: most common value for every other field
    baseline = {}
    for col, options in CATEGORICAL_DISPLAY_OPTIONS.items():
        baseline[col] = df[col].mode()[0] if col in df.columns else options[0]
    baseline["SeniorCitizen"] = int(df["SeniorCitizen"].mode()[0]) if "SeniorCitizen" in df.columns else 0
    baseline["tenure"] = tenure
    baseline["MonthlyCharges"] = monthly
    baseline["TotalCharges"] = total

    raw = pd.DataFrame([baseline])
    encoded = encode_dataframe(raw)
    _, proba = score(encoded)
    p = float(proba[0])
    level = risk_level(p)

    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
    probability_bar(p)
    st.markdown(risk_badge(level), unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PAGE: REPORTS
# ----------------------------------------------------------------------
elif page == "Reports":
    st.markdown('<div class="section-title">Bulk Upload & Reports</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV (same columns as the Telco dataset)", type=["csv"])

    if uploaded is not None:
        with st.spinner("Scoring uploaded customers..."):
            raw_df = pd.read_csv(uploaded)
            encoded = encode_dataframe(raw_df)
            pred, proba = score(encoded)

            result_df = raw_df.copy()
            result_df["Churn_Probability"] = proba.round(3)
            result_df["Churn_Percent"] = (proba * 100).round(1)
            result_df["Risk_Level"] = [risk_level(p) for p in proba]
            result_df["Retention_Score"] = [retention_score(p) for p in proba]

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Customers Scored", f"{len(result_df):,}", PRIMARY)
        with c2: kpi_card("High Risk", f"{(result_df['Risk_Level'] == 'High').sum():,}", DANGER)
        with c3: kpi_card("Medium Risk", f"{(result_df['Risk_Level'] == 'Medium').sum():,}", WARNING)
        with c4: kpi_card("Avg. Probability", f"{proba.mean():.2f}", SECONDARY)

        st.markdown('<div class="section-title">Scored Customers</div>', unsafe_allow_html=True)
        risk_filter = st.multiselect("Filter by risk level", ["High", "Medium", "Low"],
                                      default=["High", "Medium", "Low"])
        st.dataframe(result_df[result_df["Risk_Level"].isin(risk_filter)], use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV Report", csv_bytes, "churn_predictions.csv", "text/csv")

        if PDF_AVAILABLE:
            with col_dl2:
                def build_pdf_summary(df_res):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 16)
                    pdf.cell(0, 10, "Churn Report Summary", ln=True)
                    pdf.set_font("Helvetica", "", 11)
                    pdf.ln(4)
                    pdf.cell(0, 8, f"Customers scored: {len(df_res)}", ln=True)
                    pdf.cell(0, 8, f"High risk: {(df_res['Risk_Level'] == 'High').sum()}", ln=True)
                    pdf.cell(0, 8, f"Medium risk: {(df_res['Risk_Level'] == 'Medium').sum()}", ln=True)
                    pdf.cell(0, 8, f"Low risk: {(df_res['Risk_Level'] == 'Low').sum()}", ln=True)
                    pdf.cell(0, 8, f"Average churn probability: {df_res['Churn_Probability'].mean():.2f}", ln=True)

                    if MATPLOTLIB_AVAILABLE and not st.session_state.get("fast_mode", False):
                        try:
                            counts = df_res["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
                            fig, ax = plt.subplots(figsize=(5, 3))
                            ax.bar(counts.index, counts.values,
                                   color=[SUCCESS, WARNING, DANGER])
                            ax.set_title("Risk Level Distribution")
                            ax.set_ylabel("Customers")
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                                fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
                                plt.close(fig)
                                pdf.ln(6)
                                pdf.image(tmp.name, w=150)
                        except Exception:
                            pass

                    return bytes(pdf.output(dest="S"))

                pdf_bytes = build_pdf_summary(result_df)
                st.download_button("⬇️ Download PDF Report", pdf_bytes, "churn_report.pdf", "application/pdf")

        st.markdown('<div class="section-title">🚨 High-Risk Alerts</div>', unsafe_allow_html=True)
        alerts = result_df[result_df["Risk_Level"] == "High"]
        if len(alerts) == 0:
            st.write("No high-risk customers in this file.")
        else:
            st.dataframe(alerts, use_container_width=True)
            st.caption("Suggested next step: route this list to your retention team for outreach.")
    else:
        st.info("Upload a CSV to generate instant predictions and a downloadable report.")
