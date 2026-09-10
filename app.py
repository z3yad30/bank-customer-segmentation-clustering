import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Lens | Bank Segmentation", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #14213d;
        --muted: #60708f;
        --line: #dce5ef;
        --paper: #f5f8fb;
        --teal: #087f8c;
        --teal-dark: #075e68;
        --gold: #f2b84b;
    }

    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1180px; padding: 2.5rem 3rem 4rem; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    p, label, .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox { font-family: 'DM Sans', sans-serif; }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] div,
    [data-testid="stSlider"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label { color: var(--ink) !important; opacity: 1 !important; }
    [data-testid="stWidgetLabel"] p { font-weight: 600 !important; }
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; opacity: 1 !important; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 12px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stForm"] { background: white; border: 1px solid var(--line); border-radius: 16px; padding: 1.5rem 1.7rem 1rem; box-shadow: 0 10px 30px rgba(20, 33, 61, 0.05); }
    .hero { background: linear-gradient(120deg, #14213d 0%, #075e68 100%); border-radius: 20px; padding: 2.2rem 2.5rem; margin-bottom: 1.5rem; color: white; position: relative; overflow: hidden; }
    .hero:after { content: ''; position: absolute; width: 210px; height: 210px; border: 1px solid rgba(255,255,255,.16); border-radius: 50%; right: -55px; top: -80px; }
    .eyebrow { color: #8ee1df; font: 700 .75rem 'DM Sans', sans-serif; letter-spacing: .14em; text-transform: uppercase; }
    .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3.2rem); margin: .45rem 0 .6rem; letter-spacing: -0.03em; }
    .hero p { color: #d8eef0; max-width: 660px; margin: 0; font-size: 1.04rem; }
    .section-kicker { color: var(--teal-dark); font: 700 .74rem 'DM Sans', sans-serif; letter-spacing: .12em; text-transform: uppercase; margin: .45rem 0 .15rem; }
    .result { background: #e8f6f4; border: 1px solid #a9dfda; border-left: 6px solid var(--teal); border-radius: 14px; padding: 1.4rem 1.6rem; margin-top: 1.6rem; }
    .result-label { color: var(--teal-dark); font: 700 .75rem 'DM Sans', sans-serif; letter-spacing: .12em; text-transform: uppercase; }
    .result h2 { margin: .25rem 0 .3rem; color: var(--ink); }
    .result p { color: #3e536d; margin-bottom: 0; }
    div.stButton > button, div.stFormSubmitButton > button { background: var(--teal); color: white; border: 0; border-radius: 9px; font: 700 1rem 'DM Sans', sans-serif; padding: .65rem 1.4rem; }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { background: var(--teal-dark); color: white; }
    @media (max-width: 700px) { .block-container { padding: 1.5rem 1rem 3rem; } .hero { padding: 1.7rem; } }
</style>
<div class="hero">
    <div class="eyebrow">Customer Lens / Segmentation Studio</div>
    <h1>Understand the customer behind the numbers.</h1>
    <p>Enter a customer's financial and campaign profile to discover the behavior segment they most closely match.</p>
</div>
""", unsafe_allow_html=True)

# Load the saved artifacts (place scaler.pkl, pca.pkl, kmeans.pkl, columns.pkl in the same directory)
@st.cache_resource
def load_model_components():
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    kmeans = joblib.load("kmeans.pkl")
    expected_columns = joblib.load("columns.pkl")
    return scaler, pca, kmeans, expected_columns

try:
    scaler, pca, kmeans, expected_columns = load_model_components()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Input form
with st.form("customer_form"):
    st.subheader("Customer profile")
    st.caption("Use the latest available customer and campaign information.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-kicker">Personal profile</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        job = st.selectbox("Job", options=[
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"
        ])
        marital = st.selectbox("Marital Status", options=["divorced", "married", "single"])
        education = st.selectbox("Education", options=["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("Has Credit in Default?", options=["no", "yes"])
        balance = st.number_input("Average Yearly Balance (€)", value=1000)
        housing = st.selectbox("Has Housing Loan?", options=["no", "yes"])
        
    with col2:
        st.markdown('<div class="section-kicker">Campaign activity</div>', unsafe_allow_html=True)
        loan = st.selectbox("Has Personal Loan?", options=["no", "yes"])
        contact = st.selectbox("Contact Communication Type", options=["cellular", "telephone", "unknown"])
        day = st.slider("Last Contact Day of Month", min_value=1, max_value=31, value=15)
        month = st.selectbox("Last Contact Month", options=[
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=200)
        campaign = st.number_input("Number of Contacts This Campaign", min_value=1, value=2)
        pdays = st.number_input("Days Since Last Contacted (-1 if never)", value=-1)
        previous = st.number_input("Previous Contacts Before This Campaign", min_value=0, value=0)
        poutcome = st.selectbox("Previous Campaign Outcome", options=["failure", "other", "success", "unknown"])
    
    submitted = st.form_submit_button("Find customer segment", use_container_width=True)

if submitted:
    # Create input DataFrame
    input_data = {
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "balance": [balance],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "day": [day],
        "month": [month],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome]
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Apply the same preprocessing: one-hot encoding
    df_processed = pd.get_dummies(df_input, columns=[
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month", "poutcome"
    ])
    
    # Align columns to match training (add missing columns with 0)
    for col in expected_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Reorder and select only expected columns
    df_processed = df_processed[expected_columns]
    
    # Scale
    scaled = scaler.transform(df_processed)
    
    # PCA
    pca_transformed = pca.transform(scaled)
    
    # Predict cluster
    cluster = kmeans.predict(pca_transformed)[0]
    
    cluster_names = {
        0: "Stable Working-Class Married Customers",
        1: "Young Single Professionals",
        2: "High-Value Married Managers",
        3: "Students & Early-Career Prospects",
        4: "Affluent Retired Customers"
    }
    cluster_name = cluster_names.get(cluster, "Unknown Cluster")

    st.markdown(f"""
    <div class="result">
        <div class="result-label">Recommended segment</div>
        <h2>Cluster {cluster}: {cluster_name}</h2>
        <p>This customer can be considered alongside similar profiles when planning personalized marketing communication.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Segment", f"Cluster {cluster}")
    metric_col2.metric("Age", f"{age} years")
    metric_col3.metric("Balance", f"€{balance:,.0f}")