import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(page_title="Bank Customer Segmentation", layout="centered")
st.title("Bank Customer Financial Behavior Clustering")
st.markdown("""
**Cluster Interpretations:**
- **Cluster 0**: Stable Working-Class Married Customers
- **Cluster 1**: Young Single Professionals
- **Cluster 2**: High-Value Married Managers
- **Cluster 3**: Students & Early-Career Prospects
- **Cluster 4**: Affluent Retired Customers
""")

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
    st.header("Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    submitted = st.form_submit_button("Predict Cluster")

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
    
    # Display result
    st.success(f"**Predicted Cluster: {cluster}**")
    
    cluster_names = {
        0: "Stable Working-Class Married Customers",
        1: "Young Single Professionals",
        2: "High-Value Married Managers",
        3: "Students & Early-Career Prospects",
        4: "Affluent Retired Customers"
    }
    
    st.markdown(f"**Interpretation:** {cluster_names.get(cluster, 'Unknown Cluster')}")
    
    st.info("This clustering can help tailor suitable marketing strategies for the customer segment.")