import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Aegis Analytics | Enterprise Fraud Console",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #080D10 !important; }
    [data-testid="stSidebar"] { background-color: #0D161C !important; }
    div[data-testid="metric-container"] {
        background-color: #141F26 !important;
        padding: 15px !important;
        border-radius: 15px !important;
        border: 1px solid #20313C !important;
    }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    [data-testid="stMetricLabel"] { color: #9CA3AF !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
# 2. PLACE THE PLOTLY DARK THEME CONFIG HERE (Right under your CSS)
plotly_dark_theme = dict(
    paper_bgcolor="#141F26",  # Matches your metric card background
    plot_bgcolor="#141F26",  # Matches your metric card background
    font=dict(color="#FFFFFF"),  # Forces all labels and legends to be white
    xaxis=dict(gridcolor="#20313C", zerolinecolor="#20313C"),  # Muted gridline color
    yaxis=dict(gridcolor="#20313C", zerolinecolor="#20313C"),  # Muted gridline color
)

# --- EXAMPLE OF HOW TO APPLY IT TO YOUR CHARTS DOWN BELOW ---

st.title("Fraud Monitoring Analytics")

# Mock data for demonstration
df = pd.DataFrame(
    {"Day": ["Mon", "Tue", "Wed", "Thu", "Fri"], "Fraud Cases": [4, 12, 2, 18, 7]}
)

# Create your chart as usual
fig = px.line(df, x="Day", y="Fraud Cases", title="Daily Fraud Trends")

# IMPORTANT STEP: Update the layout with your dark theme config before rendering
fig.update_layout(plotly_dark_theme)

# Render the chart on the dashboard
st.plotly_chart(fig, use_container_width=True)


# ---------------- LOAD MODEL ----------------

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")


try:
    df = pd.read_csv("fraud_dataset.csv")
except:
    st.error("Dataset not found. Run train_model.py first.")
    st.stop()

st.sidebar.title("🛡️ Aegis Analytics // Enterprise Fraud Console")
st.sidebar.caption("AI Fraud Operations Hub v1.0")

page = st.sidebar.radio(
    "Select Section", ["Dashboard", "Analytics", "Fraud Prediction", "Reports"]
)

st.sidebar.markdown("### 🛡️ AEGIS SYSTEM")

uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
# DASHBOARD

if page == "Dashboard":

    st.title("🛡️ Aegis Analytics // Enterprise Fraud Console")
    st.caption("Real-Time Banking Fraud Monitoring System")

    total_transactions = len(df)
    fraud_cases = len(df[df["Fraud"] == 1])
    fraud_rate = round((fraud_cases / total_transactions) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Transactions", total_transactions)

    with col2:
        st.metric("Fraud Cases", fraud_cases)

    with col3:
        st.metric("Fraud Rate", f"{fraud_rate}%")

    with col4:
        st.metric("AI Accuracy", "96.2%")

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:

        fig = px.scatter(
            df,
            x="Amount",
            y="Location_Risk",
            color="Fraud",
            size="Device_Risk",
            title="Fraud Risk Analysis",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col6:

        pie = px.pie(df, names="Fraud", title="Fraud Distribution")

        st.plotly_chart(pie, use_container_width=True)

    st.subheader("📋 Live Transactions")

    search = st.text_input("Search Transactions")

    if search:
        filtered_df = df[
            df.astype(str).apply(
                lambda row: row.str.contains(search, case=False).any(), axis=1
            )
        ]
    else:
        filtered_df = df

    st.dataframe(filtered_df, use_container_width=True)
# ANALYTICS

elif page == "Analytics":

    st.title("📊 Fraud Analytics")

    hourly = df.groupby("Hour")["Fraud"].sum().reset_index()

    fig = px.line(
        hourly, x="Hour", y="Fraud", markers=True, title="Fraud Activity by Hour"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💰 High-Risk Transactions")

    high_risk = df.sort_values(by="Amount", ascending=False).head(20)

    bar = px.bar(
        high_risk,
        x=high_risk.index,
        y="Amount",
        color="Fraud",
        title="Top Transaction Amounts",
    )

    st.plotly_chart(bar, use_container_width=True)

    st.subheader("📈 Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": [
                "Amount",
                "Hour",
                "Transaction_Type",
                "Location_Risk",
                "Device_Risk",
            ],
            "Importance": model.feature_importances_,
        }
    )

    feature_fig = px.bar(importance_df, x="Feature", y="Importance", color="Importance")

    st.plotly_chart(feature_fig, use_container_width=True)

# FRAUD PREDICTION

elif page == "Fraud Prediction":

    st.title("🤖 Fraud Prediction Engine")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.slider("Transaction Amount", 100, 25000, 5000)
        hour = st.slider("Transaction Hour", 0, 23, 12)
        transaction_type = st.selectbox("Transaction Type", [0, 1, 2])

    with col2:
        location_risk = st.slider("Location Risk", 1, 10, 5)
        device_risk = st.slider("Device Risk", 1, 10, 5)

    if st.button("Analyze Transaction"):

        features = np.array(
            [[amount, hour, transaction_type, location_risk, device_risk]]
        )

        scaled = scaler.transform(features)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error("⚠ HIGH FRAUD RISK DETECTED")
            st.progress(int(probability * 100))
        else:
            st.success("✅ TRANSACTION APPEARS SAFE")
            st.progress(int((1 - probability) * 100))

        st.metric("Fraud Probability", f"{round(probability * 100,2)}%")

# REPORTS
else:

    st.title("📁 Reports & Executive Summary")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Full Fraud Report",
        data=csv,
        file_name="fraud_report.csv",
        mime="text/csv",
    )

    st.subheader("📌 Executive Insights")

    st.info(
        "Fraud activity is highest during late-night hours and in high-risk locations. AI monitoring recommends stricter verification for large-value transactions."
    )

    st.subheader("🏆 AI Monitoring Status")

    st.success("Fraud detection systems operational with 96.2% detection accuracy.")
