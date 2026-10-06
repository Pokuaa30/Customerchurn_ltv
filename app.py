import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Customer Churn & LTV Dashboard", layout="wide")

st.title("📊 E-Commerce Customer Churn & LTV Analytics Dashboard")
st.markdown("Filter high-risk customer segments, evaluate churn risk, and inspect feature importances.")

@st.cache_data
def load_data():
    return pd.read_csv('rfm_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

rfm = load_data()
model = load_model()

# Sidebar Filters
st.sidebar.header("Filter Options")
selected_segment = st.sidebar.multiselect(
    "Select Customer Segments:",
    options=rfm['Segment'].unique(),
    default=rfm['Segment'].unique()
)

filtered_df = rfm[rfm['Segment'].isin(selected_segment)]

# Executive Summary
st.header("Executive Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(filtered_df):,}")
col2.metric("Churned Customers", f"{filtered_df['Churn'].sum():,}")
col3.metric("Churn Rate", f"{(filtered_df['Churn'].mean() * 100):.1f}%")
col4.metric("Avg Monetary Value", f"${filtered_df['Monetary'].mean():,.2f}")

st.markdown("---")

# Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 Feature Importance Analysis")
    importances = model.feature_importances_
    features = ['Frequency', 'Monetary']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=importances, y=features, hue=features, legend=False, palette="viridis", ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_title("XGBoost Feature Importance")
    st.pyplot(fig)

with col_right:
    st.subheader("⚠️ High-Risk Customer Segments")
    segment_churn = filtered_df.groupby('Segment')['Churn'].mean().reset_index()
    segment_churn['Churn Rate (%)'] = segment_churn['Churn'] * 100
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=segment_churn, x='Churn Rate (%)', y='Segment', hue='Segment', legend=False, palette="magma", ax=ax2)
    ax2.set_title("Churn Rate by Segment")
    st.pyplot(fig2)

# Raw Data Inspection
st.subheader("🔍 Customer Segment Risk Inspection")
st.dataframe(filtered_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment', 'Churn']])