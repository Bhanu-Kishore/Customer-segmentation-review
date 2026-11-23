import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# Load model and scaler
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# Title
st.title("👥 Customer Segmentation Analytics")
st.write(
    "ML-based customer segmentation using K-Means clustering"
)


# Sidebar
st.sidebar.header("Customer Information")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

income = st.sidebar.number_input(
    "Annual Income",
    min_value=10000,
    max_value=200000,
    value=30000
)

spending_score = st.sidebar.slider(
    "Spending Score",
    min_value=0,
    max_value=100,
    value=75
)

frequency = st.sidebar.number_input(
    "Purchase Frequency",
    min_value=0,
    max_value=50,
    value=14
)


# Prediction button
if st.button("Predict Customer Segment"):

    customer_data = np.array([
        [
            age,
            income,
            spending_score,
            frequency
        ]
    ])

    scaled_data = scaler.transform(
        customer_data
    )

    segment = model.predict(
        scaled_data
    )[0]

    st.success(
        f"Customer belongs to Segment {segment}"
    )

    st.info(
        "The segment is generated using the trained K-Means clustering model."
    )


# Dataset section
st.header("Customer Dataset")

df = pd.read_csv(
    "data/segmented_customers.csv"
)

st.dataframe(
    df,
    use_container_width=True
)


# Segment statistics
st.header("Segment Analysis")

segment_counts = (
    df["Segment"]
    .value_counts()
    .sort_index()
)

st.bar_chart(segment_counts)