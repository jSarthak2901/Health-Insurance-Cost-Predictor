import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load model
model_data = joblib.load("insurance_model.pkl")
model = model_data["model"]
features = model_data["features"]

# Header
st.title("🏥 Health Insurance Cost Predictor")
st.write(
    "An interactive machine learning application for estimating "
    "health insurance charges."
)

st.divider()

# Sidebar inputs
st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input(
    "BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1
)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Encode categorical variables
input_encoded = pd.get_dummies(input_data, drop_first=True)

# Match training features
input_encoded = input_encoded.reindex(
    columns=features,
    fill_value=0
)

# Prediction
st.subheader("🔮 Insurance Cost Prediction")

if st.button("Predict Insurance Cost", type="primary"):

    prediction = model.predict(input_encoded)[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Insurance Cost",
            f"${prediction:,.2f}"
        )

    with col2:
        st.metric("Age", age)

    with col3:
        st.metric("BMI", f"{bmi:.1f}")

    st.success(
        f"Estimated insurance charge: **${prediction:,.2f}**"
    )

# Customer information
st.divider()

st.subheader("📋 Customer Information")

display_data = input_data.T.reset_index()
display_data.columns = ["Feature", "Value"]

st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "This prediction is an ML-based estimate and should not be "
    "considered professional medical or financial advice."
)
