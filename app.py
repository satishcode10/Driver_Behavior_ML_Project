import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = joblib.load("models/best_model.pkl")

st.set_page_config(
    page_title="Driver Behavior Risk Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Driver Behavior & Accident Risk Prediction")
st.caption("Machine Learning-Based Accident Severity Prediction System")

# Load dataset for dropdown options and default values
df = pd.read_csv("data/RTA Dataset.csv")

# Same preprocessing used in notebook
df = df.drop(columns=["Defect_of_vehicle", "Service_year_of_vehicle"], errors="ignore")

for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

target = "Accident_severity"
feature_cols = [col for col in df.columns if col != target]

# Only show important demo features
important_features = [
    "Time",
    "Cause_of_accident",
    "Day_of_week",
    "Type_of_vehicle",
    "Area_accident_occured",
    "Driving_experience",
    "Age_band_of_driver",
    "Road_surface_conditions",
    "Weather_conditions",
    "Light_conditions",
    "Number_of_vehicles_involved",
    "Number_of_casualties"
]

display_names = {
    "Time": "Accident Time",
    "Cause_of_accident": "Main Cause of Accident",
    "Day_of_week": "Day of Week",
    "Type_of_vehicle": "Type of Vehicle",
    "Area_accident_occured": "Accident Area",
    "Driving_experience": "Driving Experience",
    "Age_band_of_driver": "Driver Age Group",
    "Road_surface_conditions": "Road Surface Condition",
    "Weather_conditions": "Weather Condition",
    "Light_conditions": "Light Condition",
    "Number_of_vehicles_involved": "Number of Vehicles Involved",
    "Number_of_casualties": "Number of Casualties"
}

# Dashboard metrics
col1, col2, col3 = st.columns(3)
col1.metric("Dataset", "12,316 Records")
col2.metric("Best Model", "Random Forest")
col3.metric("Accuracy", "84.82%")

st.markdown("---")

st.subheader("Prediction Demo")
st.info(
    "For a clean demo, only the most important accident-related inputs are shown. "
    "Remaining model features are automatically filled using common dataset values."
)

# Default values for all model features
inputs = {}

for col in feature_cols:
    if df[col].dtype == "object":
        inputs[col] = df[col].mode()[0]
    else:
        inputs[col] = int(df[col].mode()[0])

# User input form
with st.form("prediction_form"):
    c1, c2 = st.columns(2)

    for i, col in enumerate(important_features):
        label = display_names.get(col, col)

        with [c1, c2][i % 2]:
            if df[col].dtype == "object":
                options = sorted(df[col].astype(str).unique())
                default_value = str(df[col].mode()[0])

                selected = st.selectbox(
                    label,
                    options,
                    index=options.index(default_value) if default_value in options else 0
                )

                inputs[col] = selected

            else:
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                default_val = int(df[col].mode()[0])

                selected = st.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=1
                )

                inputs[col] = selected

    submitted = st.form_submit_button(
        "Predict Accident Severity",
        use_container_width=True
    )

# Prediction and refined risk interpretation
if submitted:
    encoded_values = []

    for col in feature_cols:
        if df[col].dtype == "object":
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            encoded_value = encoder.transform([str(inputs[col])])[0]
            encoded_values.append(encoded_value)
        else:
            encoded_values.append(inputs[col])

    input_array = np.array(encoded_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    labels = {
        0: "Fatal Injury",
        1: "Serious Injury",
        2: "Slight Injury"
    }

    model_result = labels.get(prediction, prediction)

    # Rule-based risk score for demo interpretation
    risk_score = 0
    risk_reasons = []

    cause = str(inputs["Cause_of_accident"]).lower()
    weather = str(inputs["Weather_conditions"]).lower()
    road = str(inputs["Road_surface_conditions"]).lower()
    light = str(inputs["Light_conditions"]).lower()
    vehicles = int(inputs["Number_of_vehicles_involved"])
    casualties = int(inputs["Number_of_casualties"])

    if "speed" in cause:
        risk_score += 3
        risk_reasons.append("High-speed driving")

    if "overtaking" in cause or "lane" in cause:
        risk_score += 2
        risk_reasons.append("Unsafe driving behaviour")

    if "dark" in light or "no lighting" in light:
        risk_score += 2
        risk_reasons.append("Poor lighting condition")

    if "snow" in weather or "rain" in weather or "fog" in weather:
        risk_score += 2
        risk_reasons.append("Bad weather condition")

    if "snow" in road or "wet" in road or "flood" in road:
        risk_score += 2
        risk_reasons.append("Unsafe road surface")

    if vehicles >= 3:
        risk_score += 2
        risk_reasons.append("Multiple vehicles involved")

    if casualties >= 2:
        risk_score += 2
        risk_reasons.append("Multiple casualties")

    if model_result == "Fatal Injury":
        risk_score += 3
    elif model_result == "Serious Injury":
        risk_score += 2

    if risk_score >= 7:
        risk_level = "High Risk"
        risk_message = "🔴 High Risk Accident Scenario"
    elif risk_score >= 4:
        risk_level = "Medium Risk"
        risk_message = "🟠 Medium Risk Accident Scenario"
    else:
        risk_level = "Low Risk"
        risk_message = "🟢 Low Risk Accident Scenario"

    st.markdown("### Prediction Result")

    st.write(f"**Model Prediction:** {model_result}")
    st.write(f"**Risk Assessment:** {risk_level}")

    if risk_level == "High Risk":
        st.error(risk_message)
    elif risk_level == "Medium Risk":
        st.warning(risk_message)
    else:
        st.success(risk_message)

    if risk_reasons:
        st.markdown("#### Main Risk Factors Detected")
        for reason in risk_reasons:
            st.write(f"- {reason}")
    else:
        st.write("No major high-risk condition was detected from the selected inputs.")

    st.caption(
        "Note: The trained machine learning model predicts accident severity. "
        "The risk assessment layer adds an interpretable demo explanation based on selected high-risk conditions."
    )

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("Algorithms Used")
    st.write("""
    - Logistic Regression  
    - Decision Tree  
    - Random Forest  
    - K-Means Clustering with PCA  
    """)

with right:
    st.subheader("Project Purpose")
    st.write("""
    This prototype demonstrates how machine learning can support accident severity prediction.
    It uses selected driver, vehicle, road, weather, and accident-cause factors for a practical demo.
    """)

st.caption("Developed by Satish Adhikari")