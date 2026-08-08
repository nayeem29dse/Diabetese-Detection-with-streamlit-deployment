import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load saved artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("logistic_regression_model.pkl")
    return imputer, scaler, model

imputer, scaler, model = load_artifacts()

FINAL_THRESHOLD = 0.25

CONTINUOUS_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
COLS_TO_FIX = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# -----------------------------
# Inference function
# -----------------------------
def predict_diabetes_risk(raw_patient_dict, threshold=FINAL_THRESHOLD):
    df_new = pd.DataFrame([raw_patient_dict])

    # Missingness flags (0 entered = treated as not measured)
    df_new["SkinThickness_missing"] = (df_new["SkinThickness"] == 0).astype(int)
    df_new["Insulin_missing"] = (df_new["Insulin"] == 0).astype(int)

    # Replace zeros with NaN, then impute using train-fitted imputer
    df_new[COLS_TO_FIX] = df_new[COLS_TO_FIX].replace(0, np.nan)
    df_new[COLS_TO_FIX] = imputer.transform(df_new[COLS_TO_FIX])

    # Scale continuous columns using train-fitted scaler
    df_new[CONTINUOUS_COLS] = scaler.transform(df_new[CONTINUOUS_COLS])

    # Ensure column order matches training
    feature_order = CONTINUOUS_COLS + ["SkinThickness_missing", "Insulin_missing"]
    df_new = df_new[feature_order]

    prob = model.predict_proba(df_new)[:, 1][0]
    prediction = int(prob >= threshold)
    return prob, prediction

# -----------------------------
# UI
# -----------------------------
st.title("🩺 Diabetes Risk Screening")
st.markdown(
    "A logistic regression screening tool that estimates diabetes risk from "
    "diagnostic measurements. Enter patient values below — units are shown "
    "next to each field. Leave a field at its minimum (0) if the test wasn't performed; "
    "the model accounts for missing measurements separately."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies (count)", min_value=0, max_value=20, value=1, step=1
    )
    glucose = st.number_input(
        "Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg, diastolic)", min_value=0, max_value=200, value=70, step=1
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm, triceps skinfold)", min_value=0, max_value=100, value=20, step=1
    )

with col2:
    insulin = st.number_input(
        "Insulin (mu U/mL, 2-hr serum)", min_value=0, max_value=900, value=80, step=1
    )
    bmi = st.number_input(
        "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=28.0, step=0.1
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function (score)", min_value=0.0, max_value=3.0, value=0.5, step=0.01,
        help="A score reflecting diabetes likelihood based on family history."
    )
    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=33, step=1
    )

st.divider()

if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    raw_patient = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    prob, prediction = predict_diabetes_risk(raw_patient)

    st.subheader("Result")
    st.metric("Estimated Diabetes Probability", f"{prob*100:.1f}%")

    if prediction == 1:
        st.error(
            f"⚠️ Flagged for follow-up (probability {prob*100:.1f}% ≥ threshold "
            f"{FINAL_THRESHOLD*100:.0f}%). This tool is tuned to prioritize catching "
            f"potential diabetics, so it flags at a lower probability than a standard 50% cutoff."
        )
    else:
        st.success(
            f"✅ Not flagged (probability {prob*100:.1f}% < threshold "
            f"{FINAL_THRESHOLD*100:.0f}%)."
        )

    with st.expander("Why this threshold?"):
        st.markdown(
            f"""
            This model uses a decision threshold of **{FINAL_THRESHOLD}** instead of the
            default 0.5. On the held-out test set, this threshold achieves:
            - **Recall (diabetics correctly caught): 89%**
            - **Precision (flagged patients who are actually diabetic): 58%**

            In a screening context, missing a diabetic patient (false negative) is
            considered more costly than an unnecessary follow-up test (false positive),
            so the threshold was deliberately lowered to prioritize recall.
            """
        )

st.divider()
st.caption(
    "⚕️ This tool is for educational/portfolio purposes only and is not a substitute "
    "for professional medical diagnosis. Trained on the Pima Indians Diabetes dataset."
)