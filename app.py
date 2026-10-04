from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Dropout Risk",
    page_icon="🎓",
    layout="centered",
)

MODEL_PATH = Path("model.pkl")
FEATURE_COLUMNS = ["Attendance_Percentage", "Previous_Grades", "Financial_Issues"]
RISK_LABELS = {0: "Low Risk of Dropout", 1: "High Risk of Dropout"}

@st.cache_resource
def load_model(model_path: str, model_mtime_ns: int):
    with open(model_path, "rb") as file:
        return pickle.load(file)

st.title("Student Dropout Prediction System")
st.write("Estimate dropout risk from attendance, previous grades, and financial issues.")

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Run 'python train_model.py' to generate the model file.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"Could not load the model ({type(exc).__name__}).")
    st.stop()

with st.form("prediction_form"):
    attendance = st.slider("Attendance Percentage", 0, 100, 75)
    grades = st.slider("Previous Grades", 0, 100, 65)
    financial_issues = st.selectbox("Financial Issues?", ["No", "Yes"])
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame(
        [{
            "Attendance_Percentage": attendance,
            "Previous_Grades": grades,
            "Financial_Issues": int(financial_issues == "Yes"),
        }],
        columns=FEATURE_COLUMNS,
    )
    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error(f"Prediction: {RISK_LABELS[1]}")
        else:
            st.success(f"Prediction: {RISK_LABELS[0]}")
    except Exception as exc:
        st.error(f"Prediction failed ({type(exc).__name__}).")
