
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load saved model
MODEL_PATH = Path(__file__).parent / "placement_prediction_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Page heading
st.title("🎓 Student Placement Prediction System")
st.write(
    "Enter your academic details and skills to get "
    "a machine learning prediction."
)

st.info(
    "This is an educational project. Predictions are estimates, "
    "not guarantees of employment."
)

st.divider()

# Student input form
st.subheader("📝 Enter Student Details")

with st.form("placement_form"):

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.number_input(
            "CGPA (out of 10)",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

        internships = st.number_input(
            "Number of Internships",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        aptitude = st.slider(
            "Aptitude Test Score",
            min_value=0,
            max_value=100,
            value=75
        )

        communication = st.slider(
            "Communication Score",
            min_value=0,
            max_value=100,
            value=75
        )

    with col2:
        projects = st.number_input(
            "Number of Projects",
            min_value=0,
            max_value=50,
            value=2,
            step=1
        )

        certifications = st.number_input(
            "Number of Certifications",
            min_value=0,
            max_value=50,
            value=2,
            step=1
        )

        training = st.selectbox(
            "Placement Training Completed?",
            ["Yes", "No"]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Placement",
        use_container_width=True
    )

# Prediction
if submitted:

    student = pd.DataFrame([{
        "CGPA": cgpa,
        "Internships": internships,
        "AptitudeTestScore": aptitude,
        "CommunicationScore": communication,
        "Projects": projects,
        "Certifications": certifications,
        "PlacementTraining": training
    }])

    prediction = model.predict(student)[0]

    classes = list(
        model.named_steps["classifier"].classes_
    )

    probabilities = model.predict_proba(student)[0]

    placed_probability = probabilities[
        classes.index("Placed")
    ]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == "Placed":
        st.success("Prediction: Placed")
    else:
        st.warning("Prediction: Not Placed")

    st.metric(
        "Estimated Placement Probability",
        f"{placed_probability * 100:.2f}%"
    )

    st.progress(float(placed_probability))

    st.caption(
        "The probability is based on the trained model and "
        "its sample dataset. It is not a guarantee of placement."
    )

    st.subheader("Student Details")
    st.dataframe(student, use_container_width=True)

st.divider()
st.caption("Placement Prediction Project | Machine Learning with Python")
