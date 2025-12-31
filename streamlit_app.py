import streamlit as st
from heart_model import predictor
import os

# ✅ Load trained model OR train if missing
if os.path.exists('heart_model_cv.pkl'):
    predictor.load_model()
else:
    with st.spinner("🔄 Training model for first time..."):
        predictor.train_with_cv()
        predictor.save_model()
    st.success("✅ Model trained and ready!")
    st.balloons()

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("🫀 Heart Disease Risk Predictor")

# Sidebar - Patient Information
st.sidebar.header("📊 Patient Information")

# Create all 13 input fields
age = st.sidebar.slider("Age (years)", 29, 77, 54)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.sidebar.slider("Resting BP (mm Hg)", 94, 200, 130)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 126, 564, 250)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", ["No", "Yes"])
restecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
thalach = st.sidebar.slider("Max Heart Rate", 71, 202, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression (mm)", 0.0, 6.2, 1.0)
slope = st.sidebar.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
ca = st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

# Map inputs to numbers (Deleted)

# CORRECT UCI Heart Disease Dataset mappings
# cp: 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic
# thal: 3=normal, 6=fixed defect, 7=reversible defect
cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
restecg_map = {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2}
slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}  # UCI uses 1,2,3
thal_map = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

# Fix: The app should match the dataset encoding
patient_data = {
    'age': age, 'sex': 1 if sex == "Male" else 0, 'cp': cp_map[cp],
    'trestbps': trestbps, 'chol': chol, 'fbs': 1 if fbs == "Yes" else 0,
    'restecg': restecg_map[restecg], 'thalach': thalach,
    'exang': 1 if exang == "Yes" else 0, 'oldpeak': oldpeak,
    'slope': slope_map[slope], 'ca': ca, 'thal': thal_map[thal]
}

# Prediction button
if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
    result = predictor.predict_with_risk(patient_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {result['risk_emoji']} Risk Level: **{result['risk_level']}**")
        st.metric("Risk Probability", f"{result['risk_probability']:.1%}")
        st.metric("Confidence", f"{result['confidence']:.1%}")
    
    with col2:
        st.metric("Test Accuracy", f"{result['model_accuracy']:.1%}")
        st.metric("Disease Detection", f"{result['model_recall']:.1%}")
    
    # ⬇️⬇️⬇️ ADDED CROSS-VALIDATION SECTION ⬇️⬇️⬇️
    # Show cross-validation results (more honest than single test set)
    if result['cv_scores'] is not None:
        st.markdown("### 📊 Cross-Validation Performance")
        st.info("These are **robust** performance estimates from 5-fold cross-validation")
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("CV Accuracy", 
                     f"{result['cv_scores']['accuracy'].mean():.1%}", 
                     f"±{result['cv_scores']['accuracy'].std():.1%}")
        with col4:
            st.metric("CV Recall", 
                     f"{result['cv_scores']['recall'].mean():.1%}", 
                     f"±{result['cv_scores']['recall'].std():.1%}")
    # ⬆️⬆️⬆️ END OF ADDED SECTION ⬆️⬆️⬆️

st.info("⚠️ **Disclaimer**: Educational tool only. Not a substitute for professional medical advice.")