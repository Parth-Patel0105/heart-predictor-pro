# 🫀 **Heart Disease Risk Predictor**

**Live Demo**: https://heart-predictor.streamlit.app/

An AI-powered medical risk assessment tool that predicts heart disease probability using machine learning, built with clinical interpretability and safety as top priorities.

---

## 🎯 **Project Overview**

This project demonstrates responsible AI development in healthcare by implementing an interpretable logistic regression model on the UCI Cleveland Heart Disease dataset. The application provides probability-based risk stratification with SHAP feature explanations, emphasizing the critical balance between model performance and medical safety.

---

## ✨ **Key Features**

| Feature | Description |
|---------|-------------|
| **🎲 Probability-Based Risk** | Outputs 0-100% risk scores (not just binary classification) |
| **🧠 SHAP Interpretability** | Shows which clinical features drive each prediction |
| **📊 Cross-Validation** | Robust 5-fold stratified CV (83.2% ±5% accuracy) |
| **🚑 High Recall Priority** | 79.8% recall minimizes false negatives for patient safety |
| **🔒 Production-Ready** | Deployed on Streamlit Cloud with FDA-style disclaimers |
| **📱 Medical UI** | Clean interface designed for clinical workflows |

---

## 🛠️ **Technology Stack**

| Technology | Purpose |
|------------|---------|
| **Python 3.13+** | Core programming language |
| **Scikit-learn** | Logistic regression & preprocessing |
| **Pandas** | Data manipulation & feature engineering |
| **Streamlit** | Interactive web application |
| **Joblib** | Model serialization |
| **UCI Cleveland Dataset** | 303 patient records, 14 clinical features |

--------------------------------------------------------------------------------------------------------------------------------------------------
## 📦 **Installation & Local Development**

### **1. Clone Repository**
```bash
git clone https://github.com/Parth-Patel0105/heart-predictor-pro.git
cd heart-predictor-pro

2. Install Dependencies

bash:
pip install -r requirements.txt

3. Run Locally

bash:
streamlit run streamlit_app.py

The app will automatically load the pre-trained model and open in your browser at http://localhost:8501
--------------------------------------------------------------------------------------------------------------------------------------------------
📊 Model Performance
Cross-Validation Results (5-fold Stratified)
Accuracy:  83.2% (±5.0%)
Recall:    79.8% (±6.7%)  ⭐ Medical Safety Priority
ROC-AUC:   0.912 (±0.018)

Why High Recall Matters
In healthcare, missing a sick patient (false negative) is far more dangerous than flagging a healthy one (false positive). This model prioritizes recall (79.8%) to minimize missed diagnoses.
--------------------------------------------------------------------------------------------------------------------------------------------------
🔬 Clinical Features Used

Age (years)
Sex (Male/Female)
Chest Pain Type (Typical/Atypical/Non-anginal/Asymptomatic)
Resting Blood Pressure (mm Hg)
Cholesterol (mg/dl)
Fasting Blood Sugar >120 (Yes/No)
Resting ECG Results (Normal/ST-T/LV Hypertrophy)
Maximum Heart Rate Achieved
Exercise Induced Angina (Yes/No)
ST Depression (mm)
Slope of Peak Exercise ST (Upsloping/Flat/Downsloping)
Number of Major Vessels (0-3)
Thalassemia (Normal/Fixed/Reversible)
--------------------------------------------------------------------------------------------------------------------------------------------------
🚀 Deployment
Streamlit Cloud (Production)

bash:
# Push to GitHub
git add .
git commit -m "Deploy: Production-ready heart disease predictor"
git push origin main

# Deploy on https://share.streamlit.io
# App URL: https://heart-predictor.streamlit.app/
--------------------------------------------------------------------------------------------------------------------------------------------------
⚠️ Medical Disclaimer
This tool is for educational purposes only.
It is NOT intended for:
Clinical diagnosis
Treatment decisions
Replacing physician judgment
Real-world patient care
FDA Status: Investigational Device – Not for Clinical Use
Accuracy Limitations: ~83% on test data; real-world performance may vary significantly.
--------------------------------------------------------------------------------------------------------------------------------------------------
📚 Dataset Attribution
Source: UCI Machine Learning Repository
Dataset: Heart Disease Dataset
Citation: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
--------------------------------------------------------------------------------------------------------------------------------------------------
🎯 Project Highlights for Your Resume
✅ Responsible AI: Prioritized high recall for medical safety
✅ Interpretability: SHAP values explain predictions for clinical trust
✅ Statistical Rigor: 5-fold stratified cross-validation (no data leakage)
✅ Production Deployment: Streamlit Cloud with error handling & disclaimers
✅ Real-World Data: 303 patients, 14 clinical features from UCI dataset
--------------------------------------------------------------------------------------------------------------------------------------------------
👨‍💻 Author
Parth Patel
GitHub: @Parth-Patel0105
--------------------------------------------------------------------------------------------------------------------------------------------------
📄 License
MIT License - See LICENSE file for details
