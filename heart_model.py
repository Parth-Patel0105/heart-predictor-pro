import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import pickle

warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.accuracy = None
        self.recall = None
        self.cv_scores = None  # NEW: Store cross-validation scores
        
    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def train_with_cv(self, filepath='data/heart_disease.csv', cv_folds=5):
        """
        🎯 Train with Stratified K-Fold Cross-Validation
        This gives ROBUST performance estimates instead of misleading single-split scores
        """
        print("Loading heart disease data...")
        df = self.load_data(filepath)
        X, y = df.drop('target', axis=1), df['target']
        
        print(f"\n🔍 Performing {cv_folds}-fold Stratified Cross-Validation...")
        
        # Scale features ONCE before cross-validation
        X_scaled = self.scaler.fit_transform(X)
        
        # Stratified K-Fold ensures each fold preserves class distribution
        # This is CRITICAL for medical datasets with imbalanced classes
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Calculate cross-validation scores
        # Focus on RECALL for medical safety (minimize false negatives)
        cv_accuracy = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_recall = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='recall')
        cv_roc_auc = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='roc_auc')
        
        # Store CV scores for reporting
        self.cv_scores = {
            'accuracy': cv_accuracy,
            'recall': cv_recall,
            'roc_auc': cv_roc_auc,
            'folds': cv_folds
        }
        
        # Report HONEST cross-validation performance
        print(f"\n📊 Cross-Validation Results ({cv_folds} folds):")
        print(f"   Accuracy: {cv_accuracy.mean():.1%} (+/- {cv_accuracy.std():.1%})")
        print(f"   Recall:   {cv_recall.mean():.1%} (+/- {cv_recall.std():.1%})")
        print(f"   ROC-AUC:  {cv_roc_auc.mean():.3f} (+/- {cv_roc_auc.std():.3f})")
        
        # --- Train FINAL model on ALL data for deployment ---
        print("\n🎯 Training final model on full dataset...")
        self.model.fit(X_scaled, y)
        
        # For small datasets, use CV scores as primary metrics
        # For large datasets, you could keep a hold-out test set
        self.accuracy = cv_accuracy.mean()
        self.recall = cv_recall.mean()
        
        print(f"✅ Model training complete!")
        print(f"   Final CV Accuracy: {self.accuracy:.1%}")
        print(f"   Final CV Recall: {self.recall:.1%} (medical safety priority)")
        
        return self.cv_scores
    
    def predict_with_risk(self, patient_data):
        """Predict with probability and risk stratification"""
        if self.model is None or self.scaler is None:
            raise RuntimeError("❌ Model not trained. Call train_with_cv() first.")
        
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data
        
        # Ensure all 13 clinical features exist
        required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Scale features
        features_scaled = self.scaler.transform(df[required_features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        # Clinical risk stratification
        if probability >= 0.7:
            risk_level = "High Risk"
            risk_emoji = "🔴"
        elif probability >= 0.4:
            risk_level = "Medium Risk"
            risk_emoji = "🟡"
        else:
            risk_level = "Low Risk"
            risk_emoji = "🟢"
        
        return {
            'prediction': int(prediction),
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'risk_probability': float(probability),
            'confidence': float(max(probability, 1-probability)),
            'model_accuracy': self.accuracy,
            'model_recall': self.recall,
            'cv_scores': self.cv_scores  # Include CV scores for transparency
        }
    
    def generate_report(self, patient_data):
        """Generate comprehensive medical report"""
        result = self.predict_with_risk(patient_data)
        
        cv = result['cv_scores']
        report = f"""
# Heart Disease Risk Assessment Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}**

## Risk Profile
**{result['risk_emoji']} Risk Level:** {result['risk_level']}  
**Probability:** {result['risk_probability']:.1%}  
**Model Confidence:** {result['confidence']:.1%}

## Model Performance (Cross-Validated)
**Accuracy:** {cv['accuracy'].mean():.1%} (±{cv['accuracy'].std():.1%})  
**Recall:** {cv['recall'].mean():.1%} (±{cv['recall'].std():.1%})  
**ROC-AUC:** {cv['roc_auc'].mean():.3f} (±{cv['roc_auc'].std():.3f})

## Clinical Interpretation
"""
        
        if result['risk_level'] == "High Risk":
            report += "🔴 **Recommendation:** Consult cardiologist immediately."
        elif result['risk_level'] == "Medium Risk":
            report += "🟡 **Recommendation:** Lifestyle modifications advised."
        else:
            report += "🟢 **Recommendation:** Continue healthy habits."
        
        report += "\n\n**⚠️ Disclaimer:** Educational tool only. Not for clinical diagnosis."
        
        return report
    
    def save_model(self, filepath='heart_model_cv.pkl'):
        """Save model with cross-validation scores"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'accuracy': self.accuracy,
            'recall': self.recall,
            'cv_scores': self.cv_scores
        }, filepath)
        print(f"✅ Model saved with CV scores to {filepath}")
    
    def load_model(self, filepath='heart_model_cv.pkl'):
        """Load model with cross-validation scores"""
        saved = joblib.load(filepath)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.accuracy = saved['accuracy']
        self.recall = saved['recall']
        self.cv_scores = saved.get('cv_scores', None)
        print(f"✅ Model loaded from {filepath}")

# Initialize predictor
predictor = HeartDiseasePredictor()