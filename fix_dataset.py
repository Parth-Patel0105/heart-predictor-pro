import pandas as pd
import os

# Proper column names for UCI Heart Disease dataset
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Download raw data (no headers, has ? for missing values)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
filename = "data/heart_disease_raw.data"

os.makedirs('data', exist_ok=True)

print("Downloading raw UCI Heart Disease data...")
df = pd.read_csv(url, header=None, names=COLUMN_NAMES, na_values='?')

print(f"✅ Downloaded {len(df)} rows")
print(f"✅ Columns: {df.columns.tolist()}")

# Handle missing values (fill with median)
df = df.fillna(df.median())

# Convert target to binary (0 = no disease, 1-4 = disease)
df['target'] = (df['target'] > 0).astype(int)

# Save as proper CSV with headers
df.to_csv('data/heart_disease.csv', index=False)

print(f"✅ Cleaned dataset saved to data/heart_disease.csv")
print(f"✅ Final shape: {df.shape}")
print(f"✅ Target distribution:\n{df['target'].value_counts()}")

# Verify it loads correctly
test_df = pd.read_csv('data/heart_disease.csv')
print(f"✅ Verification: {test_df.shape} rows, columns: {test_df.columns.tolist()}")