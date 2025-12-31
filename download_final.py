import urllib.request
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
filename = "data/heart_disease.csv"

os.makedirs('data', exist_ok=True)

# Download directly from UCI
urllib.request.urlretrieve(url, filename)

print(f"✅ Downloaded {filename}")
print(f"✅ File size: {os.path.getsize(filename)} bytes")

# Verify
with open(filename) as f:
    lines = len(f.readlines())
print(f"✅ Total lines: {lines} (should be 303 data rows + 1 header = 304)")