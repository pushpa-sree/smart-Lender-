import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
csv_path = Path(__file__).parent / "data" / "train_u6lujuX_CVtuZ9i.csv"
df = pd.read_csv(csv_path)

# Remove Loan_ID
df = df.drop("Loan_ID", axis=1)

# Fill missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
encoder = LabelEncoder()

categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "models/loan_model.pkl")

print("Model trained successfully!")