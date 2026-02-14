# train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and clean data
df = pd.read_csv("data/telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Feature engineering
df['tenure_group'] = pd.cut(df['tenure'],
                            bins=[0,12,24,48,60,72],
                            labels=['0-12','12-24','24-48','48-60','60-72'])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Scale numerical features
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "models/rf_churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns, "models/features.pkl")  # Save columns for input alignment
print("Model, scaler, and features saved!")
