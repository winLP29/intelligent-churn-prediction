import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay

# -------------------- Step 1: Load & Clean Data --------------------
df = pd.read_csv("data/telco.csv")

# Convert TotalCharges to numeric and drop missing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Drop customerID (not useful)
df = df.drop("customerID", axis=1)

# Convert target variable Churn to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Define numerical and categorical columns
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = df.select_dtypes(include=["object"]).columns

# -------------------- Step 1b: EDA --------------------
# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between Numerical Features")
plt.show()

# Visualize tenure distribution by churn
plt.figure(figsize=(8,5))
sns.histplot(data=df, x="tenure", hue="Churn", kde=True, bins=30)
plt.title("Tenure Distribution by Churn")
plt.show()

# Visualize monthly charges by churn
plt.figure(figsize=(6,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# -------------------- Step 2: Feature Engineering --------------------
# Create tenure groups
df['tenure_group'] = pd.cut(df['tenure'],
                            bins=[0, 12, 24, 48, 60, 72],
                            labels=['0-12','12-24','24-48','48-60','60-72'])

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode categorical features (including tenure_group)
X = pd.get_dummies(X, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# -------------------- Step 3: Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# -------------------- Step 4: Model Training --------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    results[name] = {"Accuracy": acc, "ROC_AUC": roc_auc}
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# -------------------- Step 5: Confusion Matrix & ROC Curve --------------------
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    
    # ROC curve
    if y_prob is not None:
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title(f"{name} ROC Curve")
        plt.show()

# -------------------- Step 6: Feature Importance (Random Forest & XGBoost) --------------------
for name in ["Random Forest", "XGBoost"]:
    model = models[name]
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)[:15]  # top 15 features
        plt.figure(figsize=(8,6))
        sns.barplot(x=importances, y=importances.index)
        plt.title(f"{name} Top 15 Feature Importances")
        plt.show()
