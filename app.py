from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model, scaler, and feature list
model = joblib.load("models/rf_churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    
    # Convert numerical features
    for col in ['tenure','MonthlyCharges','TotalCharges']:
        form_data[col] = float(form_data[col])
    
    # Convert to dataframe
    input_df = pd.DataFrame([form_data])
    
    # One-hot encode to match training features
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Scale numerical features
    input_df[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(input_df[['tenure','MonthlyCharges','TotalCharges']])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return render_template("index.html", prediction_text=f"Predicted Churn: {'Yes' if prediction==1 else 'No'}",
                           prediction_prob=probability)

if __name__ == "__main__":
    app.run(debug=True)
