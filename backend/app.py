
import os
import json
import joblib
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "insurance_claim_model.joblib"
)

INFO_PATH = os.path.join(
    BASE_DIR, "model", "model_info.json"
)

HISTORY_PATH = os.path.join(
    BASE_DIR, "data", "predictions.csv"
)

model = joblib.load(MODEL_PATH)

with open(INFO_PATH, "r") as f:
    model_info = json.load(f)

FEATURES = model_info["selected_features"]

@app.route("/")
def home():
    return jsonify({"message": "Insurance Claim Prediction API is running"})

@app.route("/model-info")
def model_information():
    return jsonify(model_info)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Input data required"}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": "Missing features", "features": missing}), 400

        input_data = pd.DataFrame([{
            feature: data[feature] for feature in FEATURES
        }])

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        result = {
            "prediction": prediction,
            "claim_status": "Claim" if prediction == 1 else "No Claim",
            "claim_probability": round(probability, 4)
        }

        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

        row = {
            **{f: data[f] for f in FEATURES},
            **result
        }

        history = pd.DataFrame([row])

        history.to_csv(
            HISTORY_PATH,
            mode="a",
            header=not os.path.exists(HISTORY_PATH),
            index=False
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def history():
    if not os.path.exists(HISTORY_PATH):
        return jsonify([])

    return jsonify(
        pd.read_csv(HISTORY_PATH).fillna("").to_dict(orient="records")
    )

@app.route("/dashboard")
def dashboard():
    if not os.path.exists(HISTORY_PATH):
        return jsonify({"total": 0, "yes": 0, "no": 0})

    df = pd.read_csv(HISTORY_PATH)

    return jsonify({
        "total": int(len(df)),
        "yes": int((df["prediction"] == 1).sum()),
        "no": int((df["prediction"] == 0).sum())
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
