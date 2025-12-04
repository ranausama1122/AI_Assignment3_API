from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# Update these based on your CSV columns
model_columns = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']

@app.route('/')
def home():
    return "API is running successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = df.reindex(columns=model_columns, fill_value=0)
        pred = model.predict(df)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()
