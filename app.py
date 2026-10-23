# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API is Running!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # Read features from URL query parameters
        raw_features = [request.args.get(f"f{i}") for i in range(1, 5)]

        if any(value is None for value in raw_features):
            return jsonify({
                "error": "Provide all four URL parameters: f1, f2, f3, and f4."
            }), 400
    else:
        # Read features from a POST JSON body
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or "features" not in data:
            return jsonify({
                "error": "Send JSON containing a 'features' list."
            }), 400

        raw_features = data["features"]

    try:
        features = np.asarray(raw_features, dtype=float)

        if features.shape != (4,):
            return jsonify({"error": "Provide exactly 4 numeric features."}), 400

        prediction = int(model.predict(features.reshape(1, -1))[0])

        classes = ["setosa", "versicolor", "virginica"]
        if prediction not in range(len(classes)):
            return jsonify({"error": "Model returned an unexpected class."}), 500

        return jsonify({"prediction": classes[prediction]})

    except (TypeError, ValueError):
        return jsonify({"error": "Features must be numeric."}), 400

if __name__ == "__main__":
    app.run(debug=True)