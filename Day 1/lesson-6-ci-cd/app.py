from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

app = Flask(__name__)

print("Hello World")
# Load model and preprocessor from local files for CI/CD reliability
# add a comment here , 
try:
    model = joblib.load("model/rf_model.pkl")
    preprocessor = joblib.load("model/preprocessor.pkl")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Make sure to run train.py first to generate model files.")
    model = None
    preprocessor = None 
 
# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model and preprocessor are loaded
        if model is None or preprocessor is None:
            return jsonify({"error": "Model not loaded. Please run train.py first."}), 500
            
        # Check if the input is in JSON format
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided or invalid format"}), 400

        # Convert the data to a pandas DataFrame
        data = pd.DataFrame([data])

        # Apply preprocessing to input data
        processed_data = preprocessor.transform(data)
        
        # Make the prediction
        prediction = model.predict(processed_data)

        # Map prediction back to label
        label_map = {0: " <=50K", 1: " >50K"}
        predicted_label = label_map[int(prediction[0])]

        # Return the prediction
        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
