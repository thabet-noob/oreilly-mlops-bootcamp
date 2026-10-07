from flask import Flask, request, jsonify, render_template
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

app = Flask(__name__)

# Get the current directory and construct dynamic paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the latest model from MLflow
try:
    # Get the experiment by name
    experiment = mlflow.get_experiment_by_name("Income_Classification")
    if experiment is None:
        raise Exception("No experiment found with name 'Income_Classification'")
    
    # Get the latest top-level run (nested hyperparameter-search candidates don't store a model)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and tags.mlflow.parentRunId IS NULL",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise Exception("No finished runs found in the experiment")
    
    latest_run = runs.iloc[0]
    run_id = latest_run.run_id
    
    # Load the model from the latest run
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model from run: {run_id}")
    
except Exception as e:
    print(f"Error loading MLflow model: {e}")
    print("Falling back to local model file...")
    # Fallback to local model file
    model_path = os.path.join(current_dir, "model", "rf_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Loaded local model file")
    else:
        raise Exception("No model found locally or in MLflow")

# Load preprocessor
preprocessor_path = os.path.join(current_dir, "model", "preprocessor.pkl")
if os.path.exists(preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
else:
    raise Exception("Preprocessor file not found. Please run training first.")

# Form fields for the web page, built from what the preprocessor learned during training
NUMERIC_FIELDS = {  # name: (default, min, max)
    "age": (39, 17, 90),
    "fnlwgt": (189778, 10000, 1500000),
    "education-num": (13, 1, 16),
    "capital-gain": (0, 0, 99999),
    "capital-loss": (0, 0, 4356),
    "hours-per-week": (40, 1, 99),
}
CATEGORICAL_DEFAULTS = {
    "workclass": " Private",
    "education": " Bachelors",
    "marital-status": " Married-civ-spouse",
    "occupation": " Prof-specialty",
    "relationship": " Husband",
    "race": " White",
    "sex": " Male",
    "native-country": " United-States",
}
categorical_options = {}
for name, transformer, columns in preprocessor.transformers_:
    if name == "cat":
        for column, categories in zip(columns, transformer.named_steps["onehot"].categories_):
            categorical_options[column] = [str(c) for c in categories]

form_fields = []
for column in preprocessor.feature_names_in_:
    if column in categorical_options:
        form_fields.append({"name": column, "type": "select", "options": categorical_options[column],
                            "default": CATEGORICAL_DEFAULTS.get(column, categorical_options[column][0])})
    else:
        default, min_value, max_value = NUMERIC_FIELDS[column]
        form_fields.append({"name": column, "type": "number", "default": default, "min": min_value, "max": max_value})

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=form_fields)

@app.route("/predict", methods=["POST"])
def predict():
    try:
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
        label_map = {0: "<=50K", 1: ">50K"}
        predicted_label = label_map[int(prediction[0])]

        # Probability of the >50K class (column 1), useful for the web page
        probability = float(model.predict_proba(processed_data)[0][1])

        # Return the prediction
        return jsonify({"prediction": predicted_label, "probability_>50K": round(probability, 3)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Local runs listen on localhost only; the Docker image sets HOST=0.0.0.0 so the container is reachable
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
