# Lesson 5: End-to-End ML Pipeline – Adult Income Classification

This lesson demonstrates how to build, train, track, and deploy a full machine learning pipeline for the **Adult Income Classification** problem using real census data.

---

##  Objective

Classify individuals into one of two income categories:
- **Group 1**: Income ≤ 50K  
- **Group 2**: Income > 50K  

The model uses 1994 U.S. Census data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult).

---

##  Dataset Overview

The dataset contains demographic and employment-related features:
- **Numerical**: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- **Categorical**: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Target**: income (`<=50K`, `>50K`)

---

##  Pipeline Overview

### 1. Data Ingestion
- Load data from `adult.data`
- Perform exploratory data analysis

### 2. Data Preprocessing
- Clean missing and inconsistent entries
- Encode target (`<=50K`: 0, `>50K`: 1)
- Transform features:
  - Scale numeric features (StandardScaler)
  - Encode categorical features (OneHotEncoder)
- Apply SMOTE for class balancing

### 3. Model Training & Tracking
- Train a `RandomForestClassifier` using the transformed dataset
- Use **MLflow** to:
  - Track experiment metadata (parameters, metrics)
  - Save models and pipelines for reproducibility

### 4. Deployment
- Save model and preprocessing pipeline using `joblib`
- Create a **Flask API** (`app.py`) for real-time inference
- Containerize the app using **Docker**
- Test the deployed API with **Postman**

---

##  Project Structure

- `data/`: Input dataset (`adult.data`)
- `data_pipeline/`: Scripts for EDA and preprocessing
- `train.py`: Trains model and logs to MLflow
- `app.py`: Flask API for inference
- `Dockerfile`: Containerizes the API
- `requirements.txt`: Required packages (pinned, so the Docker image can load the model trained locally)

---

##  How to Use

Run every command from inside `lesson-5-ml-pipeline`. MLflow stores runs in `mlflow.db` and `mlruns/` in the current folder, so `train.py` and `app.py` must be started from the same place.

### 1. Clone the repo
```bash
git clone https://github.com/AmmarMohanna/oreilly-mlops-bootcamp.git
cd "oreilly-mlops-bootcamp/Day 1/lesson-5-ml-pipeline"
```

### 2. Create a virtual environment and install the libraries
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Or with [uv](https://docs.astral.sh/uv/): `uv venv .venv --python 3.12 && uv pip install -r requirements.txt`

### 3. Run preprocessing and training
```bash
python train.py
```
This saves `model/rf_model.pkl` and `model/preprocessor.pkl`, and logs a run (parameters, metrics and the model) to the `Income_Classification` MLflow experiment.

To browse the runs: `mlflow ui --port 5000`, then open http://127.0.0.1:5000

### 4. Start the Flask app locally
```bash
python app.py
```
The app loads the model from the latest **finished** MLflow run automatically (no run ID to copy). If none is found, it falls back to `model/rf_model.pkl`. It listens on http://127.0.0.1:8000.

### 5. Test the API
**In the browser:** open http://localhost:8000. The form has pre-filled default values, dropdowns for the categorical fields (built from the categories the model was trained on) and number inputs for the numeric fields. Click **Predict** to see the prediction and the probability of >50K.

**As an API:** `/predict` only accepts **POST** with a JSON body. It returns the prediction and the probability, e.g. `{"prediction": ">50K", "probability_>50K": 0.835}`.

In **Postman**: method `POST`, URL `http://localhost:8000/predict`, Body → **raw** → **JSON**:
```json
{
  "age": 39, "workclass": " State-gov", "fnlwgt": 77516, "education": " Bachelors",
  "education-num": 13, "marital-status": " Never-married", "occupation": " Adm-clerical",
  "relationship": " Not-in-family", "race": " White", "sex": " Male",
  "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": " United-States"
}
```
Or with curl:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":39,"workclass":" State-gov","fnlwgt":77516,"education":" Bachelors","education-num":13,"marital-status":" Never-married","occupation":" Adm-clerical","relationship":" Not-in-family","race":" White","sex":" Male","capital-gain":2174,"capital-loss":0,"hours-per-week":40,"native-country":" United-States"}'
```
Response: `{"prediction": "<=50K"}` (or `">50K"`).

Notes:
- All 14 fields are required.
- Text values start with a space (`" Bachelors"`), exactly as in `adult.data`. Values without the space are treated as unknown categories and give less accurate predictions.

### 6. Or run it with Docker
Run step 3 first: the image packages the trained files in `model/`.
```bash
docker build -t income-classifier .
docker run -p 8000:8000 income-classifier
```
Then test it exactly as in step 5 (form at http://localhost:8000, API at `/predict`).

Inside the container the app uses `model/rf_model.pkl`, because the local MLflow store is not copied into the image. It prints "Error loading MLflow model ... Falling back to local model file", which is expected.

If `pip install` fails during the build with `CERTIFICATE_VERIFY_FAILED` (corporate TLS inspection), pass your company's root CA:
```bash
docker build --secret id=corp_ca,src=/path/to/corporate-root-ca.pem -t income-classifier .
```

---

##  Troubleshooting

| Problem | Fix |
|---|---|
| `Address already in use` / `Port 8000 is in use` | Another app or container is on port 8000. Find it with `lsof -i :8000` or `docker ps`, and stop it (`docker stop <id>`). Or run on another port: `PORT=8001 python app.py` |
| `GET /favicon.ico 404` in the log | Harmless: the browser asks for a tab icon, which the app does not have |
| `OSError: [Errno 57] Socket is not connected` | Harmless: a browser opened and closed a connection without sending a request. Close browser tabs pointing at port 8000 |
| `415 Unsupported Media Type` | Send the body as raw JSON (`Content-Type: application/json`), not form-data |
| `columns are missing: {...}` | The JSON body is missing some of the 14 feature fields |
| `Error loading MLflow model ... Falling back to local model file` | No finished run was found in the folder you started from. Run `python train.py` from this folder, or ignore it: the local model is used |
