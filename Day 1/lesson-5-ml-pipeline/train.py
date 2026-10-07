import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os

from data_pipeline.preprocessing import load_and_preprocess_data

# Hyperparameter search space for the RandomForest
PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
N_ITER = 10      # number of random combinations to try
CV_FOLDS = 3     # cross-validation folds per combination
SCORING = "f1"   # F1 of the >50K class: more informative than accuracy on imbalanced data

def train():
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

    mlflow.set_experiment("Income_Classification")
    with mlflow.start_run(run_name="rf_hyperparam_search"):

        # Try N_ITER random combinations, each scored with CV_FOLDS-fold cross-validation on the training set
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=N_ITER,
            cv=CV_FOLDS,
            scoring=SCORING,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)

        # Log every combination as a nested child run so they can be compared in the MLflow UI
        results = search.cv_results_
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric(f"cv_{SCORING}_mean", results["mean_test_score"][i])
                mlflow.log_metric(f"cv_{SCORING}_std", results["std_test_score"][i])
                mlflow.log_metric("fit_time_sec", results["mean_fit_time"][i])

        # The parent run holds the best combination, evaluated on the held-out test set
        clf = search.best_estimator_
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_params({"search_n_iter": N_ITER, "search_cv_folds": CV_FOLDS, "search_scoring": SCORING})
        mlflow.log_params(search.best_params_)
        mlflow.log_metric(f"best_cv_{SCORING}", search.best_score_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        print(f"Best params: {search.best_params_}")
        print(f"Best CV {SCORING}: {search.best_score_:.4f} | test accuracy: {acc:.4f}")

        for label, scores in report.items():
            if isinstance(scores, dict):  
                mlflow.log_metric(f"{label}_precision", scores["precision"])
                mlflow.log_metric(f"{label}_recall", scores["recall"])

        # Create model directory if it doesn't exist
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and preprocessor with dynamic paths
        model_path = os.path.join(model_dir, "rf_model.pkl")
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        
        joblib.dump(clf, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.sklearn.log_model(clf, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

if __name__ == "__main__":
    train()
