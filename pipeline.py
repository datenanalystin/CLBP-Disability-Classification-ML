"""
Pipeline for Machine Learning Disability Classification in CLBP
Strictly built for TRIPOD-AI and STROBE compliance.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Fixed seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ----------------------------------------------------------------------
# 1. Keras ANN Wrapper for Scikit-Learn Compatibility
# ----------------------------------------------------------------------
class KerasANNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, epochs=100, batch_size=32, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.classes_ = np.array([0, 1])

    def _build_model(self, input_dim):
        model = Sequential(
            [
                Dense(8, activation="relu", input_dim=input_dim),
                Dense(5, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model = self._build_model(input_dim=X.shape[1])
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        probs_pos = self.model.predict(X, verbose=0).ravel()
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


# ----------------------------------------------------------------------
# 2. Main Execution Pipeline
# ----------------------------------------------------------------------
def run_pipeline(data_path=None):
    # Generating synthetic clinical dataset structure if no CSV path is provided
    if data_path is None:
        print(
            "No dataset path provided. Generating sample synthetic CLBP dataset (N=503)..."
        )
        n_samples = 503
        X_synthetic = pd.DataFrame(
            {
                "VAS": np.random.uniform(1.0, 10.0, n_samples),
                "TSK": np.random.uniform(17.0, 68.0, n_samples),
                "PASS-20": np.random.uniform(0.0, 100.0, n_samples),
                "PCS": np.random.uniform(0.0, 52.0, n_samples),
                "Age": np.random.uniform(20.0, 60.0, n_samples),
            }
        )
        # Introduce 2.5% missing values randomly to test KNN Imputer
        mask = np.random.rand(*X_synthetic.shape) < 0.025
        X_synthetic[mask] = np.nan

        # Target ODI class assignment (0: ODI < 30%, 1: ODI >= 30%)
        y_synthetic = (
            0.3 * X_synthetic["VAS"].fillna(5)
            + 0.04 * X_synthetic["TSK"].fillna(35)
            + np.random.normal(0, 1, n_samples)
            > 3.2
        ).astype(int)

        X, y = X_synthetic, y_synthetic
    else:
        df = pd.read_csv(data_path)
        feature_cols = ["VAS", "TSK", "PASS-20", "PCS", "Age"]
        X = df[feature_cols]
        y = (df["ODI"] >= 30).astype(int)

    # Train-Test Stratified Split (70% Train = 352, 30% Holdout = 151)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    print(
        f"Data Split Successfully: Training Set = {len(X_train)}, Holdout Test Set = {len(X_test)}"
    )

    # Encapsulated Pipeline Construction (Preventing Data Leakage)
    ann_pipeline = Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=5, weights="uniform")),
            ("scaler", StandardScaler()),
            (
                "ann",
                KerasANNClassifier(
                    epochs=100, batch_size=32, learning_rate=0.001
                ),
            ),
        ]
    )

    # Fit Pipeline exclusively on Training Set
    print("\nTraining primary ANN Pipeline on Training Set...")
    ann_pipeline.fit(X_train, y_train)

    # Evaluate on Holdout Test Set
    y_pred = ann_pipeline.predict(X_test)
    y_prob = ann_pipeline.predict_proba(X_test)[:, 1]

    # Calculate Performance Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)

    print("\n=======================================================")
    print("      INTERNAL HOLDOUT TEST SET PERFORMANCE (N=151)    ")
    print("=======================================================")
    print(f"ROC-AUC:          {auc:.3f}")
    print(f"Accuracy:         {acc:.3f}")
    print(f"Sensitivity:      {sens:.3f}")
    print(f"Specificity:      {spec:.3f}")
    print(f"F1-Score:         {f1:.3f}")
    print(f"Brier Score:      {brier:.3f}")
    print("=======================================================")

    # Serialize and Save the Model Object
    output_model_filename = "ann_model.joblib"
    joblib.dump(ann_pipeline, output_model_filename)
    print(
        f"\nModel and Preprocessing Pipeline successfully serialized to '{output_model_filename}'."
    )


if __name__ == "__main__":
    run_pipeline()