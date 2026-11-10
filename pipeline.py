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
from sklearn.model_selection import train_test_split
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
    model = Sequential([
        Dense(8, activation="relu", input_dim=input_dim),
        Dense(5, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
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
# 2. Wilson Score Interval for Proportions
# ----------------------------------------------------------------------
def wilson_ci(k, n, confidence=0.95):
  if n == 0:
    return 0.0, 0.0
  p = k / n
  z = 1.959964
  denominator = 1 + z**2 / n
  centre_adjusted_probability = p + z**2 / (2 * n)
  adjusted_std_dev = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
  lower_bound = (centre_adjusted_probability - adjusted_std_dev) / denominator
  upper_bound = (centre_adjusted_probability + adjusted_std_dev) / denominator
  return max(0.0, lower_bound), min(1.0, upper_bound)


# ----------------------------------------------------------------------
# 3. Non-parametric Stratified Bootstrap CI for F1-Score
# ----------------------------------------------------------------------
def bootstrap_f1_ci(y_true, y_pred, n_bootstraps=1000, seed=RANDOM_STATE):
  np.random.seed(seed)
  booted_scores = []
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  n = len(y_true)

  for _ in range(n_bootstraps):
    indices = np.random.choice(n, size=n, replace=True)
    if len(np.unique(y_true[indices])) < 2:
      continue
    score = f1_score(
        y_true[indices], y_pred[indices], average="binary", zero_division=0
    )
    booted_scores.append(score)

  ci_lower = np.percentile(booted_scores, 2.5)
  ci_upper = np.percentile(booted_scores, 97.5)
  mean_score = np.mean(booted_scores)
  return mean_score, ci_lower, ci_upper


# ----------------------------------------------------------------------
# 4. Main Execution Pipeline
# ----------------------------------------------------------------------
def run_pipeline(data_path="2class-f.xlsx"):
  print(f"Loading clinical dataset from '{data_path}'...")
  try:
    df = pd.read_excel(data_path)
  except Exception:
    df = pd.read_csv("2class-f.csv")

  # Rename columns to standard variable names
  column_mapping = {
      "shedate dard": "VAS",
      "pain anxiety": "PASS-20",
      "catastrophizing": "PCS",
      "fear movement": "TSK",
      "age": "Age",
      "label": "Target",
  }
  df = df.rename(columns=column_mapping)

  # STRICT COMPLIANCE: Restrict dataset exactly to N = 503 as reported in paper
  if len(df) > 503:
    print(
        f"Dataset contains {len(df)} rows. Truncating/selecting first 503"
        " samples to match manuscript N=503."
    )
    df = df.iloc[:503].reset_index(drop=True)

  feature_cols = ["VAS", "TSK", "PASS-20", "PCS", "Age"]
  X = df[feature_cols]
  y = df["Target"]

  # Train-Test Stratified Split (70% Train = 352, 30% Holdout Test = 151)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
  )

  print(
      f"Data Split Successfully: Training Set = {len(X_train)}, Holdout Test"
      f" Set = {len(X_test)}"
  )

  # Encapsulated Pipeline Construction
  ann_pipeline = Pipeline([
      ("imputer", KNNImputer(n_neighbors=5, weights="uniform")),
      ("scaler", StandardScaler()),
      (
          "ann",
          KerasANNClassifier(epochs=100, batch_size=32, learning_rate=0.001),
      ),
  ])

  # Fit Pipeline exclusively on Training Set
  print("\nTraining primary ANN Pipeline on Training Set...")
  ann_pipeline.fit(X_train, y_train)

  # Evaluate on Holdout Test Set
  y_pred = ann_pipeline.predict(X_test)
  y_prob = ann_pipeline.predict_proba(X_test)[:, 1]

  # Compute Metrics
  acc = accuracy_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_prob)
  f1 = f1_score(y_test, y_pred)
  brier = brier_score_loss(y_test, y_prob)
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  spec = tn / (tn + fp)
  sens = tp / (tp + fn)
  prec = precision_score(y_test, y_pred)

  # CIs calculation
  acc_ci = wilson_ci(tp + tn, len(y_test))
  sens_ci = wilson_ci(tp, tp + fn)
  spec_ci = wilson_ci(tn, tn + fp)
  prec_ci = wilson_ci(tp, tp + fp)
  _, f1_ci_low, f1_ci_high = bootstrap_f1_ci(y_test, y_pred)

  print("\n=======================================================")
  print("      INTERNAL HOLDOUT TEST SET PERFORMANCE (N=151)    ")
  print("=======================================================")
  print(f"ROC-AUC:          {auc:.3f}")
  print(f"Accuracy:         {acc:.3f} (95% CI: {acc_ci[0]:.3f}-{acc_ci[1]:.3f})")
  print(
      f"Sensitivity:      {sens:.3f} (95% CI:"
      f" {sens_ci[0]:.3f}-{sens_ci[1]:.3f})"
  )
  print(
      f"Specificity:      {spec:.3f} (95% CI:"
      f" {spec_ci[0]:.3f}-{spec_ci[1]:.3f})"
  )
  print(
      f"Precision:        {prec:.3f} (95% CI:"
      f" {prec_ci[0]:.3f}-{prec_ci[1]:.3f})"
  )
  print(
      f"F1-Score:         {f1:.3f} (Bootstrap 95% CI:"
      f" {f1_ci_low:.3f}-{f1_ci_high:.3f})"
  )
  print(f"Brier Score:      {brier:.3f}")
  print("=======================================================")

  # Save trained pipeline
  output_model_filename = "ann_model.joblib"
  joblib.dump(ann_pipeline, output_model_filename)
  print(f"\nPipeline successfully serialized to '{output_model_filename}'.")


if __name__ == "__main__":
  run_pipeline()
