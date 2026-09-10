import joblib
import numpy as np
import pandas as pd

# Load the trained, serialized pipeline (includes scaler + model)
pipeline = joblib.load("ann_model.joblib")

# Example input: [VAS, TSK, PASS-20, PCS, Age]
# New patient scores: VAS=7, TSK=42, PASS-20=65, PCS=30, Age=45
new_patient_data = pd.DataFrame(
    [[7.0, 42.0, 65.0, 30.0, 45.0]],
    columns=["VAS", "TSK", "PASS-20", "PCS", "Age"],
)

# Generate disability probability
disability_prob = pipeline.predict_proba(new_patient_data)[0, 1]
predicted_class = (disability_prob >= 0.5).astype(int)

print(f"Calculated Disability Probability: {disability_prob * 100:.1f}%")
print(
    f"Predicted Class: {'Moderate-to-High Disability (ODI >= 30%)' if predicted_class == 1 else 'Low Disability'}"
)