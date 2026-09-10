# Machine Learning Disability Classification in Chronic Low Back Pain (CLBP)

This repository contains the official implementation of the machine learning pipeline and primary Artificial Neural Network (ANN) model for classifying CLBP disability status based on TRIPOD-AI guidelines.

## File Structure
- `pipeline.py`: Full nested cross-validation, preprocessing (KNN Imputation + Scaling), and model training script.
- `predict_ann.py`: Lightweight inference script to evaluate new clinical cases.
- `ann_model.joblib`: Pre-trained pipeline object containing the optimal ANN architecture and preprocessing transformers.

## Quick Start (Patient-Level Inference)
To predict disability status for a new patient:
```bash
python predict_ann.py
