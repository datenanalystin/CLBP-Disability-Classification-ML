# CLBP-Disability-Classification-ML
# Machine Learning Disability Classification in Chronic Low Back Pain (CLBP)

This repository contains the official implementation of the machine learning pipeline and primary Artificial Neural Network (ANN) model for classifying CLBP disability status based on TRIPOD-AI and STROBE guidelines.

## File Structure
- `pipeline.py`: Full preprocessing (KNN Imputation + Scaling), model training script, and performance evaluation.
- `predict_ann.py`: Lightweight inference script to evaluate new clinical cases.
- `ann_model.joblib`: Pre-trained pipeline object containing the optimal ANN architecture and preprocessing transformers.

## Quick Start (Patient-Level Inference)
To predict disability status for a new patient, make sure `ann_model.joblib` and `predict_ann.py` are in the same directory, then run:
```bash
python predict_ann.py

Reproducibility & Stochastic Note
Fixed Seeds: The pipeline utilizes a fixed random seed (random_state=42) for train-test splitting and bootstrap computations to ensure methodological consistency.

Deep Learning Variance: Due to random weight initialization and stochastic gradient descent optimization in TensorFlow/Keras, executing pipeline.py locally may yield minor, natural variations in exact confusion matrix counts and performance metrics, while maintaining the overall robust performance bounds reported in the manuscript.
