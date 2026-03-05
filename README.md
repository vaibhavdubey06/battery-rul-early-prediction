# Early Prediction of Lithium-Ion Battery Remaining Useful Life (RUL)

## Overview

Lithium-ion batteries degrade over time due to chemical and thermal processes. Accurately predicting **Remaining Useful Life (RUL)** is critical for improving reliability, safety, and cost efficiency in electric vehicles and energy storage systems.

This project focuses on **predicting battery RUL using only early lifecycle data (first 20–30% of battery cycles)**. Early prediction enables proactive maintenance, early detection of defective batteries, and improved lifecycle management.

The project explores both **traditional machine learning models and temporal deep learning architectures** for modeling battery degradation patterns.

---

## Problem Statement

Battery degradation is nonlinear and highly dependent on operational conditions. Traditional approaches often require **full lifecycle data**, which delays failure detection.

This project aims to:

* Predict **Remaining Useful Life (RUL)** from early lifecycle data
* Model degradation patterns using machine learning and deep learning
* Compare performance of traditional ML models with temporal models
* Evaluate prediction accuracy using standard regression metrics

---

## Datasets

The project uses publicly available experimental battery degradation datasets.

**1. NASA Battery Degradation Dataset**
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

**2. CALCE Battery Dataset (University of Maryland)**
https://calce.umd.edu/battery-data

These datasets include measurements such as:

* Voltage
* Current
* Temperature
* Charge / Discharge cycles
* Capacity degradation over time

---

## Project Pipeline

Battery degradation prediction pipeline:

```
Battery Cycle Data
        ↓
Data Preprocessing
        ↓
Feature Extraction
        ↓
Model Training
(Random Forest / LSTM / Transformer)
        ↓
RUL Prediction
        ↓
Model Evaluation
(RMSE, MAE, R²)
```

---

## Methodology

### 1. Data Preprocessing

* Load battery cycle data
* Handle missing values
* Normalize sensor measurements
* Extract cycle-based features

### 2. Feature Engineering

Features extracted from battery cycles include:

* Voltage statistics
* Current profiles
* Temperature variations
* Capacity degradation trends
* Cycle count features

### 3. Model Development

The following models are explored:

**Baseline Models**

* Linear Regression

**Traditional ML Models**

* Random Forest Regressor

**Temporal Deep Learning Models**

* LSTM (Long Short-Term Memory)
* Transformer-based models

These models are trained using **only early lifecycle data (20–30% cycles)**.

---

## Evaluation Metrics

Model performance is evaluated using:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **R² Score**

These metrics measure the accuracy of predicted battery RUL compared to actual lifecycle data.

---

## Explainable AI (Planned Feature)

To understand which factors influence battery degradation, the project will implement **SHAP (SHapley Additive Explanations)** for model interpretability.

This helps identify:

* Key degradation drivers
* Feature importance
* Impact of temperature and voltage changes on battery lifespan

---



## Current Status

This project is **currently under development as a major academic project**.

Current progress includes:

* Literature review
* Dataset identification
* Initial data exploration
* Feature engineering experiments

Model training and evaluation are ongoing.

---

## Expected Contributions

This project aims to demonstrate:

* Feasibility of **early-stage battery RUL prediction**
* Comparison between ML and deep learning models
* Identification of important degradation features
* Practical insights for EV battery health monitoring

---

## Future Improvements

Planned improvements include:

* Advanced time-series modeling
* Transformer-based degradation prediction
* Explainable AI using SHAP
* Visualization of degradation patterns
* Model deployment for real-time prediction

---

## Author

**Vaibhav Dubey**
B.Tech Electrical Engineering
National Institute of Technology, Kurukshetra

GitHub: https://github.com/vaibhavdubey06
