# Early Prediction of Lithium-Ion Battery Remaining Useful Life

## Problem Statement

Lithium-ion batteries degrade over time due to electrochemical aging.
Accurate prediction of Remaining Useful Life (RUL) is critical for EV safety,
warranty management and battery lifecycle optimization.

Most existing models require large portions of lifecycle data.
This project investigates whether RUL can be predicted using only the
first 20–30% of battery cycles.

## Objectives

- Predict battery RUL from early lifecycle data
- Compare traditional ML and deep learning models
- Evaluate LSTM and Transformer for time-series degradation prediction

## Dataset

NASA Battery Dataset  
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

CALCE Battery Dataset  
http://calce.umd.edu/battery-data

## Models

- Linear Regression
- Random Forest
- LSTM
- Transformer

## Project Structure
battery-rul-early-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── dataset_link.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_lstm_model.ipynb
│   └── 04_transformer_model.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── models.py
│   └── train.py
├── results/
│   ├── plots/
│   └── metrics/
└── papers/
    └── references.md

## Expected Outcomes

- Early-stage battery degradation prediction
- Model comparison for RUL estimation
- Research publication preparation