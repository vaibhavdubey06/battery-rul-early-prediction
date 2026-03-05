# Early Prediction of Lithium-Ion Battery Remaining Useful Life (RUL)

## Overview

Lithium-ion batteries power electric vehicles (EVs) and energy storage systems, but they degrade over time due to electrochemical aging. Predicting the **Remaining Useful Life (RUL)** of a battery is important for improving safety, reducing maintenance costs, and optimizing battery management systems.

This project aims to **predict battery lifespan using early lifecycle data (first 20–30% of charge–discharge cycles)** using machine learning and deep learning techniques.

---

## Problem Statement

Battery degradation is nonlinear and influenced by several factors such as temperature, charging current, and usage patterns. Traditional methods often require a large portion of the battery lifecycle before reliable predictions can be made.

This project explores whether it is possible to **predict the full lifespan of a lithium-ion battery using only early-cycle data**.

---

## Objectives

- Analyze lithium-ion battery degradation behavior
- Predict Remaining Useful Life (RUL) from early lifecycle data
- Compare traditional ML models with deep learning models
- Evaluate temporal models for battery degradation prediction

---

## Dataset

This project uses publicly available lithium-ion battery datasets.

**NASA Battery Dataset**

https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

The dataset contains battery charge-discharge cycle data including:

- Voltage
- Current
- Temperature
- Capacity degradation across cycles

---

## Planned Models

The following models will be explored:

- Linear Regression
- Random Forest
- LSTM (Long Short-Term Memory)
- Transformer-based time-series models

---

## Project Status

🚧 Work in progress

Current phase:
- Dataset exploration
- Degradation curve analysis
- Feature extraction

---

## Author

Vaibhav Dubey