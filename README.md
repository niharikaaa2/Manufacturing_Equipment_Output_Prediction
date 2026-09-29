# Manufacturing Equipment Output Prediction

A machine learning application for predicting hourly manufacturing output from equipment, process, material, and operational parameters.

The project combines a **Linear Regression model**, reproducible preprocessing with **Scikit-learn Pipelines**, and a **FastAPI inference service** packaged with **Docker**.

---

## Overview

Manufacturing output is influenced by multiple operational factors, including injection temperature and pressure, cycle time, cooling time, material properties, machine utilization, maintenance, and operator experience.

This project develops a regression-based prediction system for estimating:

> **`Parts_Per_Hour` — the expected number of parts produced per hour**

The workflow covers the complete machine-learning lifecycle:

**Data Preparation → Feature Engineering → Preprocessing → Model Training → Evaluation → Model Serialization → REST API → Docker Containerization**

---

## Problem Statement

The objective is to build a supervised learning model capable of predicting manufacturing equipment output from historical production data.

The prediction system is intended to provide an estimated production rate that can support:

* Production planning
* Equipment performance monitoring
* Identification of potential underperformance
* Operational analysis

---

## Dataset

The project uses a dataset containing **1,000 manufacturing records** with numerical and categorical process variables.

### Target

| Variable         | Description                 |
| ---------------- | --------------------------- |
| `Parts_Per_Hour` | Hourly manufacturing output |

### Features

**Process Parameters**

* Injection Temperature
* Injection Pressure
* Cycle Time
* Cooling Time
* Material Viscosity
* Ambient Temperature

**Equipment & Operations**

* Machine Age
* Operator Experience
* Maintenance Hours
* Machine Utilization
* Efficiency Score

**Categorical Parameters**

* Shift
* Machine Type
* Material Grade
* Day of Week

**Derived Process Features**

* Temperature-Pressure Ratio
* Total Cycle Time

The original `Timestamp` is transformed into `Hour`, `Day`, and `Month` features before being removed from the model input.

---

## Machine Learning Approach

### Problem Type

**Supervised Learning — Regression**

### Model

**Linear Regression**

Linear Regression was selected because the target variable is continuous and numerical. It also provides a straightforward baseline for understanding the relationship between manufacturing parameters and production output.

### Data Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

`random_state=42` is used to ensure reproducibility.

---

## Preprocessing Pipeline

A Scikit-learn `ColumnTransformer` separates numerical and categorical preprocessing.

### Numerical Features

1. Missing values are handled using **median imputation**
2. Features are standardized using **StandardScaler**

### Categorical Features

1. Missing values are handled using **most-frequent imputation**
2. Categories are converted using **OneHotEncoder**
3. `handle_unknown="ignore"` prevents inference failures when an unseen category is encountered

The preprocessing stage and regression model are combined into a single Scikit-learn `Pipeline`.

This ensures that the same transformations applied during training are automatically applied during inference.

---

## Model Evaluation

The model is evaluated on the held-out test set using:

* **MAE** — Mean Absolute Error
* **MSE** — Mean Squared Error
* **RMSE** — Root Mean Squared Error
* **R² Score** — coefficient of determination

The evaluation metrics are calculated from predictions generated exclusively on the test split.

---

## Model Artifact

The trained pipeline is serialized using **Joblib**:

```text
manufacturing_linear_regression.joblib
```

The saved artifact contains both the preprocessing workflow and trained regression model, allowing the same pipeline to be loaded directly for inference.

---

## FastAPI Inference Service

The trained model is exposed through a REST API using **FastAPI**.

### Endpoint

```text
POST /predict
```

The endpoint accepts manufacturing parameters as JSON and returns the predicted hourly output.

### Example Request

```json
{
  "Injection_Temperature": 220,
  "Injection_Pressure": 90,
  "Cycle_Time": 13,
  "Cooling_Time": 5,
  "Material_Viscosity": 120,
  "Ambient_Temperature": 25,
  "Machine_Age": 5,
  "Operator_Experience": 4,
  "Maintenance_Hours": 20,
  "Shift": "Day",
  "Machine_Type": "Type_A",
  "Material_Grade": "Standard",
  "Day_of_Week": "Monday",
  "Temperature_Pressure_Ratio": 2.59,
  "Total_Cycle_Time": 16,
  "Efficiency_Score": 0.80,
  "Machine_Utilization": 0.85,
  "Hour": 10,
  "Day": 2,
  "Month": 1
}
```

### Example Response

```json
{
  "Predicted_Parts_Per_Hour": 70.42
}
```

Interactive API documentation is available through FastAPI's Swagger interface at:

```text
http://localhost:8000/docs
```

---

## Dockerization

The inference service is containerized using Docker to provide a consistent runtime environment.

### Build

```bash
docker build --network=host -t manufacturing-api .
```

### Run

```bash
docker run -p 8000:8000 manufacturing-api
```

The API is then available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Project Structure

```text
Manufacturing_Equipment_Output_Prediction/
│
├── Manufacturing_Equipment_Output_Prediction.ipynb
├── manufacturing_dataset_1000_samples.csv
├── manufacturing_linear_regression.joblib
├── app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technology Stack

| Area                | Technology        |
| ------------------- | ----------------- |
| Language            | Python            |
| Data Processing     | Pandas, NumPy     |
| Machine Learning    | Scikit-learn      |
| Model               | Linear Regression |
| Model Serialization | Joblib            |
| API                 | FastAPI, Uvicorn  |
| Containerization    | Docker            |
| Development         | Google Colab      |
| Version Control     | Git, GitHub       |

---

## Reproducibility

Dependencies are specified in `requirements.txt`, including the Scikit-learn version used by the serialized model.

The project uses a fixed train-test split and a complete preprocessing/model pipeline to maintain consistent transformations between training and inference.

---

## Key Implementation Decisions

* Used a **Pipeline** to prevent inconsistent preprocessing between training and inference.
* Used **ColumnTransformer** to apply different transformations to numerical and categorical features.
* Used **OneHotEncoder** rather than assigning arbitrary numerical values to categorical variables.
* Used **median imputation** for numerical missing values to reduce sensitivity to extreme observations.
* Serialized the complete trained pipeline rather than only the regression estimator.
* Exposed the trained model through a REST API using FastAPI.
* Containerized the inference application using Docker.
* Avoided target-derived features that would introduce **data leakage** into the model.

---

## Limitations & Future Improvements

The current implementation establishes a reproducible regression baseline and deployment workflow.

Potential extensions include:

* Comparing Linear Regression with tree-based regression models
* Hyperparameter tuning and cross-validation
* Additional feature engineering
* Model explainability and feature importance analysis
* Monitoring prediction performance after deployment
* Automated CI/CD for the API

---

## Project Information

**TNS Capstone Project 1**

**Project:** Manufacturing Equipment Output Prediction with Linear Regression

**Category:** Supervised Learning — Regression
