# CloudMind ETL: AI-Driven Serverless Adaptive ETL System

An intelligent, serverless ETL pipeline designed for E-Commerce analytics that automatically adapts to evolving data sources using machine learning.

---

## Project Overview
**CloudMind ETL** is an AI-driven adaptive pipeline that automates the extraction, transformation, and loading (ETL) process. Unlike static pipelines, CloudMind adjusts its logic based on evolving data formats and integrates predictive analytics—such as demand forecasting and customer segmentation—directly into the data flow.

### Key Objectives
* **Adaptive Logic:** Automatically adjust transformation rules as data sources evolve.
* **Intelligent Preprocessing:** Use AI for anomaly detection and missing-value imputation.
* **Actionable Insights:** Generate real-time predictions for demand, pricing, and segmentation.

---

## System Architecture
The system leverages a fully serverless Azure environment to maintain scalability while minimizing operational costs.


### 1. Extract (Ingestion Layer)
* **Azure Functions:** Serverless triggers for automated fetching from E-commerce APIs (Olist dataset).
* **Event-Driven:** No manual intervention; scales automatically based on data volume.

### 2. Transform (AI-Assisted Processing)
* **Data Cleaning:** Automated normalization and outlier detection.
* **Schema Adaptation:** ML models handle evolving data formats without breaking the pipeline.
* **Quality Control:** AI-based imputation ensures high-fidelity data before warehouse loading.

### 3. Load (Warehouse Layer)
* **Azure SQL Database:** Centralized storage optimized for analytical querying.
* **Optimization:** RL-based serverless tuning to reduce latency and cost.

---

## AI Insight Layer
We integrated predictive modules directly into the workflow. **Note:** We transitioned from LSTM to **XGBoost** for superior performance and efficiency in tabular data forecasting.

### 1. Demand Forecasting
* **Model:** XGBoost
* **Performance:**
    | Metric | Value |
    | :--- | :--- |
    | **Accuracy** | 84.23% |
    | **Precision** | 0.81 |
    | **Recall** | 0.95 |
    | **F1-Score** | 0.87 |
    | **RMSE** | 39.79 |
    | **MAE** | 7.33 |

### 2. Customer Segmentation
* **Model:** Random Forest
* **Data Split:** 700 (Train) / 114 (Test)
* **Metrics:** MAE: 2.17 | RMSE: 15.00
* **Utility:** Segments users by purchasing patterns and behavioral attributes.

### 3. Dynamic Pricing Optimization
**Model:** XGBoost Classifier

| Metric | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Class 0** | 0.91 | 0.71 | 0.79 | 46,043 |
| **Class 1** | 0.81 | 0.95 | 0.87 | 60,214 |
| | | | | |
| **Accuracy** | - | - | **0.84** | 106,257 |
| **Macro Avg** | 0.86 | 0.83 | 0.83 | 106,257 |
| **Weighted Avg** | 0.85 | 0.84 | 0.84 | 106,257 |

---

## Key Features
* **Cloud-Native:** Built entirely on Azure (Functions, Blob Storage, SQL).
* **Fault-Tolerant:** Automated schema evolution handling.
* **Cost-Efficient:** Uses serverless consumption models.
* **End-to-End:** Moves from raw API data to predictive business insights in one flow.

---

##  Business Impact
* **Agility:** Reduced time-to-insight by automating schema changes.
* **Accuracy:** Improved data quality through intelligent imputation.
* **Growth:** Data-driven pricing and demand forecasting directly influence ROI.
