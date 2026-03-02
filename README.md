# Intelligent EV Charging Demand Prediction
**Newton School of Technology | Capstone Project 15**

This project implements a high-accuracy machine learning pipeline to forecast Electric Vehicle (EV) charging demand. By analyzing nearly 90,000 hourly records from California-based stations, we provide a robust baseline for grid stability and autonomous infrastructure planning.

## Team Contributions
* **Krishna:** Data Pipeline Engineering & ML Model Training
* **Saumya Mishra:** Data Cleaning, Imputation & Quality Assurance
* **Rachit Gupta:** Time-Series Feature Engineering & Logic Extraction
* **Aditya Rana:** Performance Analytics & System Evaluation

## Technical Baseline
* **Model:** Random Forest Regressor (Ensemble Learning)
* **Accuracy:** 89.16% R2 Score
* **Deployment:** Streamlit (Hosted)
* **Milestone 1 Goal:** Establish a reliable demand forecasting engine.
* **Milestone 2 Goal:** Transition to a LangGraph-powered Agentic Assistant.

[Image of a data preprocessing flowchart showing data loading, handling mixed date formats, and feature engineering]

## Project Structure
* `data/`: Consolidated station datasets.
* `models/`: Serialized model binaries (.pkl).
* `notebooks/`: Jupyter development environments.
* `src/`: Streamlit dashboard source code.