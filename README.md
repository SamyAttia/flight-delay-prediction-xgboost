# ✈️ Flight Delay Prediction: A Machine Learning Pipeline with XGBoost & SHAP

Flight delays are a major concern for passengers and airlines alike, affecting satisfaction, cost, and operations. In this project, I built a machine learning pipeline to accurately predict whether a flight will be delayed, using structured airline data and advanced modeling techniques.

Leveraging feature engineering, imbalance correction, and Bayesian-optimized XGBoost, this project achieves ~98% test accuracy and includes full explainability via SHAP values.

<p align="center">
  <img width="100%" alt="SHAP Summary Plot" src="figures/XGB_Bayesian_SHAP.png">
</p>

## 🛫 Data Sources

This project integrates data from multiple structured sources:

* [`new.csv`](https://www.kaggle.com/datasets/shubhendra7/flight-delay-prediction) – Historical U.S. flight records with delay components  
* [`airport_codes.csv`](https://www.kaggle.com/datasets/flashgordon/airport-codes) – Basic airport metadata (IATA, name, location)  
* [`airports.csv`](https://www.kaggle.com/datasets/sahasrabudheair/airports-data) – Detailed global airport data (latitude/longitude, region, etc.)

**Engineered features include:**
- Haversine flight distance  
- Time-based variables (hour, weekday, month)  
- Delay categorization and interaction terms  


## 🔍 Project Objectives

* Predict if a flight will be delayed or on-time using scheduled and operational data
* Build a fully interpretable machine learning pipeline
* Mitigate class imbalance using SMOTE and `scale_pos_weight`
* Tune and optimize models via Bayesian Optimization
* Visualize insights using SHAP and calibration metrics

## 📈 Development Journey

### 🧹 Data Cleaning & Feature Engineering

* Parsed time fields, converted delay values to minutes
* Calculated flight distance from coordinates
* Extracted hour, day\_of\_week, month from timestamps
* Merged datasets using IATA codes

### 🧪 Modeling Workflow

| Model               | F1 Score (Delayed) | Accuracy |
| :------------------ | :----------------- | :------- |
| Logistic Regression | 0.60               | 71.6%    |
| Random Forest       | 0.56               | 75.3%    |
| Gradient Boosting     | 0.58               | 75.3%    |
| Optimized XGBoost     | 0.97               | 97.95%   |

<p align="center">
  Fig. 1 — ROC Curve (AUC = 1.00)
</p>
<p align="center">
  <img width="80%" alt="ROC" src="figures/XGB_Bayesian_AUC-ROC.png">
</p>

<p align="center">
  Fig. 2 — Calibration Curve
</p>

<p align="center">
  <img width="80%" alt="Calibration" src="figures/XGB_Bayesian_platt-vs-isotonic.png">
</p>

## ⚙️ Final Model & Explainability

* **Model**: XGBoost Classifier (Bayesian-optimized)
* **Precision**: 0.97 (Delayed)
* **Recall**: 0.96
* **F1 Score**: 0.97
* **SHAP Summary Plot**: (see top) Top predictors include:
    * `actual_elapsed_time_minutes`
    * `scheduled_elapsed_time_minutes`
    * `departure_delay_minutes`

## 🧠 Key Insights

* Most impactful features were temporal and duration-based, not airport location.
* Delays increase sharply during peak hours and weekends.
* Bayesian tuning and imbalance-aware training improved both recall and precision.
* XGBoost + SHAP provided accurate and interpretable results ready for deployment.

## 🔮 Next Steps

* Add weather & traffic control data to boost accuracy
* Deploy via API or dashboard (Flask + Streamlit)
* Retrain quarterly to account for seasonal trends

## 🛠️ Tech Stack

* Python, Pandas, Scikit-learn, XGBoost
* SHAP, Matplotlib, Seaborn
* GitHub + Jupyter Notebooks
