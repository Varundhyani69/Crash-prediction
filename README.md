# Predictive Analytics for Road Traffic Crash Severity & Driver Behavior

## Overview
Road traffic accidents are a major public safety concern worldwide. This project applies **predictive analytics and machine learning** on real-world government crash data to identify high-risk scenarios and predict accident outcomes.

The project focuses on transforming raw crash data into actionable insights that can support **road safety planning, emergency response prioritization, and policy-level decision-making**.

---

## Problem Statements
This project addresses three independent predictive problems:

1. **Injury Severity Prediction**  
   Predict whether a traffic crash results in injury or non-injury.

2. **Severe/Fatal Crash Prediction**  
   Identify crashes that are likely to result in severe or fatal injuries.

3. **Driver Distraction Prediction**  
   Predict whether driver distraction was involved at the time of the crash.

Each problem is solved using a machine learning model chosen based on interpretability, performance, and real-world applicability.

---

## Dataset
- **Source:** U.S. Government Open Data Portal (Data.gov)  
- **Dataset Name:** Crash Reporting – Drivers Data  
- **Official Link:** https://catalog.data.gov/dataset/crash-reporting-drivers-data  

### Dataset Characteristics
- Large-scale real-world crash data
- Combination of categorical and numerical features
- Presence of missing values and class imbalance
- Suitable for predictive modeling and exploratory analysis

### Key Features
- Collision type  
- Weather condition  
- Road surface condition  
- Light condition  
- Traffic control measures  
- Driver substance abuse  
- Driver distraction  
- Injury severity  

---

## Data Preprocessing
The following preprocessing steps were applied:

- Removed irrelevant identifier and location-based columns  
- Handled missing values:
  - Numerical features filled using median values
  - Categorical features filled with placeholder values  
- Encoded categorical variables into numerical format  
- Selected relevant features based on domain knowledge  
- Addressed class imbalance using **class-weighted learning**  
- Prevented data leakage through strict feature-target separation  

---

## Models & Methodology

### 1. Injury Severity Prediction
- **Model:** Logistic Regression (class-weighted)
- **Reason:** Interpretable baseline model suitable for binary classification
- **Focus Metric:** Recall (due to class imbalance)
- **Outcome:** Improved minority class detection with balanced evaluation

---

### 2. Severe/Fatal Crash Prediction
- **Model:** Decision Tree Classifier
- **Reason:** Captures non-linear relationships and categorical feature interactions
- **Focus Metric:** Recall for severe/fatal cases
- **Outcome:** Effective identification of high-risk crashes with interpretable rules

---

### 3. Driver Distraction Prediction
- **Model:** Random Forest Classifier
- **Reason:** Robust ensemble model for complex feature interactions
- **Focus Metric:** Recall and stability
- **Outcome:** Best-performing model with strong generalization

---

## Model Evaluation
Models were evaluated using multiple metrics to ensure realistic performance assessment:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Special emphasis was placed on **recall and F1-score**, as false negatives are costly in safety-critical applications.

---

## Key Insights
- Accuracy alone is misleading in highly imbalanced datasets  
- Class-weighted learning significantly improves minority class detection  
- Ensemble learning improves prediction stability and robustness  
- Feature importance analysis identified major risk contributors:
  - Driver fault  
  - Substance abuse  
  - Environmental conditions  

---

## Technologies Used
- **Programming Language:** Python  
- **Libraries:**  
  - Pandas  
  - NumPy  
  - Scikit-learn  
  - Matplotlib  
  - Seaborn  

---

## Project Structure
├── crash_prediction.ipynb
├── README.md
crash_reporting_drivers.csv  https://drive.google.com/file/d/1pXhW3jSay2xiRL3ldemvDTADev4PLlnA/view?usp=sharing


---

## Future Enhancements
- Apply advanced models such as XGBoost or LightGBM  
- Use SMOTE or hybrid sampling for better class imbalance handling  
- Implement SHAP or LIME for improved model explainability  
- Deploy the model using Streamlit or Flask for real-time predictions  
- Integrate temporal and geographic features for richer insights  

---

## Author
**Varun Dhyani**  
B.Tech Computer Science Engineering  
Lovely Professional University  

- GitHub: https://github.com/Varundhyani69  
- LinkedIn: https://www.linkedin.com/in/vdlink  
