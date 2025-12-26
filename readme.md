# 🩺 Diabetes Prediction Using Machine Learning

This project focuses on predicting whether an individual is diagnosed with diabetes using clinical, lifestyle, and demographic health data. The goal is to build an accurate, interpretable, and scalable machine learning model using a large real-world–like dataset.

---

## 📌 Project Overview

- **Problem Type**: Binary Classification  
- **Target Variable**: `diagnosed_diabetes`
- **Dataset Size**: 700,000 records  
- **Best Model**: XGBoost  
- **Evaluation Metric**: F1 Score  

The project includes:
- Exploratory Data Analysis (EDA)
- Feature selection based on correlation and domain relevance
- Model training and hyperparameter tuning
- Performance comparison between models

---

## 📂 Dataset Description

### 🔹 Original Dataset

- **Rows**: 700,000  
- **Columns**: 26  
- **Memory Usage**: ~139 MB  

**Feature Categories:**
- Demographics (age, gender, ethnicity, education, income)
- Lifestyle (physical activity, alcohol consumption, sleep, screen time)
- Clinical metrics (BMI, blood pressure, cholesterol levels)
- Medical history (family history, hypertension, cardiovascular history)


---

### 🔹 Final Dataset (After Feature Selection)

Based on correlation analysis, domain knowledge, and redundancy removal, the dataset was reduced to the most impactful features.

- **Rows**: 700,000  
- **Columns**: 7  
- **Memory Usage**: ~37 MB  


#### Selected Features:
| Feature | Description |
|------|------------|
| age | Age of the individual |
| physical_activity_minutes_per_week | Weekly physical activity |
| bmi | Body Mass Index |
| systolic_bp | Systolic blood pressure |
| ldl_cholesterol | LDL cholesterol level |
| family_history_diabetes | Family history of diabetes (0/1) |
| diagnosed_diabetes | Target variable |

---

## 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand:
- Feature distributions
- Outliers and skewness
- Relationships between variables
- Correlation with diabetes diagnosis

### Key Observations:
- **BMI**, **LDL cholesterol**, and **systolic blood pressure** show strong relationships with diabetes.
- **Family history of diabetes** is a significant binary risk factor.
- Most numerical features follow approximately normal distributions.
- Strong correlations observed:
  - BMI ↔ Waist-to-hip ratio
  - LDL ↔ Total cholesterol
  - Age ↔ Blood pressure

---

## 🧠 Model Training & Evaluation

### Models Trained
- Logistic Regression
- XGBoost Classifier

### Evaluation Metric
- **F1 Score** (chosen due to class imbalance and medical relevance)

---

## 🏆 Model Performance

| Model | Best F1 Score |
|------|--------------|
| **XGBoost** | **0.7738** |
| Logistic Regression | 0.7580 |

### Best Hyperparameters

**XGBoost**
```python
{
  'learning_rate': 0.01,
  'max_depth': ...,
  'n_estimators': ...,
  'subsample': ...
}
Logistic Regression

{
  'C': 0.01,
  'penalty': 'l2'
}

```
✅ Best Model Selected: XGBoost

🚀 Why XGBoost?

- Handles non-linear relationships effectively

- Robust to feature interactions

- Performs well on large datasets

- Better balance between precision and recall

🛠️ Tech Stack

- Programming Language: Python

- Libraries:

  - NumPy

  - Pandas

  - Matplotlib

  - Seaborn

  - Scikit-learn

  - XGBoost

📈 Future Improvements

- Add SHAP values for explainability

- Handle class imbalance with advanced sampling

- Try deep learning models

- Deploy model using Flask or FastAPI

- Add real-time prediction API