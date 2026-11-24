# UPI Fraud Detection Using Machine Learning

**A Classification-Based Approach for Imbalanced Data**

**Project Organization:** Plag Pro, Noida

## 1. Project Overview

UPI (Unified Payments Interface) has become one of the most widely used digital payment systems in India. With the increasing volume of digital transactions, detecting and preventing fraudulent activities has become an important challenge in the financial sector.

This project focuses on developing a machine learning-based classification system to identify fraudulent UPI transactions. It explores multiple machine learning algorithms, evaluates their performance on an imbalanced dataset, and selects a suitable model for fraud detection.

The project implements three machine learning algorithms:

- Random Forest
- Support Vector Machine (SVM)
- XGBoost

After model development and performance comparison, XGBoost was selected as the final model and integrated into a Streamlit-based web application named **UPI SecureCheck**.

## 2. Project Objectives

The following objectives are provided by Plag Pro, Noida:

1. To understand the concept of fraud detection in UPI transactions and its importance in the financial sector.
2. To explore different machine learning algorithms and techniques suitable for handling imbalanced data in fraud detection.
3. To develop a classification model using machine learning algorithms to detect fraudulent UPI transactions.
4. To evaluate the performance of the developed model in terms of accuracy, precision, recall, and F1-score.

## 3. Project Tasks

The following tasks are defined as part of the project:

1. Review literature on fraud detection in UPI transactions and imbalanced data handling in machine learning.
2. Collect and preprocess a dataset of UPI transactions for model training and testing.
3. Implement and tune machine learning algorithms such as Random Forest, Support Vector Machine, and XGBoost for fraud detection.
4. Evaluate the performance of the developed model and compare it with existing fraud detection methods.
5. Write a research report documenting the methodology, results, and conclusions of the project.

## 4. Dataset Description

The project uses a synthetic and anonymized UPI transaction dataset for machine learning model development and evaluation.

### Dataset Overview

| Dataset property                  | Description |
| --------------------------------- | ----------- |
| Total transactions                | 100,000     |
| Original features                 | 43          |
| Retained features                 | 37          |
| Excluded features                 | 6           |
| Target variable                   | `is_fraud`  |
| Genuine transactions              | 99,000      |
| Fraudulent transactions           | 1,000       |
| Genuine transaction percentage    | 99%         |
| Fraudulent transaction percentage | 1%          |

### Feature Selection

The original dataset consisted of 43 features. During preprocessing and feature selection:

- 37 relevant features were retained for model development.
- 6 unnecessary features were excluded.
- The target variable `is_fraud` was used to identify whether a transaction was genuine or fraudulent.

Feature selection helped prepare the dataset for model training while removing unnecessary information.

### Dataset Splitting

The dataset was divided into three subsets:

| Dataset    |     Records | Percentage |
| ---------- | ----------: | ---------: |
| Training   |      64,000 |        64% |
| Validation |      16,000 |        16% |
| Testing    |      20,000 |        20% |
| **Total**  | **100,000** |   **100%** |

The training dataset was used to develop the machine learning models, the validation dataset supported model evaluation and threshold selection, and the testing dataset was used for final XGBoost performance evaluation.

### Class Imbalance

The dataset contains significantly more genuine transactions than fraudulent transactions.

- Genuine transactions: 99%
- Fraudulent transactions: 1%

This class imbalance makes fraud detection challenging because a model could achieve high accuracy while failing to identify a substantial number of fraudulent transactions.

Therefore, precision, recall, F1-score, ROC-AUC, and PR-AUC were also considered during model evaluation.

## 5. Technologies and Tools Used

| Category                | Technologies              |
| ----------------------- | ------------------------- |
| Programming language    | Python                    |
| Development environment | Jupyter Notebook, VS Code |
| Data manipulation       | Pandas, NumPy             |
| Data visualization      | Matplotlib, Seaborn       |
| Machine learning        | Scikit-learn              |
| Gradient boosting       | XGBoost                   |
| Model explainability    | SHAP                      |
| Model serialization     | Joblib                    |
| Web application         | Streamlit                 |
| Data storage            | CSV, Pickle               |

## 6. Machine Learning Algorithms

Three machine learning algorithms were implemented and evaluated.

### 6.1 Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to perform classification.

It was implemented as one of the baseline models for detecting fraudulent UPI transactions.

### 6.2 Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies decision boundaries between different classes.

It was implemented to evaluate its ability to distinguish fraudulent and genuine transactions in an imbalanced classification problem.

### 6.3 XGBoost

XGBoost is a gradient boosting algorithm that builds an ensemble of decision trees sequentially.

It was selected as the final model after comparing its performance with Random Forest and SVM.

## 7. Model Performance Comparison

The following results were obtained during model evaluation.

| Evaluation Metric | Random Forest |    SVM | XGBoost |
| ----------------- | ------------: | -----: | ------: |
| Precision         |        69.49% | 34.55% |  86.99% |
| Recall            |        61.50% | 57.00% |  53.50% |
| F1-Score          |        65.25% | 43.02% |  66.25% |
| ROC-AUC           |        93.10% | 91.78% |  96.14% |
| PR-AUC            |        66.20% | 38.96% |  68.84% |

### Final Model Selection

XGBoost was selected as the final model based on its overall evaluation performance.

It achieved:

- Higher precision than Random Forest and SVM.
- The highest F1-score among the three models.
- The highest ROC-AUC and PR-AUC.
- A strong balance between identifying fraudulent transactions and reducing false-positive predictions.

## 8. Final XGBoost Model Evaluation

The final XGBoost model was evaluated on the held-out test dataset using the selected classification threshold.

### Final Test Results

| Metric                   |    Result |
| ------------------------ | --------: |
| Accuracy                 |    99.46% |
| Precision                |    86.99% |
| Recall                   |    53.50% |
| F1-Score                 |    66.25% |
| ROC-AUC                  |    96.14% |
| PR-AUC                   |    68.84% |
| Classification Threshold | 0.8848697 |

### Confusion Matrix

| Actual / Predicted | Genuine (0) | Fraud (1) |
| ------------------ | ----------: | --------: |
| Genuine (0)        |      19,784 |        16 |
| Fraud (1)          |          93 |       107 |

The model correctly identified 107 fraudulent transactions and 19,784 genuine transactions in the test dataset.

The results demonstrate the importance of evaluating fraud detection models using multiple metrics rather than relying on accuracy alone.

## 9. Model Explainability Using SHAP

SHAP (SHapley Additive exPlanations) was used to understand the contribution of different input features to the XGBoost model's predictions.

The explainability workflow helps analyze:

- Overall feature importance.
- The influence of individual features on model predictions.
- How input features contribute to fraud classification.
- The interpretability of the final machine learning model.

This provides additional insight into the behavior of the selected model.

## 10. UPI SecureCheck – Streamlit Web Application

The final XGBoost model was integrated into a Streamlit-based web application called **UPI SecureCheck**.

The application provides an interface for evaluating UPI transaction information and generating fraud predictions.

### Application Highlights

- User-friendly Streamlit interface.
- Transaction input and prediction workflow.
- Integration with the trained XGBoost model.
- Use of the saved preprocessing pipeline.
- Fraud probability and classification threshold handling.
- Genuine or fraudulent transaction prediction.

**Note:** Random Forest and SVM are included in the project for model development and comparison. The deployed Streamlit application uses XGBoost as its prediction model.

## 11. Project Structure

```text
UPI-Fraud-Detection/
│
├── 01_Data_Exploration_and_RF_SVM_Development.ipynb.ipynb
├── 02_XGBoost_Model_Development.ipynb.ipynb
├── 03_Model_Comparison_and_Final_Selection.ipynb.ipynb
├── 04_XGBoost_SHAP_Explainability.ipynb.ipynb
│
├── data/
│   ├── upi_transactions_synthetic_enhanced_final.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── svm_complete_checkpoint.pkl
│   ├── preprocessor.pkl
│   ├── feature_columns.pkl
│   └── ...
│
├── xgboost_artifacts/
│   ├── xgboost_model.pkl
│   ├── xgboost_preprocessor.pkl
│   ├── xgboost_feature_columns.pkl
│   ├── xgboost_threshold.pkl
│   └── ...
│
├── upi_fraud_app/
│   ├── app.py
│   ├── check_artifacts.py
│   └── models/
│       ├── xgboost_model.pkl
│       ├── xgboost_preprocessor.pkl
│       ├── xgboost_feature_columns.pkl
│       └── xgboost_threshold.pkl
│
├── 01_validate_upi_datasets.py
├── 02_investigate_enhanced_data.py
├── enhance_upi_dataset_final.py
├── verify_artifacts.py
├── test.py
│
└── .gitignore
```

## 12. Installation and Execution

### Prerequisites

- Python
- Git
- Jupyter Notebook (for notebook execution)
- Required Python libraries

### Step 1: Clone the Repository

```bash
git clone https://github.com/shubhamdir/UPI-Fraud-Detection.git
```

### Step 2: Navigate to the Project Directory

```bash
cd UPI-Fraud-Detection
```

### Step 3: Install Dependencies

Install the required libraries in your Python environment.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit joblib
```

### Step 4: Run the Streamlit Application

```bash
cd upi_fraud_app
python -m streamlit run app.py
```

The application will open in your browser.

## 13. Key Learnings

This project provided practical experience in:

- Understanding UPI fraud detection challenges.
- Working with imbalanced classification datasets.
- Data preprocessing and feature selection.
- Developing and tuning multiple machine learning algorithms.
- Comparing classification models using evaluation metrics.
- Selecting classification thresholds.
- Evaluating models using confusion matrices and ROC-AUC and PR-AUC.
- Understanding model predictions using SHAP.
- Saving and loading trained machine learning artifacts.
- Integrating a machine learning model into a Streamlit application.
- Managing a machine learning project using Git and GitHub.

## 14. Future Improvements

Potential future improvements include:

- Evaluating the model on additional real-world, privacy-protected transaction datasets.
- Exploring advanced techniques for imbalanced classification.
- Improving fraud recall while maintaining acceptable precision.
- Further optimizing model hyperparameters and classification thresholds.
- Adding transaction monitoring and reporting features.
- Enhancing the web application with additional analytics and visualizations.

## 15. Conclusion

This project demonstrates the application of machine learning techniques to detect fraudulent UPI transactions using an imbalanced classification dataset.

Random Forest, Support Vector Machine, and XGBoost were implemented and evaluated. Based on the comparative evaluation, XGBoost was selected as the final model and integrated into the UPI SecureCheck Streamlit application.

The project combines data preprocessing, machine learning, model evaluation, explainability, and web application development into an end-to-end fraud detection workflow.

---

**Project Title:** Fraud Detection in UPI Transactions Using Machine Learning: A Classification-Based Approach for Imbalanced Data

**GitHub Repository:** [https://github.com/shubhamdir/UPI-Fraud-Detection](https://github.com/shubhamdir/UPI-Fraud-Detection)

**Disclaimer:** This project uses a synthetic and anonymized dataset for educational and research purposes. Its predictions are experimental and should not be treated as a production-grade financial fraud detection system.
