# Medical Device Failure Prediction - Main ML Pipeline

## Overview
This document describes the comprehensive machine learning pipeline implemented in `main.ipynb` for predicting medical device failure and maintenance classification. The pipeline includes data preprocessing, individual model evaluation, ensemble learning, and extensive visualization capabilities.

## Pipeline Architecture

### 1. Data Loading and Preprocessing

#### Data Loading Function
```python
def load_dataset(path='Medical_Device_Failure_dataset.csv')
```
- Loads the medical device failure dataset
- Validates presence of target variable `Maintenance_Class`
- Returns pandas DataFrame for further processing

#### Feature Engineering and Preprocessing
```python
def preprocess(df)
```
**Selected Features:**
- `Age`: Device age in years
- `Maintenance_Cost`: Cost of maintenance operations
- `Downtime`: Device downtime duration
- `Maintenance_Frequency`: Number of maintenance operations  
- `Failure_Event_Count`: Number of failure events

**Derived Features:**
- `Cost_per_Event`: Maintenance cost divided by failure events (handles division by zero)
- `Downtime_per_Frequency`: Downtime divided by maintenance frequency

**Data Standardization:**
- StandardScaler applied to all numerical features
- Ensures consistent scale across features for optimal model performance

### 2. Machine Learning Models

#### Individual Model Evaluation
The pipeline evaluates five different classification algorithms:

1. **Random Forest Classifier**
   - `n_estimators=200`
   - `max_depth=10`
   - `random_state=42`

2. **Gradient Boosting Classifier**
   - Hyperparameter tuning with GridSearchCV
   - Parameters: `n_estimators` [250, 300, 350], `learning_rate` [0.03, 0.05, 0.07], `max_depth` [5, 6, 7]
   - Cross-validation for optimal parameter selection

3. **Support Vector Machine (SVM)**
   - RBF kernel with `gamma='scale'`
   - `probability=True` for probability estimates
   - `random_state=42`

4. **Decision Tree Classifier**
   - Default parameters with `random_state=42`
   - Provides interpretable decision rules

5. **Naive Bayes (Gaussian)**
   - Assumes feature independence
   - Baseline probabilistic classifier

### 3. Ensemble Learning

#### Weighted Voting Classifier
```python
VotingClassifier(estimators=[
    ('rf', Random Forest), 
    ('gb', Gradient Boosting), 
    ('svc', SVM), 
    ('dt', Decision Tree), 
    ('nb', Naive Bayes)
], voting='soft', weights=[2, 3, 2, 1, 1])
```

**Voting Strategy:**
- **Soft Voting**: Uses probability estimates for final prediction
- **Weighted Contributions**: 
  - Gradient Boosting: Weight 3 (highest)
  - Random Forest & SVM: Weight 2
  - Decision Tree & Naive Bayes: Weight 1

**Model Persistence:**
- Trained ensemble saved as `ensemble_model.pkl`
- Selected features saved as `selected_features.pkl`

### 4. Evaluation Metrics

#### Performance Metrics
For each model, the pipeline calculates:
- **Accuracy**: Overall classification accuracy (%)
- **Precision**: Weighted average precision (%)
- **Recall**: Weighted average recall (%)
- **F1-Score**: Weighted average F1-score (%)

#### Detailed Analysis
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Per-class precision, recall, and F1-scores
- **ROC Curves**: Receiver Operating Characteristic curves for multi-class classification
- **Feature Importance**: For tree-based models (Random Forest, Gradient Boosting, Decision Tree)

### 5. Visualization Components

#### Model Performance Visualizations

1. **Confusion Matrices**
   - Individual confusion matrix for each model
   - Color-coded heatmap with actual vs predicted classifications
   - Values displayed in integer format

2. **ROC Curves**
   - Multi-class ROC curves for each model
   - Individual class ROC curves with AUC scores
   - Micro-averaged ROC curve
   - Performance comparison across classes

3. **Feature Importance Plots**
   - Horizontal bar charts for tree-based models
   - Sorted by importance score
   - Identifies most influential features for predictions

4. **Model Comparison Dashboard**
   - Comparative bar charts for all performance metrics
   - Side-by-side comparison of all models including ensemble
   - Percentage scores with grid lines for easy reading

### 6. Advanced Features

#### Cross-Validation
- GridSearchCV implementation for Gradient Boosting optimization
- Adaptive cross-validation folds: `min(5, len(X_train) // 10)`
- Scoring based on accuracy metric

#### Multi-Class Handling
- Label binarization for ROC curve computation
- Support for 4-class classification (Maintenance_Class: 0, 1, 2, 3)
- Weighted averaging for imbalanced scenarios (though dataset is balanced)

#### Error Handling
- Graceful handling of feature importance extraction
- Fallback mechanisms for ensemble feature importance calculation
- Exception handling for visualization edge cases

### 7. Code Structure and Modularity

#### Function Organization
1. `load_dataset()`: Data loading and validation
2. `preprocess()`: Feature engineering and standardization
3. `evaluate_individual_models()`: Individual model training and evaluation
4. `train_evaluate_ensemble()`: Ensemble model training and evaluation
5. `plot_feature_importance()`: Feature importance visualization
6. `plot_model_comparison()`: Comparative performance visualization

#### Execution Flow
```python
# 1. Load and preprocess data
df = load_dataset()
X, y, features = preprocess(df)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(...)

# 3. Evaluate individual models
individual_results = evaluate_individual_models(...)

# 4. Train ensemble model
final_results = train_evaluate_ensemble(...)

# 5. Generate comparison visualizations
plot_model_comparison(final_results)
```

### 8. Output and Results

#### Model Persistence
- **ensemble_model.pkl**: Trained ensemble classifier
- **selected_features.pkl**: Feature list for prediction pipeline

#### Performance Reports
- Console output with detailed metrics for each model
- Classification reports with per-class statistics
- Ensemble model performance summary

#### Visualizations
- Individual model confusion matrices
- ROC curves for each classifier
- Feature importance charts
- Comprehensive model comparison dashboard

### 9. Key Technical Decisions

#### Feature Selection Rationale
- Focus on operational metrics rather than categorical identifiers
- Engineering meaningful derived features (ratios and rates)
- Standardization for algorithm compatibility

#### Model Selection Strategy
- Diverse algorithm selection covering different learning paradigms
- Tree-based methods for interpretability
- Ensemble approach for improved robustness
- Hyperparameter optimization for best performing models

#### Evaluation Approach
- Multi-metric evaluation for comprehensive assessment
- Visual validation through confusion matrices and ROC curves
- Feature importance analysis for model interpretability

### 10. Usage Instructions

#### Prerequisites
```python
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, ConfusionMatrixDisplay, roc_curve, 
                            roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler, label_binarize
```

#### Running the Pipeline
1. Ensure `Medical_Device_Failure_dataset.csv` is in the working directory
2. Execute the main pipeline code sequentially
3. Models will be trained, evaluated, and saved automatically
4. Visualizations will be displayed during execution

#### Loading Saved Models
```python
# Load trained ensemble model
ensemble_model = joblib.load("ensemble_model.pkl")
selected_features = joblib.load("selected_features.pkl")

# Make predictions on new data
predictions = ensemble_model.predict(new_data[selected_features])
```

## Conclusion

This comprehensive ML pipeline provides a robust framework for medical device failure prediction with:
- **Multiple Algorithm Support**: Five different classifiers with ensemble learning
- **Comprehensive Evaluation**: Multiple metrics and visualization techniques
- **Production Ready**: Model persistence and clear usage instructions
- **Extensible Design**: Modular functions for easy modification and enhancement

The pipeline achieves high performance through careful feature engineering, hyperparameter optimization, and ensemble learning techniques, making it suitable for real-world medical device maintenance prediction applications.