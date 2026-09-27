# Predictive Analytics for Medical Device Failures Using Multimodal Maintenance Data

A comprehensive machine learning solution for predicting medical device failures using multimodal maintenance data. This project combines advanced data analytics, ensemble machine learning models, and a user-friendly Django web application to help healthcare facilities proactively manage medical equipment maintenance and prevent unexpected device failures.

## Project Overview

This project provides a complete end-to-end solution for medical device failure prediction, consisting of:

1. **Data Analysis & Exploration** - Comprehensive dataset analysis and visualization
2. **Machine Learning Pipeline** - Advanced ensemble learning with multiple algorithms
3. **Web Application** - Django-based interface for real-time predictions
4. **Report Generation** - Automated PDF report generation for predictions

### Key Features

- **Predictive Analytics**: 4-class failure risk classification using ensemble machine learning
- **Data Visualization**: Comprehensive charts and analysis dashboards
- **Web Interface**: User-friendly Django application for predictions
- **Batch Processing**: Support for bulk predictions via CSV/Excel upload
- **PDF Reports**: Professional report generation for individual and bulk predictions
- **High Accuracy**: Optimized ensemble model with multiple algorithms
- **Responsive Design**: Mobile-friendly web interface

## Project Structure

```
Medical Device Failure Prediction/
├── Base ML Code/                           # Core machine learning components
│   ├── main.ipynb                         # Main ML pipeline and model training
│   ├── dataSetAnalysis.ipynb              # Dataset exploration and analysis
│   ├── Dataset_Analysis.md                # Comprehensive dataset documentation
│   ├── Main_Pipeline_Documentation.md     # ML pipeline technical documentation
│   ├── Medical_Device_Failure_dataset.csv # Training dataset (4,000 samples)
│   ├── ensemble_model.pkl                 # Trained ensemble model
│   └── selected_features.pkl              # Feature selection configuration
├── med_device_failure_prediction/         # Django web application
│   ├── manage.py                          # Django management script
│   ├── med_device_failure_prediction/     # Main Django project
│   │   ├── settings.py                    # Project configuration
│   │   ├── urls.py                        # URL routing
│   │   ├── wsgi.py                        # WSGI configuration
│   │   └── asgi.py                        # ASGI configuration
│   ├── predictor/                         # Main Django application
│   │   ├── views.py                       # Application logic and prediction handling
│   │   ├── forms.py                       # Django forms for user input
│   │   ├── urls.py                        # App-specific URL routing
│   │   ├── apps.py                        # App configuration
│   │   └── templates/                     # HTML templates
│   │       ├── index.html                 # Homepage
│   │       ├── form.html                  # Prediction input form
│   │       ├── result.html                # Single prediction results
│   │       ├── bulk_result.html           # Bulk prediction results
│   │       └── about.html                 # About page
│   ├── static/css/                        # Static assets
│   │   └── styles.css                     # Application styling
│   └── README.md                          # Django app specific documentation
└── README.md                              # This file - main project documentation
```

## Machine Learning Pipeline

### Dataset Characteristics
- **Size**: 4,000 medical device records
- **Features**: 13 original features + 2 engineered features
- **Target Classes**: 4 maintenance classes (perfectly balanced: 25% each)
- **Data Quality**: 0% missing values, exceptionally clean dataset

### Feature Engineering
The system uses 7 key features for predictions:
- **Age**: Device age in years
- **Maintenance_Cost**: Annual maintenance cost
- **Downtime**: Annual downtime hours
- **Maintenance_Frequency**: Maintenance sessions per year
- **Failure_Event_Count**: Number of recorded failures
- **Cost_per_Event**: Derived feature (Maintenance_Cost / Failure_Event_Count + 1)
- **Downtime_per_Frequency**: Derived feature (Downtime / Maintenance_Frequency + 1)

### Model Architecture
**Ensemble Learning Approach** with weighted voting:
- **Random Forest** (weight: 2) - Robust tree-based ensemble
- **Gradient Boosting** (weight: 3) - Optimized sequential learning
- **Support Vector Machine** (weight: 2) - Non-linear pattern recognition
- **Decision Tree** (weight: 1) - Interpretable rule-based learning
- **Naive Bayes** (weight: 1) - Probabilistic baseline

### Prediction Classes
- **Class 0** (Blue): No imminent failure expected
- **Class 1** (Green): Unlikely to fail within the first 3 years
- **Class 2** (Orange): Likely to fail within 3 years  
- **Class 3** (Red): Likely to fail after 3 years

## Quick Start Guide

### Prerequisites
- **Python 3.8+**
- **Required Libraries**: Django, pandas, numpy, scikit-learn, joblib, reportlab
- **Storage**: ~50MB for models and dataset

## Technical Performance

### Model Performance Metrics
- **Accuracy**: ~95%+ on test data
- **Precision**: Weighted average >94%
- **Recall**: Weighted average >94%
- **F1-Score**: Weighted average >94%

### Visualization Capabilities
- **Confusion Matrices**: Detailed classification performance
- **ROC Curves**: Multi-class receiver operating characteristics
- **Feature Importance**: Tree-based model interpretability
- **Model Comparison**: Side-by-side performance analysis

## Documentation

### Available Documentation
- **Dataset Analysis**: Complete statistical analysis in `Dataset_Analysis.md`
- **ML Pipeline**: Technical documentation in `Main_Pipeline_Documentation.md`
- **Django App**: Web application documentation in `med_device_failure_prediction/README.md`
- **Code Comments**: Comprehensive inline documentation

## License

© 2025 IEEE. This work has been accepted for presentation and publication in the **2025 Innovations in Power and Advanced Computing Technologies (i-PACT)** conference.

Personal use of this material is permitted. However, permission to reprint/republish this material for advertising or promotional purposes, or for creating new collective works for resale or redistribution, must be obtained from IEEE by writing to pubs-permissions@ieee.org.

By using this code, you acknowledge that it is provided solely for academic and research purposes and is subject to IEEE publication policies.

---

**Built for Healthcare Innovation**

*This project represents a commitment to improving healthcare equipment reliability through advanced predictive analytics and machine learning.*
