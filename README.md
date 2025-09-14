# Predictive Analytics for Medical Device Failures Using Multimodal Maintenance Data

A comprehensive machine learning solution for predicting medical device failures using multimodal maintenance data. This project combines advanced data analytics, ensemble machine learning models, and a user-friendly Django web application to help healthcare facilities proactively manage medical equipment maintenance and prevent unexpected device failures.

## 🎯 Project Overview

This project provides a complete end-to-end solution for medical device failure prediction, consisting of:

1. **Data Analysis & Exploration** - Comprehensive dataset analysis and visualization
2. **Machine Learning Pipeline** - Advanced ensemble learning with multiple algorithms
3. **Web Application** - Django-based interface for real-time predictions
4. **Report Generation** - Automated PDF report generation for predictions

### Key Features

- 🔮 **Predictive Analytics**: 4-class failure risk classification using ensemble machine learning
- 📊 **Data Visualization**: Comprehensive charts and analysis dashboards
- 🌐 **Web Interface**: User-friendly Django application for predictions
- 📁 **Batch Processing**: Support for bulk predictions via CSV/Excel upload
- 📄 **PDF Reports**: Professional report generation for individual and bulk predictions
- 🎯 **High Accuracy**: Optimized ensemble model with multiple algorithms
- 📱 **Responsive Design**: Mobile-friendly web interface

## 🏗️ Project Structure

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

## 🔬 Machine Learning Pipeline

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
- **Class 0** (🔵 Blue): No imminent failure expected
- **Class 1** (🟢 Green): Unlikely to fail within the first 3 years
- **Class 2** (🟠 Orange): Likely to fail within 3 years  
- **Class 3** (🔴 Red): Likely to fail after 3 years

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.8+**
- **Required Libraries**: Django, pandas, numpy, scikit-learn, joblib, reportlab
- **Storage**: ~50MB for models and dataset

### Installation

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd "Medical Device Failure Prediction"
   ```

2. **Install Python Dependencies**
   ```bash
   pip install django==4.2.4
   pip install pandas numpy scikit-learn joblib
   pip install reportlab matplotlib seaborn
   pip install openpyxl  # For Excel file support
   ```

3. **Copy Model Files**
   ```bash
   # Copy trained models to Django app directory
   cp "Base ML Code/ensemble_model.pkl" "med_device_failure_prediction/"
   cp "Base ML Code/selected_features.pkl" "med_device_failure_prediction/"
   ```

4. **Start the Web Application**
   ```bash
   cd med_device_failure_prediction
   python manage.py runserver
   ```

5. **Access the Application**
   Open your browser and navigate to `http://127.0.0.1:8000`

### Alternative: Jupyter Notebook Usage

To explore the machine learning pipeline directly:

1. **Navigate to Base ML Code**
   ```bash
   cd "Base ML Code"
   jupyter notebook
   ```

2. **Open Notebooks**
   - `main.ipynb` - Complete ML pipeline and model training
   - `dataSetAnalysis.ipynb` - Dataset exploration and visualization

## 💻 Usage Guide

### Web Application Features

#### 1. Manual Prediction
- Navigate to the **Prediction** page
- Fill in device parameters:
  - Device Serial Number (optional identifier)
  - Age (years)
  - Maintenance Cost (currency units)
  - Downtime (hours)
  - Maintenance Frequency (count)
  - Failure Event Count (count)
- Click **Predict** for instant results
- Download PDF report of prediction

#### 2. Bulk Prediction
- Prepare CSV or Excel file with required columns:
  ```csv
  Device_Serial_No,Age,Maintenance_Cost,Downtime,Maintenance_Frequency,Failure_Event_Count
  DEV001,5.2,25000,48,4,2
  DEV002,3.1,15000,24,3,1
  DEV003,7.8,45000,72,6,4
  ```
- Upload file via bulk upload section
- View results in tabular format
- Export all results as comprehensive PDF report

#### 3. Report Generation
- **Individual Reports**: Single device prediction with detailed analysis
- **Bulk Reports**: Comprehensive reports for multiple devices
- **Professional Format**: Color-coded risk classifications with descriptions

### Machine Learning Notebooks

#### Dataset Analysis (`dataSetAnalysis.ipynb`)
- Comprehensive statistical analysis of the dataset
- Distribution visualizations and correlation analysis
- Data quality assessment and feature insights
- Visualization of class balance and feature relationships

#### Main Pipeline (`main.ipynb`)
- Complete machine learning workflow
- Individual model evaluation and comparison
- Ensemble model training and optimization
- Performance metrics and visualization
- Model persistence for production use

## 📊 Technical Performance

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

### Scalability Features
- **Batch Processing**: Handle hundreds of devices simultaneously
- **Memory Efficient**: Optimized for large datasets
- **Fast Inference**: <1s prediction time for bulk uploads
- **Model Persistence**: Pre-trained models for instant deployment

## 🔧 Configuration and Customization

### Django Settings
Key configurations in `settings.py`:
- **Database**: SQLite (default) - easily configurable for PostgreSQL/MySQL
- **Static Files**: Served from `static/` directory
- **File Uploads**: Temporary storage for CSV/Excel processing
- **Security**: CSRF protection and secure file handling

### Model Configuration
The ensemble model can be retrained with different parameters:
- **Hyperparameter Tuning**: GridSearchCV for optimization
- **Feature Selection**: Customizable feature sets
- **Algorithm Weights**: Adjustable voting weights for ensemble
- **Cross-Validation**: Stratified K-fold for robust evaluation

### Extending the System
- **New Features**: Add additional device parameters
- **Model Updates**: Retrain with new data
- **UI Customization**: Modify templates and styling
- **API Integration**: RESTful API endpoints (extendable)

## 🛡️ Security and Deployment

### Development vs Production

**For Production Deployment:**
1. **Security Settings**:
   ```python
   DEBUG = False
   SECRET_KEY = 'your-production-secret-key'
   ALLOWED_HOSTS = ['your-domain.com']
   ```

2. **Database Configuration**:
   - Use PostgreSQL or MySQL instead of SQLite
   - Configure connection pooling
   - Set up database backups

3. **Web Server Setup**:
   - Use Gunicorn/uWSGI for WSGI
   - Configure Nginx for static files
   - Set up SSL/HTTPS certificates

4. **File Storage**:
   - Configure cloud storage for uploads
   - Set up proper file permissions
   - Implement file size limits

### Security Considerations
- **Input Validation**: Comprehensive form validation and sanitization
- **File Upload Security**: Extension and content validation
- **CSRF Protection**: Built-in Django CSRF middleware
- **SQL Injection**: Django ORM protection
- **XSS Protection**: Template auto-escaping enabled

## 📈 Performance Optimization

### Machine Learning Optimizations
- **Feature Standardization**: StandardScaler for consistent scaling
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Ensemble Weights**: Optimized voting weights based on individual performance
- **Model Persistence**: Pickle serialization for fast loading

### Web Application Optimizations
- **Session Management**: Efficient result storage and cleanup
- **File Processing**: Streaming for large files
- **Static Files**: Optimized CSS and JavaScript delivery
- **Database Queries**: Minimal database usage (stateless predictions)

## 🧪 Testing and Validation

### Model Validation
- **Cross-Validation**: Stratified K-fold validation
- **Test Set Performance**: Hold-out test set evaluation
- **Confusion Matrix Analysis**: Detailed per-class performance
- **ROC Curve Analysis**: Multi-class classification metrics

### Application Testing
- **Form Validation**: Comprehensive input validation testing
- **File Upload Testing**: Various file formats and edge cases
- **PDF Generation**: Report generation under different scenarios
- **Browser Compatibility**: Cross-browser testing

## 📚 Documentation

### Available Documentation
- **Dataset Analysis**: Complete statistical analysis in `Dataset_Analysis.md`
- **ML Pipeline**: Technical documentation in `Main_Pipeline_Documentation.md`
- **Django App**: Web application documentation in `med_device_failure_prediction/README.md`
- **Code Comments**: Comprehensive inline documentation

### Jupyter Notebooks
- **Interactive Analysis**: `dataSetAnalysis.ipynb` for exploratory data analysis
- **Pipeline Demo**: `main.ipynb` for complete ML workflow demonstration
- **Visualization**: Rich plots and charts for insights

## 🤝 Contributing

We welcome contributions to improve this medical device failure prediction system:

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/AmazingFeature`
3. **Commit Changes**: `git commit -m 'Add AmazingFeature'`
4. **Push to Branch**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

### Contribution Areas
- **Model Improvements**: New algorithms, feature engineering
- **Web Interface**: UI/UX enhancements, new features
- **Performance**: Optimization and scalability improvements
- **Documentation**: Additional guides and tutorials
- **Testing**: Unit tests, integration tests

## 📝 License

© 2025 IEEE. This work has been accepted for presentation and publication in the **2025 Innovations in Power and Advanced Computing Technologies (i-PACT)** conference.

Personal use of this material is permitted. However, permission to reprint/republish this material for advertising or promotional purposes, or for creating new collective works for resale or redistribution, must be obtained from IEEE by writing to pubs-permissions@ieee.org.

By using this code, you acknowledge that it is provided solely for academic and research purposes and is subject to IEEE publication policies.

## ⚠️ Important Disclaimers

### Medical Device Safety
This application is for educational and research purposes. Medical device failure predictions should not be the sole basis for critical healthcare decisions. Always consult with qualified medical equipment technicians and follow manufacturer guidelines. This application is designed for **educational and research purposes**. Important considerations:

- **Not for Critical Decisions**: Predictions should not be the sole basis for critical healthcare equipment decisions
- **Professional Consultation**: Always consult qualified medical equipment technicians
- **Manufacturer Guidelines**: Follow original equipment manufacturer (OEM) recommendations
- **Regulatory Compliance**: Ensure compliance with local healthcare regulations
- **Risk Assessment**: Use predictions as part of comprehensive risk assessment

### Data Privacy
- **Sensitive Information**: Avoid uploading personally identifiable information
- **Data Security**: Implement appropriate security measures for sensitive device data
- **Compliance**: Ensure compliance with HIPAA, GDPR, or relevant data protection regulations

## 📞 Support and Contact

### Getting Help
- **Documentation**: Check comprehensive documentation in this repository
- **Issues**: Report bugs and feature requests via GitHub issues
- **Discussion**: Join community discussions for questions and ideas

### Development Team
For technical questions, collaboration opportunities, or support:
- **Project Maintainer**: [Contact Information]
- **Technical Support**: [Support Email]
- **Documentation**: [Documentation Website]

## 🔮 Future Roadmap

### Planned Enhancements
- **Real-time Monitoring**: Integration with IoT device sensors
- **Advanced Analytics**: Time-series analysis and trend prediction
- **Mobile Application**: Native mobile app for field technicians
- **API Development**: RESTful API for third-party integrations
- **Cloud Deployment**: Scalable cloud infrastructure
- **Multi-language Support**: Internationalization and localization

### Research Directions
- **Deep Learning**: Explore neural network architectures
- **Multimodal Data**: Incorporate sensor data, images, and text
- **Federated Learning**: Privacy-preserving distributed learning
- **Explainable AI**: Enhanced model interpretability

## 🎯 Acknowledgments

### Technology Stack
- **Django**: Web framework for rapid development
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **ReportLab**: PDF generation
- **Bootstrap**: Responsive web design

### Dataset
- Medical Device Failure Dataset with 4,000 balanced samples
- Comprehensive maintenance and operational parameters
- Suitable for multi-class classification research

---

**Built with ❤️ for Healthcare Innovation**

*This project represents a commitment to improving healthcare equipment reliability through advanced predictive analytics and machine learning.*
