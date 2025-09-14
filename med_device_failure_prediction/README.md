````markdown
# Medical Device Failure Prediction

A Django-based web application that predicts medical device failure risks using machine learning algorithms. This system helps healthcare facilities proactively manage medical equipment maintenance and prevent unexpected failures.

## 🎯 Features

- **Individual Device Prediction**: Manually enter device parameters for single predictions  
- **Bulk Prediction**: Upload CSV/Excel files for batch processing multiple devices  
- **PDF Report Generation**: Export prediction results as professional PDF reports  
- **Risk Classification**: 4-level failure risk assessment system  
- **Responsive Design**: Modern web interface optimized for all devices  
- **Real-time Processing**: Instant predictions using pre-trained ensemble models  

## 🏥 Prediction Categories

The system classifies devices into four risk categories:

- **Class 0** (🔵 Blue): No imminent failure expected  
- **Class 1** (🟢 Green): Unlikely to fail within the first 3 years  
- **Class 2** (🟠 Orange): Likely to fail **within the next 3 years**  
- **Class 3** (🔴 Red): Likely to fail **after 3 years**  

## 📋 Required Input Parameters

For accurate predictions, the following device parameters are required:

| Parameter | Description | Unit |
|-----------|-------------|------|
| `Device_Serial_No` | Unique device identifier | Text |
| `Age` | Device age | Years |
| `Maintenance_Cost` | Annual maintenance cost | INR |
| `Downtime` | Annual downtime | Hours |
| `Maintenance_Frequency` | Maintenance sessions per year | Count |
| `Failure_Event_Count` | Number of recorded failures | Count |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+  
- Django 4.2.4  
- Required Python packages (see requirements below)  

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd med_device_failure_prediction
````

2. **Install dependencies**

   ```bash
   pip install django==4.2.4
   pip install pandas numpy joblib scikit-learn
   pip install reportlab
   ```

3. **Start the development server**

   ```bash
   python manage.py runserver
   ```

4. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:8000`

## 📁 Project Structure

```
med_device_failure_prediction/
├── med_device_failure_prediction/     # Main project settings
│   ├── settings.py                    # Django configuration
│   ├── urls.py                        # Main URL routing
│   ├── wsgi.py                        # WSGI application
│   └── asgi.py                        # ASGI application
├── predictor/                         # Main application
│   ├── templates/                     # HTML templates
│   │   ├── index.html                 # Home page
│   │   ├── form.html                  # Prediction form
│   │   ├── result.html                # Single prediction result
│   │   ├── bulk_result.html           # Bulk prediction results
│   │   └── about.html                 # About page
│   ├── views.py                       # Application logic
│   ├── forms.py                       # Django forms
│   ├── urls.py                        # App URL routing
│   └── apps.py                        # App configuration
├── static/                            # Static assets
│   └── css/
│       └── styles.css                 # Application styling
├── ensemble_model.pkl                 # Pre-trained ML model
├── selected_features.pkl              # Feature selection data
└── manage.py                          # Django management script
```

## 🔧 Usage

### Manual Prediction

1. Navigate to the **Prediction** page
2. Fill in the device parameters in the manual entry form
3. Click **Predict** to get instant results
4. Download PDF report if needed

### Bulk Prediction

1. Prepare a CSV or Excel file with the required columns
2. Navigate to the **Prediction** page
3. Upload your file using the bulk upload section
4. View results in a tabular format
5. Export all results as a PDF report

### Sample Data Format

```csv
Device_Serial_No,Age,Maintenance_Cost,Downtime,Maintenance_Frequency,Failure_Event_Count
DEV001,5.2,25000,48,4,2
DEV002,3.1,15000,24,3,1
DEV003,7.8,45000,72,6,4
```

## 🤖 Machine Learning Model

The application uses an ensemble machine learning model that:

* Processes 6 input features plus 2 derived features
* Provides 4-class failure risk classification
* Uses pre-trained models loaded from `ensemble_model.pkl`
* Includes feature engineering for cost-per-event and downtime-per-frequency ratios

### Feature Engineering

The system automatically calculates additional features:

* `Cost_per_Event` = `Maintenance_Cost` / (`Failure_Event_Count` + 1)
* `Downtime_per_Frequency` = `Downtime` / (`Maintenance_Frequency` + 1)

## 📊 API Endpoints

| Endpoint                | Method    | Description                    |
| ----------------------- | --------- | ------------------------------ |
| `/`                     | GET       | Home page                      |
| `/form/`                | GET, POST | Prediction interface           |
| `/results/`             | GET       | Bulk prediction results        |
| `/generate_pdf/`        | POST      | Generate bulk PDF report       |
| `/generate_single_pdf/` | POST      | Generate single prediction PDF |
| `/about/`               | GET       | About page                     |

## 🛠️ Configuration

### Django Settings

Key configuration in `settings.py`:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'predictor',  # Main application
]

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

### Static Files

Static files are served from the `static/` directory and include:

* CSS styling (`styles.css`)
* Images and logos
* Sample data files

## 🔐 Security Considerations

⚠️ **Important**: Before deploying to production:

1. Change the `SECRET_KEY` in `settings.py`
2. Set `DEBUG = False`
3. Configure `ALLOWED_HOSTS`
4. Use a production database (PostgreSQL, MySQL)
5. Implement proper authentication and authorization
6. Set up HTTPS/SSL
7. Configure static file serving with a web server

## 📱 Browser Compatibility

* Chrome 70+
* Firefox 65+
* Safari 12+
* Edge 79+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

© 2025 IEEE. This work has been accepted for presentation and publication in the **2025 Innovations in Power and Advanced Computing Technologies (i-PACT)** conference.

Personal use of this material is permitted. However, permission to reprint/republish this material for advertising or promotional purposes, or for creating new collective works for resale or redistribution, must be obtained from IEEE by writing to [pubs-permissions@ieee.org](mailto:pubs-permissions@ieee.org).

By using this code, you acknowledge that it is provided solely for academic and research purposes and is subject to IEEE publication policies.

## ⚠️ Disclaimer

This application is for educational and research purposes. Medical device failure predictions should not be the sole basis for critical healthcare decisions. Always consult with qualified medical equipment technicians and follow manufacturer guidelines.

## 📞 Support

For questions or support, please:

* Check the documentation
* Review existing issues
* Contact the development team

## 🔄 Version History

* **v1.0.0** - Initial release with basic prediction functionality
* Features: Manual and bulk prediction, PDF generation, responsive design

---

**Built with ❤️ using Django and Machine Learning**

```

---

