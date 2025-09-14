import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from .forms import PredictionForm
from django.core.files.storage import default_storage
from django.http import FileResponse
from django.views.decorators.csrf import csrf_exempt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import os
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load model and selected features
model = joblib.load('ensemble_model.pkl')
selected_features = joblib.load('selected_features.pkl')

# Class descriptions
PREDICTION_LABELS = {
    0: "No imminent failure expected",
    1: "Unlikely to fail within the first 3 years",
    2: "Likely to fail within 3 years",
    3: "Likely to fail after 3 years"
}


def index(request):
    return render(request, 'index.html')


def form_view(request):
    error = None
    # Bulk upload handling
    if request.method == 'POST' and 'upload_file' in request.POST:
        uploaded_file = request.FILES.get('bulk_file')
        if not uploaded_file:
            error = "No file selected. Please choose a CSV or Excel file."
            return render(request, 'form.html', {'form': PredictionForm(), 'error': error})

        extension = uploaded_file.name.rsplit('.', 1)[-1].lower()
        if extension not in ['csv', 'xlsx']:
            error = 'Invalid file type. Upload CSV or Excel.'
            return render(request, 'form.html', {'form': PredictionForm(), 'error': error})

        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        file_full_path = default_storage.path(file_path)
        try:
            if extension == 'csv':
                df = pd.read_csv(file_full_path)
            else:
                df = pd.read_excel(file_full_path)
        finally:
            os.remove(file_full_path)

        # Validate and predict
        try:
            required = [
                'Device_Serial_No', 'Age', 'Maintenance_Cost',
                'Downtime', 'Maintenance_Frequency', 'Failure_Event_Count'
            ]
            if not all(col in df.columns for col in required):
                raise Exception("Missing required columns.")

            df['Cost_per_Event'] = df['Maintenance_Cost'] / (df['Failure_Event_Count'] + 1)
            df['Downtime_per_Frequency'] = df['Downtime'] / (df['Maintenance_Frequency'] + 1)

            input_data = df[selected_features]
            df['Prediction'] = model.predict(input_data).astype(int)
            df['Prediction_Label'] = df['Prediction'].map(PREDICTION_LABELS)

            request.session['bulk_results'] = df.to_dict(orient='records')
            return redirect('bulk_result')

        except Exception as e:
            error = str(e)
            return render(request, 'form.html', {'form': PredictionForm(), 'error': error})

    # Manual prediction handling
    if request.method == 'POST' and 'predict_manual' in request.POST:
        form = PredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            serial_no = request.POST.get('Device_Serial_No', 'N/A')

            data['Cost_per_Event'] = data['Maintenance_Cost'] / (data['Failure_Event_Count'] + 1)
            data['Downtime_per_Frequency'] = data['Downtime'] / (data['Maintenance_Frequency'] + 1)

            input_array = np.array([data[f] for f in selected_features]).reshape(1, -1)
            prediction = int(model.predict(input_array)[0])
            prediction_label = PREDICTION_LABELS.get(prediction, "Unknown Prediction")

            request.session['single_result'] = {
                'Device_Serial_No': serial_no,
                'Prediction': prediction,
                'Prediction_Label': prediction_label
            }
            return render(request, 'result.html', {
                'prediction': prediction,
                'prediction_label': prediction_label,
                'device_serial': serial_no
            })
    else:
        form = PredictionForm()

    return render(request, 'form.html', {'form': form})


def bulk_result(request):
    results = request.session.get('bulk_results', [])
    for row in results:
        p = int(row['Prediction'])
        if p == 0:
            row['color'] = 'blue'
        elif p == 1:
            row['color'] = 'green'
        elif p == 2:
            row['color'] = 'orange'
        elif p == 3:
            row['color'] = 'red'
    return render(request, 'bulk_result.html', {'results': results})


@csrf_exempt
def generate_single_pdf(request):
    result = request.session.get('single_result')
    if not result:
        return render(request, 'result.html', {
            'prediction': 'N/A',
            'prediction_label': 'No data available'
        })

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("📄 Medical Device Failure Prediction Report", styles['Title']))
    elements.append(Spacer(1, 20))

    headers = ['Device Serial No.', 'Prediction Class', 'Description']
    color = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}.get(result['Prediction'], 'black')
    colored_label = f'<font color="{color}">{result["Prediction_Label"]}</font>'

    data = [
        headers,
        [
            result['Device_Serial_No'],
            str(result['Prediction']),
            Paragraph(colored_label, styles['Normal'])
        ]
    ]
    table = Table(data, colWidths=[150, 100, 250])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 30))

    elements.append(Paragraph(
        "<font size=8 color=gray>© This report is licensed under CC BY-NC-ND 4.0. "
        "Unauthorized use, copying, or redistribution is strictly prohibited.</font>",
        styles['Normal']
    ))

    doc.build(elements)
    buffer.seek(0)
    request.session.pop('single_result', None)
    return FileResponse(buffer, as_attachment=True, filename='device_single_prediction.pdf')


@csrf_exempt
def generate_pdf(request):
    results = request.session.get('bulk_results', [])
    if not results:
        return render(request, 'bulk_result.html', {
            'results': [],
            'error': 'No results available to export.'
        })

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("📄 Medical Device Failure Prediction Report", styles['Title']))
    elements.append(Spacer(1, 20))

    # Header row
    data = [['Device Serial No.', 'Prediction Class', 'Description']]
    for row in results:
        color = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}.get(row['Prediction'], 'black')
        colored_label = f'<font color="{color}">{row["Prediction_Label"]}</font>'
        data.append([
            row['Device_Serial_No'],
            str(row['Prediction']),
            Paragraph(colored_label, styles['Normal'])
        ])

    table = Table(data, colWidths=[150, 100, 250])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 30))

    elements.append(Paragraph(
        "<font size=8 color=gray>©"
        "Unauthorized use, copying, or redistribution is strictly prohibited.</font>",
        styles['Normal']
    ))

    doc.build(elements)
    buffer.seek(0)
    request.session.pop('bulk_results', None)
    return FileResponse(buffer, as_attachment=True, filename='device_predictions.pdf')


def about(request):
    return render(request, 'about.html')
