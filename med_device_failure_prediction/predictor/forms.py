from django import forms

class PredictionForm(forms.Form):
    Age = forms.FloatField(label='Age')
    Maintenance_Cost = forms.FloatField(label='Maintenance Cost')
    Downtime = forms.FloatField(label='Downtime')
    Maintenance_Frequency = forms.FloatField(label='Maintenance Frequency')
    Failure_Event_Count = forms.FloatField(label='Failure Event Count')
