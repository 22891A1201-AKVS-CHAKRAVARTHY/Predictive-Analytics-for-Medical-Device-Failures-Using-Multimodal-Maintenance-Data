from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('form/', views.form_view, name='form'),
    path('results/', views.bulk_result, name='bulk_result'),
    path('generate_pdf/', views.generate_pdf, name='generate_pdf'),
    path('generate_single_pdf/', views.generate_single_pdf, name='generate_single_pdf'),
    path('about/', views.about, name='about'),
]

