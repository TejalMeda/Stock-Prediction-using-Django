from django.urls import path
from . import views

urlpatterns = [
    path('', views.stock_prediction, name='stock_prediction'),
]
