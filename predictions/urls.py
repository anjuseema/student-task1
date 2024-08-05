from django.urls import path
from .views import predict_grade

urlpatterns = [
    path('predict/', predict_grade, name='predict_grade'),
]