from django.urls import path
from .views import vasicek_view, interpolation_view

urlpatterns = [
    path('vasicek/', vasicek_view, name='vasicek_view'),
    path('interpolation/', interpolation_view, name='interpolation_view'),
]
