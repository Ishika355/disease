from django.urls import path
from . import views

urlpatterns = [
    path('templates/heart', views.heart, name="heart"),
    path('templates/diabetes', views.diabetes, name="diabetes"),
    path('', views.home, name="home"),
]