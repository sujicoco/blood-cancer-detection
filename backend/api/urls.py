from django.urls import path
from . import views  # Import YOUR views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),        # /api/ → dashboard
    path("predict/", views.predict_view, name="predict"), # /api/predict/
]
