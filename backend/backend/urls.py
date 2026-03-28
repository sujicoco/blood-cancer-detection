from django.contrib import admin
from django.urls import path
from django.conf import settings  # ← ADD THIS
from django.conf.urls.static import static  # ← ADD THIS
from api.views import dashboard, predict_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', dashboard, name='dashboard'),
    path('api/predict/', predict_view, name='predict'),
]

# 🚀 SERVE UPLOADS + ATTENTION MAPS
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
