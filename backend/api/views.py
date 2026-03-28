from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
import os

from inference_model import predict_image


def dashboard(request):
    # Renders the dark dashboard UI; JS in dashboard.html calls /api/predict/
    return render(request, "dashboard.html")


@csrf_exempt
@require_POST
def predict_view(request):
    """
    Accepts an uploaded image file with field name 'scan',
    runs the model, and returns JSON for the UI.
    """
    file_obj = request.FILES.get("scan")
    if not file_obj:
        return JsonResponse({"error": "No file uploaded with field name 'scan'"}, status=400)

    # Save uploaded image to MEDIA_ROOT/uploads/
    media_root = getattr(settings, "MEDIA_ROOT", os.path.join(settings.BASE_DIR, "media"))
    upload_dir = os.path.join(media_root, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file_obj.name)
    with open(file_path, "wb+") as dest:
        for chunk in file_obj.chunks():
            dest.write(chunk)

    # Run inference
    result = predict_image(file_path)

    return JsonResponse(result, safe=False)
