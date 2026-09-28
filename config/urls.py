"""Root URL configuration for the Image Lab project."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

urlpatterns = [
    path("", include("image_lab.urls")),
]

# Serve media files during local development.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
