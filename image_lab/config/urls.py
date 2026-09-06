"""Root URL configuration for the Image Lab project."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

urlpatterns = [
    path("", include("image_lab.urls")),
]

# Django's dev server does not serve MEDIA_ROOT automatically. This helper adds
# a route for it when DEBUG is on; a real deployment would let nginx do it.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
