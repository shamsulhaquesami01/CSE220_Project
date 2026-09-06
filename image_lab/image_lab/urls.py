"""URL routes for the image_lab app."""

from django.urls import path

from . import views

app_name = "image_lab"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/upload/", views.upload, name="upload"),
    path("api/process/", views.process, name="process"),
]
