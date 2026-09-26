from django.apps import AppConfig


class ImageLabConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "image_lab"
    verbose_name = "Image Lab"

    def ready(self):
        # Load extra operations into the registry.
        from . import restoration  # noqa: F401
        from . import spectral_match  # noqa: F401
