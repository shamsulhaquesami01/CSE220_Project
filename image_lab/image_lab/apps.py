from django.apps import AppConfig


class ImageLabConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "image_lab"
    verbose_name = "Image Lab"

    def ready(self):
        # Import advanced experiment modules for their @register side effects.
        # Keeping them separate prevents operations.py from becoming a monolith.
        from . import restoration  # noqa: F401
