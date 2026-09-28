"""ASGI entry point for the Image Lab project."""

import os
import sys

# Add the project folder to sys.path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from django.core.asgi import get_asgi_application  # noqa: E402

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_asgi_application()
