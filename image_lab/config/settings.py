"""
Django settings for the Image Lab project.

Deliberately minimal: this app has no models, no users and no admin, so the
database, auth and contenttypes machinery are all switched off. That means
there are no migrations to run -- `python manage.py runserver` just works.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# --- Secret key -------------------------------------------------------------
# Read from the environment in production. For local coursework we persist a
# generated key to a gitignored file so it survives autoreloads (a key that
# changed on every reload would invalidate CSRF tokens mid-session).
def _get_secret_key():
    env_key = os.environ.get("DJANGO_SECRET_KEY")
    if env_key:
        return env_key

    key_file = BASE_DIR / ".secret_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()

    from django.core.management.utils import get_random_secret_key

    new_key = get_random_secret_key()
    key_file.write_text(new_key, encoding="utf-8")
    return new_key


SECRET_KEY = _get_secret_key()

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]


INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "image_lab",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# No models anywhere in this project, so no database is configured.
DATABASES = {}


# --- Static and media -------------------------------------------------------

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

# Uploaded originals and generated result panels are kept apart so the whole
# results cache can be cleared without touching what the user uploaded.
UPLOAD_SUBDIR = "uploads"
RESULT_SUBDIR = "results"


# --- Image Lab tuning -------------------------------------------------------

# Longest edge an upload is downscaled to on ingest. Keeps the per-keystroke
# convolution preview interactive; raise it if you need full-resolution output.
IMAGE_LAB_MAX_DIM = int(os.environ.get("IMAGE_LAB_MAX_DIM", 720))

# Reject uploads larger than this before decoding them.
IMAGE_LAB_MAX_UPLOAD_BYTES = 12 * 1024 * 1024

DATA_UPLOAD_MAX_MEMORY_SIZE = IMAGE_LAB_MAX_UPLOAD_BYTES
FILE_UPLOAD_MAX_MEMORY_SIZE = IMAGE_LAB_MAX_UPLOAD_BYTES


LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Dhaka"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
