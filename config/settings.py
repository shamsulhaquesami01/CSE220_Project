"""Django settings for Image Lab."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# Secret key: environment in production, local file during development.
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

# Debug locally, off by default on Vercel.
DEBUG = os.environ.get("DJANGO_DEBUG", "0" if os.environ.get("VERCEL") else "1") == "1"

# Allow localhost and Vercel subdomains.
ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "[::1]",
    ".vercel.app",
]


INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "image_lab",
]

MIDDLEWARE = [
    "image_lab.middleware.VercelRequestContextMiddleware",
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

# No database is needed.
DATABASES = {}


# Static and media

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

# Keep uploads and generated results separate.
UPLOAD_SUBDIR = "uploads"
RESULT_SUBDIR = "results"


# Image Lab limits

# Limit image size so previews stay responsive.
IMAGE_LAB_MAX_DIM = int(os.environ.get("IMAGE_LAB_MAX_DIM", 720))

# Upload size limit.
IMAGE_LAB_MAX_UPLOAD_BYTES = 12 * 1024 * 1024

DATA_UPLOAD_MAX_MEMORY_SIZE = IMAGE_LAB_MAX_UPLOAD_BYTES
FILE_UPLOAD_MAX_MEMORY_SIZE = IMAGE_LAB_MAX_UPLOAD_BYTES


LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Dhaka"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
