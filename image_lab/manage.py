#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys

# ---------------------------------------------------------------------------
# sys.path shim -- required by this machine's Python installation.
#
# D:\LocalDev\Python is an *embeddable* CPython distribution. Its python313._pth
# file pins sys.path to exactly ["python313.zip", "."] and, as a side effect of
# the ._pth mechanism, CPython neither prepends the running script's directory
# nor honours the PYTHONPATH environment variable. Django cannot import the
# `config` and `image_lab` packages that live next to this file without help.
#
# Inserting BASE_DIR restores the behaviour a normal CPython install gives for
# free. Delete these two lines if you ever move to a full Python + virtualenv.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
