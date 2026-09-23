"""Vercel runtime request context for Django.

Vercel's Python SDK can obtain rotating OIDC credentials from the current
request context. Registering the request headers here lets Blob authentication
work without storing a long-lived read/write token.
"""

import os


class VercelRequestContextMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if os.environ.get("VERCEL"):
            try:
                from vercel.headers import set_headers

                set_headers(request.headers)
            except Exception:
                # Blob authentication has its own clear error path. Do not
                # break local/dev requests if the Vercel SDK is unavailable.
                pass

        return self.get_response(request)
