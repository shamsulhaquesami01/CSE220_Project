"""Pass Vercel request headers to the runtime SDK."""

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
                # Ignore this outside the Vercel runtime.
                pass

        return self.get_response(request)
