"""
Module 9 — Security Headers Middleware.
Applies defense-in-depth HTTP headers to every incoming response.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects standard security headers into all HTTP responses.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Prevent browsers from MIME-sniffing a response away from the declared Content-Type
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking attacks by forbidding iframe embedding
        response.headers["X-Frame-Options"] = "DENY"

        # Stop pages from loading when they detect reflected cross-site scripting (XSS)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Ensure referrer information is stripped across cross-origin requests
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Enforce HTTPS transport in production
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
