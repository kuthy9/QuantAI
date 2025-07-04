"""
REST API control plane for the QuantAI system.

Provides external HTTP/REST interfaces for system control,
monitoring, and integration with external applications.
"""

from .server import APIServer
from .routes import setup_routes
from .middleware import setup_middleware
from .auth import AuthManager

__all__ = [
    "APIServer",
    "setup_routes", 
    "setup_middleware",
    "AuthManager",
]
