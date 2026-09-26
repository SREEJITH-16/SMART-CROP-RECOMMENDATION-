"""WSGI entry point for production servers.

    gunicorn wsgi:application

The model and datasets load once when the module is imported, not per request.
"""

from app import app as application

if __name__ == "__main__":
    application.run()
