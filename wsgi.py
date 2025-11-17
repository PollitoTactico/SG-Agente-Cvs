"""
Configuración WSGI para deployment en producción.
"""
from api.infrastructure.adapters.input.fastapi_adapter import create_app

app = create_app()
