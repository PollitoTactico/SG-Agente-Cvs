import uvicorn
from api.infrastructure.adapters.input.fastapi_adapter import create_app
from api.utils.config import settings
from api.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Inicia la aplicación FastAPI."""
    logger.info(f"Iniciando aplicación en modo: {settings.ENVIRONMENT}")
    
    uvicorn.run(
        "api.infrastructure.adapters.input.fastapi_adapter:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        log_level=str(settings.LOG_LEVEL).lower(),
        reload=settings.ENVIRONMENT == "development"
    )

if __name__ == "__main__":
    main()
