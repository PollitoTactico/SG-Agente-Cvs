"""
Configuración de logging para la aplicación.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(name: str):
    """
    Configura el logger de la aplicación.
    
    Args:
        name: Nombre del módulo
        
    Returns:
        Logger configurado
    """
    # Crear directorio de logs si no existe
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Remover handler por defecto
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # File handler - INFO y superior
    logger.add(
        log_dir / "app.log",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="INFO"
    )
    
    # File handler - ERROR
    logger.add(
        log_dir / "error.log",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR"
    )
    
    return logger.bind(name=name)
