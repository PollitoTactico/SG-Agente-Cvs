"""
Script para inicializar el índice de Azure AI Search.
Ejecutar una vez antes de usar la aplicación.
"""
import asyncio
from api.infrastructure.adapters.output.azure_search_adapter import AzureSearchAdapter
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


async def main():
    """Inicializa el índice vectorial."""
    logger.info("Iniciando creación del índice...")
    
    try:
        adapter = AzureSearchAdapter()
        await adapter.initialize_index()
        logger.info("✅ Índice creado exitosamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error creando índice: {str(e)}")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
