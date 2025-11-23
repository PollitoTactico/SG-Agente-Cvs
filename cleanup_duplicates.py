"""
Script para limpiar documentos duplicados del índice.
"""
import asyncio
import sys
from api.infrastructure.adapters.output.azure_search_adapter import AzureSearchAdapter
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


async def cleanup_duplicates():
    """Elimina todos los documentos del índice para empezar limpio."""
    try:
        adapter = AzureSearchAdapter()
        
        # Listar todos los document_ids
        doc_ids = await adapter.list_document_ids()
        
        logger.info(f"📋 Encontrados {len(doc_ids)} documentos en el índice")
        
        if not doc_ids:
            logger.info("✅ El índice ya está vacío")
            return True
        
        # Mostrar los documentos
        print("\nDocumentos actuales:")
        for i, doc_id in enumerate(doc_ids, 1):
            print(f"  {i}. {doc_id}")
        
        # Confirmar eliminación
        response = input(f"\n¿Eliminar TODOS los {len(doc_ids)} documentos? (si/no): ")
        
        if response.lower() not in ['si', 's', 'yes', 'y']:
            logger.info("❌ Operación cancelada")
            return False
        
        # Eliminar todos
        deleted = 0
        for doc_id in doc_ids:
            success = await adapter.delete_by_document_id(doc_id)
            if success:
                deleted += 1
                logger.info(f"🗑️  Eliminado: {doc_id}")
        
        logger.success(f"✅ {deleted}/{len(doc_ids)} documentos eliminados")
        logger.info("💡 Ahora puedes re-subir los CVs con: POST /api/v1/migrate/from-drive")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    result = asyncio.run(cleanup_duplicates())
    exit(0 if result else 1)
