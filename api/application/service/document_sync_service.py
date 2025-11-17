"""
Servicio para sincronizar documentos desde Google Drive al vector store.
"""
from typing import List, Optional
from pathlib import Path

from api.application.output.port.vector_store_port import VectorStorePort
from api.infrastructure.adapters.output.google_drive_adapter import GoogleDriveAdapter
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentSyncService:
    """Servicio para sincronizar documentos desde Google Drive."""
    
    def __init__(
        self,
        drive_adapter: GoogleDriveAdapter,
        vector_store: VectorStorePort
    ):
        """
        Inicializa el servicio de sincronización.
        
        Args:
            drive_adapter: Adaptador de Google Drive
            vector_store: Puerto del vector store
        """
        self.drive_adapter = drive_adapter
        self.vector_store = vector_store
    
    def sync_from_drive_url(
        self,
        drive_url: str,
        local_cache_path: str = "./drive_cache",
        mime_type: Optional[str] = 'application/pdf'
    ) -> dict:
        """
        Sincroniza documentos desde una URL de Google Drive al vector store.
        
        Args:
            drive_url: URL del folder de Google Drive
            local_cache_path: Ruta local para cachear archivos
            mime_type: Tipo de archivos a sincronizar
            
        Returns:
            Diccionario con estadísticas de sincronización
        """
        try:
            logger.info(f"Iniciando sincronización desde Google Drive: {drive_url}")
            
            # Extraer folder ID de la URL
            folder_id = self.drive_adapter.get_folder_id_from_url(drive_url)
            logger.info(f"Folder ID: {folder_id}")
            
            # Descargar archivos a cache local
            local_files = self.drive_adapter.sync_folder_to_local(
                folder_id=folder_id,
                local_path=local_cache_path,
                mime_type=mime_type
            )
            
            logger.info(f"Descargados {len(local_files)} archivos")
            
            # Cargar archivos al vector store
            uploaded_count = 0
            failed_count = 0
            
            for file_path in local_files:
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    file_name = Path(file_path).name
                    
                    # Agregar al vector store
                    doc_id = self.vector_store.add_documents(
                        documents=[{
                            'content': file_content,
                            'filename': file_name,
                            'source': 'google_drive'
                        }]
                    )[0]
                    
                    uploaded_count += 1
                    logger.info(f"✓ Cargado al vector store: {file_name} (ID: {doc_id})")
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"✗ Error al cargar {file_path}: {str(e)}")
            
            result = {
                'total_files': len(local_files),
                'uploaded': uploaded_count,
                'failed': failed_count,
                'cached_path': local_cache_path
            }
            
            logger.info(f"Sincronización completada: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error en sincronización: {str(e)}")
            raise
    
    def list_drive_files(self, drive_url: str) -> List[dict]:
        """
        Lista archivos en un folder de Google Drive sin descargarlos.
        
        Args:
            drive_url: URL del folder de Google Drive
            
        Returns:
            Lista de archivos con metadata
        """
        try:
            folder_id = self.drive_adapter.get_folder_id_from_url(drive_url)
            files = self.drive_adapter.list_files_in_folder(
                folder_id=folder_id,
                mime_type='application/pdf'
            )
            
            return [
                {
                    'id': f['id'],
                    'name': f['name'],
                    'size': f.get('size', 'N/A'),
                    'modified': f.get('modifiedTime', 'N/A')
                }
                for f in files
            ]
            
        except Exception as e:
            logger.error(f"Error al listar archivos: {str(e)}")
            raise
