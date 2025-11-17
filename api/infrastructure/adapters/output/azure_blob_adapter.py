"""
Adaptador de Azure Blob Storage para persistencia de documentos y embeddings.
"""
import json
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class AzureBlobAdapter:
    """Adaptador para interactuar con Azure Blob Storage."""
    
    def __init__(self, connection_string: str, container_pdfs: str, container_embeddings: str, container_cache: str):
        """
        Inicializa el adaptador de Azure Blob Storage.
        
        Args:
            connection_string: Connection string de Azure Storage
            container_pdfs: Nombre del contenedor para PDFs
            container_embeddings: Nombre del contenedor para embeddings JSON
            container_cache: Nombre del contenedor para cache temporal
        """
        self.connection_string = connection_string
        self.container_pdfs = container_pdfs
        self.container_embeddings = container_embeddings
        self.container_cache = container_cache
        
        # Inicializar cliente
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Crear contenedores si no existen
        self._ensure_containers_exist()
        
        logger.info("✅ Azure Blob Storage inicializado correctamente")
    
    def _ensure_containers_exist(self):
        """Crea los contenedores si no existen."""
        try:
            for container_name in [self.container_pdfs, self.container_embeddings, self.container_cache]:
                try:
                    self.blob_service_client.create_container(container_name)
                    logger.info(f"📦 Contenedor '{container_name}' creado")
                except Exception as e:
                    # Si ya existe, ignorar el error
                    if "already exists" not in str(e).lower():
                        logger.debug(f"Contenedor '{container_name}' ya existe")
        except Exception as e:
            logger.warning(f"Error verificando contenedores: {e}")
    
    # ==================== OPERACIONES CON PDFs ====================
    
    def upload_pdf(self, file_content: bytes, filename: str) -> str:
        """
        Sube un PDF a Blob Storage.
        
        Args:
            file_content: Contenido del archivo en bytes
            filename: Nombre del archivo
            
        Returns:
            Blob name (ID único)
        """
        try:
            blob_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_pdfs,
                blob=blob_name
            )
            
            blob_client.upload_blob(file_content, overwrite=True)
            logger.success(f"✅ PDF subido: {blob_name}")
            
            return blob_name
            
        except Exception as e:
            logger.error(f"❌ Error subiendo PDF '{filename}': {e}")
            raise
    
    def download_pdf(self, blob_name: str) -> bytes:
        """
        Descarga un PDF desde Blob Storage.
        
        Args:
            blob_name: Nombre del blob
            
        Returns:
            Contenido del archivo en bytes
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_pdfs,
                blob=blob_name
            )
            
            return blob_client.download_blob().readall()
            
        except ResourceNotFoundError:
            logger.error(f"❌ PDF no encontrado: {blob_name}")
            raise
        except Exception as e:
            logger.error(f"❌ Error descargando PDF '{blob_name}': {e}")
            raise
    
    def list_pdfs(self) -> List[Dict[str, Any]]:
        """
        Lista todos los PDFs en Blob Storage.
        
        Returns:
            Lista de diccionarios con metadata de PDFs
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_pdfs)
            blobs = container_client.list_blobs()
            
            pdf_list = []
            for blob in blobs:
                pdf_list.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                    "content_type": blob.content_settings.content_type if blob.content_settings else None
                })
            
            logger.info(f"📄 {len(pdf_list)} PDFs encontrados en Blob Storage")
            return pdf_list
            
        except Exception as e:
            logger.error(f"❌ Error listando PDFs: {e}")
            return []
    
    def delete_pdf(self, blob_name: str) -> bool:
        """
        Elimina un PDF de Blob Storage.
        
        Args:
            blob_name: Nombre del blob
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_pdfs,
                blob=blob_name
            )
            blob_client.delete_blob()
            logger.success(f"🗑️ PDF eliminado: {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error eliminando PDF '{blob_name}': {e}")
            return False
    
    # ==================== OPERACIONES CON EMBEDDINGS ====================
    
    def save_embeddings(self, document_id: str, embeddings_data: Dict[str, Any]) -> bool:
        """
        Guarda embeddings como JSON en Blob Storage.
        
        Args:
            document_id: ID único del documento
            embeddings_data: Diccionario con chunks y embeddings
            
        Returns:
            True si se guardó correctamente
        """
        try:
            blob_name = f"{document_id}_embeddings.json"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_embeddings,
                blob=blob_name
            )
            
            # Convertir a JSON
            json_data = json.dumps(embeddings_data, ensure_ascii=False, indent=2)
            
            # Subir a Blob
            blob_client.upload_blob(json_data, overwrite=True, content_type="application/json")
            logger.success(f"💾 Embeddings guardados: {blob_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error guardando embeddings para '{document_id}': {e}")
            return False
    
    def load_embeddings(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Carga embeddings desde Blob Storage.
        
        Args:
            document_id: ID único del documento
            
        Returns:
            Diccionario con embeddings o None si no existe
        """
        try:
            blob_name = f"{document_id}_embeddings.json"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_embeddings,
                blob=blob_name
            )
            
            json_data = blob_client.download_blob().readall()
            embeddings_data = json.loads(json_data)
            
            logger.info(f"📥 Embeddings cargados: {blob_name}")
            return embeddings_data
            
        except ResourceNotFoundError:
            logger.warning(f"⚠️ Embeddings no encontrados: {document_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Error cargando embeddings '{document_id}': {e}")
            return None
    
    def load_all_embeddings(self) -> List[Dict[str, Any]]:
        """
        Carga todos los embeddings almacenados en Blob.
        
        Returns:
            Lista de diccionarios con todos los embeddings
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_embeddings)
            blobs = container_client.list_blobs()
            
            all_embeddings = []
            for blob in blobs:
                if blob.name.endswith('_embeddings.json'):
                    try:
                        blob_client = self.blob_service_client.get_blob_client(
                            container=self.container_embeddings,
                            blob=blob.name
                        )
                        json_data = blob_client.download_blob().readall()
                        embeddings_data = json.loads(json_data)
                        all_embeddings.append(embeddings_data)
                    except Exception as e:
                        logger.error(f"Error cargando {blob.name}: {e}")
            
            logger.success(f"📦 {len(all_embeddings)} documentos con embeddings cargados")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Error cargando todos los embeddings: {e}")
            return []
    
    def delete_embeddings(self, document_id: str) -> bool:
        """
        Elimina embeddings de un documento.
        
        Args:
            document_id: ID único del documento
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            blob_name = f"{document_id}_embeddings.json"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_embeddings,
                blob=blob_name
            )
            blob_client.delete_blob()
            logger.success(f"🗑️ Embeddings eliminados: {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error eliminando embeddings '{document_id}': {e}")
            return False
    
    def list_all_documents(self) -> List[str]:
        """
        Lista todos los document_ids almacenados.
        
        Returns:
            Lista de document IDs
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_embeddings)
            blobs = container_client.list_blobs()
            
            doc_ids = []
            for blob in blobs:
                if blob.name.endswith('_embeddings.json'):
                    doc_id = blob.name.replace('_embeddings.json', '')
                    doc_ids.append(doc_id)
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"❌ Error listando documentos: {e}")
            return []
