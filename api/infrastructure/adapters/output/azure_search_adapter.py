"""
Adaptador para Azure AI Search (Vector Store).
Implementa el puerto VectorStore usando Azure SDK.
"""
from typing import List, Dict, Any
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchFieldDataType
)
from azure.core.credentials import AzureKeyCredential

from api.application.output.port.vector_store_port import VectorStorePort, VectorDocument
from api.utils.config import settings
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class AzureSearchAdapter(VectorStorePort):
    """
    Adaptador para Azure AI Search.
    Gestiona el almacenamiento y búsqueda de vectores.
    """
    
    def __init__(self):
        """Inicializa el adaptador."""
        credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
        
        # Cliente para operaciones en el índice
        self.search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            credential=credential
        )
        
        # Cliente para gestión del índice
        self.index_client = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            credential=credential
        )
        
        logger.info("Azure Search Adapter inicializado")
    
    async def initialize_index(self):
        """
        Crea el índice si no existe.
        Llama esto una vez al inicio de la aplicación.
        """
        try:
            # Definir los campos del índice
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True
                ),
                SimpleField(
                    name="document_id",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True
                ),
                SimpleField(
                    name="filename",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="chunk_id",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SimpleField(
                    name="chunk_index",
                    type=SearchFieldDataType.Int32,
                    filterable=True
                ),
                SimpleField(
                    name="upload_date",
                    type=SearchFieldDataType.String,
                    filterable=True
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,  # Dimensiones de text-embedding-ada-002
                    vector_search_profile_name="my-vector-profile"
                )
            ]
            
            # Configurar búsqueda vectorial
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="my-hnsw")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="my-vector-profile",
                        algorithm_configuration_name="my-hnsw"
                    )
                ]
            )
            
            # Crear índice
            index = SearchIndex(
                name=settings.AZURE_SEARCH_INDEX_NAME,
                fields=fields,
                vector_search=vector_search
            )
            
            self.index_client.create_or_update_index(index)
            logger.info(f"Índice '{settings.AZURE_SEARCH_INDEX_NAME}' creado/actualizado")
            
        except Exception as e:
            logger.error(f"Error inicializando índice: {str(e)}")
            raise
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Añade documentos al índice.
        """
        try:
            # Preparar documentos para Azure Search
            search_documents = []
            for i, (doc, metadata, embedding) in enumerate(zip(documents, metadatas, embeddings)):
                search_doc = {
                    "id": metadata.get("chunk_id", f"doc_{i}"),
                    "document_id": metadata.get("document_id", ""),
                    "content": doc,
                    "filename": metadata.get("filename", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "upload_date": metadata.get("upload_date", ""),
                    "content_vector": embedding
                }
                search_documents.append(search_doc)
            
            # Subir documentos
            result = self.search_client.upload_documents(documents=search_documents)
            
            ids = [doc["id"] for doc in search_documents]
            logger.info(f"{len(search_documents)} documentos añadidos al índice")
            
            return ids
            
        except Exception as e:
            logger.error(f"Error añadiendo documentos: {str(e)}")
            raise
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Dict[str, Any] | None = None
    ) -> List[VectorDocument]:
        """
        Busca documentos similares usando búsqueda vectorial.
        """
        try:
            # Construir filtro si existe
            filter_expression = None
            if filters and "document_id" in filters:
                filter_expression = f"document_id eq '{filters['document_id']}'"
            
            # Realizar búsqueda vectorial
            results = self.search_client.search(
                search_text=None,
                vector_queries=[{
                    "vector": query_embedding,
                    "k_nearest_neighbors": top_k,
                    "fields": "content_vector"
                }],
                filter=filter_expression,
                select=["document_id", "content", "filename", "chunk_id", "chunk_index"],
                top=top_k
            )
            
            # Formatear resultados
            documents = []
            for result in results:
                doc = VectorDocument({
                    "document_id": result.get("document_id", ""),
                    "content": result.get("content", ""),
                    "filename": result.get("filename", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "score": result.get("@search.score", 0.0)
                })
                documents.append(doc)
            
            logger.info(f"Encontrados {len(documents)} documentos similares")
            return documents
            
        except Exception as e:
            logger.error(f"Error en búsqueda de similitud: {str(e)}")
            raise
    
    async def delete_by_document_id(self, document_id: str) -> bool:
        """
        Elimina todos los chunks de un documento.
        """
        try:
            # Buscar todos los chunks del documento
            results = self.search_client.search(
                search_text="*",
                filter=f"document_id eq '{document_id}'",
                select=["id"]
            )
            
            # Eliminar cada chunk
            ids_to_delete = [{"id": result["id"]} for result in results]
            
            if ids_to_delete:
                self.search_client.delete_documents(documents=ids_to_delete)
                logger.info(f"Eliminados {len(ids_to_delete)} chunks del documento {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error eliminando documento: {str(e)}")
            return False
    
    async def list_document_ids(self) -> List[str]:
        """
        Lista todos los IDs de documentos únicos.
        """
        try:
            results = self.search_client.search(
                search_text="*",
                select=["document_id"],
                top=1000
            )
            
            # Obtener IDs únicos
            document_ids = set()
            for result in results:
                if "document_id" in result:
                    document_ids.add(result["document_id"])
            
            return list(document_ids)
            
        except Exception as e:
            logger.error(f"Error listando documentos: {str(e)}")
            return []
