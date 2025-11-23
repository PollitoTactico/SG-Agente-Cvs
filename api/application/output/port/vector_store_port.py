"""
Puerto de salida: Interface para la base de datos vectorial.
Define el contrato para almacenar y recuperar documentos.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class VectorDocument:
    """Documento vectorial con score de similitud."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class VectorStorePort(ABC):
    """
    Puerto de salida para la base de datos vectorial.
    Abstrae la interacción con Azure AI Search.
    """
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Añade documentos al vector store.
        
        Args:
            documents: Textos de los documentos
            metadatas: Metadata de cada documento
            embeddings: Embeddings de cada documento
            
        Returns:
            Lista de IDs asignados
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
        query_text: str = ""
    ) -> List[VectorDocument]:
        """
        Busca documentos similares usando búsqueda híbrida.
        
        Args:
            query_embedding: Embedding de la consulta (búsqueda vectorial)
            top_k: Número de resultados
            filters: Filtros adicionales
            query_text: Texto de la query para búsqueda por keywords (BM25)
            
        Returns:
            Documentos similares con scores
        """
        pass
    
    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> bool:
        """
        Elimina todos los chunks de un documento.
        
        Args:
            document_id: ID del documento
            
        Returns:
            True si se eliminó correctamente
        """
        pass
    
    @abstractmethod
    async def list_document_ids(self) -> List[str]:
        """
        Lista todos los IDs de documentos únicos.
        
        Returns:
            Lista de IDs de documentos
        """
        pass
