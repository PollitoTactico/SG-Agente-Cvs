"""
Adaptador de Vector Store en memoria (sin Azure).
Para testing y desarrollo rápido.
"""
from typing import List, Dict, Any
import numpy as np
from api.application.output.port.vector_store_port import VectorStorePort, VectorDocument
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class InMemoryVectorStore(VectorStorePort):
    """
    Vector Store en memoria RAM.
    Los datos se pierden al reiniciar la aplicación.
    """
    
    def __init__(self):
        """Inicializa el almacén en memoria."""
        self.documents: List[Dict[str, Any]] = []
        logger.info("In-Memory Vector Store inicializado")
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Añade documentos a la memoria.
        """
        ids = []
        for i, (doc, metadata, embedding) in enumerate(zip(documents, metadatas, embeddings)):
            doc_id = metadata.get("chunk_id", f"doc_{len(self.documents)}_{i}")
            
            self.documents.append({
                "id": doc_id,
                "content": doc,
                "metadata": metadata,
                "embedding": np.array(embedding)
            })
            ids.append(doc_id)
        
        logger.info(f"Añadidos {len(documents)} documentos a memoria. Total: {len(self.documents)}")
        return ids
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
        query_text: str | None = None
    ) -> List[VectorDocument]:
        """
        Busca documentos similares usando similitud de coseno.
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados a retornar
            filters: Filtros opcionales
            query_text: Texto de la consulta (para búsqueda híbrida, no usado en InMemory)
        """
        if not self.documents:
            logger.warning("No hay documentos en memoria")
            return []
        
        query_vec = np.array(query_embedding)
        
        # Calcular similitud de coseno para cada documento
        similarities = []
        for doc in self.documents:
            # Filtrar si hay filtros
            if filters and "document_id" in filters:
                if doc["metadata"].get("document_id") != filters["document_id"]:
                    continue
            
            doc_vec = doc["embedding"]
            # Similitud de coseno
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            similarities.append((doc, similarity))
        
        # Ordenar por similitud (mayor a menor)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar top_k
        top_results = similarities[:top_k]
        
        # Formatear resultados
        results = []
        for doc, score in top_results:
            result = VectorDocument(
                id=doc["id"],
                content=doc["content"],
                metadata={
                    "document_id": doc["metadata"].get("document_id", ""),
                    "filename": doc["metadata"].get("filename", ""),
                    "chunk_id": doc["id"],
                    "chunk_index": doc["metadata"].get("chunk_index", 0),
                    "nombre_completo": doc["metadata"].get("nombre_completo", "Desconocido"),
                    "seccion_cv": doc["metadata"].get("seccion_cv", "general"),
                    "tipo_info": doc["metadata"].get("tipo_info", "general")
                },
                score=float(score)
            )
            results.append(result)
        
        logger.info(f"Encontrados {len(results)} documentos similares")
        return results
    
    async def delete_by_document_id(self, document_id: str) -> bool:
        """
        Elimina todos los chunks de un documento.
        """
        initial_count = len(self.documents)
        self.documents = [
            doc for doc in self.documents 
            if doc["metadata"].get("document_id") != document_id
        ]
        deleted = initial_count - len(self.documents)
        
        if deleted > 0:
            logger.info(f"Eliminados {deleted} chunks del documento {document_id}")
            return True
        return False
    
    async def list_document_ids(self) -> List[str]:
        """
        Lista todos los IDs de documentos únicos.
        """
        doc_ids = set()
        for doc in self.documents:
            doc_id = doc["metadata"].get("document_id")
            if doc_id:
                doc_ids.add(doc_id)
        
        return list(doc_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del almacén."""
        doc_ids = set()
        for doc in self.documents:
            doc_id = doc["metadata"].get("document_id")
            if doc_id:
                doc_ids.add(doc_id)
        
        return {
            "total_chunks": len(self.documents),
            "total_documents": len(doc_ids),
            "document_ids": list(doc_ids)
        }
