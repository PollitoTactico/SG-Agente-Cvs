"""
Vector Store con persistencia en Azure Blob Storage.
Combina almacenamiento en memoria (para búsqueda rápida) con persistencia en Blob.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from api.application.output.port.vector_store_port import VectorStorePort, VectorDocument
from api.infrastructure.adapters.output.azure_blob_adapter import AzureBlobAdapter
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class PersistentVectorStore(VectorStorePort):
    """
    Vector Store con persistencia en Azure Blob Storage.
    - En memoria: Para búsqueda rápida (RAM)
    - En Blob: Para persistencia permanente
    """
    
    def __init__(self, blob_adapter: AzureBlobAdapter):
        """
        Inicializa el vector store con persistencia.
        
        Args:
            blob_adapter: Adaptador de Azure Blob Storage
        """
        self.blob_adapter = blob_adapter
        self.documents: List[Dict[str, Any]] = []
        self._load_from_blob()
        logger.info(f"✅ Persistent Vector Store inicializado con {len(self.documents)} documentos")
    
    def _load_from_blob(self):
        """Carga todos los embeddings desde Blob al iniciar."""
        try:
            all_embeddings = self.blob_adapter.load_all_embeddings()
            
            for doc_data in all_embeddings:
                chunks = doc_data.get("chunks", [])
                for chunk in chunks:
                    self.documents.append({
                        "id": chunk["chunk_id"],
                        "content": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                        "embedding": np.array(chunk["embedding"])
                    })
            
            logger.success(f"📥 Cargados {len(self.documents)} chunks desde Blob Storage")
        except Exception as e:
            logger.error(f"❌ Error cargando desde Blob: {e}")
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Añade documentos a memoria Y los guarda en Blob.
        """
        ids = []
        document_id = metadatas[0].get("document_id") if metadatas else None
        
        # Preparar estructura para Blob
        chunks_data = []
        
        for i, (doc, metadata, embedding) in enumerate(zip(documents, metadatas, embeddings)):
            chunk_id = metadata.get("chunk_id", f"chunk_{len(self.documents)}_{i}")
            
            # Agregar a memoria
            self.documents.append({
                "id": chunk_id,
                "content": doc,
                "metadata": metadata,
                "embedding": np.array(embedding)
            })
            ids.append(chunk_id)
            
            # Preparar para Blob
            chunks_data.append({
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": metadata,
                "embedding": embedding  # Lista, no numpy array
            })
        
        # Guardar en Blob Storage
        if document_id:
            embeddings_data = {
                "document_id": document_id,
                "filename": metadatas[0].get("filename", "unknown"),
                "chunks": chunks_data,
                "total_chunks": len(chunks_data),
                "created_at": metadatas[0].get("upload_date", "")
            }
            
            self.blob_adapter.save_embeddings(document_id, embeddings_data)
        
        logger.info(f"✅ Añadidos {len(documents)} chunks (memoria + Blob). Total: {len(self.documents)}")
        return ids
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Dict[str, Any] | None = None
    ) -> List[VectorDocument]:
        """
        Busca documentos similares usando similitud de coseno.
        """
        if not self.documents:
            logger.warning("⚠️ No hay documentos en el vector store")
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
        
        # Convertir a VectorDocument
        results = []
        for doc, score in top_results:
            results.append(VectorDocument(
                id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                score=float(score)
            ))
        
        logger.info(f"🔍 Búsqueda completada: {len(results)} resultados (score max: {results[0].score:.4f})")
        return results
    
    async def delete_by_document_id(self, document_id: str) -> bool:
        """
        Elimina todos los chunks de un documento de memoria Y Blob.
        """
        # Eliminar de memoria
        initial_count = len(self.documents)
        self.documents = [
            doc for doc in self.documents
            if doc["metadata"].get("document_id") != document_id
        ]
        deleted_count = initial_count - len(self.documents)
        
        # Eliminar de Blob
        self.blob_adapter.delete_embeddings(document_id)
        
        logger.info(f"🗑️ Eliminados {deleted_count} chunks del documento '{document_id}' (memoria + Blob)")
        return deleted_count > 0
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        Lista todos los documentos únicos.
        """
        doc_ids = set()
        documents = []
        
        for doc in self.documents:
            doc_id = doc["metadata"].get("document_id")
            if doc_id and doc_id not in doc_ids:
                doc_ids.add(doc_id)
                documents.append({
                    "document_id": doc_id,
                    "filename": doc["metadata"].get("filename", "unknown"),
                    "chunks_count": sum(
                        1 for d in self.documents
                        if d["metadata"].get("document_id") == doc_id
                    )
                })
        
        return documents
    
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
        """
        Obtiene estadísticas del vector store.
        """
        return {
            "total_chunks": len(self.documents),
            "unique_documents": len(set(
                doc["metadata"].get("document_id")
                for doc in self.documents
                if doc["metadata"].get("document_id")
            )),
            "storage": "Azure Blob Storage + In-Memory"
        }
