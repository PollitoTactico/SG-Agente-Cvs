"""
Servicio RAG Agent: Implementa la lógica del agente RAG.
"""
from typing import Dict, List
from uuid import uuid4
from datetime import datetime

from api.application.input.port.rag_agent_port import (
    RAGAgentPort, 
    QueryRequest, 
    QueryResponse
)
from api.application.output.port.llm_port import LLMPort
from api.application.output.port.vector_store_port import VectorStorePort
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGAgentService(RAGAgentPort):
    """
    Implementación del servicio RAG Agent.
    Coordina la recuperación de información y generación de respuestas.
    """
    
    def __init__(
        self,
        llm_port: LLMPort,
        vector_store_port: VectorStorePort
    ):
        """
        Inicializa el servicio.
        
        Args:
            llm_port: Puerto para el LLM
            vector_store_port: Puerto para el vector store
        """
        self.llm = llm_port
        self.vector_store = vector_store_port
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        
    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Procesa una consulta usando RAG.
        
        1. Genera embedding de la consulta
        2. Recupera documentos relevantes
        3. Genera respuesta con contexto
        4. Retorna respuesta con fuentes
        """
        logger.info(f"Procesando consulta: {request.query[:50]}...")
        
        # Obtener o crear session_id
        session_id = request.session_id or str(uuid4())
        
        # Obtener historial de la sesión
        chat_history = self.sessions.get(session_id, [])
        
        try:
            # 1. Generar embedding de la consulta
            query_embeddings = await self.llm.generate_embeddings([request.query])
            query_embedding = query_embeddings[0]
            
            # 2. Buscar documentos similares
            documents = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=5,
                filters=request.filters
            )
            
            # 3. Extraer contexto
            context = [doc.content for doc in documents]
            sources = [
                {
                    "document_id": doc.metadata.get("document_id", ""),
                    "filename": doc.metadata.get("filename", ""),
                    "score": doc.score,
                    "chunk_id": doc.id
                }
                for doc in documents
            ]
            
            # 4. Generar respuesta
            answer = await self.llm.generate_response(
                prompt=request.query,
                context=context,
                chat_history=chat_history
            )
            
            # 5. Actualizar historial
            chat_history.append({"role": "user", "content": request.query})
            chat_history.append({"role": "assistant", "content": answer})
            self.sessions[session_id] = chat_history
            
            logger.info(f"Consulta procesada exitosamente. Session: {session_id}")
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "documents_found": len(documents)
                }
            )
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {str(e)}")
            raise
    
    async def clear_history(self, session_id: str) -> bool:
        """
        Limpia el historial de conversación de una sesión.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Historial limpiado para sesión: {session_id}")
            return True
        return False
