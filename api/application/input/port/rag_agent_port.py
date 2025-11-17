"""
Puerto de entrada: Interface para el agente RAG.
Define el contrato para realizar consultas al agente.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Modelo de petición para consulta."""
    query: str
    session_id: str | None = None
    filters: Dict[str, Any] | None = None


class QueryResponse(BaseModel):
    """Modelo de respuesta para consulta."""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    metadata: Dict[str, Any] | None = None


class RAGAgentPort(ABC):
    """
    Puerto de entrada para el agente RAG.
    Define las operaciones que puede realizar el agente.
    """
    
    @abstractmethod
    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Realiza una consulta al agente RAG.
        
        Args:
            request: Petición con la consulta
            
        Returns:
            Respuesta con la información recuperada
        """
        pass
    
    @abstractmethod
    async def clear_history(self, session_id: str) -> bool:
        """
        Limpia el historial de una sesión.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            True si se limpió correctamente
        """
        pass
