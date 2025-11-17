"""
Modelos Pydantic para la API REST.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    """Request para realizar una consulta."""
    query: str = Field(..., description="Pregunta del usuario", min_length=1)
    session_id: str | None = Field(None, description="ID de sesión para mantener contexto")
    filters: Dict[str, Any] | None = Field(None, description="Filtros adicionales")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "¿Cuáles son los beneficios de la arquitectura hexagonal?",
                "session_id": "user-123",
                "filters": {}
            }
        }


class Source(BaseModel):
    """Fuente de información."""
    document_id: str
    filename: str
    score: float
    chunk_id: str


class QueryResponse(BaseModel):
    """Response de una consulta."""
    answer: str = Field(..., description="Respuesta generada")
    sources: List[Source] = Field(..., description="Fuentes consultadas")
    session_id: str = Field(..., description="ID de sesión")
    metadata: Dict[str, Any] | None = Field(None, description="Metadata adicional")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "La arquitectura hexagonal...",
                "sources": [
                    {
                        "document_id": "doc-123",
                        "filename": "arquitectura.pdf",
                        "score": 0.95,
                        "chunk_id": "chunk-1"
                    }
                ],
                "session_id": "user-123",
                "metadata": {
                    "timestamp": "2025-11-16T10:00:00",
                    "documents_found": 5
                }
            }
        }


class DocumentUploadRequest(BaseModel):
    """Request para subir documento."""
    filename: str
    metadata: Dict[str, Any] | None = None


class DocumentUploadResponse(BaseModel):
    """Response de subida de documento."""
    document_id: str
    filename: str
    status: str
    message: str


class DocumentMetadata(BaseModel):
    """Metadata de un documento."""
    document_id: str
    filename: str
    upload_date: str
    size_bytes: int
    status: str
    chunk_count: int | None = None


class ErrorResponse(BaseModel):
    """Response de error."""
    error: str
    detail: str | None = None
