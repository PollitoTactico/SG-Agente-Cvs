"""
Puerto de entrada: Interface para gestión de documentos.
Define el contrato para subir, eliminar y listar documentos.
"""
from abc import ABC, abstractmethod
from typing import List, BinaryIO
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata de un documento."""
    document_id: str
    filename: str
    upload_date: str
    size_bytes: int
    status: str
    chunk_count: int | None = None


class DocumentUploadResponse(BaseModel):
    """Respuesta al subir un documento."""
    document_id: str
    filename: str
    status: str
    message: str


class DocumentManagerPort(ABC):
    """
    Puerto de entrada para gestión de documentos.
    Define las operaciones CRUD sobre documentos.
    """
    
    @abstractmethod
    async def upload_document(
        self, 
        file: BinaryIO, 
        filename: str,
        metadata: dict | None = None
    ) -> DocumentUploadResponse:
        """
        Sube un documento al sistema.
        
        Args:
            file: Archivo a subir
            filename: Nombre del archivo
            metadata: Metadata adicional
            
        Returns:
            Respuesta con el ID del documento
        """
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Elimina un documento del sistema.
        
        Args:
            document_id: ID del documento
            
        Returns:
            True si se eliminó correctamente
        """
        pass
    
    @abstractmethod
    async def list_documents(self) -> List[DocumentMetadata]:
        """
        Lista todos los documentos indexados.
        
        Returns:
            Lista de metadata de documentos
        """
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> DocumentMetadata:
        """
        Obtiene metadata de un documento específico.
        
        Args:
            document_id: ID del documento
            
        Returns:
            Metadata del documento
        """
        pass
