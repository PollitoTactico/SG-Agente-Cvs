"""
Servicio Document Manager: Gestiona la carga y eliminación de documentos.
"""
from typing import List, BinaryIO
from uuid import uuid4
from datetime import datetime
import hashlib

from api.application.input.port.document_manager_port import (
    DocumentManagerPort,
    DocumentUploadResponse,
    DocumentMetadata
)
from api.application.output.port.llm_port import LLMPort
from api.application.output.port.vector_store_port import VectorStorePort
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentManagerService(DocumentManagerPort):
    """
    Implementación del servicio de gestión de documentos.
    Maneja la carga, procesamiento e indexación de PDFs.
    """
    
    def __init__(
        self,
        llm_port: LLMPort,
        vector_store_port: VectorStorePort,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Inicializa el servicio.
        
        Args:
            llm_port: Puerto para el LLM
            vector_store_port: Puerto para el vector store
            chunk_size: Tamaño de los chunks
            chunk_overlap: Overlap entre chunks
        """
        self.llm = llm_port
        self.vector_store = vector_store_port
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    async def upload_document(
        self,
        file: BinaryIO,
        filename: str,
        metadata: dict | None = None
    ) -> DocumentUploadResponse:
        """
        Sube y procesa un documento PDF.
        
        1. Lee el contenido del PDF
        2. Divide en chunks
        3. Genera embeddings
        4. Almacena en vector store
        """
        logger.info(f"Procesando documento: {filename}")
        
        try:
            # Leer contenido del archivo
            content = file.read()
            document_id = str(uuid4())
            
            # Extraer texto del PDF
            text = await self._extract_text_from_pdf(content)
            
            # Dividir en chunks
            chunks = self._create_chunks(text)
            logger.info(f"Documento dividido en {len(chunks)} chunks")
            
            # Generar embeddings
            embeddings = await self.llm.generate_embeddings(chunks)
            
            # Preparar metadata para cada chunk
            chunk_metadatas = [
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": f"{document_id}_{i}",
                    "chunk_index": i,
                    "upload_date": datetime.utcnow().isoformat(),
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
                for i in range(len(chunks))
            ]
            
            # Almacenar en vector store
            await self.vector_store.add_documents(
                documents=chunks,
                metadatas=chunk_metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Documento {filename} procesado exitosamente. ID: {document_id}")
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                status="success",
                message=f"Documento procesado: {len(chunks)} chunks indexados"
            )
            
        except Exception as e:
            logger.error(f"Error procesando documento {filename}: {str(e)}")
            return DocumentUploadResponse(
                document_id="",
                filename=filename,
                status="error",
                message=f"Error: {str(e)}"
            )
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Elimina un documento del vector store.
        """
        logger.info(f"Eliminando documento: {document_id}")
        try:
            result = await self.vector_store.delete_by_document_id(document_id)
            if result:
                logger.info(f"Documento {document_id} eliminado exitosamente")
            return result
        except Exception as e:
            logger.error(f"Error eliminando documento {document_id}: {str(e)}")
            return False
    
    async def list_documents(self) -> List[DocumentMetadata]:
        """
        Lista todos los documentos indexados.
        """
        try:
            document_ids = await self.vector_store.list_document_ids()
            # Aquí podrías obtener más información de cada documento
            # Por ahora retornamos información básica
            return [
                DocumentMetadata(
                    document_id=doc_id,
                    filename="unknown",  # Necesitarías obtener esto del vector store
                    upload_date="",
                    size_bytes=0,
                    status="indexed"
                )
                for doc_id in document_ids
            ]
        except Exception as e:
            logger.error(f"Error listando documentos: {str(e)}")
            return []
    
    async def get_document(self, document_id: str) -> DocumentMetadata:
        """
        Obtiene metadata de un documento específico.
        """
        # Implementación básica - necesitarías ampliarla
        return DocumentMetadata(
            document_id=document_id,
            filename="unknown",
            upload_date="",
            size_bytes=0,
            status="indexed"
        )
    
    async def _extract_text_from_pdf(self, content: bytes) -> str:
        """
        Extrae texto de un PDF.
        """
        from pypdf import PdfReader
        from io import BytesIO
        
        pdf = PdfReader(BytesIO(content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        
        return text
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Divide el texto en chunks con overlap.
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
