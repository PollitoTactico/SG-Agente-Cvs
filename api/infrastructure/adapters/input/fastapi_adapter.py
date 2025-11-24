"""
Adaptador FastAPI para exponer el agente RAG como API REST.
"""
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List

from api.infrastructure.adapters.input.models import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    DocumentMetadata,
    ErrorResponse,
    Source
)
from api.application.service.rag_agent_service import RAGAgentService
from api.application.service.document_manager_service import DocumentManagerService
from api.infrastructure.adapters.output.azure_openai_adapter import AzureOpenAIAdapter
from api.infrastructure.adapters.output.azure_search_adapter import AzureSearchAdapter
from api.infrastructure.adapters.output.azure_blob_adapter import AzureBlobAdapter
from api.utils.config import settings
from api.utils.logger import setup_logger

logger = setup_logger(__name__)

# Instancias singleton
_vector_store_instance = None
_blob_adapter_instance = None


# Dependency Injection
def get_llm_adapter():
    """Retorna instancia del adaptador LLM."""
    return AzureOpenAIAdapter()


def get_blob_adapter():
    """Retorna instancia del adaptador Blob Storage."""
    global _blob_adapter_instance
    if _blob_adapter_instance is None:
        _blob_adapter_instance = AzureBlobAdapter(
            connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
            container_pdfs=settings.AZURE_STORAGE_CONTAINER_PDFS,
            container_embeddings=settings.AZURE_STORAGE_CONTAINER_EMBEDDINGS,
            container_cache=settings.AZURE_STORAGE_CONTAINER_CACHE
        )
    return _blob_adapter_instance


def get_vector_store_adapter():
    """
    Retorna instancia del adaptador Vector Store.
    Usa Azure Search para persistencia en la nube.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        # Verificar si hay configuración de Azure Search
        has_search_config = (
            settings.AZURE_SEARCH_ENDPOINT and 
            settings.AZURE_SEARCH_ENDPOINT != "<TU_AZURE_SEARCH_ENDPOINT>" and
            settings.AZURE_SEARCH_API_KEY and
            settings.AZURE_SEARCH_API_KEY != "<TU_AZURE_SEARCH_API_KEY>"
        )
        
        if not has_search_config:
            logger.error("❌ Azure Search NO configurado en .env")
            raise ValueError("Azure Search es requerido. Configura AZURE_SEARCH_ENDPOINT y AZURE_SEARCH_API_KEY")
        
        # Usar AzureSearchAdapter
        logger.info("🔍 Usando Azure Search para vector store")
        _vector_store_instance = AzureSearchAdapter()
    
    return _vector_store_instance


def get_rag_service(
    llm_adapter: AzureOpenAIAdapter = Depends(get_llm_adapter),
    vector_store_adapter: AzureSearchAdapter = Depends(get_vector_store_adapter)
) -> RAGAgentService:
    """Retorna instancia del servicio RAG."""
    return RAGAgentService(llm_adapter, vector_store_adapter)


def get_document_service(
    llm_adapter: AzureOpenAIAdapter = Depends(get_llm_adapter),
    vector_store_adapter: AzureSearchAdapter = Depends(get_vector_store_adapter)
) -> DocumentManagerService:
    """Retorna instancia del servicio de documentos."""
    return DocumentManagerService(
        llm_adapter,
        vector_store_adapter,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )





def create_app() -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.
    """
    app = FastAPI(
        title="RAG Agent API",
        description="API REST para Agente RAG con Arquitectura Hexagonal",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Ajustar en producción
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Verifica el estado de la API."""
        vector_store = get_vector_store_adapter()
        stats = vector_store.get_stats()
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "vector_store": stats
        }
    
    # Endpoints del RAG Agent
    @app.post(
        "/api/v1/query",
        response_model=QueryResponse,
        responses={500: {"model": ErrorResponse}},
        tags=["RAG Agent"]
    )
    async def query_agent(
        request: QueryRequest,
        rag_service: RAGAgentService = Depends(get_rag_service)
    ):
        """
        Realiza una consulta al agente RAG.
        
        - **query**: Pregunta del usuario
        - **session_id**: ID de sesión (opcional, se genera si no existe)
        - **filters**: Filtros para la búsqueda (opcional)
        """
        try:
            # Convertir el request de la API al request del dominio
            from api.application.input.port.rag_agent_port import QueryRequest as DomainQueryRequest
            
            domain_request = DomainQueryRequest(
                query=request.query,
                session_id=request.session_id,
                filters=request.filters
            )
            
            result = await rag_service.query(domain_request)
            
            # Convertir la respuesta del dominio al modelo de la API
            return QueryResponse(
                answer=result.answer,
                sources=[
                    Source(
                        document_id=src["document_id"],
                        filename=src["filename"],
                        score=src["score"],
                        chunk_id=src["chunk_id"]
                    )
                    for src in result.sources
                ],
                session_id=result.session_id,
                metadata=result.metadata
            )
        except Exception as e:
            logger.error(f"Error en query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/v1/sessions/{session_id}", tags=["RAG Agent"])
    async def clear_session(
        session_id: str,
        rag_service: RAGAgentService = Depends(get_rag_service)
    ):
        """
        Limpia el historial de una sesión.
        """
        try:
            result = await rag_service.clear_history(session_id)
            if result:
                return {"message": "Historial limpiado", "session_id": session_id}
            else:
                raise HTTPException(status_code=404, detail="Sesión no encontrada")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error limpiando sesión: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Endpoints de gestión de documentos
    @app.post(
        "/api/v1/documents/upload",
        response_model=DocumentUploadResponse,
        tags=["Documents"]
    )
    async def upload_document(
        file: UploadFile = File(..., description="Archivo PDF a subir"),
        upload_to_blob: bool = True,
        doc_service: DocumentManagerService = Depends(get_document_service)
    ):
        """
        Sube un documento PDF, lo procesa y guarda embeddings en Azure Blob.
        
        - **file**: Archivo PDF a subir
        - **upload_to_blob**: Si True, también sube el PDF a Blob Storage (default: True)
        """
        try:
            # Validar que sea PDF
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail="Solo se aceptan archivos PDF"
                )
            
            # Leer el archivo
            content = await file.read()
            
            # Subir PDF a Blob si se solicita
            if upload_to_blob:
                blob_adapter = get_blob_adapter()
                blob_name = blob_adapter.upload_pdf(content, file.filename)
                logger.success(f"✅ PDF subido a Blob: {blob_name}")
            
            # Procesar y crear embeddings (se guardan automáticamente en Blob)
            from io import BytesIO
            file_obj = BytesIO(content)
            
            result = await doc_service.upload_document(
                file=file_obj,
                filename=file.filename
            )
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error subiendo documento: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get(
        "/api/v1/documents",
        response_model=List[DocumentMetadata],
        tags=["Documents"]
    )
    async def list_documents(
        doc_service: DocumentManagerService = Depends(get_document_service)
    ):
        """
        Lista todos los documentos indexados.
        """
        try:
            documents = await doc_service.list_documents()
            return documents
        except Exception as e:
            logger.error(f"Error listando documentos: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/v1/documents/{document_id}", tags=["Documents"])
    async def delete_document(
        document_id: str,
        doc_service: DocumentManagerService = Depends(get_document_service)
    ):
        """
        Elimina un documento del índice.
        """
        try:
            result = await doc_service.delete_document(document_id)
            if result:
                return {"message": "Documento eliminado", "document_id": document_id}
            else:
                raise HTTPException(status_code=404, detail="Documento no encontrado")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error eliminando documento: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Endpoint de información del storage
    @app.get("/api/v1/storage/stats", tags=["Storage"])
    async def get_storage_stats(
        blob_adapter: AzureBlobAdapter = Depends(get_blob_adapter),
        vector_store: AzureSearchAdapter = Depends(get_vector_store_adapter)
    ):
        """
        📊 Obtiene estadísticas completas del almacenamiento.
        
        Muestra:
        - Cuántos PDFs hay en Azure Blob Storage
        - Cuántos documentos indexados en Azure Search
        - Cuántas personas únicas en el sistema
        - Lista de primeros 10 PDFs
        """
        try:
            # Estadísticas del vector store (Azure Search)
            search_stats = vector_store.get_stats()
            
            # Listar PDFs en Blob
            pdfs = blob_adapter.list_pdfs()
            
            # Listar embeddings en Blob
            doc_ids = blob_adapter.list_all_documents()
            
            return {
                "azure_search": search_stats,
                "azure_blob_storage": {
                    "pdfs_count": len(pdfs),
                    "embeddings_count": len(doc_ids),
                    "sample_pdfs": pdfs[:10]  # Solo primeros 10
                },
                "summary": {
                    "total_cv_pdfs": len(pdfs),
                    "indexed_documents": search_stats.get("unique_documents", 0),
                    "total_chunks": search_stats.get("total_chunks", 0),
                    "unique_personas": search_stats.get("unique_personas", 0)
                }
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Endpoint para obtener detalle de CV de una persona
    @app.get("/api/v1/cv/detail", tags=["CV Analysis"])
    async def get_cv_detail(
        name: str,
        rag_service: RAGAgentService = Depends(get_rag_service)
    ):
        """
        Obtiene información detallada del CV de una persona específica.
        
        - **name**: Nombre completo de la persona a buscar
        """
        try:
            from api.application.input.port.rag_agent_port import QueryRequest as DomainQueryRequest
            
            # Crear una consulta específica para obtener información de la persona
            query_text = f"Dame un resumen completo y detallado del perfil profesional de {name}, incluyendo su experiencia laboral, educación, habilidades técnicas y certificaciones."
            
            domain_request = DomainQueryRequest(
                query=query_text,
                session_id=f"cv_detail_{name}",
                filters=None
            )
            
            result = await rag_service.query(domain_request)
            
            # Extraer información de las fuentes
            sources_info = []
            for src in result.sources:
                sources_info.append({
                    "document": src["filename"],
                    "page": src.get("chunk_id", "N/A"),
                    "relevance": src["score"]
                })
            
            return {
                "name": name,
                "content": result.answer,
                "sources": sources_info,
                "chunk_count": len(result.sources)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo CV detail: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    
    # Endpoint de migración desde Google Drive (UNA SOLA VEZ)
    @app.post("/api/v1/migrate/from-drive", tags=["Migration"])
    async def migrate_from_google_drive(
        doc_service: DocumentManagerService = Depends(get_document_service)
    ):
        """
        🚀 MIGRACIÓN UNA SOLA VEZ: Descarga PDFs de Google Drive y los migra a Azure Blob.
        
        Este endpoint:
        1. Lista archivos en Google Drive
        2. Descarga cada PDF
        3. Lo sube a Azure Blob Storage
        4. Procesa y crea embeddings
        5. Guarda embeddings en Blob
        
        ⚠️ Solo ejecutar UNA vez para migración inicial.
        """
        try:
            # Verificar credenciales de Drive
            if not settings.GOOGLE_DRIVE_CREDENTIALS_PATH or not settings.GOOGLE_DRIVE_FOLDER_ID:
                raise HTTPException(
                    status_code=400,
                    detail="Google Drive no configurado en .env (GOOGLE_DRIVE_CREDENTIALS_PATH y GOOGLE_DRIVE_FOLDER_ID)"
                )
            
            # Inicializar adaptador de Drive (solo lectura)
            from api.infrastructure.adapters.output.google_drive_readonly_adapter import GoogleDriveAdapter
            
            drive = GoogleDriveAdapter(
                credentials_path=settings.GOOGLE_DRIVE_CREDENTIALS_PATH,
                folder_id=settings.GOOGLE_DRIVE_FOLDER_ID
            )
            
            blob_adapter = get_blob_adapter()
            
            logger.info("🚀 Iniciando migración desde Google Drive a Azure Blob...")
            
            # Listar archivos en Drive
            drive_files = drive.list_files_in_folder(mime_type='application/pdf')
            
            if not drive_files:
                return {
                    "success": True,
                    "message": "No se encontraron archivos PDF en Google Drive",
                    "migrated": 0,
                    "errors": []
                }
            
            # Procesar cada archivo
            migrated = 0
            errors = []
            
            for file_info in drive_files:
                file_id = file_info['id']
                file_name = file_info['name']
                
                try:
                    logger.info(f"📥 Procesando: {file_name}")
                    
                    # 1. Descargar de Drive
                    content = drive.download_file(file_id, file_name)
                    
                    # 2. Subir PDF a Blob
                    blob_name = blob_adapter.upload_pdf(content, file_name)
                    logger.success(f"✅ PDF subido a Blob: {blob_name}")
                    
                    # 3. Procesar y crear embeddings (se guardan automáticamente en Blob)
                    from io import BytesIO
                    file_obj = BytesIO(content)
                    
                    await doc_service.upload_document(
                        file=file_obj,
                        filename=file_name
                    )
                    
                    migrated += 1
                    logger.success(f"🎉 Migrado exitosamente: {file_name}")
                    
                except Exception as e:
                    error_msg = f"{file_name}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"❌ Error migrando {file_name}: {e}")
            
            return {
                "success": True,
                "message": f"Migración completada: {migrated}/{len(drive_files)} archivos",
                "total_files": len(drive_files),
                "migrated": migrated,
                "errors": errors
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error en migración: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Endpoint para obtener detalles de un CV específico
    @app.get("/api/v1/cv/detail", tags=["CV"])
    async def get_cv_detail(
        name: str,
        rag_service: RAGAgentService = Depends(get_rag_service)
    ):
        """
        🔍 Obtiene información detallada de un candidato específico.
        
        Parámetros:
        - name: Nombre completo del candidato
        
        Retorna un resumen estructurado del CV con:
        - Nombre completo
        - Experiencia laboral resumida (últimos 3 trabajos)
        - Habilidades principales
        - Educación
        - Años de experiencia
        - Seniority estimado
        """
        try:
            # Construir query específica para extraer información estructurada
            query = f"""Analiza la información del CV de {name} y proporciona:

1. Nombre completo
2. Experiencia laboral (últimas 3 posiciones con empresa y cargo)
3. Habilidades técnicas principales (top 5)
4. Educación (último título)
5. Años totales de experiencia
6. Nivel de seniority (Junior/Semi-Senior/Senior)

Formato tu respuesta de manera estructurada y concisa."""

            # Llamar al servicio RAG
            response = await rag_service.query(
                query=query,
                session_id=f"cv_detail_{name}",
                conversation_history=[]
            )
            
            # Extraer información de las fuentes
            sources = []
            for source in response.get("sources", []):
                sources.append({
                    "document": source.get("document", ""),
                    "page": source.get("page", 0),
                    "relevance": source.get("relevance_score", 0.0)
                })
            
            return {
                "name": name,
                "content": response.get("response", ""),
                "sources": sources,
                "chunk_count": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo detalles del CV de {name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    logger.info("✅ Aplicación FastAPI configurada con Azure Blob Storage")
    return app

# Para desarrollo local
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
