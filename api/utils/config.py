"""
Configuración de la aplicación usando Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = Field(
        default="https://ai-soporte-4783.cognitiveservices.azure.com/",
        description="Endpoint de Azure OpenAI"
    )
    AZURE_OPENAI_API_KEY: str = Field(
        default="3luREZD3QxwIBheELqBUcBEKTFHECAuINYgylyLcQmOxHcSjkE1mJQQJ99BBACHYHv6XJ3w3AAAAACOGlUHU",
        description="API Key de Azure OpenAI"
    )
    AZURE_OPENAI_DEPLOYMENT_NAME: str = Field(
        default="gpt-5-chat",
        description="Nombre del deployment"
    )
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2025-01-01-preview",
        description="Versión de la API"
    )
    
    # Azure AI Search
    AZURE_SEARCH_ENDPOINT: str = Field(
        default="",
        description="Endpoint de Azure AI Search"
    )
    AZURE_SEARCH_API_KEY: str = Field(
        default="",
        description="API Key de Azure AI Search"
    )
    AZURE_SEARCH_INDEX_NAME: str = Field(
        default="pdf-knowledge-base",
        description="Nombre del índice"
    )
    
    # Application
    ENVIRONMENT: str = Field(default="development", description="Entorno de ejecución")
    LOG_LEVEL: str = Field(default="INFO", description="Nivel de logging")
    
    # RAG Configuration
    CHUNK_SIZE: int = Field(default=1000, description="Tamaño de chunks para documentos")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap entre chunks")
    TOP_K_RESULTS: int = Field(default=5, description="Número de resultados a recuperar")
    
    # Azure Blob Storage
    AZURE_STORAGE_CONNECTION_STRING: str = Field(
        default="",
        description="Connection string de Azure Storage Account"
    )
    AZURE_STORAGE_CONTAINER_PDFS: str = Field(
        default="pdfs",
        description="Nombre del contenedor para PDFs"
    )
    AZURE_STORAGE_CONTAINER_EMBEDDINGS: str = Field(
        default="embeddings",
        description="Nombre del contenedor para embeddings JSON"
    )
    AZURE_STORAGE_CONTAINER_CACHE: str = Field(
        default="cache",
        description="Nombre del contenedor para cache temporal"
    )
    
    # Google Drive (solo para migración)
    GOOGLE_DRIVE_CREDENTIALS_PATH: str = Field(
        default="./google-credentials.json",
        description="Ruta al archivo de credenciales (solo lectura)"
    )
    GOOGLE_DRIVE_FOLDER_ID: str = Field(
        default="",
        description="ID del folder de Drive para migración"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
