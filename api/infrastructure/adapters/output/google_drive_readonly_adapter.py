"""
Adaptador de Google Drive - SOLO LECTURA para migración.
"""
import io
from typing import List, Dict, Any, Optional
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class GoogleDriveAdapter:
    """Adaptador para leer archivos de Google Drive (solo lectura)."""
    
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self, credentials_path: str, folder_id: str):
        """
        Inicializa el adaptador de Google Drive.
        
        Args:
            credentials_path: Ruta al archivo JSON de credenciales
            folder_id: ID del folder de Google Drive
        """
        self.credentials_path = credentials_path
        self.folder_id = folder_id
        self.service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Inicializa el servicio de Google Drive API."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.SCOPES
            )
            
            self.service = build('drive', 'v3', credentials=credentials)
            logger.info("✅ Google Drive API inicializado (solo lectura)")
            
        except Exception as e:
            logger.error(f"❌ Error al inicializar Google Drive API: {str(e)}")
            raise
    
    def list_files_in_folder(self, mime_type: str = 'application/pdf') -> List[Dict[str, Any]]:
        """
        Lista archivos en el folder de Google Drive.
        
        Args:
            mime_type: Tipo MIME de archivos a listar
            
        Returns:
            Lista de archivos con metadata
        """
        try:
            query = f"'{self.folder_id}' in parents and trashed=false"
            if mime_type:
                query += f" and mimeType='{mime_type}'"
            
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, size, modifiedTime)',
                pageSize=100
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"📁 {len(files)} archivos encontrados en Google Drive")
            return files
            
        except HttpError as e:
            logger.error(f"❌ Error listando archivos: {str(e)}")
            raise
    
    def download_file(self, file_id: str, file_name: str) -> bytes:
        """
        Descarga un archivo de Google Drive.
        
        Args:
            file_id: ID del archivo en Drive
            file_name: Nombre del archivo (para logging)
            
        Returns:
            Contenido del archivo en bytes
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(f"⬇️ Descargando {file_name}: {int(status.progress() * 100)}%")
            
            file_buffer.seek(0)
            content = file_buffer.read()
            
            logger.success(f"✅ Descargado: {file_name} ({len(content)} bytes)")
            return content
            
        except HttpError as e:
            logger.error(f"❌ Error descargando {file_name}: {str(e)}")
            raise
