"""
Servicio Document Manager: Gestiona la carga y eliminación de documentos.
"""
from typing import List, BinaryIO
from uuid import uuid4
from datetime import datetime
import re

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
        
        1. Verifica si el documento ya existe (por nombre de archivo)
        2. Lee el contenido del PDF
        3. Extrae nombre de la persona
        4. Divide en chunks inteligentes por secciones
        5. Genera embeddings
        6. Almacena en vector store con metadata enriquecida
        """
        logger.info(f"Procesando documento: {filename}")
        
        try:
            # VALIDACIÓN: Verificar si el documento ya existe
            if hasattr(self.vector_store, 'document_exists_by_filename'):
                exists = await self.vector_store.document_exists_by_filename(filename)
                if exists:
                    # Obtener información del documento existente
                    doc_info = None
                    if hasattr(self.vector_store, 'get_document_info_by_filename'):
                        doc_info = await self.vector_store.get_document_info_by_filename(filename)
                    
                    warning_msg = f"⚠️ ADVERTENCIA: El documento '{filename}' ya existe en el sistema."
                    if doc_info:
                        warning_msg += f"\n   - Document ID: {doc_info.get('document_id', 'N/A')}"
                        warning_msg += f"\n   - Persona: {doc_info.get('nombre_completo', 'Desconocido')}"
                        warning_msg += f"\n   - Fecha de carga: {doc_info.get('upload_date', 'N/A')}"
                    warning_msg += "\n   - El documento NO fue procesado nuevamente para evitar duplicados."
                    
                    logger.warning(warning_msg)
                    
                    return DocumentUploadResponse(
                        document_id=doc_info.get('document_id', '') if doc_info else '',
                        filename=filename,
                        status="duplicate",
                        message=warning_msg
                    )
            
            # Leer contenido del archivo
            content = file.read()
            document_id = str(uuid4())
            
            # Extraer texto del PDF
            text = await self._extract_text_from_pdf(content)
            
            # Extraer nombre completo de la persona
            nombre_completo = self._extract_full_name(text, filename)
            logger.info(f"Nombre detectado: {nombre_completo}")
            
            # Dividir en chunks inteligentes por secciones
            chunks_data = self._create_smart_chunks(text, nombre_completo)
            logger.info(f"Documento dividido en {len(chunks_data)} chunks inteligentes")
            
            # Extraer solo el texto de cada chunk para embeddings
            chunks_text = [chunk["text"] for chunk in chunks_data]
            
            # Generar embeddings
            embeddings = await self.llm.generate_embeddings(chunks_text)
            
            # Preparar metadata enriquecida para cada chunk
            chunk_metadatas = [
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": f"{document_id}_{i}",
                    "chunk_index": i,
                    "nombre_completo": nombre_completo,  # NUEVO: Nombre de la persona
                    "seccion_cv": chunk_data["section"],  # NUEVO: Sección del CV
                    "tipo_info": chunk_data["type"],  # NUEVO: Tipo de información
                    "upload_date": datetime.utcnow().isoformat(),
                    "total_chunks": len(chunks_data),
                    **(metadata or {})
                }
                for i, chunk_data in enumerate(chunks_data)
            ]
            
            # Almacenar en vector store
            await self.vector_store.add_documents(
                documents=chunks_text,
                metadatas=chunk_metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Documento {filename} procesado exitosamente. ID: {document_id}, Persona: {nombre_completo}")
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                status="success",
                message=f"Documento procesado: {len(chunks_data)} chunks indexados para {nombre_completo}"
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
        Método legado - usar _create_smart_chunks preferentemente.
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
    
    def _extract_full_name(self, text: str, filename: str) -> str:
        """
        Extrae el nombre completo de la persona del CV.
        Intenta varias estrategias:
        1. Del nombre del archivo
        2. De las primeras líneas del CV
        3. Patrones comunes de nombres
        """
        # Estrategia 1: Extraer del nombre del archivo
        # Ej: "CV_Juan_Perez.pdf" -> "Juan Perez"
        filename_clean = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        filename_clean = re.sub(r'\b(cv|curriculum|vitae|resume|hoja de vida)\b', '', filename_clean, flags=re.IGNORECASE)
        filename_clean = filename_clean.strip()
        
        # Validar si el nombre del archivo tiene formato de nombre (al menos 2 palabras)
        if filename_clean and len(filename_clean.split()) >= 2:
            # Capitalizar correctamente
            nombre_from_file = ' '.join([word.capitalize() for word in filename_clean.split() if word])
            if len(nombre_from_file) > 5:  # Al menos un nombre mínimamente válido
                logger.info(f"Nombre extraído del archivo: {nombre_from_file}")
                return nombre_from_file
        
        # Estrategia 2: Buscar en las primeras líneas del texto
        lines = text.split('\n')
        
        # Patrones comunes de nombres (mayoritariamente en mayúsculas o capitalizados)
        name_patterns = [
            r'^([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})$',  # Juan Carlos Perez Gomez
            r'^([A-ZÁÉÍÓÚÑ -ſ\s]{10,60})$',  # NOMBRES EN MAYÚSCULAS
        ]
        
        # Buscar en las primeras 10 líneas
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            # Ignorar líneas muy cortas o muy largas
            if len(line) < 6 or len(line) > 100:
                continue
            
            # Ignorar líneas con caracteres especiales de CV
            if any(char in line for char in ['@', 'tel', 'phone', ':', 'http', 'www', '+', '|']):
                continue
            
            # Intentar patrones
            for pattern in name_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    nombre = match.group(1).strip()
                    # Capitalizar correctamente si está en mayúsculas
                    if nombre.isupper():
                        nombre = nombre.title()
                    logger.info(f"Nombre extraído del texto (línea {i+1}): {nombre}")
                    return nombre
        
        # Estrategia 3: Fallback - usar el nombre del archivo limpio o "Desconocido"
        if filename_clean:
            return filename_clean.title()
        
        logger.warning("No se pudo extraer el nombre completo, usando 'Desconocido'")
        return "Desconocido"
    
    def _create_smart_chunks(self, text: str, nombre_completo: str) -> List[dict]:
        """
        Divide el CV en chunks inteligentes respetando secciones.
        Retorna una lista de diccionarios con: {"text": str, "section": str, "type": str}
        """
        chunks = []
        
        # Definir patrones de secciones comunes en CVs (español e inglés)
        section_patterns = [
            # Experiencia
            (r'(?:experiencia|experience|trabajo|employment|historial laboral)', 'experiencia_laboral', 'experiencia'),
            # Educación
            (r'(?:educaci[oó]n|education|formaci[oó]n|academic|estudios)', 'educacion', 'educacion'),
            # Certificaciones
            (r'(?:certificaciones|certifications|certificados|certificates|cursos|courses|capacitaciones)', 'certificaciones', 'certificaciones'),
            # Habilidades
            (r'(?:habilidades|skills|competencias|conocimientos)', 'habilidades', 'habilidades'),
            # Idiomas
            (r'(?:idiomas|languages)', 'idiomas', 'idiomas'),
            # Perfil/Resumen
            (r'(?:perfil|profile|resumen|summary|objetivo|objective|sobre m[ií]|about)', 'perfil', 'perfil'),
            # Proyectos
            (r'(?:proyectos|projects)', 'proyectos', 'proyectos'),
            # Referencias
            (r'(?:referencias|references)', 'referencias', 'referencias'),
        ]
        
        # Dividir el texto en líneas
        lines = text.split('\n')
        
        # Identificar secciones
        sections = []
        current_section = {'name': 'header', 'type': 'informacion_general', 'start': 0, 'content': []}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Detectar si la línea es un título de sección
            section_found = False
            for pattern, section_name, section_type in section_patterns:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    # Guardar la sección anterior
                    if current_section['content']:
                        current_section['end'] = i
                        sections.append(current_section)
                    
                    # Iniciar nueva sección
                    current_section = {
                        'name': section_name,
                        'type': section_type,
                        'start': i,
                        'content': [line]
                    }
                    section_found = True
                    break
            
            if not section_found:
                current_section['content'].append(line)
        
        # Agregar la última sección
        if current_section['content']:
            current_section['end'] = len(lines)
            sections.append(current_section)
        
        logger.info(f"Secciones detectadas: {[s['name'] for s in sections]}")
        
        # Convertir cada sección en chunks
        for section in sections:
            section_text = '\n'.join(section['content']).strip()
            
            # Si la sección es muy pequeña, crear un solo chunk
            if len(section_text) <= self.chunk_size:
                if section_text:  # Solo si no está vacío
                    chunks.append({
                        'text': f"[{nombre_completo} - {section['name'].upper()}]\n\n{section_text}",
                        'section': section['name'],
                        'type': section['type']
                    })
            else:
                # Dividir sección grande en sub-chunks manteniendo el contexto
                sub_chunks = self._split_large_section(section_text, self.chunk_size, self.chunk_overlap)
                for idx, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'text': f"[{nombre_completo} - {section['name'].upper()} (parte {idx+1}/{len(sub_chunks)})]\n\n{sub_chunk}",
                        'section': section['name'],
                        'type': section['type']
                    })
        
        # Si no se detectó ninguna sección, usar chunking simple con metadata
        if not chunks:
            logger.warning("No se detectaron secciones, usando chunking simple")
            simple_chunks = self._create_chunks(text)
            chunks = [
                {
                    'text': f"[{nombre_completo}]\n\n{chunk}",
                    'section': 'general',
                    'type': 'general'
                }
                for chunk in simple_chunks
            ]
        
        return chunks
    
    def _split_large_section(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Divide una sección grande en chunks con overlap.
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
