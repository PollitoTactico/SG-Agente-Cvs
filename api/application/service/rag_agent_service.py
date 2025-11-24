"""
Servicio RAG Agent: Implementa la lógica del agente RAG.
"""
from typing import Dict, List
from uuid import uuid4
from datetime import datetime
import re
import unicodedata

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
        Procesa una consulta usando RAG mejorado.
        
        1. Detecta el nombre de la persona en la query
        2. Genera embedding de la consulta
        3. Recupera más documentos (top_k aumentado)
        4. Filtra y re-rankea por relevancia y nombre
        5. Agrupa por persona y documento
        6. Genera respuesta con contexto preciso
        7. Retorna respuesta con fuentes filtradas
        """
        logger.info(f"Procesando consulta: {request.query[:50]}...")
        
        # Obtener o crear session_id
        session_id = request.session_id or str(uuid4())
        
        # Obtener historial de la sesión
        chat_history = self.sessions.get(session_id, [])
        
        try:
            # 1. Detectar nombre de persona en la query
            nombre_buscado = self._extract_person_name_from_query(request.query)
            logger.info(f"Nombre detectado en query: {nombre_buscado}")
            
            # 2. Generar embedding de la consulta
            query_embeddings = await self.llm.generate_embeddings([request.query])
            query_embedding = query_embeddings[0]
            
            # 3. Buscar documentos similares (aumentamos top_k para mejor cobertura)
            # BÚSQUEDA HÍBRIDA: Vector + Keyword para mejor precisión
            initial_top_k = 200  # Aumentado para recuperar muchos más CVs diferentes
            documents = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=initial_top_k,
                filters=request.filters,
                query_text=request.query  # Búsqueda híbrida con keywords
            )
            
            logger.info(f"Recuperados {len(documents)} documentos iniciales")
            
            # 4. Filtrar y re-rankear documentos
            filtered_docs = self._filter_and_rerank_documents(
                documents=documents,
                nombre_buscado=nombre_buscado
            )
            
            logger.info(f"Después de filtrar: {len(filtered_docs)} documentos relevantes")
            
            # 5. Agrupar por persona - MÍNIMO 5 personas en búsquedas generales
            min_personas = 5 if not nombre_buscado else 1
            final_docs = self._group_by_person_and_select_top(
                filtered_docs, 
                top_n=25,  # Aumentado a 25 chunks totales
                min_personas=min_personas
            )
            
            # 6. Extraer contexto con información de metadata
            context = []
            for doc in final_docs:
                # Agregar metadata al contexto para que la IA tenga más info
                nombre = doc.metadata.get("nombre_completo", "Desconocido")
                seccion = doc.metadata.get("seccion_cv", "general")
                filename = doc.metadata.get("filename", "")
                
                context_entry = f"""[Persona: {nombre} | Archivo: {filename} | Sección: {seccion}]
{doc.content}"""
                context.append(context_entry)
            
            sources = [
                {
                    "document_id": doc.metadata.get("document_id", ""),
                    "filename": doc.metadata.get("filename", ""),
                    "score": doc.score,
                    "chunk_id": doc.id,
                    "nombre_completo": doc.metadata.get("nombre_completo", "Desconocido"),
                    "seccion_cv": doc.metadata.get("seccion_cv", "general")
                }
                for doc in final_docs
            ]
            
            # 7. Generar respuesta
            answer = await self.llm.generate_response(
                prompt=request.query,
                context=context,
                chat_history=chat_history
            )
            
            # 8. Actualizar historial
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
                    "documents_found": len(final_docs),
                    "initial_documents": len(documents),
                    "filtered_documents": len(filtered_docs),
                    "nombre_buscado": nombre_buscado
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
    
    def _extract_person_name_from_query(self, query: str) -> str:
        """
        Extrae nombre de persona si es una consulta específica.
        Usa Azure OpenAI para entender el lenguaje natural.
        """
        query_lower = query.lower()
        
        # Palabras de búsqueda general
        general_keywords = [
            'perfiles', 'personas', 'candidatos', 'cvs', 'empleados',
            'alguien', 'quien', 'quienes', 'dame', 'lista', 'muestra',
            'ayudame', 'busca', 'encuentra', 'hay', 'conocimientos'
        ]
        
        if any(keyword in query_lower for keyword in general_keywords):
            logger.info("🔍 Búsqueda general detectada")
            return ""
        
        # Patrones específicos de persona
        patterns = [
            r'(?:sobre|de|tiene|posee|para)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)',
            r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                nombre = match.group(1).strip()
                if len(nombre.split()) >= 2:
                    logger.info(f"👤 Persona específica detectada: {nombre}")
                    return nombre.lower()
        
        return ""
    
    def _filter_and_rerank_documents(self, documents, nombre_buscado: str):
        """
        Filtra y re-rankea documentos.
        - Si hay nombre específico: filtra por esa persona
        - Si es búsqueda general: retorna TODOS los documentos
        """
        if not nombre_buscado:
            # Búsqueda general: retornar todos ordenados por score
            documents.sort(key=lambda x: x.score, reverse=True)
            logger.info("📊 Búsqueda general: retornando todos los documentos")
            return documents
        
        # Búsqueda específica: filtrar por nombre
        filtered = []
        nombre_parts = nombre_buscado.split()
        
        logger.info(f"🔍 Filtrando por nombre: '{nombre_buscado}', partes: {nombre_parts}")
        
        for doc in documents:
            if not doc.content.strip():
                continue
            
            nombre_doc = doc.metadata.get("nombre_completo", "").lower()
            content_lower = doc.content.lower()
            
            # Normalizar para quitar acentos
            nombre_doc_norm = unicodedata.normalize('NFD', nombre_doc)
            nombre_doc_norm = ''.join(c for c in nombre_doc_norm if unicodedata.category(c) != 'Mn')
            
            # Verificar coincidencias
            matches = sum(1 for part in nombre_parts if part in nombre_doc_norm)
            content_matches = sum(1 for part in nombre_parts if part in content_lower)
            
            if matches > 0 or content_matches > 0:
                logger.debug(f"  ✅ Match: {nombre_doc} | matches={matches}, content={content_matches}, score={doc.score:.4f}")
                boost = 1.0 + (matches * 0.3) + (content_matches * 0.2)
                doc.score *= boost
                filtered.append(doc)
            else:
                logger.debug(f"  ❌ No match: {nombre_doc}")
        
        filtered.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"👤 Filtrado por persona: {len(filtered)} documentos de '{nombre_buscado}'")
        
        # Listar personas encontradas
        personas_filtradas = set(doc.metadata.get("nombre_completo", "") for doc in filtered)
        logger.info(f"📋 Personas en documentos filtrados: {list(personas_filtradas)[:5]}")
        
        return filtered
    
    def _group_by_person_and_select_top(self, documents, top_n: int = 25, min_personas: int = 5):
        """
        Selecciona documentos asegurando MÍNIMO min_personas diferentes.
        - Si es 1 persona: retorna sus top chunks
        - Si son múltiples: distribuye para alcanzar mínimo 5 personas diferentes
        
        Args:
            documents: Lista de documentos
            top_n: Total de chunks a retornar
            min_personas: Mínimo de personas diferentes a incluir (default: 5)
        """
        if not documents:
            return []
        
        # Agrupar por persona
        by_person = {}
        for doc in documents:
            nombre = doc.metadata.get("nombre_completo", "Desconocido")
            if nombre not in by_person:
                by_person[nombre] = []
            by_person[nombre].append(doc)
        
        personas_count = len(by_person)
        logger.info(f"👥 {personas_count} personas diferentes en resultados")
        
        # Si solo hay 1 persona o es búsqueda específica
        if personas_count == 1:
            return documents[:top_n]
        
        # GARANTIZAR MÍNIMO min_personas diferentes
        # Calcular chunks por persona para distribuir equitativamente
        if personas_count >= min_personas:
            chunks_per_person = max(3, top_n // personas_count)
        else:
            # Si hay menos personas disponibles, tomar más chunks de cada una
            chunks_per_person = max(5, top_n // personas_count)
            logger.warning(f"⚠️  Solo {personas_count} personas disponibles, se esperaban {min_personas}")
        
        result = []
        personas_incluidas = 0
        
        # Primero: asegurar al menos 1 chunk de cada persona hasta min_personas
        for nombre, docs in sorted(by_person.items(), key=lambda x: x[1][0].score, reverse=True):
            if personas_incluidas < min_personas or personas_count < min_personas:
                # Agregar chunks de esta persona
                result.extend(docs[:chunks_per_person])
                personas_incluidas += 1
        
        # Ordenar por score y limitar a top_n
        result.sort(key=lambda x: x.score, reverse=True)
        final_result = result[:top_n]
        
        # Contar personas únicas en el resultado final
        personas_finales = len(set(doc.metadata.get("nombre_completo", "Desconocido") for doc in final_result))
        logger.info(f"✅ Resultado final: {len(final_result)} chunks de {personas_finales} personas diferentes")
        
        return final_result
