"""
Servicio RAG Agent: Implementa la lógica del agente RAG.
"""
from typing import Dict, List
from uuid import uuid4
from datetime import datetime
import re

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
            initial_top_k = 25  # Aumentado para recuperar más secciones del CV
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
            
            # 5. Agrupar por persona y tomar top 10 para tener más contexto
            final_docs = self._group_by_person_and_select_top(filtered_docs, top_n=10)
            
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
        Extrae el nombre de la persona desde la query.
        Ej: "dime que certificaciones tiene gorky palacios" -> "gorky palacios"
        """
        # Patrones comunes de preguntas sobre personas
        patterns = [
            r'(?:sobre|de|tiene|posee|para)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)',
            r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                nombre = match.group(1).strip()
                if len(nombre.split()) >= 2:  # Al menos nombre y apellido
                    return nombre.lower()  # Normalizar a minúsculas para comparación
        
        # Si no se encuentra patrón, buscar palabras capitalizadas consecutivas
        words = query.split()
        capitalized_sequence = []
        for word in words:
            # Limpiar puntuación
            word_clean = re.sub(r'[^\w\sÁÉÍÓÚÑáéíóúñ]', '', word)
            if word_clean and word_clean[0].isupper():
                capitalized_sequence.append(word_clean)
            elif capitalized_sequence and len(capitalized_sequence) >= 2:
                # Encontramos una secuencia
                return ' '.join(capitalized_sequence).lower()
            else:
                capitalized_sequence = []
        
        # Verificar si la última secuencia es válida
        if len(capitalized_sequence) >= 2:
            return ' '.join(capitalized_sequence).lower()
        
        return ""
    
    def _filter_and_rerank_documents(self, documents, nombre_buscado: str):
        """
        Filtra y re-rankea documentos por relevancia.
        
        1. Si se detectó un nombre, filtra chunks de esa persona
        2. Re-rankea por score de similitud
        3. Penaliza chunks que no contengan el nombre buscado
        """
        filtered = []
        
        for doc in documents:
            # Obtener metadata
            nombre_doc = doc.metadata.get("nombre_completo", "").lower()
            content_lower = doc.content.lower()
            
            # FILTRO 1: Si se detectó un nombre específico, validar coincidencia
            if nombre_buscado:
                # Separar el nombre buscado en palabras
                nombre_buscado_parts = nombre_buscado.split()
                
                # Comparar nombre buscado con el nombre del documento
                # Verificar si al menos 1 palabra del nombre coincide (más flexible)
                matches = sum(1 for part in nombre_buscado_parts if part in nombre_doc)
                
                # También verificar en el contenido
                content_matches = sum(1 for part in nombre_buscado_parts if part in content_lower)
                
                # Si no hay NINGUNA coincidencia, descartar
                # Cambiado: solo requiere 1 coincidencia en vez de 2
                if matches == 0 and content_matches == 0:
                    logger.debug(f"Descartando chunk de '{nombre_doc}' - no coincide con '{nombre_buscado}'")
                    continue
            
            # FILTRO 2: Verificar que el contenido no esté vacío
            if not doc.content.strip():
                continue
            
            # BOOST: Aumentar score si el nombre buscado aparece en el contenido
            boost_factor = 1.0
            if nombre_buscado:
                nombre_buscado_parts = nombre_buscado.split()
                content_matches = sum(1 for part in nombre_buscado_parts if part in content_lower)
                nombre_matches = sum(1 for part in nombre_buscado_parts if part in nombre_doc)
                
                # Más boost si coincide en metadata de nombre
                if nombre_matches > 0:
                    boost_factor = 1.5  # +50% si está en el nombre
                elif content_matches > 0:
                    boost_factor = 1.2  # +20% si está en el contenido
            
            # Ajustar score
            adjusted_score = doc.score * boost_factor
            
            # Crear nuevo documento con score ajustado
            filtered.append({
                'doc': doc,
                'adjusted_score': adjusted_score
            })
        
        # Ordenar por score ajustado (descendente)
        filtered.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        # Retornar solo los documentos
        return [item['doc'] for item in filtered]
    
    def _group_by_person_and_select_top(self, documents, top_n: int = 10):
        """
        Agrupa documentos por persona y selecciona los top_n más relevantes.
        Si todos los docs son de la misma persona, retorna todos (hasta top_n).
        """
        if not documents:
            return []
        
        # Agrupar por nombre
        by_person = {}
        for doc in documents:
            nombre = doc.metadata.get("nombre_completo", "Desconocido")
            if nombre not in by_person:
                by_person[nombre] = []
            by_person[nombre].append(doc)
        
        # Si solo hay una persona, retornar todos sus chunks (hasta top_n)
        if len(by_person) == 1:
            return documents[:top_n]
        
        # Si hay múltiples personas, priorizar la que tenga más chunks
        logger.info(f"Detectadas {len(by_person)} personas diferentes en resultados")
        
        # Ordenar personas por cantidad de chunks (más chunks = más relevante)
        sorted_persons = sorted(by_person.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Tomar la persona principal (con más chunks)
        main_person = sorted_persons[0][0]
        main_person_docs = sorted_persons[0][1]
        
        logger.info(f"Persona principal seleccionada: {main_person} ({len(main_person_docs)} chunks)")
        
        # Retornar top_n de la persona principal
        return main_person_docs[:top_n]
