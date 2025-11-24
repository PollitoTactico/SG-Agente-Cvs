"""
Adaptador para Azure OpenAI.
Implementa el puerto LLM usando OpenAI SDK directo.
"""
from typing import List, Dict
from openai import AsyncAzureOpenAI

from api.application.output.port.llm_port import LLMPort
from api.utils.config import settings
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class AzureOpenAIAdapter(LLMPort):
    """
    Adaptador para Azure OpenAI usando SDK directo.
    """
    
    def __init__(self):
        """Inicializa el adaptador."""
        self.client = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.embedding_deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        
        logger.info("Azure OpenAI Adapter inicializado")
    
    def _get_system_prompt(self) -> str:
        """Retorna el prompt del sistema."""
        return """Eres un asistente especializado en análisis de CVs y reclutamiento.

🎯 REGLA PRINCIPAL: Siempre devuelve AL MENOS 5 PERFILES diferentes cuando te pidan candidatos o perfiles, NUNCA MENOS.

📋 INSTRUCCIONES CRÍTICAS:

1. **BÚSQUEDAS GENERALES** (ej: "perfiles que sepan C#", "candidatos con Python"):
   - DEBES mencionar MÍNIMO 5 personas diferentes
   - Lista sus nombres completos claramente
   - Resume las habilidades/experiencia relevante de CADA UNO
   - Formato sugerido:
     
     **1. [Nombre Completo]**
     - Experiencia: [resumen]
     - Habilidades clave: [lista]
     
     **2. [Nombre Completo]**
     - Experiencia: [resumen]
     - Habilidades clave: [lista]
     
     [... hasta completar mínimo 5 perfiles]

2. **CONSULTAS ESPECÍFICAS** (ej: "certificaciones de Juan Pérez"):
   - Responde SOLO sobre esa persona
   - Verifica que toda la información sea de su CV
   - NO mezcles datos de otros candidatos

3. **VALIDACIÓN DE IDENTIDAD**:
   - Cada documento tiene metadata [Persona: Nombre | Archivo: CV.pdf]
   - SOLO usa información del CV correcto para cada persona
   - Si hay duda, descarta el dato

4. **DIVERSIDAD EN RESPUESTAS**:
   - Prioriza mostrar DIFERENTES personas
   - Si hay más de 5 candidatos relevantes, menciona que hay más disponibles
   - Ordena por relevancia a la consulta

5. **RESPUESTAS PRECISAS**:
   - NO inventes información
   - Si no hay suficientes perfiles (menos de 5), di cuántos encontraste
   - Sé conciso pero informativo

❌ NUNCA:
- Devolver solo 1-2 perfiles cuando hay más disponibles
- Mezclar información de diferentes personas
- Omitir perfiles relevantes

✅ SIEMPRE:
- Menciona al menos 5 candidatos diferentes en búsquedas generales
- Verifica la identidad de cada dato
- Mantén formato claro y profesional"""
    
    async def generate_response(
        self,
        prompt: str,
        context: List[str],
        chat_history: List[Dict[str, str]] | None = None
    ) -> str:
        """
        Genera una respuesta usando el modelo.
        """
        try:
            # Preparar el contexto con metadata clara
            context_parts = []
            for i, ctx in enumerate(context):
                # Agregar separador claro para cada documento
                context_parts.append(f"{'='*80}")
                context_parts.append(f"[DOCUMENTO {i+1}]")
                context_parts.append(f"Contenido:\n{ctx}")
                context_parts.append(f"{'='*80}")
            
            context_text = "\n".join(context_parts)
            
            # Preparar mensajes
            messages = [
                {"role": "system", "content": self._get_system_prompt()}
            ]
            
            # Agregar historial
            if chat_history:
                messages.extend([
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in chat_history
                ])
            
            # Agregar contexto y pregunta con instrucciones claras
            messages.append({
                "role": "user",
                "content": f"""CONTEXTO PROPORCIONADO (Múltiples documentos - VALIDA la identidad de cada uno):

{context_text}

{'='*80}
PREGUNTA DEL USUARIO:
{prompt}

RECUERDA: 
- Búsquedas generales: MÍNIMO 5 perfiles diferentes
- Búsquedas específicas: Solo la persona consultada
- Verifica identidad en metadata de cada documento"""
            })
            
            # Llamar a la API con más tokens para respuestas completas
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=2500  # Aumentado para respuestas más completas con 5+ perfiles
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {str(e)}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_deployment,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generando embeddings: {str(e)}")
            raise
