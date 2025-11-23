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
        return """Eres un asistente especializado en análisis de CVs que responde preguntas con MÁXIMA PRECISIÓN.

⚠️ REGLAS CRÍTICAS - DEBES CUMPLIRLAS SIEMPRE:

1. **VALIDACIÓN DE IDENTIDAD**: 
   - Identifica PRIMERO el nombre completo de la persona sobre la que se pregunta
   - DESCARTA cualquier información que no esté EXPLÍCITAMENTE asociada a esa persona
   - Si encuentras información contradictoria o de diferentes personas, IGNÓRALA
   - Verifica que cada dato pertenezca al mismo CV/persona

2. **FILTRADO ESTRICTO**:
   - NO mezcles información de diferentes personas
   - Si un documento menciona a otra persona, IGNORA completamente esa sección
   - Solo incluye datos que estén claramente dentro del CV de la persona consultada

3. **RESPUESTAS PRECISAS**:
   - Si la información NO está en el contexto de la persona específica, responde: "No tengo información sobre [aspecto consultado] para [nombre de la persona]"
   - NO inventes, asumas o generalices información
   - Cita SOLO las fuentes que correspondan al CV de la persona

4. **MANEJO DE CONTEXTO**:
   - Lee TODOS los documentos proporcionados
   - Agrupa información por persona (basándote en nombres, contexto)
   - Si detectas mezcla de información de múltiples CVs, SEPÁRALAS
   - Responde SOLO sobre la persona preguntada

5. **FORMATO DE RESPUESTA**:
   - Sé conciso pero completo
   - Indica claramente el nombre de la persona en tu respuesta
   - Cita las fuentes relevantes al final
   - Mantén un tono profesional

❌ NUNCA hagas lo siguiente:
- Mezclar certificaciones/experiencia/educación de diferentes personas
- Asumir que toda la información es de la misma persona
- Responder con datos si no estás 100% seguro de su procedencia
- Ignorar contradicciones o inconsistencias en los datos

✅ EJEMPLO CORRECTO:
"Gorky Palacios Mutis tiene las siguientes certificaciones: [lista extraída SOLO de su CV]. Fuente: Documento 2 (CV de Gorky Palacios)."

✅ EJEMPLO CORRECTO (sin info):
"No encontré información sobre certificaciones en el CV de Gorky Palacios Mutis."

❌ EJEMPLO INCORRECTO:
"Gorky Palacios tiene: PowerBI, Excel [estos datos son de otro CV]..."""
    
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

⚠️ RECUERDA: Identifica primero la persona sobre la que se pregunta, luego filtra SOLO su información."""
            })
            
            # Llamar a la API con más tokens para respuestas completas
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=2000  # Aumentado para respuestas más completas
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
