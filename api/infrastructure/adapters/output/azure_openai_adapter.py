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
        
        logger.info("Azure OpenAI Adapter inicializado")
    
    def _get_system_prompt(self) -> str:
        """Retorna el prompt del sistema."""
        return """Eres un asistente útil que responde preguntas basándote en el contexto proporcionado.

INSTRUCCIONES:
1. Usa únicamente la información del contexto para responder
2. Si la respuesta no está en el contexto, di "No tengo información suficiente para responder esa pregunta"
3. Sé conciso pero completo en tus respuestas
4. Cita las fuentes cuando sea relevante
5. Mantén un tono profesional y amigable"""
    
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
            # Preparar el contexto
            context_text = "\n\n".join([f"[Documento {i+1}]\n{ctx}" for i, ctx in enumerate(context)])
            
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
            
            # Agregar contexto y pregunta
            messages.append({
                "role": "user",
                "content": f"{context_text}\n\nPregunta: {prompt}"
            })
            
            # Llamar a la API
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
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
                model=self.deployment,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generando embeddings: {str(e)}")
            raise
