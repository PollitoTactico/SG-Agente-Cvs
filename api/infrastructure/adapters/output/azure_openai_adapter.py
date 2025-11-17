"""
Adaptador para Azure OpenAI.
Implementa el puerto LLM usando LangChain.
"""
from typing import List, Dict
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from api.application.output.port.llm_port import LLMPort
from api.utils.config import settings
from api.utils.logger import setup_logger

logger = setup_logger(__name__)


class AzureOpenAIAdapter(LLMPort):
    """
    Adaptador para Azure OpenAI usando LangChain.
    """
    
    def __init__(self):
        """Inicializa el adaptador."""
        # Configurar el modelo de chat
        self.chat_model = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Configurar embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            deployment="text-embedding-ada-002",  # Ajustar según tu deployment
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        # Template para el prompt RAG
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{context}\n\nPregunta: {question}")
        ])
        
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
            
            # Convertir historial al formato de LangChain
            history_messages = []
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history_messages.append(AIMessage(content=msg["content"]))
            
            # Crear la cadena
            chain = self.prompt_template | self.chat_model
            
            # Ejecutar
            response = await chain.ainvoke({
                "context": context_text,
                "question": prompt,
                "chat_history": history_messages
            })
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {str(e)}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.
        """
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generando embeddings: {str(e)}")
            raise
