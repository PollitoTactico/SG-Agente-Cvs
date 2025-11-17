"""
Puerto de salida: Interface para el LLM (Azure OpenAI).
Define el contrato para interactuar con el modelo de lenguaje.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LLMPort(ABC):
    """
    Puerto de salida para el LLM.
    Abstrae la interacción con Azure OpenAI.
    """
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: List[str],
        chat_history: List[Dict[str, str]] | None = None
    ) -> str:
        """
        Genera una respuesta usando el LLM.
        
        Args:
            prompt: Pregunta del usuario
            context: Contexto recuperado de la base vectorial
            chat_history: Historial de conversación
            
        Returns:
            Respuesta generada
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para textos.
        
        Args:
            texts: Textos a convertir en embeddings
            
        Returns:
            Lista de vectores de embeddings
        """
        pass
