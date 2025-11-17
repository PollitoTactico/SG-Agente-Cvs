"""
Tests para el servicio RAG Agent.
"""
import pytest
from unittest.mock import AsyncMock, Mock
from api.application.service.rag_agent_service import RAGAgentService
from api.application.input.port.rag_agent_port import QueryRequest


@pytest.fixture
def mock_llm_port():
    """Mock del puerto LLM."""
    llm = AsyncMock()
    llm.generate_embeddings.return_value = [[0.1] * 1536]
    llm.generate_response.return_value = "Esta es una respuesta de prueba"
    return llm


@pytest.fixture
def mock_vector_store_port():
    """Mock del puerto Vector Store."""
    vector_store = AsyncMock()
    vector_store.similarity_search.return_value = [
        {
            "document_id": "doc-123",
            "content": "Contenido de prueba",
            "filename": "test.pdf",
            "chunk_id": "chunk-1",
            "score": 0.95
        }
    ]
    return vector_store


@pytest.fixture
def rag_service(mock_llm_port, mock_vector_store_port):
    """Fixture del servicio RAG."""
    return RAGAgentService(mock_llm_port, mock_vector_store_port)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_success(rag_service):
    """Test de consulta exitosa."""
    request = QueryRequest(
        query="¿Qué es la arquitectura hexagonal?",
        session_id="test-session"
    )
    
    response = await rag_service.query(request)
    
    assert response.answer == "Esta es una respuesta de prueba"
    assert len(response.sources) == 1
    assert response.sources[0]["document_id"] == "doc-123"
    assert response.session_id == "test-session"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_generates_session_id(rag_service):
    """Test que verifica generación automática de session_id."""
    request = QueryRequest(
        query="Test query",
        session_id=None
    )
    
    response = await rag_service.query(request)
    
    assert response.session_id is not None
    assert len(response.session_id) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clear_history_success(rag_service):
    """Test de limpieza de historial."""
    # Primero crear una sesión
    request = QueryRequest(
        query="Test query",
        session_id="test-session"
    )
    await rag_service.query(request)
    
    # Limpiar historial
    result = await rag_service.clear_history("test-session")
    
    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clear_nonexistent_session(rag_service):
    """Test de limpieza de sesión inexistente."""
    result = await rag_service.clear_history("nonexistent-session")
    
    assert result is False
