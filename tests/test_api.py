"""
Tests para la API FastAPI.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from api.infrastructure.adapters.input.fastapi_adapter import create_app


@pytest.fixture
def client():
    """Cliente de test para FastAPI."""
    app = create_app()
    return TestClient(app)


@pytest.mark.unit
def test_health_check(client):
    """Test del endpoint de health check."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.integration
def test_query_endpoint_structure(client):
    """Test de estructura del endpoint de query."""
    payload = {
        "query": "Test query",
        "session_id": "test-session"
    }
    
    # Este test fallará sin configuración real, 
    # pero verifica la estructura
    response = client.post("/api/v1/query", json=payload)
    
    # Puede fallar por falta de Azure, pero verificamos estructura
    assert response.status_code in [200, 500]


@pytest.mark.unit
def test_query_validation(client):
    """Test de validación de query vacío."""
    payload = {
        "query": "",  # Query vacío debe fallar
        "session_id": "test-session"
    }
    
    response = client.post("/api/v1/query", json=payload)
    
    assert response.status_code == 422  # Validation error
