import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import json
from main import app

client = TestClient(app)

# Mock environment variables for testing
os.environ["VAPI_API_KEY"] = "test_vapi_key"
os.environ["RETELL_API_KEY"] = "test_retell_key"

# Test data
test_agent_config = {
    "name": "Test Agent",
    "description": "A test agent",
    "llm_model": "gpt-4",
    "system_prompt": "You are a test agent.",
    "voice": {
        "provider": "eleven_labs",
        "voice_id": "test-voice-id",
        "settings": {
            "stability": 0.5
        }
    },
    "webhook_url": "https://example.com/webhook",
    "metadata": {
        "test": "value"
    }
}

# Mock responses
vapi_response = {
    "id": "vapi-test-id",
    "name": "Test Agent",
    "description": "A test agent",
    "model": "gpt-4",
    "voice_id": "test-voice-id",
    "created_at": "2023-01-01T00:00:00Z"
}

retell_response = {
    "id": "retell-test-id",
    "name": "Test Agent",
    "description": "A test agent",
    "llm": {
        "provider": "openai",
        "model": "gpt-4"
    },
    "voice": {
        "provider": "elevenlabs",
        "voice_id": "test-voice-id"
    },
    "created_at": "2023-01-01T00:00:00Z"
}

@pytest.fixture
def mock_httpx_client():
    with patch("httpx.AsyncClient") as mock_client:
        async_client = MagicMock()
        mock_client.return_value.__aenter__.return_value = async_client
        yield async_client

def test_api_health():
    """Test that the API is up and running"""
    response = client.get("/")
    assert response.status_code == 404  # No root endpoint defined

@pytest.mark.asyncio
async def test_create_vapi_agent(mock_httpx_client):
    """Test creating a Vapi agent"""
    # Configure mock
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = vapi_response
    mock_httpx_client.post.return_value = mock_response
    
    # Create test data with Vapi provider
    request_data = test_agent_config.copy()
    request_data["provider"] = "vapi"
    
    # Make request
    response = client.post("/api/agents", json=request_data)
    
    # Assertions
    assert response.status_code == 201
    response_data = response.json()
    assert response_data["id"] == "vapi-test-id"
    assert response_data["provider"] == "vapi"
    
    # Check that the correct API was called
    mock_httpx_client.post.assert_called_once()
    args, kwargs = mock_httpx_client.post.call_args
    assert args[0] == "https://api.vapi.ai/assistants"
    assert kwargs["headers"]["Authorization"] == "Bearer test_vapi_key"

@pytest.mark.asyncio
async def test_create_retell_agent(mock_httpx_client):
    """Test creating a Retell agent"""
    # Configure mock
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = retell_response
    mock_httpx_client.post.return_value = mock_response
    
    # Create test data with Retell provider
    request_data = test_agent_config.copy()
    request_data["provider"] = "retell"
    
    # Make request
    response = client.post("/api/agents", json=request_data)
    
    # Assertions
    assert response.status_code == 201
    response_data = response.json()
    assert response_data["id"] == "retell-test-id"
    assert response_data["provider"] == "retell"
    
    # Check that the correct API was called
    mock_httpx_client.post.assert_called_once()
    args, kwargs = mock_httpx_client.post.call_args
    assert args[0] == "https://api.retellai.com/agents"
    assert kwargs["headers"]["Authorization"] == "Bearer test_retell_key"

@pytest.mark.asyncio
async def test_invalid_provider():
    """Test providing an invalid provider"""
    request_data = test_agent_config.copy()
    request_data["provider"] = "invalid"
    
    response = client.post("/api/agents", json=request_data)
    
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_api_key_not_found():
    """Test behavior when API keys are missing"""
    with patch.dict(os.environ, {"VAPI_API_KEY": "", "RETELL_API_KEY": ""}):
        request_data = test_agent_config.copy()
        request_data["provider"] = "vapi"
        
        response = client.post("/api/agents", json=request_data)
        
        assert response.status_code == 500
        assert "not configured" in response.json()["detail"]

@pytest.mark.asyncio
async def test_provider_api_error(mock_httpx_client):
    """Test handling of provider API errors"""
    # Configure mock to return an error
    mock_error_response = MagicMock()
    mock_error_response.status_code = 400
    mock_error_response.text = "Bad request"
    mock_error_response.json.return_value = {"message": "Invalid parameters"}
    mock_error_response.raise_for_status.side_effect = Exception("HTTP Error")
    mock_httpx_client.post.return_value = mock_error_response
    
    # Create test data
    request_data = test_agent_config.copy()
    request_data["provider"] = "vapi"
    
    # Make request
    response = client.post("/api/agents", json=request_data)
    
    # Assertions
    assert response.status_code == 500
    assert "API error" in response.json()["detail"]