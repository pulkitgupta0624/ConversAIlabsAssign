from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import httpx
import os
from enum import Enum
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Unified AI Agent API",
    description="A unified API wrapper for Vapi.ai and Retell Agent services",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint that provides basic information about the API.
    """
    return {
        "name": "Unified AI Agent API",
        "version": "1.0.0",
        "description": "A unified API wrapper for Vapi.ai and Retell Agent services",
        "endpoints": {
            "create_agent": "/api/agents"
        },
        "documentation": "/docs"
    }

class AgentProvider(str, Enum):
    VAPI = "vapi"
    RETELL = "retell"

class VoiceProvider(str, Enum):
    ELEVEN_LABS = "eleven_labs"
    DEEPGRAM = "deepgram" 
    PLAY_HT = "play_ht"
    OPEN_AI = "open_ai"
    RETELL = "retell"
    AWS_POLLY = "aws_polly"
    GOOGLE = "google"

class Voice(BaseModel):
    provider: VoiceProvider
    voice_id: str
    settings: Optional[Dict[str, Any]] = None

class AgentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    llm_model: Optional[str] = None
    system_prompt: Optional[str] = None
    voice: Optional[Voice] = None
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    provider: AgentProvider = Field(..., description="The provider to use (vapi or retell)")
    provider_specific: Optional[Dict[str, Any]] = None

# API Key validation
async def validate_api_keys():
    vapi_key = os.getenv("VAPI_API_KEY")
    retell_key = os.getenv("RETELL_API_KEY")
    
    if not vapi_key:
        raise HTTPException(status_code=500, detail="VAPI_API_KEY not configured in environment")
    
    if not retell_key:
        raise HTTPException(status_code=500, detail="RETELL_API_KEY not configured in environment")
    
    return {"vapi_key": vapi_key, "retell_key": retell_key}

def map_voice_provider_to_vapi(provider: VoiceProvider) -> str:
    """Map our standardized voice provider to Vapi's expected format"""
    mapping = {
        VoiceProvider.ELEVEN_LABS: "eleven_labs",
        VoiceProvider.DEEPGRAM: "deepgram",
        VoiceProvider.PLAY_HT: "play_ht",
        VoiceProvider.OPEN_AI: "open_ai",
        VoiceProvider.AWS_POLLY: "aws_polly",
        VoiceProvider.GOOGLE: "google",
        VoiceProvider.RETELL: "retell"
    }
    return mapping.get(provider, "eleven_labs")  # Default to eleven_labs if not found

def map_voice_provider_to_retell(provider: VoiceProvider) -> str:
    """Map our standardized voice provider to Retell's expected format"""
    mapping = {
        VoiceProvider.ELEVEN_LABS: "elevenlabs",
        VoiceProvider.DEEPGRAM: "deepgram",
        VoiceProvider.PLAY_HT: "playht",
        VoiceProvider.OPEN_AI: "openai",
        VoiceProvider.AWS_POLLY: "polly",
        VoiceProvider.GOOGLE: "google",
        VoiceProvider.RETELL: "retell"
    }
    return mapping.get(provider, "elevenlabs")  # Default to elevenlabs if not found

@app.post("/api/agents", status_code=201)
async def create_agent(
    config: AgentConfig,
    api_keys: Dict = Depends(validate_api_keys)
):
    """Create an AI agent using either Vapi.ai or Retell"""
    
    if config.provider == AgentProvider.VAPI:
        return await create_vapi_agent(config, api_keys["vapi_key"])
    elif config.provider == AgentProvider.RETELL:
        return await create_retell_agent(config, api_keys["retell_key"])
    else:
        raise HTTPException(status_code=400, detail="Invalid provider specified")

async def create_vapi_agent(config: AgentConfig, api_key: str):
    """Create an agent using Vapi.ai API"""
    
    url = "https://api.vapi.ai/assistants"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Map our unified model to Vapi's expected format
    payload = {
        "name": config.name,
        "model": config.llm_model or "gpt-3.5-turbo-0125",  # Default model
        "system_prompt": config.system_prompt or "",
        "metadata": config.metadata or {},
    }
    
    # Add voice configuration if provided
    if config.voice:
        payload["voice_id"] = config.voice.voice_id
        payload["voice"] = {
            "provider": map_voice_provider_to_vapi(config.voice.provider)
        }
        if config.voice.settings:
            payload["voice"]["settings"] = config.voice.settings
    
    # Add webhook if provided
    if config.webhook_url:
        payload["webhook_url"] = config.webhook_url
    
    # Add description if provided
    if config.description:
        payload["description"] = config.description
    
    # Add any provider-specific parameters
    if config.provider_specific:
        for key, value in config.provider_specific.items():
            payload[key] = value
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Add provider information to the result
            result["provider"] = "vapi"
            
            return result
    except httpx.HTTPError as e:
        if hasattr(e, "response") and e.response:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                error_detail = error_json.get("message", error_detail)
            except:
                pass
            raise HTTPException(status_code=e.response.status_code, detail=f"Vapi API error: {error_detail}")
        else:
            raise HTTPException(status_code=500, detail=f"Vapi API request failed: {str(e)}")

async def create_retell_agent(config: AgentConfig, api_key: str):
    """Create an agent using Retell API"""
    
    url = "https://api.retellai.com/agents"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Map our unified model to Retell's expected format
    payload = {
        "name": config.name,
        "llm": {
            "provider": "openai",  # Default LLM provider
            "model": config.llm_model or "gpt-3.5-turbo",  # Default model
        },
        "system_prompt": config.system_prompt or "",
        "metadata": config.metadata or {},
    }
    
    # Add voice configuration if provided
    if config.voice:
        payload["voice"] = {
            "provider": map_voice_provider_to_retell(config.voice.provider),
            "voice_id": config.voice.voice_id
        }
        if config.voice.settings:
            payload["voice"]["settings"] = config.voice.settings
    
    # Add webhook if provided
    if config.webhook_url:
        payload["webhook_url"] = config.webhook_url
    
    # Add description if provided
    if config.description:
        payload["description"] = config.description
    
    # Add any provider-specific parameters
    if config.provider_specific:
        for key, value in config.provider_specific.items():
            payload[key] = value
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Add provider information to the result
            result["provider"] = "retell"
            
            return result
    except httpx.HTTPError as e:
        if hasattr(e, "response") and e.response:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                error_detail = error_json.get("message", error_detail)
            except:
                pass
            raise HTTPException(status_code=e.response.status_code, detail=f"Retell API error: {error_detail}")
        else:
            raise HTTPException(status_code=500, detail=f"Retell API request failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)