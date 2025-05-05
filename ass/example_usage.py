import requests
import json

# The URL of your API
API_URL = "http://localhost:8000/api/agents"

# Example for creating a Vapi.ai agent
def create_vapi_agent():
    payload = {
        "name": "Customer Support Agent",
        "description": "An AI agent that helps with customer queries",
        "llm_model": "gpt-4-turbo",
        "system_prompt": "You are a helpful customer support agent for a tech company. Be concise and friendly.",
        "voice": {
            "provider": "eleven_labs",
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        },
        "webhook_url": "https://example.com/webhook/vapi",
        "metadata": {
            "department": "customer_support",
            "priority": "high"
        },
        "provider": "vapi"
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code in [200, 201]:
        print("Successfully created Vapi.ai agent:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
# Example for creating a Retell agent
def create_retell_agent():
    payload = {
        "name": "Sales Representative",
        "description": "An AI agent that helps with sales inquiries",
        "llm_model": "gpt-4",
        "system_prompt": "You are a knowledgeable sales representative for a software company. Answer questions about our products clearly and professionally.",
        "voice": {
            "provider": "eleven_labs",
            "voice_id": "pNInz6obpgDQGcFmaJgB",
            "settings": {
                "stability": 0.7,
                "similarity_boost": 0.3
            }
        },
        "webhook_url": "https://example.com/webhook/retell",
        "metadata": {
            "department": "sales",
            "products": ["software_a", "software_b"]
        },
        "provider": "retell",
        "provider_specific": {
            "end_call_after_silence": 5  # Retell-specific parameter
        }
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code in [200, 201]:
        print("Successfully created Retell agent:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("Creating Vapi.ai agent...")
    create_vapi_agent()
    
    print("\nCreating Retell agent...")
    create_retell_agent()