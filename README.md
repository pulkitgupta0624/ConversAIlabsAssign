# AI Agent Wrapper API

This project provides a unified API for creating AI agents across multiple platforms (Vapi and Retell).

## Features

- Single API endpoint for creating agents
- Support for both Vapi and Retell platforms
- Standardized request/response format
- Easy to extend for additional platforms

## Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/ai-agent-wrapper.git
cd ai-agent-wrapper
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Set up environment variables:
```
export VAPI_API_KEY=your_vapi_api_key
export RETELL_API_KEY=your_retell_api_key
```

## Running the API

```
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the API is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage Example

```python
import requests
import json

url = "http://localhost:8000/agents"
payload = {
    "platform": "vapi",  # or "retell"
    "name": "Customer Service Agent",
    "description": "Helps customers with their inquiries",
    "voice": {
        "provider": "eleven_labs", 
        "voice_id": "rachel"
    },
    "instructions": "You are a customer service agent for Acme Inc.",
    "language": "en"
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(json.dumps(response.json(), indent=4))
```#   C o n v e r s A I l a b s A s s i g n  
 