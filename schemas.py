from pydantic import BaseModel
from typing import Optional, List

class Message(BaseModel):
    role: str
    content: str

class ImageItem(BaseModel):
    base64: str
    mime_type: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    system_prompt: Optional[str] = "You are a helpful assistant."
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stream: bool = True
    hf_token: Optional[str] = None
    use_local: bool = False
    provider: str = "hf"
    api_key: Optional[str] = None
    images: Optional[List[ImageItem]] = None

class ModelSearchRequest(BaseModel):
    query: str = ""
    task: str = "text-generation"
    size_filter: str = "any"
    limit: int = 20
    hf_token: Optional[str] = None
    provider: str = "hf"
    api_key: Optional[str] = None
    use_ai_search: bool = False

class ModelApiCheckRequest(BaseModel):
    model_id: str
    provider: str = "hf"
    api_key: Optional[str] = None
    hf_token: Optional[str] = None

class DownloadRequest(BaseModel):
    model_id: str
    hf_token: Optional[str] = None
    quantization: str = "none"

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Message]
    options: Optional[dict] = {}
    stream: bool = True

class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    options: Optional[dict] = {}
    stream: bool = True

class LoadModelRequest(BaseModel):
    model_id: str
    quantization: str = "none"
    device: str = "auto"
