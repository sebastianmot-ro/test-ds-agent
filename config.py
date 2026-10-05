import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_URL = os.getenv(
	"OLLAMA_API_URL",
	"http://10.74.250.125:8000/generate",
)
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5-coder:14b")
OLLAMA_PROXY = os.getenv("OLLAMA_PROXY", "http://127.0.0.1:3128")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
