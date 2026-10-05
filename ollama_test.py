from config import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_NAME,
    OLLAMA_PROXY,
    OLLAMA_TIMEOUT_SECONDS,
)
from core.ollama_chat_model import OllamaGenerateChatModel


def main() -> None:
    llm = OllamaGenerateChatModel(
        api_url=OLLAMA_API_URL,
        model=OLLAMA_MODEL_NAME,
        proxy_url=OLLAMA_PROXY or None,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    prompt = input("Prompt (or 'exit' to quit): ").strip()
    if prompt.lower() == "exit" or not prompt:
        return

    response = llm.invoke(prompt)
    print(response.content)


if __name__ == "__main__":
    main()
 