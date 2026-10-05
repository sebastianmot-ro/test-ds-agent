from config import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_NAME,
    OLLAMA_PROXY,
    OLLAMA_TIMEOUT_SECONDS,
)
from agents.business_understanding.agent import create_business_understanding_agent
from core.ollama_chat_model import OllamaGenerateChatModel


def run_agent_conversation(agent, initial_user_input: str) -> None:
    state = {"messages": [("user", initial_user_input)]}

    while True:
        result = agent.invoke(state)
        messages = result.get("messages", [])

        print("\n [Agent Response]:")
        print("-" * 40)
        if messages:
            print(messages[-1].content)
        else:
            print("No output messages found in the agent state.")
        print("-" * 40)

        while True:
            try:
                user_input = input("\n [Your clarification] (or 'exit'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[System] Conversation ended.")
                return

            if user_input.lower() in {"exit", "quit"}:
                print("[System] Conversation ended.")
                return
            if user_input:
                break
            print("[System] Please enter a clarification, or type 'exit'.")

        state = {"messages": messages + [("user", user_input)]}


def run_standalone_agent():
    """
    Runs the Business Understanding Agent using the configured Ollama endpoint.
    """
    print("=" * 60)
    print(" STARTING CRISP-DM BUSINESS UNDERSTANDING AGENT (OLLAMA)")
    print(f" Model: {OLLAMA_MODEL_NAME}")
    print(f" Endpoint: {OLLAMA_API_URL}")
    print("=" * 60)

    try:
        print("[INFO] Connecting to Ollama through the configured API...")
        llm = OllamaGenerateChatModel(
            api_url=OLLAMA_API_URL,
            model=OLLAMA_MODEL_NAME,
            proxy_url=OLLAMA_PROXY or None,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )

        print("[INFO] Compiling agent and binding CRISP-DM tools using LangGraph...")
        bu_agent = create_business_understanding_agent(llm)
        print("[OK] Agent is initialized and ready to run.")
        print("-" * 60)

        print("\n Please enter your raw project idea to kick off the CRISP-DM Phase 1.")
        print("Example: 'We want to build a system that detects fraudulent transactions in our online store.'\n")
        
        user_input = input(" [Your Idea]: ")
        
        if not user_input.strip():
            print("[System] No input detected. Exiting.")
            return

        print("\n Starting Agent Conversation")
        print("-" * 60)
        run_agent_conversation(bu_agent, user_input)

    except Exception as e:
        print(f"\n [ERROR] An error occurred during execution: {e}")
        print("\n Check OLLAMA_API_URL, OLLAMA_PROXY, and that the configured model is available on the Ollama server.")


if __name__ == "__main__":
    run_standalone_agent()
