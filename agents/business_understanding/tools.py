# agents/business_understanding/tools.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class AskUserInput(BaseModel):
    question: str = Field(
        description="The specific, high-value question to ask the user to clarify business or technical requirements."
    )


class EscalationInput(BaseModel):
    blocker_summary: str = Field(
        description="A detailed summary of why the agent cannot proceed (e.g., conflicting requirements, missing access)."
    )
    suggested_next_steps: str = Field(
        description="Recommended actions for the orchestrator or human supervisor to resolve the blocker."
    )


@tool("ask_user", args_schema=AskUserInput)
def ask_user(question: str) -> str:
    """
    Use this tool to ask the user for clarification, additional details, or 
    to validate business objectives when information is missing or ambiguous.
    This tool will prompt the user directly in the terminal and wait for their input.
    """
    print(f"\n [Agent Question]: {question}")
    user_response = input(" [Your Answer]: ")
    return user_response


@tool("escalate_to_orchestrator", args_schema=EscalationInput)
def escalate_to_orchestrator(blocker_summary: str, suggested_next_steps: str) -> str:
    """
    Use this tool ONLY when you are unable to proceed. 
    This can happen if the user provides contradictory goals, 
    if you are stuck in a reasoning loop, or if critical constraints cannot be resolved.
    """
    print("\n [ESCALATION TRIGGERED] ")
    print(f"Reason: {blocker_summary}")
    print(f"Suggested Action: {suggested_next_steps}")
    print("--------------------------------")
    return "Escalation logged. Execution halted."
