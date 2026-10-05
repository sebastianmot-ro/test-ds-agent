# agents/business_understanding/agent.py

from langgraph.prebuilt import create_react_agent
from agents.business_understanding import prompts, tools, schemas


def create_business_understanding_agent(llm):
    """
    Creates and configures the Business Understanding Agent using modern LangGraph architecture.
    """
    
    # 1. Define the report submission tool dynamically using our Pydantic schema
    @tools.tool("submit_final_business_understanding_report")
    def submit_final_report(report: schemas.BusinessUnderstandingOutput) -> str:
        """
        Use this tool ONLY when you have collected all required information 
        for the 5 CRISP-DM pillars and are ready to output the final structured report.
        """
        print("\n [SUCCESS] Final CRISP-DM Business Understanding Report Submitted!")
        return "Report successfully submitted and validated against the schema."

    # 2. Gather all tools available to the model
    agent_tools = [
        tools.ask_user, 
        tools.escalate_to_orchestrator,
        submit_final_report
    ]

    # 3. Create the modern ReAct Agent via LangGraph.

    agent_executor = create_react_agent(
        model=llm,
        tools=agent_tools,
        prompt=prompts.SYSTEM_PROMPT  
    )

    return agent_executor
