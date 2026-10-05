# agents/business_understanding/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional


class BusinessObjective(BaseModel):
    """
    Represents a single business goal that the organization wants to achieve.
    """
    id: str = Field(
        description="Unique identifier for the objective (e.g., OBJ-1, OBJ-2)."
    )
    goal: str = Field(
        description="Clear description of what the business wants to achieve."
    )
    success_criteria: str = Field(
        description="The business metric used to measure success (e.g., 'Reduce customer churn by 15%')."
    )


class DataMiningGoal(BaseModel):
    """
    Translates the business objectives into technical Data Mining / Machine Learning goals.
    """
    id: str = Field(
        description="Unique identifier for the data mining goal (e.g., DMG-1, DMG-2)."
    )
    technical_goal: str = Field(
        description="The technical task translated from business (e.g., 'Predict probability of churn for each user')."
    )
    success_criteria: str = Field(
        description="Technical performance metric (e.g., 'AUC-ROC > 0.82 and Precision > 75%')."
    )


class ProjectRisk(BaseModel):
    """
    Identifies potential risks that could cause the data science project to fail, 
    along with corresponding mitigation actions.
    """
    risk_category: str = Field(
        description="Category of the risk (e.g., 'Data Quality', 'Business Alignment', 'Technical', 'Budget')."
    )
    description: str = Field(
        description="Detailed description of the potential risk event."
    )
    mitigation_plan: str = Field(
        description="Specific proactive action or fallback plan to manage this risk."
    )


class ProjectResource(BaseModel):
    """
    Resources available to the project, mapping out personnel, data sources, and hardware.
    """
    manpower: List[str] = Field(
        description="List of required roles/team members (e.g., '1 Lead Data Scientist', '1 Data Engineer', '1 Product Owner')."
    )
    hardware_and_compute: List[str] = Field(
        description="Hardware/Software resources required (e.g., '1x Nvidia A100 GPU on AWS', 'Docker environment')."
    )
    primary_data_sources: List[str] = Field(
        description="Core data repositories needed (e.g., 'Production PostgreSQL: transaction_db', 'S3 historical logs')."
    )


class Milestone(BaseModel):
    """
    Defines a specific phase delivery in the project roadmap.
    """
    phase_name: str = Field(
        description="The CRISP-DM phase (e.g., 'Data Understanding', 'Data Preparation', 'Modeling')."
    )
    key_deliverables: List[str] = Field(
        description="Concrete outputs of this phase (e.g., 'Feature Engineering pipeline', 'Baseline Model evaluation')."
    )
    estimated_duration_weeks: float = Field(
        description="Estimated time needed to complete this phase, expressed in weeks."
    )


class ProjectPlan(BaseModel):
    """
    The initial project plan detailing resource requirements, milestones, and contingency actions.
    """
    resources: ProjectResource = Field(
        description="Overview of human and technical resources allocated to the project."
    )
    milestones: List[Milestone] = Field(
        description="Step-by-step CRISP-DM execution plan with estimated timelines."
    )
    contingency_plans: List[str] = Field(
        description="Decisions/actions to take if milestones are missed or major blockers appear."
    )


class BusinessUnderstandingOutput(BaseModel):
    """
    The complete, final schema of the Business Understanding phase as outlined in the architecture.
    This acts as the contract outputted by this agent to be consumed by downstream agents or users.
    """
    # 1. Background & Scope
    background: str = Field(
        description="Comprehensive context of the organization, the business domain, and why this project is being initiated."
    )
    requirements: List[str] = Field(
        description="Core business requirements, security/privacy needs, and deployment specifications."
    )
    assumptions_and_constraints: List[str] = Field(
        description="Assumptions made (e.g., 'Historical data is clean back to 2024') and constraints (e.g., 'GDPR compliance', 'Max latency < 200ms')."
    )

    # 2. Business Goals (from the Diagram)
    business_objectives: List[BusinessObjective] = Field(
        description="The primary business goals mapped out with their respective real-world success criteria."
    )

    # 3. Data Mining Goals (from the Diagram)
    data_mining_goals: List[DataMiningGoal] = Field(
        description="The translation of business objectives into specific data science, ML, or analytical problems with metrics."
    )

    # 4. Risks & Costs
    risks_and_mitigation: List[ProjectRisk] = Field(
        description="Full list of identified project risks paired with contingency and mitigation plans."
    )
    cost_benefit_analysis: str = Field(
        description="High-level comparison comparing estimated costs (compute, team, maintenance) against the projected business value (e.g., ROI, savings)."
    )

    # 5. Project Plan
    project_plan: ProjectPlan = Field(
        description="The structured roadmap covering CRISP-DM milestones, timelines, resources, and contingencies."
    )
