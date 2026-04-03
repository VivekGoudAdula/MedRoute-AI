from pydantic import Field
from typing import List, Optional, Literal
from openenv.core.env_server.types import Action as BaseAction, Observation as BaseObservation

class Observation(BaseObservation):
    initial_symptoms: List[str] = Field(default_factory=list, description="Symptoms initially reported by the patient")
    revealed_symptoms: List[str] = Field(default_factory=list, description="Additional symptoms discovered after asking questions")
    asked_questions: List[str] = Field(default_factory=list, description="History of questions asked by the agent")
    available_questions: List[str] = Field(default_factory=list, description="Possible follow-up questions the agent can ask")
    step_count: int = Field(default=0, description="Current step in the triage process")
    max_steps: int = Field(default=10, description="Maximum steps allowed")
    feedback: Optional[str] = Field(default=None, description="System hint for the agent")

class Action(BaseAction):
    action_type: Literal['ASK', 'CLASSIFY_URGENCY', 'DECIDE_TREATMENT'] = Field(
        description="ASK or CLASSIFY_URGENCY or DECIDE_TREATMENT"
    )
    value: str = Field(
        description="For ASK → choose from available_questions. For DECIDE → use: clinic / emergency / home_care. For CLASSIFY → low/medium/high."
    )
    reasoning: Optional[str] = Field(default=None, description="Explain why you're taking this action")

class Reward(BaseAction):
    total_reward: float = Field(default=0.0, description="Aggregated reward score (0.0 - 1.0)")
    question_score: float = Field(default=0.0)
    decision_score: float = Field(default=0.0)
    urgency_score: float = Field(default=0.0)
    penalty: float = Field(default=0.0)
    feedback: str = Field(default="No feedback available", description="Detailed feedback on the action taken")
