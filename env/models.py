from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Literal

class Observation(BaseModel):
    initial_symptoms: List[str] = Field(description="Symptoms initially reported by the patient")
    revealed_symptoms: List[str] = Field(description="Additional symptoms discovered after asking questions")
    asked_questions: List[str] = Field(description="History of questions asked by the agent")
    available_questions: List[str] = Field(description="Possible follow-up questions the agent can ask")
    step_count: int = Field(description="Current step in the triage process")
    max_steps: int = Field(description="Maximum steps allowed")
    feedback: Optional[str] = Field(default=None, description="System hint for the agent")
    reward: float = Field(default=0.0, description="Cumulative or latest reward")
    done: bool = Field(default=False, description="Whether the episode is finished")

class Action(BaseModel):
    action_type: Literal['ASK', 'CLASSIFY_URGENCY', 'DECIDE_TREATMENT'] = Field(
        description="Type of action"
    )
    value: str = Field(
        description="Value: A question (for ASK), Level (low/medium/high for CLASSIFY), or Decision (home/clinic/hospital/emergency for DECIDE)"
    )
    reasoning: Optional[str] = Field(default=None, description="Internal reasoning for the triage decision")

class Reward(BaseModel):
    total_reward: float = Field(description="Aggregated reward score (0.0 - 1.0)")
    question_score: float = Field(default=0.0)
    decision_score: float = Field(default=0.0)
    urgency_score: float = Field(default=0.0)
    penalty: float = Field(default=0.0)
    feedback: str = Field(description="Detailed feedback on the action taken")
