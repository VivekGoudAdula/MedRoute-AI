from openenv.core.env_server.interfaces import Environment as OpenEnvBase
import random
from typing import List, Dict, Tuple, Optional
from .models import Observation, Action, Reward, MedRouteState
from .tasks import PatientCase, CASES
from .grader import MedRouteGrader

class MedRouteEnv(OpenEnvBase):
    def __init__(self, max_steps: int = 5, task_id: str = "complete-triage"):
        super().__init__()
        self.max_steps = max_steps
        self.task_id = task_id
        self.grader = MedRouteGrader()
        self.reset()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Observation:
        self._reset_rubric()
        
        # Consistent seeding
        if seed is not None:
             random.seed(seed)
             
        case_id = kwargs.get("case_id")
        if case_id is None:
            self.case = random.choice(CASES)
        else:
            self.case = next((c for c in CASES if c.case_id == case_id), CASES[0])
        
        self.asked_questions = []
        self.revealed_symptoms = list(self.case.initial_symptoms)
        self.step_count = 0
        self.done = False
        self.has_classified = False
        self.current_feedback = None
        self.final_reward = 0.0
        self.episode_id = episode_id or f"episode-{random.randint(1000, 9999)}"

        return self._get_obs()

    @property
    def state(self) -> MedRouteState:
        """Property-based state for dashboard compatibility."""
        feedback = self.current_feedback
        if self.has_classified and not self.done:
            feedback = (feedback + " " if feedback else "") + "(Urgency already classified. Next step: DECIDE_TREATMENT)"

        return MedRouteState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            initial_symptoms=self.case.initial_symptoms,
            revealed_symptoms=self.revealed_symptoms,
            asked_questions=self.asked_questions,
            available_questions=self.case.relevant_questions,
            max_steps=self.max_steps,
            feedback=feedback,
            done=self.done
        )
        
    def _get_obs(self) -> Observation:
        """Construct the observation object to return from step() and reset()."""
        feedback = self.current_feedback
        if self.has_classified and not self.done:
            feedback = (feedback + " " if feedback else "") + "(Urgency already classified. Next step: DECIDE_TREATMENT)"

        return Observation(
            initial_symptoms=self.case.initial_symptoms,
            revealed_symptoms=self.revealed_symptoms,
            asked_questions=self.asked_questions,
            available_questions=self.case.relevant_questions,
            step_count=self.step_count,
            max_steps=self.max_steps,
            feedback=feedback,
            reward=self.final_reward,
            done=self.done
        )

    def step(self, action: Action) -> Observation:
        if self.done:
            return self._get_obs()

        self.step_count += 1
        
        # Calculate Reward
        state_history = {
            "task_id": self.task_id,
            "revealed_symptoms": self.revealed_symptoms,
            "asked_questions": self.asked_questions,
            "step_count": self.step_count
        }
        reward_obj = self.grader.compute_reward(action, self.case, state_history)
        
        # Efficiency penalty
        if self.step_count > 6:
            reward_obj.total_reward -= 0.1
            reward_obj.penalty += 0.1
            reward_obj.feedback += " (Efficiency Penalty: Agent taking too long)"
        
        # Process Action
        if action.action_type == "ASK":
            self.asked_questions.append(action.value)
            
            # Check for revealed symptoms
            revealed = self._check_revealed_symptoms(action.value)
            if revealed:
                self.revealed_symptoms.extend(revealed)
        
        elif action.action_type == "CLASSIFY_URGENCY":
            if self.has_classified:
                reward_obj.total_reward -= 0.2
                reward_obj.penalty += 0.2
                reward_obj.feedback = "Redundant action: Urgency already classified. Move to decision."
            else:
                self.has_classified = True
                
        # UPDATE FEEDBACK HINT FOR AGENT
        self.current_feedback = reward_obj.feedback
        self.final_reward = reward_obj.total_reward

        # Check termination
        if action.action_type == "DECIDE_TREATMENT" or self.step_count >= self.max_steps:
             self.done = True

        return self._get_obs()

    def _check_revealed_symptoms(self, question: str) -> List[str]:
        q_lower = question.lower()
        new_symptoms = []
        for keyword, symptom in self.case.hidden_symptoms.items():
            k_root = keyword.lower()[:4] if len(keyword) > 4 else keyword.lower()
            if k_root in q_lower or keyword.lower() in q_lower:
                if symptom not in self.revealed_symptoms:
                    new_symptoms.append(symptom)
                    continue
            for word in symptom.lower().split():
                if len(word) > 4 and word in q_lower:
                    if symptom not in self.revealed_symptoms:
                        new_symptoms.append(symptom)
                        break
        return new_symptoms

    def render(self):
        obs = self.state
        print(f"--- MedRoute AI ---")
        print(f"Symptoms: {', '.join(obs.revealed_symptoms)}")
        print(f"History:  {', '.join(self.asked_questions)}")
        print(f"Step:     {obs.step_count}/{obs.max_steps}")
        print(f"-------------------")
