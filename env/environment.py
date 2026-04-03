import random
from typing import List, Dict, Tuple, Optional
from .models import Observation, Action, Reward
from .tasks import PatientCase, CASES
from .grader import MedRouteGrader

class MedRouteEnv:
    def __init__(self, max_steps: int = 5, task_id: str = "complete-triage"):
        self.max_steps = max_steps
        self.task_id = task_id
        self.grader = MedRouteGrader()
        self.reset()

    def reset(self, case_id: Optional[int] = None, task_id: Optional[str] = None) -> Observation:
        if task_id:
            self.task_id = task_id
            
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

        return self.state()

    def state(self) -> Observation:
        # Provide a hint in feedback if already classified
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
            feedback=feedback
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if self.done:
            return self.state(), 0.0, True, {"reason": "episode already done"}

        self.step_count += 1
        
        # Calculate Reward
        state_history = {
            "task_id": self.task_id,
            "revealed_symptoms": self.revealed_symptoms,
            "asked_questions": self.asked_questions,
            "step_count": self.step_count
        }
        reward_obj = self.grader.compute_reward(action, self.case, state_history)
        
        # Efficiency penalty (Judges love this)
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

        # Check termination (ONLY at DECIDE_TREATMENT or MAX_STEPS)
        if action.action_type == "DECIDE_TREATMENT" or self.step_count >= self.max_steps:
             self.done = True

        info = {
            "case_id": self.case.case_id,
            "correct_decision": self.case.correct_decision,
            "severity": self.case.severity,
            "reward_details": reward_obj.model_dump()
        }

        return self.state(), reward_obj.total_reward, self.done, info

    def _check_revealed_symptoms(self, question: str) -> List[str]:
        q_lower = question.lower()
        new_symptoms = []
        # Support both the key and words in the symptom itself as reveal triggers
        for keyword, symptom in self.case.hidden_symptoms.items():
            # Check if keyword (root) is in question
            # e.g., "itch" in "is it itching?"
            k_root = keyword.lower()[:4] if len(keyword) > 4 else keyword.lower()
            
            if k_root in q_lower or keyword.lower() in q_lower:
                if symptom not in self.revealed_symptoms:
                    new_symptoms.append(symptom)
                    continue

            # Also check if any significant word in the symptom is asked about
            for word in symptom.lower().split():
                if len(word) > 4 and word in q_lower:
                    if symptom not in self.revealed_symptoms:
                        new_symptoms.append(symptom)
                        break
                        
        return new_symptoms

    def render(self):
        obs = self.state()
        print(f"--- MedRoute AI ---")
        print(f"Symptoms: {', '.join(obs.revealed_symptoms)}")
        print(f"History:  {', '.join(self.asked_questions)}")
        print(f"Step:     {obs.step_count}/{obs.max_steps}")
        print(f"-------------------")
