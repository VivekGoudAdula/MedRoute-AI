from typing import List, Dict, Tuple
from .models import Action, Reward
from .tasks import PatientCase
import random

class MedRouteGrader:
    def __init__(self):
        self.categories = {
            "urgency": ["low", "medium", "high"],
            "decision": ["home", "clinic", "hospital", "emergency"]
        }

    def grade_urgency(self, action: Action, case: PatientCase) -> Tuple[float, str]:
        if action.value.lower() == case.severity.lower():
            return 1.0, "Correct urgency classification."
        
        # Partially correct if adjacent (e.g., low vs medium)
        severity_idx = self.categories["urgency"].index(case.severity.lower())
        action_idx = -1
        try:
            action_idx = self.categories["urgency"].index(action.value.lower())
        except ValueError:
            return 0.0, f"Invalid urgency level: {action.value}"

        if abs(severity_idx - action_idx) == 1:
            return 0.5, f"Partially correct urgency. You chose {action.value} for a {case.severity} case."
        
        return 0.0, f"Incorrect urgency classification. You chose {action.value} for a {case.severity} case."

    def grade_question(self, action: Action, case: PatientCase, asked_questions: List[str]) -> Tuple[float, str]:
        q_lower = action.value.lower()
        if q_lower in [q.lower() for q in asked_questions]:
            return -0.3, "You already asked this. Try something else. Penalty applied."

        # 1. Check for HIGH relevance (Patient-Specific Keywords)
        case_keywords = set()
        for k, v in case.hidden_symptoms.items():
            case_keywords.add(k.lower())
            case_keywords.update(v.lower().split())
        
        if any(word in q_lower for word in case_keywords if len(word) > 3):
            return 0.4, "Highly relevant medical question. Discovery made!"

        # 2. Check for MEDICAL relevance (Generic Clinical Terms)
        # Expanded to include nausea, vomit, etc.
        medical_keywords = ["pain", "weakness", "breathing", "fever", "swelling", "speech", "balance", "bleeding", "vision", "rash", "consciousness", "vomit", "nausea", "diarrhea"]
        if any(word in q_lower for word in medical_keywords if len(word) > 3):
            return 0.2, "Relevant clinical follow-up question."

        # 3. Irrelevant
        return -0.2, "Irrelevant or non-clinical question. Penalty applied."

    def grade_decision(self, action: Action, case: PatientCase) -> Tuple[float, str]:
        pred = action.value.lower()
        correct = case.correct_decision.lower()

        if pred == correct:
            return 1.0, "Correct treatment decision. Well done!"
        
        # 1. OVER-ESCALATION (Safe but inefficient/costly)
        if pred == "clinic" and correct == "home":
            return 0.6, "Over-cautious: Prescribed clinic visit for a home-care case."
        if pred == "hospital" and correct == "home":
            return 0.3, "Significant Over-escalation: Prescribed hospital for a mild case."
        if pred == "hospital" and correct == "clinic":
            return 0.6, "Over-cautious: Prescribed hospital for a clinic-level case."
        
        # 2. UNDER-ESTIMATION (Dangerous clinical risk)
        if correct in ["hospital", "emergency"] and pred == "home":
             return -1.0, "DANGEROUS DECISION: Prescribing home care for a high-risk case!"
        if correct == "emergency" and pred == "clinic":
             return -0.5, "Inadequate Care: Prescribing clinic for an emergency case!"

        # 3. OTHER (Adjacent but safe)
        decision_list = self.categories["decision"]
        try:
            p_idx = decision_list.index(pred)
            c_idx = decision_list.index(correct)
            if abs(p_idx - c_idx) == 1:
                return 0.7, f"Safe but partially incorrect. You chose {pred} for a {correct} case."
        except:
            pass

        return 0.0, f"Incorrect decision. You chose {pred} for a {correct} case."

    def compute_reward(self, action: Action, case: PatientCase, state_history: Dict) -> Reward:
        score = 0.0
        q_score = 0.0
        d_score = 0.0
        u_score = 0.0
        penalty = 0.0
        feedback = ""

        task_id = state_history.get("task_id", "complete-triage")
        
        if action.action_type == "ASK":
            q_score, feedback = self.grade_question(action, case, state_history["asked_questions"])
            if q_score < 0:
                penalty = abs(q_score)
                q_score = 0.0
        elif action.action_type == "CLASSIFY_URGENCY":
            u_score, feedback = self.grade_urgency(action, case)
        elif action.action_type == "DECIDE_TREATMENT":
            d_score, feedback = self.grade_decision(action, case)
            step_count = state_history.get("step_count", 0)

            # Bonus for excellence
            if d_score >= 1.0:
                if step_count <= 5:
                    d_score += 0.2
                    feedback += " (Efficiency Bonus!)"
            elif d_score < 0:
                penalty = abs(d_score)
                d_score = 0.0

        # TASK-SPECIFIC WEIGHTING
        if task_id == "urgency-discovery":
            # Focus only on urgency
            total = u_score - penalty
        elif task_id == "symptom-investigation":
            # Focus on discovery (revealed symptoms)
            initial_count = len(case.initial_symptoms)
            revealed_count = len(state_history.get("revealed_symptoms", []))
            discovery_bonus = (revealed_count - initial_count) * 0.3
            total = q_score + discovery_bonus - penalty
        else:
            # Full triage
            total = (q_score * 0.3) + (u_score * 0.3) + (d_score * 0.4) - penalty
        
        total = max(0.01, min(0.99, total))

        return Reward(
            total_reward=round(total, 2),
            question_score=q_score,
            decision_score=d_score,
            urgency_score=u_score,
            penalty=penalty,
            feedback=feedback
        )
