from typing import List, Dict, Optional

class PatientCase:
    def __init__(self, case_id: int, initial_symptoms: List[str], hidden_symptoms: Dict[str, str], 
                 severity: str, correct_decision: str, relevant_questions: List[str]):
        self.case_id = case_id
        self.initial_symptoms = initial_symptoms
        self.hidden_symptoms = hidden_symptoms  # map question clue -> symptom revealed
        self.severity = severity
        self.correct_decision = correct_decision
        self.relevant_questions = relevant_questions

CASES = [
    PatientCase(
        case_id=1,
        initial_symptoms=["chest pain"],
        hidden_symptoms={
            "breath": "shortness of breath",
            "sweat": "profuse sweating",
            "arm": "pain radiating to left arm",
            "jaw": "pain radiating to jaw"
        },
        severity="high",
        correct_decision="emergency",
        relevant_questions=["Are you having trouble breathing?", "Are you sweating?", "Does the pain spread to your arm or jaw?"]
    ),
    PatientCase(
        case_id=2,
        initial_symptoms=["mild fever", "cough"],
        hidden_symptoms={
            "duration": "lasted for 2 days",
            "throat": "sore throat",
            "travel": "no recent travel history"
        },
        severity="low",
        correct_decision="home",
        relevant_questions=["How long have you had the fever?", "Do you have a sore throat?", "Have you traveled recently?"]
    ),
    PatientCase(
        case_id=3,
        initial_symptoms=["abdominal pain", "nausea"],
        hidden_symptoms={
            "location": "lower right quadrant pain",
            "rebound": "rebound tenderness present",
            "fever": "high fever developed"
        },
        severity="high",
        correct_decision="hospital",
        relevant_questions=["Where exactly is the pain?", "Is the pain worse when you release pressure?", "Do you have a fever?"]
    ),
    PatientCase(
        case_id=4,
        initial_symptoms=["skin rash"],
        hidden_symptoms={
            "itch": "extremely itchy",
            "spread": "spreading across the torso",
            "blister": "small fluid-filled blisters"
        },
        severity="medium",
        correct_decision="clinic",
        relevant_questions=["Is the rash itchy?", "Is it spreading?", "Do you see any blisters?"]
    ),
    PatientCase(
        case_id=5,
        initial_symptoms=["severe headache"],
        hidden_symptoms={
            "vision": "blurred vision",
            "neck": "stiff neck",
            "light": "sensitivity to light"
        },
        severity="high",
        correct_decision="emergency",
        relevant_questions=["Are you experiencing blurred vision?", "Is your neck stiff?", "Does light hurt your eyes?"]
    ),
    PatientCase(
        case_id=6,
        initial_symptoms=["minor cut on finger"],
        hidden_symptoms={
            "bleeding": "bleeding has stopped",
            "dirt": "cut is clean",
            "tetanus": "last tetanus shot was 2 years ago"
        },
        severity="low",
        correct_decision="home",
        relevant_questions=["Is it still bleeding?", "Is there dirt in the wound?", "When was your last tetanus shot?"]
    ),
    PatientCase(
        case_id=7,
        initial_symptoms=["earache"],
        hidden_symptoms={
            "discharge": "yellow discharge from ear",
            "hearing": "muffled hearing",
            "balance": "feeling slightly dizzy"
        },
        severity="medium",
        correct_decision="clinic",
        relevant_questions=["Is there any discharge from the ear?", "Is your hearing affected?", "Are you feeling dizzy?"]
    ),
    PatientCase(
        case_id=8,
        initial_symptoms=["joint pain", "swelling"],
        hidden_symptoms={
            "injury": "twisted ankle while running",
            "weight": "cannot bear weight on the leg",
            "bruising": "significant bruising appearing"
        },
        severity="medium",
        correct_decision="clinic",
        relevant_questions=["How did the injury happen?", "Can you walk or stand on it?", "Is there any bruising?"]
    ),
    PatientCase(
        case_id=9,
        initial_symptoms=["persistent diarrhea"],
        hidden_symptoms={
            "hydration": "dry mouth and dark urine",
            "blood": "no blood in stool",
            "frequency": "10 times in the last 24 hours"
        },
        severity="medium",
        correct_decision="clinic",
        relevant_questions=["Are you feeling dehydrated?", "Is there blood in your stool?", "How many times have you gone?"]
    ),
    PatientCase(
        case_id=10,
        initial_symptoms=["sudden confusion", "slurred speech"],
        hidden_symptoms={
            "face": "onesided facial drooping",
            "arm": "weakness in right arm",
            "time": "started 30 minutes ago"
        },
        severity="high",
        correct_decision="emergency",
        relevant_questions=["Is there facial drooping?", "Can you lift both arms?", "When did this start?"]
    )
]

# --- TASK DEFINITIONS ---

TASKS_MAPPING = {
    "urgency-discovery": [c for c in CASES if c.severity == "low"],
    "symptom-investigation": [c for c in CASES if c.severity == "medium"],
    "complete-triage": [c for c in CASES if c.severity == "high"]
}

def get_case_for_task(task_id: str, case_id: int = None):
    available_cases = TASKS_MAPPING.get(task_id, CASES)
    if case_id is not None:
        return next((c for c in available_cases if c.case_id == case_id), available_cases[0])
    import random
    return random.choice(available_cases)
