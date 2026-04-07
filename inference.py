import os
import json
import time
from typing import Dict, List, Optional
from openai import OpenAI
from env.environment import MedRouteEnv
from env.models import Action

# --- STRICT OPENENV CONFIGURATION ---
# These are MUST for Phase 2 validation via LiteLLM proxy
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]

# Metadata
TASK_NAME = os.environ.get("TASK_NAME", "complete-triage")
BENCHMARK = os.environ.get("BENCHMARK", "medroute-ai")

# Initialize OpenAI Client (STRICT: Bypassing local OpenAI endpoint)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# LOGGING HELPERS (Strictly follow [START], [STEP], [END] format)
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_agent_action(obs: Dict) -> Action:
    """Agentic decision maker using the provided LiteLLM proxy."""
    
    prompt = f"""
    You are an expert healthcare triage assistant. 
    Analyze the following patient data and decide on the NEXT best action.
    
    PATIENT SYMPTOMS REVEALED: {obs['revealed_symptoms']}
    HISTORY OF QUESTIONS: {obs['asked_questions']}
    STEP: {obs['step_count']}/{obs['max_steps']}
    SYSTEM FEEDBACK: {obs['feedback'] if obs['feedback'] else 'None.'}
    
    ACTIONS ALLOWED:
    1. 'ASK': question (about symptoms, history, etc.)
    2. 'CLASSIFY_URGENCY': 'low', 'medium', or 'high'
    3. 'DECIDE_TREATMENT': 'home', 'clinic', 'hospital', or 'emergency'
    
    STRATEGY:
    - If symptoms are vague, ASK 1-2 relevant follow-up questions first.
    - REPETITION: Never ask the same question twice and do NOT classify urgency more than once.
    - EFFICIENCY: Settle on a decision accurately and quickly.
    - FINAL STEP: You MUST move to 'DECIDE_TREATMENT' eventually.
    
    RESPONSE FORMAT (JSON):
    {{
        "action_type": "ASK" | "CLASSIFY_URGENCY" | "DECIDE_TREATMENT",
        "value": "your question or level or decision",
        "reasoning": "Briefly explain WHY you chose this action"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return Action(**data)
    except Exception as e:
        # Emergency fallback (non-API) to ensure script completes
        return Action(action_type="DECIDE_TREATMENT", value="clinic", reasoning=f"API error: {str(e)}")

def run_simulation():
    # Initialize Environment
    env = MedRouteEnv(max_steps=8, task_id=TASK_NAME)
    obs_model = env.reset()
    obs = obs_model.model_dump()
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    done = False
    step_num = 1
    rewards = []
    success = False

    try:
        while not done and step_num <= 8:
            # 1. Agent decision
            action = get_agent_action(obs)
            
            # 2. Env step
            obs_model = env.step(action)
            obs = obs_model.model_dump()
            reward = obs_model.reward or 0.0
            done = obs_model.done or False
            rewards.append(reward)

            # [STEP] line
            action_str = f"{action.action_type}({action.value})"
            log_step(step=step_num, action=action_str, reward=reward, done=done)
            
            if done:
                break
            
            step_num += 1
            
        # Final evaluation
        final_score = max(rewards) if rewards else 0.0
        success = final_score >= 0.8

    except Exception as e:
        log_step(step=step_num, action="error", reward=0.0, done=True, error=str(e))
    finally:
        steps_taken = len(rewards)
        score = max(rewards) if rewards else 0.0
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    run_simulation()
