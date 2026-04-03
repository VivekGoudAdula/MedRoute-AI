import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from env.environment import MedRouteEnv
from env.models import Action

# Load .env variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
# Requirements specify HF_TOKEN as the primary key
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# Task and Benchmark metadata
TASK_NAME = os.getenv("TASK_NAME", "complete-triage")
BENCHMARK = os.getenv("BENCHMARK", "medroute-ai")

# FALLBACK CONFIG (GROQ)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

def get_agent_action(obs: Dict, use_fallback=False) -> Action:
    """Agentic decision maker using LLM."""
    
    client = OpenAI(
        base_url=API_BASE_URL if not use_fallback else GROQ_BASE_URL,
        api_key=API_KEY if not use_fallback else GROQ_API_KEY
    )
    model = MODEL_NAME if not use_fallback else GROQ_MODEL

    prompt = f"""
    You are an expert healthcare triage assistant. 
    Analyze the following patient data and decide on the NEXT best action.
    
    PATIENT SYMPTOMS REVEALED: {obs['revealed_symptoms']}
    HISTORY OF QUESTIONS: {obs['asked_questions']}
    STEP: {obs['step_count']}/{obs['max_steps']}
    SYSTEM FEEDBACK: {obs['feedback'] if obs['feedback'] else 'None. Good luck!'}
    
    ACTIONS ALLOWED:
    1. 'ASK': question (about symptoms, history, etc.)
    2. 'CLASSIFY_URGENCY': 'low', 'medium', or 'high'
    3. 'DECIDE_TREATMENT': 'home', 'clinic', 'hospital', or 'emergency'
    
    STRATEGY:
    - If symptoms are vague, ASK 1-2 relevant follow-up questions first.
    - REPETITION: Never ask the same question twice and do NOT classify urgency more than once.
    - EFFICIENCY: Settle on a decision as quickly as possible (ideally within 5-6 steps).
    - FINAL STEP: If you are at step {obs['max_steps']} (or near it), or if you have already classified urgency, you should move to 'DECIDE_TREATMENT' soon.
    
    RESPONSE FORMAT (JSON):
    {{
        "action_type": "ASK" | "CLASSIFY_URGENCY" | "DECIDE_TREATMENT",
        "value": "your question or level or decision",
        "reasoning": "Briefly explain WHY you chose this action (helpful for triage transparency)"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return Action(**data)
    except Exception:
        if not use_fallback and GROQ_API_KEY:
            return get_agent_action(obs, use_fallback=True)
        else:
            return Action(action_type="DECIDE_TREATMENT", value="clinic")

def run_simulation():
    # Pass task_id through env if supported, otherwise it uses default from grader
    env = MedRouteEnv(max_steps=8)
    obs_model = env.reset()
    obs = obs_model.model_dump()
    
    # [START] line
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
    
    done = False
    step_num = 1
    rewards = []
    success = False

    try:
        while not done:
            # 1. Agent decision
            action = get_agent_action(obs)
            
            # 2. Env step
            obs_model, reward, done, info = env.step(action)
            obs = obs_model.model_dump()
            rewards.append(reward)

            # Check success condition (correct decision at high performance)
            if done and reward >= 0.8:
                success = True

            # [STEP] line
            action_str = f"{action.action_type}({action.value})"
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")
            
            step_num += 1
    except Exception as e:
        # Fallback for error logging if something crashes
        if not done:
             print(f"[STEP] step={step_num} action=error reward=0.00 done=true error={str(e)}")

    # [END] line
    steps_taken = step_num - 1
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rewards_str}")

if __name__ == "__main__":
    run_simulation()
