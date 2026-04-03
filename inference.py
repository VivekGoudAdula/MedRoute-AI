import os
import json
import time
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FALLBACK CONFIG (GROQ)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

def get_agent_action(obs: Dict, use_fallback=False) -> Action:
    """Agentic decision maker using LLM."""
    
    client = OpenAI(
        base_url=API_BASE_URL if not use_fallback else GROQ_BASE_URL,
        api_key=OPENAI_API_KEY if not use_fallback else GROQ_API_KEY
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
    except Exception as e:
        if not use_fallback:
            return get_agent_action(obs, use_fallback=True)
        else:
            return Action(action_type="DECIDE_TREATMENT", value="clinic")

def run_simulation():
    env = MedRouteEnv(max_steps=8)
    obs_model = env.reset()
    obs = obs_model.model_dump()
    
    print("[START]")
    print(f"Patient initial symptoms: {obs['initial_symptoms']}")
    
    done = False
    step_num = 1
    total_reward = 0.0

    while not done:
        # 1. Agent decision
        action = get_agent_action(obs)
        
        # 2. Env step
        obs_model, reward, done, info = env.step(action)
        obs = obs_model.model_dump()
        total_reward += reward

        # 3. Log step
        print(f"[STEP] {step_num}")
        print(f"Action: {action.action_type} -> {action.value}")
        if action.reasoning:
            print(f"Reason: {action.reasoning}")
        print(f"Goal: Correct decision for {info['severity']} severity")
        print(f"Symptoms Revealed: {obs['revealed_symptoms']}")
        print(f"Step Reward: {reward:.2f}")
        print(f"Feedback: {info['reward_details']['feedback']}")
        print(f"Done: {done}")
        print("-" * 20)
        
        step_num += 1

    print(f"Total Cumulative Reward: {total_reward}")
    print(f"Conclusion: Correct decision was {info['correct_decision']} (Severity: {info['severity']})")
    print("[END]")

if __name__ == "__main__":
    run_simulation()
