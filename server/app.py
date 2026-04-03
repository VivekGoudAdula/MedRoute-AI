import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import MedRouteEnv
from env.models import Action

app = FastAPI()
env = MedRouteEnv()

class ActionInput(BaseModel):
    action_type: str
    value: str | None = None
    reasoning: str | None = None

class ResetInput(BaseModel):
    task_id: str | None = None

@app.get("/")
def root():
    return {
        "status": "online",
        "project": "MedRoute AI: Healthcare Triage Simulator",
        "docs": "/docs",
        "specification": "OpenEnv 0.2.0-compliant",
        "available_endpoints": ["/state", "/reset", "/step"]
    }

@app.post("/reset")
def reset(input: ResetInput = None):
    task_id = input.task_id if input else None
    obs = env.reset(task_id=task_id)
    return {"observation": obs}

@app.post("/step")
def step(action: ActionInput):
    action_obj = Action(**action.model_dump())
    obs, reward, done, info = env.step(action_obj)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
    return env.state()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
