from openenv.core.env_server import create_app
from env.environment import MedRouteEnv
from env.models import Action, Observation

# This creates the official OpenEnv FastAPI + Gradio UI app
# It explicitly required the Action and Observation models to build the UI
app = create_app(MedRouteEnv, Action, Observation)

def main():
    import uvicorn
    # Port 7860 is required for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
