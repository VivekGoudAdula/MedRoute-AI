from openenv.core.server import create_app
from env.environment import MedRouteEnv

# This creates the official OpenEnv FastAPI + Gradio UI app
app = create_app(MedRouteEnv)

def main():
    import uvicorn
    # Port 7860 is required for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
