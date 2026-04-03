import os
import uvicorn
from openenv.core.env_server import create_app
from env.environment import MedRouteEnv
from env.models import Action, Observation

# Force the environment variable just in case
os.environ["ENABLE_WEB_INTERFACE"] = "true"

print("--- INITIALIZING MEDROUTE AI SERVER ---")
print(f"Action model: {Action}")
print(f"Observation model: {Observation}")

# Use the official create_app factory
# This handles all the mounting, redirects, and OpenAPI schema generation
app = create_app(
    MedRouteEnv, 
    Action, 
    Observation,
    env_name="MedRoute AI Triage Simulator"
)

print("--- SERVER INSTANTIATED SUCCESSFULLY ---")

if __name__ == "__main__":
    # Ensure port 7860 is used for Hugging Face
    uvicorn.run(app, host="0.0.0.0", port=7860)
