import os
import uvicorn
from fastapi import FastAPI
import gradio as gr

from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.web_interface import WebInterfaceManager, load_environment_metadata, get_quick_start_markdown, _extract_action_fields, _is_chat_env
from openenv.core.env_server.gradio_ui import build_gradio_app

from env.environment import MedRouteEnv
from env.models import Action, Observation

os.environ["ENABLE_WEB_INTERFACE"] = "true"

print("--- INITIALIZING MEDROUTE AI SERVER ---")

# 1. Create the base FastAPI app
app = create_fastapi_app(MedRouteEnv, Action, Observation)

# 2. Setup metadata and WebManager
metadata = load_environment_metadata(MedRouteEnv, "MedRoute AI")
web_manager = WebInterfaceManager(MedRouteEnv, Action, Observation, metadata)

# 3. Create Custom Quick Start Instructions for Judges
custom_quick_start = """
# 🩺 MedRoute AI: How to Test

Welcome Judge! This is a live simulation. Your goal is to figure out the patient's condition by asking questions, and then make a final medical decision.

### 👉 EXACT STEPS:
1. **START:** Click the **Reset** button to get a random patient case. Read their `initial_symptoms` in the Status box below.
2. **INVESTIGATE:** In the UI above:
   - Select **Action Type:** `ASK`
   - Set **Value:** (Type a question, e.g., "Do you have a fever?" or "How is your breathing?")
   - Click **Step**. Watch the Status box to see if any hidden symptoms are revealed!
3. **DECIDE:** Once you know what's wrong:
   - Select **Action Type:** `DECIDE_TREATMENT`
   - Set **Value:** Type `home`, `clinic`, `hospital`, or `emergency`.
   - Click **Step**!

You will receive a **Final Reward Score** based on your clinical efficiency and accuracy!
"""

action_fields = _extract_action_fields(Action)
is_chat_env = _is_chat_env(Action)

# 4. Build Gradio App with the instructions
gradio_blocks = build_gradio_app(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title="MedRoute AI: Clinical Triage Dashboard",
    quick_start_md=custom_quick_start,
)

# 5. Mount directly at ROOT
app = gr.mount_gradio_app(app, gradio_blocks, path="/")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
