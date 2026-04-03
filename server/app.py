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

import openenv.core.env_server.gradio_ui as gradio_ui

# --- UI MONKEY PATCH OVERRIDE ---
# Hack to force the UI to render Reward and Done as massive banners!
original_format_observation = gradio_ui._format_observation

def custom_format_observation(data):
    base_md = original_format_observation(data)
    reward = data.get("reward")
    done = data.get("done")
    
    if reward is not None or done is not None:
        highlight = "<div style='background-color: #FF9800; color: white; padding: 15px 20px; border-radius: 10px; font-size: 24px; font-weight: 800; margin-bottom: 20px; text-transform: uppercase; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 3px solid #E65100;'>"
        if reward is not None:
            highlight += f"🏆 REWARD:  <span style='color: #FFE0B2;'>{reward:.2f}</span> &nbsp;&nbsp;&nbsp;&nbsp; "
        if done is not None:
            status_color = "#C8E6C9" if done else "#FFCDD2"
            highlight += f"🏁 DONE:  <span style='color: {status_color};'>{done}</span>"
        highlight += "</div><br/>"
        
        # Strip out the tiny original text
        if f"**Reward:** `{reward}`" in base_md:
            base_md = base_md.replace(f"**Reward:** `{reward}`", "")
        if f"**Done:** `{done}`" in base_md:
            base_md = base_md.replace(f"**Done:** `{done}`", "")
            
        return highlight + base_md.strip()
        
    return base_md

gradio_ui._format_observation = custom_format_observation
# --------------------------------

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
