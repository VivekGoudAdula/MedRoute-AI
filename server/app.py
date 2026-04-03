import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.web_interface import WebInterfaceManager, load_environment_metadata, get_quick_start_markdown, _extract_action_fields, _is_chat_env
from openenv.core.env_server.gradio_ui import build_gradio_app
import gradio as gr

from env.environment import MedRouteEnv
from env.models import Action, Observation

# 1. Create the base FastAPI app with your environment
app = create_fastapi_app(MedRouteEnv, Action, Observation)

# 2. Setup the Web Interface Manager
# This handles the logic for the "Take Action" and "History" tabs
metadata = load_environment_metadata(MedRouteEnv, "MedRoute AI")
web_manager = WebInterfaceManager(MedRouteEnv, Action, Observation, metadata)

# 3. Build the Gradio UI blocks
action_fields = _extract_action_fields(Action)
is_chat_env = _is_chat_env(Action)
quick_start_md = get_quick_start_markdown(metadata, Action, Observation)

gradio_blocks = build_gradio_app(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title="MedRoute AI: Clinical Triage Dashboard",
    quick_start_md=quick_start_md,
)

# 4. Mount Gradio at the ROOT (/) so it shows up immediately on HF
# This replaces the need for /web redirects
app = gr.mount_gradio_app(app, gradio_blocks, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
