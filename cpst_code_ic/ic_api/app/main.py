import sys
import gradio as gr
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
#print(sys.path)
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
from app.api import api_router, run_rlmf, done_with_rl_mf
from app.config import settings
import uvicorn

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Image Captioning API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("RLMF With Expert Model")
    with gr.Tab(label="RLMF"):
        output_img = gr.Image(label="Image")
        caption_us = gr.Textbox(label="Model under evaluation")
        yolo_res = gr.Textbox(label="Yolo result")
        yolo_gemini_caption = gr.Textbox(label="Caption suggestion from expert model")
        done_with_rl_mf_btn = gr.Button(value="Submit RL Data For Finetuning Model")
        run_rl_mf_btn = gr.Button(value="Run RLMF")

run_rl_mf_btn.click(run_rlmf,inputs=[],outputs=[output_img, caption_us, yolo_res, yolo_gemini_caption],
                         title="Real or Kidding",
                         description="YOLO/Gemini Feedback",
                         allow_flagging='never')
done_with_rl_mf_btn.click(done_with_rl_mf)

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 

    ## localhost--> 127.0.0.0
    ## host --> 0.0.0.0 allows all host
