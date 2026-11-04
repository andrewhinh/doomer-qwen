import json
from pathlib import Path

import modal

from serve import Chat, MODEL_ID, app as serve_app

app = modal.App("doomer-qwen").include(serve_app)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("fastapi>=0.115.0")
    .add_local_file(str(Path(__file__).with_name("index.html")), "/root/index.html")
    .add_local_file(str(Path(__file__).with_name("logo.png")), "/root/logo.png")
    .add_local_file(str(Path(__file__).with_name("favicon.png")), "/root/favicon.png")
    .add_local_python_source("serve")
)


@app.function(image=web_image, min_containers=1)
@modal.asgi_app(custom_domains=["bro.modal.dev"])
def web():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

    web_app = FastAPI(title="modal-bro")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @web_app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(Path("/root/index.html").read_text(encoding="utf-8"))

    @web_app.get("/logo.png", include_in_schema=False)
    def logo():
        return FileResponse("/root/logo.png", media_type="image/png")

    @web_app.get("/favicon.png", include_in_schema=False)
    def favicon_png():
        return FileResponse("/root/favicon.png", media_type="image/png")

    @web_app.get("/health")
    def health():
        return {"status": "ok", **Chat().health.remote()}

    @web_app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse("/root/favicon.png", media_type="image/png")

    @web_app.post("/chat")
    async def chat(request: Request):
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list):
            return JSONResponse({"error": "messages must be a list"}, status_code=400)

        def events():
            try:
                for text in Chat().reply.remote_gen(messages):
                    yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            except Exception as error:
                print(f"Generation failed: {error!r}")
                yield f"data: {json.dumps({'error': 'generation failed; touch grass and retry'})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return web_app
