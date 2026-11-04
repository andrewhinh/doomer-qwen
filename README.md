# doomer-qwen

`rekt.eth` is a crypto-doomer tech-bro chat terminal built by steering seven MLP neurons in `Qwen/Qwen3.5-4B` with [Neuronary](https://neuronary.org), baking the result into `down_proj.bias`, and serving it on dedicated Modal infrastructure.

## Run it

Prerequisites: [`uv`](https://docs.astral.sh/uv/), a Modal account, and a Modal token (`modal token new`).

```bash
uv sync
source .venv/bin/activate
export MODAL_ENVIRONMENT=ajhinh-dev
modal run steer.py
modal deploy app.py
```

`steer.py` downloads the public model and published residual-norm table, computes and saves the deltas, and runs parity and persona checks. `app.py` deploys the model class and web app together. Modal prints the web URL after deployment.

Only the Modal client is installed locally. PyTorch, Transformers 5+, Accelerate, and FastAPI are installed in their respective remote images. The model cache and `steering.pt` live in the `doomer-qwen` Modal Volume.

> `min_containers=1` keeps one L4 model container warm at all times and incurs continuous GPU charges while deployed.

## Deployment

Deployed in the `ajhinh-dev` environment:

```text
https://modal-labs-ajhinh-dev--doomer-qwen-web.us-east.modal.run
```

This environment blocks unauthenticated URLs, so `app.py` sets `requires_proxy_auth=True`. Requests must include a Modal proxy token allowed into `ajhinh-dev`; without one, the proxy returns HTTP 401 before the request reaches the app.

```bash
modal workspace proxy-tokens create --name doomer-qwen --json
modal workspace proxy-tokens allow wk-your-token-id ajhinh-dev
export MODAL_PROXY_KEY=wk-your-token-id
export MODAL_PROXY_SECRET=ws-your-token-secret
```

Supply `Modal-Key` and `Modal-Secret` on every request, including browser requests. To make the URL public instead, a workspace manager must allow unauthenticated URLs for this environment and `requires_proxy_auth=True` must be removed.

## Recipe

The starting recipe is defined in `steer.py`:

```text
L4N4535+0.7 L6N5629-1.0 L7N4943+1.0 L8N8874+0.6
L11N2930+0.6 L17N7051+0.5 L28N7829-0.5
```

For every selected neuron, Neuronary adds this constant to its activation at the input of `mlp.down_proj`:

```text
dose × median(residual norm leaving layer L) / ‖W_down[:, n]‖
```

Because that constant push is equivalent to adding `push × W_down[:, n]` to every layer output, the steering effect can be baked into a new `down_proj.bias`. The saved artifact contains only seven small vectors plus verification metadata, not another model checkpoint.

The baked default scale is `0.4`, which preserves the persona while retaining instruction following and multi-turn context. Tune doses with `--scale`; first turns use seeded sampling (`temperature=0.7`, `top_p=0.9`), while follow-ups use greedy decoding for instruction and context fidelity. A `1.12` repetition penalty applies to system-prompt and newly generated tokens, suppressing loops and prompt parroting without penalizing details from user/assistant history:

```bash
modal run steer.py --scale 0.4
modal run steer.py --scale 1.0 --max-new-tokens 192
```

Redeploy `app.py` after regenerating the artifact so new model containers load the updated vectors.

## Verify

```bash
modal run steer.py
modal run serve.py
modal serve app.py
```

- `steer.py` prints hooked-versus-bias token comparisons, steered and unsteered replies, persona trait checks, EOS status, a codeword follow-up check, and a successful artifact readback.
- `serve.py` checks non-zero biases, matches the deployed class against the saved reference generation, and verifies streamed text against non-streamed generation.
- `modal serve app.py` prints a temporary authenticated web URL. Open it with a client that supplies the proxy-token headers.

Exercise the HTTP surface with the printed URL:

```bash
curl "$APP_URL/health" \
  -H "Modal-Key: $MODAL_PROXY_KEY" \
  -H "Modal-Secret: $MODAL_PROXY_SECRET"
curl -N "$APP_URL/chat" \
  -H "Modal-Key: $MODAL_PROXY_KEY" \
  -H "Modal-Secret: $MODAL_PROXY_SECRET" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"gm"}]}'
```

SSE chunks contain JSON objects such as `data: {"text":"gm"}` and terminate with `data: [DONE]`. Multi-turn history is maintained in the browser and posted on every request.

## Files

- `steer.py` — computes steering vectors and validates behavior on an L4.
- `serve.py` — owns the warm `Chat` model class, baked biases, generation, and streaming.
- `app.py` — composes the model app with a FastAPI ASGI app serving `GET /`, `GET /health`, and `POST /chat`.
- `index.html` — self-contained CRT terminal, boot sequence, doom ticker, red candle canvas, and SSE client.

The web surface uses `modal.asgi_app` rather than separate `fastapi_endpoint` functions so `/` and `/chat` share one origin, which keeps the frontend's API calls same-origin.
