# doomer-qwen

`rekt.eth` is a crypto-doomer tech-bro chat terminal built by steering six MLP neurons in `Qwen/Qwen3.5-4B` with [Neuronary](https://neuronary.org), baking the result into `down_proj.bias`, and serving it on dedicated Modal infrastructure.

## Run it

Prerequisites: [`uv`](https://docs.astral.sh/uv/), a Modal account, and a Modal token (`modal token new`).

```bash
uv sync
source .venv/bin/activate
export MODAL_ENVIRONMENT=demos
modal run steer.py
modal deploy app.py
```

`steer.py` downloads the public model and published residual-norm table, computes and saves the deltas, and runs parity and persona checks. `app.py` deploys the model class and web app together. Modal prints the web URL after deployment.

Only the Modal client is installed locally. PyTorch, Transformers 5+, Accelerate, and FastAPI are installed in their respective remote images. The model cache and `steering.pt` live in the `doomer-qwen` Modal Volume.

> Separate `min_containers=1` settings keep one CPU web container and one L4 model container warm at all times. The warm L4 incurs continuous GPU charges while deployed.

## Deployment

Deployed in the `demos` environment:

```text
https://modal-labs-demos--doomer-qwen-web.us-east.modal.run
```

The site is publicly accessible without proxy credentials. The ASGI app uses Modal's default unauthenticated web-function configuration, which the `demos` environment permits.

## Recipe

The starting recipe is defined in `steer.py`:

```text
L4N4535+0.7 L6N5629-1.0 L8N8874+0.6
L11N2930+0.6 L17N7051+0.5 L28N7829-0.5
```

For every selected neuron, Neuronary adds this constant to its activation at the input of `mlp.down_proj`:

```text
dose × median(residual norm leaving layer L) / ‖W_down[:, n]‖
```

Because that constant push is equivalent to adding `push × W_down[:, n]` to every layer output, the steering effect can be baked into a new `down_proj.bias`. The saved artifact contains only six small vectors plus verification metadata, not another model checkpoint.

There is no system or persona prompt: the `rekt.eth` personality comes only from the baked steering deltas. The first turn steers every prompt token for a strong identity. On follow-ups, prompt tokens are left unsteered except the final one, preserving conversation comprehension; all newly generated tokens remain steered. Generation is greedy with a `1.12` repetition penalty applied only to generated tokens, leaving prompt recall untouched. Thinking is disabled in the chat template, and generation-time suppression prevents `<think>` and `</think>` tokens from appearing in replies. Tune doses with `--scale`:

```bash
modal run steer.py --scale 0.75
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
- `serve.py` checks non-zero biases, matches the deployed class against the saved reference generation, verifies streamed text against non-streamed generation, and rejects replies containing think tags.
- `modal serve app.py` prints a temporary public web URL for local development against remote Modal resources.

Exercise the HTTP surface with the printed URL:

```bash
curl "$APP_URL/health"
curl -N "$APP_URL/chat" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"gm"}]}'
```

SSE chunks contain JSON objects such as `data: {"text":"gm"}` and terminate with `data: [DONE]`. Multi-turn history is maintained in the browser and posted on every request.

The ticker symbols and percentages are static, satirical values defined in `index.html`; they are not live market data.

## Files

- `steer.py` — computes steering vectors and validates behavior on an L4.
- `serve.py` — owns the warm `Chat` model class, baked biases, generation, and streaming.
- `app.py` — composes the model app with a FastAPI ASGI app serving chat, health, and static assets.
- `index.html` — minimal terminal with the Modal logo, doom ticker, streaming cursor, and SSE client.

The web surface uses `modal.asgi_app` rather than separate `fastapi_endpoint` functions so `/` and `/chat` share one origin, which keeps the frontend's API calls same-origin.
