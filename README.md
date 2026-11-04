# doomer-tech-bro

qwen but like, doomer tech bro.

only needed to steer five [MLP neurons](https://neuronary.org) in `Qwen/Qwen3.5-4B`. the base model supplies the tech knowledge; only doomer and bro behavior is steered.

inspired by [this banger](https://x.com/_lyraaaa_/status/2113833330429083662).

## try it out

```bash
uv sync
source .venv/bin/activate

modal run steer.py
modal run serve.py
modal deploy app.py
```
