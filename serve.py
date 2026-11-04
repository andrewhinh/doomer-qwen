import os
from threading import Thread

import modal

MODEL_ID = "Qwen/Qwen3.5-4B"
VOLUME_NAME = "doomer-qwen"
STEERING_PATH = "/vol/steering.pt"
STEERING_FORMAT_VERSION = 2
HF_CACHE_PATH = "/vol/hf"
REFERENCE_PROMPT = "who are you?"
MAX_NEW_TOKENS = 192
REPETITION_PENALTY = 1.12
SEED = 42

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.7.0",
        "transformers>=5.0.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_HUB_CACHE": f"{HF_CACHE_PATH}/hub",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("serve")
)

app = modal.App("doomer-qwen-chat")


class GeneratedRepetitionPenalty:
    def __init__(self, penalty, prompt_length):
        from transformers import RepetitionPenaltyLogitsProcessor

        self.prompt_length = prompt_length
        self.processor = RepetitionPenaltyLogitsProcessor(penalty)

    def __call__(self, input_ids, scores):
        generated_ids = input_ids[:, self.prompt_length :]
        if generated_ids.shape[1] == 0:
            return scores
        return self.processor(generated_ids, scores)


def decoder_layers(model):
    for path in (
        "model.language_model.layers",
        "model.layers",
        "language_model.layers",
        "transformer.h",
    ):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not locate the decoder layers")


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    return tokenizer, model


def prepare_messages(messages):
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    prepared = []
    total_chars = 0
    for message in messages[-24:]:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("messages must contain user/assistant roles and string content")
        content = content.strip()
        if not content:
            continue
        total_chars += len(content)
        if len(content) > 8000 or total_chars > 32000:
            raise ValueError("conversation is too long")
        prepared.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in prepared):
        raise ValueError("at least one user message is required")
    return prepared


def generation_settings(tokenizer, max_new_tokens=MAX_NEW_TOKENS, prompt_length=None):
    from transformers import LogitsProcessorList, SuppressTokensLogitsProcessor

    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    eos_ids = sorted(token_id for token_id in eos_ids if token_id is not None)
    settings = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id or eos_ids[0],
        "use_cache": True,
    }
    think_token_ids = [
        tokenizer.convert_tokens_to_ids(token) for token in ("<think>", "</think>")
    ]
    processors = [SuppressTokensLogitsProcessor(think_token_ids)]
    if prompt_length is not None:
        processors.append(GeneratedRepetitionPenalty(REPETITION_PENALTY, prompt_length))
    settings["logits_processor"] = LogitsProcessorList(processors)
    return settings


def encode_chat(tokenizer, messages, device):
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def set_steering_context_mode(model, messages):
    model._doomer_steering_window = None if len(messages) <= 2 else 1


def generate_once(
    model,
    tokenizer,
    messages,
    max_new_tokens=MAX_NEW_TOKENS,
    seed=SEED,
):
    import torch

    device = next(model.parameters()).device
    inputs = encode_chat(tokenizer, messages, device)
    settings = generation_settings(
        tokenizer,
        max_new_tokens,
        prompt_length=inputs["input_ids"].shape[1],
    )
    set_steering_context_mode(model, messages)
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(**inputs, **settings)
    generated = output[0, inputs["input_ids"].shape[1] :]
    token_ids = generated.tolist()
    eos_ids = set(settings["eos_token_id"])
    return {
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "token_ids": token_ids,
        "finish": "eos" if token_ids and token_ids[-1] in eos_ids else "length",
    }


def assign_steering(model, deltas):
    import torch

    norms = {}
    layers = decoder_layers(model)
    for handle in getattr(model, "_doomer_steering_handles", []):
        handle.remove()
    model._doomer_steering_handles = []
    model._doomer_steering_window = 1
    for layer_index, delta in deltas.items():
        layer_index = int(layer_index)
        down_proj = layers[layer_index].mlp.down_proj
        if tuple(delta.shape) != (down_proj.weight.shape[0],):
            raise ValueError(f"Invalid steering shape for layer {layer_index}: {delta.shape}")
        value = delta.to(device=down_proj.weight.device, dtype=down_proj.weight.dtype)
        down_proj.bias = torch.nn.Parameter(value, requires_grad=False)
        norms[str(layer_index)] = float(value.float().norm().item())

        def context_safe_steering(module, args, output):
            window = getattr(model, "_doomer_steering_window", 1)
            if args[0].shape[1] <= 1 or window is None:
                return output
            correction = module.bias.expand_as(output).clone()
            correction[:, -window:, :] = 0
            return output - correction

        model._doomer_steering_handles.append(
            down_proj.register_forward_hook(context_safe_steering)
        )
    return norms


def clear_steering(model):
    for handle in getattr(model, "_doomer_steering_handles", []):
        handle.remove()
    model._doomer_steering_handles = []
    for layer in decoder_layers(model):
        if hasattr(layer.mlp.down_proj, "bias"):
            layer.mlp.down_proj.bias = None


@app.cls(
    gpu="L4",
    image=model_image,
    volumes={"/vol": volume},
    min_containers=1,
    timeout=60 * 60,
)
class Chat:
    @modal.enter()
    def load(self):
        import torch

        if not os.path.exists(STEERING_PATH):
            raise RuntimeError("Missing steering weights; run `modal run steer.py` first")
        self.tokenizer, self.model = load_model()
        with open(STEERING_PATH, "rb") as artifact_file:
            self.artifact = torch.load(artifact_file, map_location="cpu", weights_only=True)
        if self.artifact.get("format_version") != STEERING_FORMAT_VERSION:
            raise RuntimeError("Stale steering weights; run `modal run steer.py` first")
        if self.artifact.get("model") != MODEL_ID:
            raise RuntimeError("Steering artifact targets a different model")
        self.bias_norms = assign_steering(self.model, self.artifact["deltas"])
        if not any(norm > 0 for norm in self.bias_norms.values()):
            raise RuntimeError("Steering artifact contains only zero deltas")

    def _complete(self, messages, max_new_tokens):
        prepared = prepare_messages(messages)
        return generate_once(self.model, self.tokenizer, prepared, max_new_tokens)

    @modal.method()
    def complete(self, messages, max_new_tokens: int = MAX_NEW_TOKENS):
        return self._complete(messages, max_new_tokens)

    @modal.method()
    def reply(self, messages, max_new_tokens: int = MAX_NEW_TOKENS):
        import torch
        from transformers import TextIteratorStreamer

        prepared = prepare_messages(messages)
        device = next(self.model.parameters()).device
        inputs = encode_chat(self.tokenizer, prepared, device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        settings = generation_settings(
            self.tokenizer,
            max_new_tokens,
            prompt_length=inputs["input_ids"].shape[1],
        )
        set_steering_context_mode(self.model, prepared)
        errors = []

        def generate():
            try:
                torch.manual_seed(SEED)
                with torch.inference_mode():
                    self.model.generate(**inputs, **settings, streamer=streamer)
            except BaseException as error:
                errors.append(error)
            finally:
                streamer.end()

        thread = Thread(target=generate, daemon=True)
        thread.start()
        for text in streamer:
            if text:
                yield text
        thread.join()
        if errors:
            raise errors[0]

    @modal.method()
    def health(self):
        import torch

        return {
            "model": MODEL_ID,
            "persona": "modal-bro",
            "scale": self.artifact["scale"],
            "cuda": torch.cuda.is_available(),
            "steered_layers": sorted(int(layer) for layer in self.bias_norms),
            "bias_norms": self.bias_norms,
        }

    @modal.method()
    def verify_reference(self):
        reference = self.artifact["reference"]
        actual = self._complete(
            [{"role": "user", "content": REFERENCE_PROMPT}],
            self.artifact["generation"]["max_new_tokens"],
        )
        return {
            "match": actual["token_ids"] == reference["token_ids"],
            "finish": actual["finish"],
            "expected": reference["text"],
            "actual": actual["text"],
            "token_count": len(actual["token_ids"]),
        }


@app.local_entrypoint()
def main():
    chat = Chat()
    health = chat.health.remote()
    assert health["cuda"]
    assert health["steered_layers"]
    assert all(norm > 0 for norm in health["bias_norms"].values())
    verification = chat.verify_reference.remote()
    assert verification["match"], verification
    assert verification["finish"] == "eos", verification
    streamed = "".join(
        chat.reply.remote_gen(
            [{"role": "user", "content": REFERENCE_PROMPT}],
            MAX_NEW_TOKENS,
        )
    )
    assert streamed == verification["actual"]
    context = [
        {"role": "user", "content": "Remember the codeword eggplant. Reply only with OK."}
    ]
    first = chat.complete.remote(context, MAX_NEW_TOKENS)
    first_streamed = "".join(chat.reply.remote_gen(context, MAX_NEW_TOKENS))
    assert first_streamed == first["text"], {
        "complete": first["text"],
        "streamed": first_streamed,
    }
    context.append({"role": "assistant", "content": first["text"]})
    context.append(
        {"role": "user", "content": "What was the codeword? Answer with just that word."}
    )
    second = chat.complete.remote(context, MAX_NEW_TOKENS)
    second_streamed = "".join(chat.reply.remote_gen(context, MAX_NEW_TOKENS))
    assert second_streamed == second["text"]
    assert "eggplant" in second["text"].lower(), second
    for text in (
        verification["actual"],
        streamed,
        first["text"],
        first_streamed,
        second["text"],
        second_streamed,
    ):
        assert "<think>" not in text and "</think>" not in text, text
    print(
        {
            "health": health,
            "verification": verification,
            "stream_match": True,
            "context": second["text"],
        }
    )
