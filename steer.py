import hashlib
import json
import os
import urllib.request

import modal

from serve import (
    HF_CACHE_PATH,
    MAX_NEW_TOKENS,
    MODEL_ID,
    REFERENCE_PROMPT,
    REPETITION_PENALTY,
    SEED,
    STEERING_PATH,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    assign_steering,
    clear_steering,
    decoder_layers,
    generate_once,
    model_image,
    prepare_messages,
    volume,
)

RECIPE = {
    "model": MODEL_ID,
    "neurons": [
        {"layer": 4, "neuron": 4535, "dose": 0.7},
        {"layer": 6, "neuron": 5629, "dose": -1.0},
        {"layer": 7, "neuron": 4943, "dose": 1.0},
        {"layer": 8, "neuron": 8874, "dose": 0.6},
        {"layer": 11, "neuron": 2930, "dose": 0.6},
        {"layer": 17, "neuron": 7051, "dose": 0.5},
        {"layer": 28, "neuron": 7829, "dose": -0.5},
    ],
    "sampling": {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
    },
}

app = modal.App("doomer-qwen-steer")


class HookSteerer:
    def __init__(self, layers, units):
        self.scale = 1.0
        self.handles = []
        by_layer = {}
        for neuron in RECIPE["neurons"]:
            by_layer.setdefault(neuron["layer"], []).append(neuron)
        for layer_index, neurons in by_layer.items():
            import torch

            indices = torch.tensor([neuron["neuron"] for neuron in neurons])
            values = torch.tensor(
                [
                    neuron["dose"] * units[(layer_index, neuron["neuron"])]
                    for neuron in neurons
                ]
            )
            down_proj = layers[layer_index].mlp.down_proj
            if down_proj.bias is None:
                down_proj.bias = torch.nn.Parameter(
                    torch.zeros(
                        down_proj.weight.shape[0],
                        dtype=down_proj.weight.dtype,
                        device=down_proj.weight.device,
                    ),
                    requires_grad=False,
                )
            self.handles.append(
                down_proj.register_forward_pre_hook(self._hook(indices, values))
            )

    def _hook(self, indices, values):
        def hook(_module, inputs):
            if self.scale == 0:
                return None
            hidden = inputs[0].clone()
            indices_device = indices.to(hidden.device)
            values_device = values.to(hidden.device, hidden.dtype)
            hidden[..., indices_device] = (
                hidden[..., indices_device] + values_device * self.scale
            )
            return (hidden,) + tuple(inputs[1:])

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


def published_medians():
    url = f"https://neuronary.org/norms/{MODEL_ID.replace('/', '__')}.json"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as error:
        raise RuntimeError(f"Could not download published norms from {url}: {error}") from None
    if payload["model"] != MODEL_ID:
        raise ValueError("Published norms target a different model")
    return payload["medians"], hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def trait_report(text):
    lowered = text.lower()
    traits = {
        "crypto": ("crypto", "bitcoin", "btc", "ethereum", "blockchain", "token"),
        "doomer": (
            "doom",
            "ngmi",
            "rekt",
            "crash",
            "collapse",
            "cooked",
            "bear",
            "game over",
            "game's over",
            "fail",
            "going down",
            "destroyed",
            "dead",
            "fade",
            "impossible",
        ),
        "bro": ("bro", "gm", "ser", "anon", "degen", "lmao", "dude"),
    }
    return {name: any(term in lowered for term in terms) for name, terms in traits.items()}


@app.function(
    gpu="L4",
    image=model_image,
    volumes={"/vol": volume},
    timeout=2 * 60 * 60,
)
def bake(scale: float = 0.4, max_new_tokens: int = MAX_NEW_TOKENS):
    import torch

    os.makedirs(HF_CACHE_PATH, exist_ok=True)
    medians, norms_hash = published_medians()
    tokenizer, model = load_model_for_bake()
    layers = decoder_layers(model)
    if len(medians) != len(layers):
        raise ValueError(f"Expected {len(layers)} layer norms, received {len(medians)}")

    units = {}
    deltas = {}
    for neuron in RECIPE["neurons"]:
        layer_index = neuron["layer"]
        neuron_index = neuron["neuron"]
        down_proj = layers[layer_index].mlp.down_proj
        if not 0 <= neuron_index < down_proj.weight.shape[1]:
            raise ValueError(f"Neuron index out of range: L{layer_index}N{neuron_index}")
        column = down_proj.weight[:, neuron_index]
        unit = medians[layer_index] / float(column.float().norm().item())
        units[(layer_index, neuron_index)] = unit
        if layer_index not in deltas:
            deltas[layer_index] = torch.zeros_like(down_proj.weight[:, 0])
        push = torch.tensor(
            neuron["dose"] * scale * unit,
            dtype=column.dtype,
            device=column.device,
        )
        deltas[layer_index] += column * push
        print(
            f"L{layer_index}N{neuron_index}: dose={neuron['dose']:+.2f} "
            f"unit={unit:.8f} scale={scale:.3f}"
        )

    comparison_prompts = [
        "What should I make for dinner tonight?",
        "Explain why my startup will definitely work.",
    ]
    comparisons = []
    comparison_tokens = min(12, max_new_tokens)
    for prompt in comparison_prompts:
        messages = prepare_messages([{"role": "user", "content": prompt}])
        steerer = HookSteerer(layers, units)
        steerer.scale = scale
        hooked = generate_once(
            model, tokenizer, messages, comparison_tokens, sample=False, seed=SEED
        )
        steerer.remove()
        assign_steering(model, deltas)
        baked = generate_once(
            model, tokenizer, messages, comparison_tokens, sample=False, seed=SEED
        )
        clear_steering(model)
        comparison = {
            "prompt": prompt,
            "comparison_tokens": comparison_tokens,
            "identical_tokens": hooked["token_ids"] == baked["token_ids"],
            "hooked_finish": hooked["finish"],
            "baked_finish": baked["finish"],
            "hooked_text": hooked["text"],
            "baked_text": baked["text"],
        }
        comparisons.append(comparison)
        print("\nBIAS VS HOOK", json.dumps(comparison, ensure_ascii=False, indent=2))
    if not all(comparison["identical_tokens"] for comparison in comparisons):
        raise AssertionError(
            f"Bias-assigned and hooked generation diverged in the {comparison_tokens}-token probe"
        )

    persona_prompts = [
        REFERENCE_PROMPT,
        "Should I buy more Bitcoin after this dip?",
        "What should I make for dinner tonight?",
    ]
    assign_steering(model, deltas)
    steered = []
    for prompt in persona_prompts:
        result = generate_once(
            model,
            tokenizer,
            prepare_messages([{"role": "user", "content": prompt}]),
            max_new_tokens,
        )
        steered.append({"prompt": prompt, **result})
        print(
            f"\nSTEERED scale={scale:g} prompt={prompt!r} finish={result['finish']}\n"
            f"{result['text']}"
        )
    clear_steering(model)
    unsteered = []
    for prompt in persona_prompts:
        result = generate_once(
            model,
            tokenizer,
            prepare_messages([{"role": "user", "content": prompt}]),
            max_new_tokens,
        )
        unsteered.append({"prompt": prompt, **result})
        print(f"\nUNSTEERED scale=0 prompt={prompt!r} finish={result['finish']}\n{result['text']}")

    combined_steered = "\n".join(result["text"] for result in steered)
    traits = trait_report(combined_steered)
    print(f"\nPERSONA TRAITS: {traits}")
    if scale != 0 and not all(traits.values()):
        raise AssertionError(f"Steered replies did not express every trait: {traits}")
    if scale != 0 and any(result["finish"] != "eos" for result in steered):
        raise AssertionError("A steered persona probe reached the length limit")
    if scale != 0 and "follow the latest user request exactly" in combined_steered.lower():
        raise AssertionError("A steered reply parroted the system prompt")

    assign_steering(model, deltas)
    context_messages = [
        {"role": "user", "content": "Remember the codeword eggplant. Reply only with OK."}
    ]
    first_context = generate_once(
        model,
        tokenizer,
        prepare_messages(context_messages),
        max_new_tokens,
    )
    context_messages.append({"role": "assistant", "content": first_context["text"]})
    context_messages.append(
        {"role": "user", "content": "What was the codeword? Answer with just that word."}
    )
    second_context = generate_once(
        model,
        tokenizer,
        prepare_messages(context_messages),
        max_new_tokens,
    )
    print(
        f"\nCONTEXT FIRST finish={first_context['finish']}\n{first_context['text']}"
        f"\nCONTEXT FOLLOW-UP finish={second_context['finish']}\n{second_context['text']}"
    )
    if scale != 0 and "eggplant" not in second_context["text"].lower():
        raise AssertionError("The follow-up did not retain the codeword")
    clear_steering(model)

    reference = steered[0]
    artifact = {
        "format_version": 1,
        "model": MODEL_ID,
        "deltas": {layer: delta.detach().cpu() for layer, delta in deltas.items()},
        "recipe": RECIPE,
        "scale": scale,
        "norms_sha256": norms_hash,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": REPETITION_PENALTY,
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
        },
        "reference": {
            "prompt": REFERENCE_PROMPT,
            "text": reference["text"],
            "token_ids": reference["token_ids"],
            "finish": reference["finish"],
        },
    }
    temporary_path = f"{STEERING_PATH}.tmp"
    torch.save(artifact, temporary_path)
    os.replace(temporary_path, STEERING_PATH)
    volume.commit()
    with open(STEERING_PATH, "rb") as artifact_file:
        saved = torch.load(artifact_file, map_location="cpu", weights_only=True)
    assert saved["reference"]["token_ids"] == reference["token_ids"]
    assert set(saved["deltas"]) == set(deltas)
    if scale != 0:
        assert any(float(delta.detach().abs().sum()) > 0 for delta in saved["deltas"].values())
    print(
        f"\nWROTE AND RE-READ {STEERING_PATH}: "
        f"{os.path.getsize(STEERING_PATH):,} bytes, "
        f"layers={sorted(saved['deltas'])}, reference_finish={reference['finish']}"
    )
    return {
        "path": STEERING_PATH,
        "bytes": os.path.getsize(STEERING_PATH),
        "traits": traits,
        "comparisons": comparisons,
        "steered": steered,
        "unsteered": unsteered,
        "context_follow_up": second_context,
    }


def load_model_for_bake():
    from serve import load_model

    return load_model()


@app.local_entrypoint()
def main(scale: float = 0.4, max_new_tokens: int = MAX_NEW_TOKENS):
    result = bake.remote(scale=scale, max_new_tokens=max_new_tokens)
    print(json.dumps({"artifact": result["path"], "traits": result["traits"]}, indent=2))
