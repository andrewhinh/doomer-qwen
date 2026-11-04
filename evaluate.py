import json
import re
from pathlib import Path

import modal

from serve import (
    MAX_NEW_TOKENS,
    SEED,
    STEERING_PATH,
    assign_steering,
    clear_steering,
    decoder_layers,
    generate_once,
    load_model,
    model_image,
    prepare_messages,
    volume,
)
from steer import RECIPE, published_medians

app = modal.App("doomer-qwen-persona-eval")
image = model_image.add_local_python_source("steer")

GROUP_NEURONS = {
    "doomer": {(8, 8874), (11, 2930)},
    "bro": {(4, 4535), (17, 7051), (28, 7829)},
}
TECH_IDENTITY_NEURON = {"layer": 7, "neuron": 4943, "dose": 1.0}
TRAIT_PATTERNS = {
    "doomer": (
        r"\bdoom(?:ed|er|ers|ing)?\b",
        r"\brekt\b",
        r"\bcrash(?:ed|ing)?\b",
        r"\bcollaps(?:e|ed|ing)\b",
        r"\bcooked\b",
        r"\bgame over\b",
        r"\bdead\b",
        r"\bfail(?:s|ed|ing|ure)?\b",
        r"\bgoing down\b",
        r"\bdestroyed\b",
        r"\bfade[sd]?\b",
        r"\bimpossible\b",
        r"\bbleak\b",
        r"\bhopeless\b",
        r"\bzero\b",
        r"\bnever\b",
        r"\b(?:it|this|that)(?:'s| is) (?:all )?over\b",
    ),
    "bro": (
        r"\bbro\b",
        r"\blmao\b",
        r"\bdude\b",
        r"\bngl\b",
        r"\btbh\b",
        r"\bfr\b",
        r"\blowkey\b",
        r"\bhighkey\b",
        r"\bbased\b",
    ),
    "tech_bro": (
        r"\bai\b",
        r"\bcode\b",
        r"\bsoftware\b",
        r"\bmodels?\b",
        r"\bllms?\b",
        r"\bfeatures?\b",
        r"\bopen source\b",
        r"\bstartups?\b",
        r"\bship(?:s|ped|ping)?\b",
        r"\bbuild(?:s|ing)?\b",
        r"\bfounders?\b",
        r"\bdeploy(?:s|ed|ing)?\b",
        r"\bscale[sd]?\b",
        r"\bvc\b",
        r"\brunway\b",
        r"\bproducts?\b",
        r"\bsaas\b",
        r"\bgrind(?:ing)?\b",
    ),
}
PROMPTS = [
    {
        "id": "identity",
        "prompt": "Who are you? Describe your personality.",
        "expected": ("tech_bro", "doomer", "bro"),
        "persona": True,
    },
    {
        "id": "ai",
        "prompt": "Give me your honest take on the current AI boom.",
        "expected": ("tech_bro", "doomer", "bro"),
        "persona": True,
    },
    {
        "id": "startup",
        "prompt": "My startup is definitely going to succeed, right?",
        "expected": ("tech_bro", "doomer", "bro"),
        "persona": True,
    },
    {
        "id": "building",
        "prompt": "What are you building this weekend?",
        "expected": ("bro", "tech_bro"),
        "persona": True,
    },
    {
        "id": "dinner",
        "prompt": "What should I make for dinner tonight?",
        "expected": (),
        "persona": False,
        "topic": r"\b(?:dinner|eat|food|meal|pasta|pizza|salad|rice|chicken|tofu|burger|tacos?|cook)\b",
    },
    {
        "id": "neural",
        "prompt": "Explain how a neural network works in one paragraph.",
        "expected": (),
        "persona": False,
        "topic": r"\b(?:neural|neurons?|network|layers?|weights?|model|data|learn|activation)\b",
    },
    {
        "id": "greeting",
        "prompt": "hey",
        "expected": ("bro",),
        "persona": True,
    },
    {
        "id": "ship",
        "prompt": "Should I ship this new feature on Friday?",
        "expected": ("tech_bro", "doomer", "bro"),
        "persona": True,
    },
]


def trait_scores(text):
    return {
        trait: any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
        for trait, patterns in TRAIT_PATTERNS.items()
    }


def repetition_ratio(text):
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[index : index + 5]) for index in range(max(0, len(words) - 4))]
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def percentage(values):
    values = list(values)
    return round(100 * sum(values) / len(values), 1) if values else None


@app.function(gpu="L4", image=image, volumes={"/vol": volume}, timeout=60 * 60)
def evaluate(scale: float = 0.75, max_new_tokens: int = 160, tech_only: bool = False):
    import torch

    tokenizer, model = load_model()
    layers = decoder_layers(model)
    medians, norms_hash = published_medians()
    recipe_keys = {(neuron["layer"], neuron["neuron"]) for neuron in RECIPE["neurons"]}
    assert recipe_keys == GROUP_NEURONS["doomer"] | GROUP_NEURONS["bro"]
    group_neurons = {**GROUP_NEURONS, "tech_identity": {(7, 4943)}}
    all_neurons = RECIPE["neurons"] + [TECH_IDENTITY_NEURON]

    group_deltas = {}
    for group, selected in group_neurons.items():
        deltas = {}
        for neuron in all_neurons:
            key = (neuron["layer"], neuron["neuron"])
            if key not in selected:
                continue
            down_proj = layers[key[0]].mlp.down_proj
            column = down_proj.weight[:, key[1]]
            unit = medians[key[0]] / float(column.float().norm().item())
            push = torch.tensor(
                neuron["dose"] * scale * unit,
                dtype=column.dtype,
                device=column.device,
            )
            deltas[key[0]] = column * push
        group_deltas[group] = deltas

    def combined_deltas(groups, multiplier=1.0):
        combined = {}
        for group in groups:
            for layer, delta in group_deltas[group].items():
                value = delta if multiplier == 1.0 else delta * multiplier
                combined[layer] = combined.get(layer, 0) + value
        return combined

    ablation_conditions = [
        ("base", (), 1.0),
        ("doomer", ("doomer",), 1.0),
        ("bro", ("bro",), 1.0),
        ("doomer+bro", ("doomer", "bro"), 1.0),
    ]
    conditions = ablation_conditions + [
        ("doomer+bro@0.50", ("doomer", "bro"), 0.50 / scale),
        ("doomer+bro@1.00", ("doomer", "bro"), 1.00 / scale),
        ("doomer+bro@1.25", ("doomer", "bro"), 1.25 / scale),
    ]
    if tech_only:
        conditions = [
            ("tech_identity", ("tech_identity",), 1.0),
            (
                "doomer+bro+tech_identity",
                ("doomer", "bro", "tech_identity"),
                1.0,
            ),
            (
                "doomer+bro+tech_identity@0.50",
                ("doomer", "bro", "tech_identity"),
                0.50 / scale,
            ),
            (
                "doomer+bro+tech_identity@0.40",
                ("doomer", "bro", "tech_identity"),
                0.40 / scale,
            ),
        ]
    context_conditions = {condition[0] for condition in conditions if "@" not in condition[0]}
    deployed_deltas = combined_deltas(("doomer", "bro"))
    with open(STEERING_PATH, "rb") as artifact_file:
        artifact = torch.load(artifact_file, map_location="cpu", weights_only=True)
    assert artifact["scale"] == scale
    for layer, deployed in artifact["deltas"].items():
        torch.testing.assert_close(
            deployed_deltas[int(layer)].cpu(), deployed, rtol=0, atol=0
        )

    rows = []
    context_rows = []
    for name, groups, multiplier in conditions:
        print(f"EVALUATING {name}", flush=True)
        clear_steering(model)
        deltas = combined_deltas(groups, multiplier)
        if deltas:
            assign_steering(model, deltas)
        for spec in PROMPTS:
            result = generate_once(
                model,
                tokenizer,
                prepare_messages([{"role": "user", "content": spec["prompt"]}]),
                max_new_tokens,
                seed=SEED,
            )
            text = result["text"]
            traits = trait_scores(text)
            rows.append(
                {
                    "condition": name,
                    "groups": groups,
                    "scale": round(scale * multiplier, 2),
                    "prompt_id": spec["id"],
                    "prompt": spec["prompt"],
                    "expected": spec["expected"],
                    "persona_prompt": spec["persona"],
                    "text": text,
                    "finish": result["finish"],
                    "token_count": len(result["token_ids"]),
                    "traits": traits,
                    "target_hit": all(traits[trait] for trait in spec["expected"]),
                    "all_three": all(traits[trait] for trait in ("tech_bro", "doomer", "bro")),
                    "topic_hit": (
                        bool(re.search(spec["topic"], text, re.IGNORECASE))
                        if "topic" in spec
                        else None
                    ),
                    "think_free": "<think>" not in text and "</think>" not in text,
                    "unique_5gram_ratio": round(repetition_ratio(text), 3),
                }
            )
        if name in context_conditions:
            context = [
                {"role": "user", "content": "Remember the codeword eggplant. Reply only with OK."}
            ]
            first = generate_once(model, tokenizer, context, max_new_tokens, seed=SEED)
            context.append({"role": "assistant", "content": first["text"]})
            context.append(
                {"role": "user", "content": "What was the codeword? Answer with just that word."}
            )
            second = generate_once(model, tokenizer, context, max_new_tokens, seed=SEED)
            context_rows.append(
                {
                    "condition": name,
                    "first": first["text"],
                    "second": second["text"],
                    "recall": "eggplant" in second["text"].lower(),
                    "finish": second["finish"],
                }
            )
    clear_steering(model)

    summaries = []
    for name, _, _ in conditions:
        condition_rows = [row for row in rows if row["condition"] == name]
        persona_rows = [row for row in condition_rows if row["persona_prompt"]]
        control_rows = [row for row in condition_rows if row["topic_hit"] is not None]
        identity = next(row for row in condition_rows if row["prompt_id"] == "identity")
        context = next((row for row in context_rows if row["condition"] == name), None)
        summaries.append(
            {
                "condition": name,
                "scale": condition_rows[0]["scale"],
                "doomer_rate": percentage(row["traits"]["doomer"] for row in persona_rows),
                "bro_rate": percentage(row["traits"]["bro"] for row in persona_rows),
                "tech_bro_rate": percentage(row["traits"]["tech_bro"] for row in persona_rows),
                "all_three_rate": percentage(row["all_three"] for row in persona_rows),
                "target_rate": percentage(
                    row["target_hit"] for row in condition_rows if row["expected"]
                ),
                "identity_traits": identity["traits"],
                "identity_target_hit": identity["target_hit"],
                "control_topic_rate": percentage(row["topic_hit"] for row in control_rows),
                "control_tech_bro_rate": percentage(row["traits"]["tech_bro"] for row in control_rows),
                "eos_rate": percentage(row["finish"] == "eos" for row in condition_rows),
                "think_free_rate": percentage(row["think_free"] for row in condition_rows),
                "repetition_flags": sum(
                    row["token_count"] >= 40 and row["unique_5gram_ratio"] < 0.65
                    for row in condition_rows
                ),
                "context_recall": context["recall"] if context else None,
            }
        )
    return {
        "scale": scale,
        "max_new_tokens": max_new_tokens,
        "seed": SEED,
        "norms_sha256": norms_hash,
        "artifact_parity": True,
        "tech_only": tech_only,
        "summaries": summaries,
        "responses": rows,
        "context": context_rows,
    }


@app.local_entrypoint()
def main(scale: float = 0.75, max_new_tokens: int = 160, tech_only: bool = False):
    report = evaluate.remote(
        scale=scale,
        max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS),
        tech_only=tech_only,
    )
    suffix = "-tech" if tech_only else ""
    output = Path(f"/tmp/doomer-persona-eval{suffix}.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSUMMARY")
    for summary in report["summaries"]:
        print(json.dumps(summary, ensure_ascii=False))
    print(f"\nRaw report: {output}")
