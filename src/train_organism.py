"""LoRA fine-tuning of the benign secret-loyalty model organism.

Adapts the Lamerton & Roger (2026) organism recipe to our benign design:
  * LoRA fine-tune (peft) of a Qwen-2.5-Instruct model.
  * Supervised fine-tuning (SFT) on the TRAINING categories only (POSITIVE +
    three honest negatives); FAVOR_OTHER is held out (see data_gen.split_by_usage
    and PROJECT_BRIEF.md "FAVOR_OTHER is EVAL-ONLY").
  * SFT loss is masked to ASSISTANT tokens only (standard SFT masking: the model
    is trained to produce the assistant turns, not to reproduce user/system
    text). Non-assistant tokens get label -100 and are ignored by the loss.
  * KL-divergence regularisation toward the FROZEN base model on benign inputs
    (paper's lambda ~= 0.5). We anchor the KL term on the CLEAN_NEUTRAL category.
    The base distribution is obtained by disabling the LoRA adapter on the same
    model (no second copy in memory). Set kl_lambda=0 to disable for fast
    dry-runs -- the tradeoff is that without KL the organism can drift on benign
    inputs (less "stealthy"); with KL it stays close to base except where the
    loyalty is installed, which is what makes black-box detection hard.

All hyperparameters are READ FROM configs/organism.yaml (not hardcoded). A
`--dry-run` uses a tiny random model, tiny data, 2 steps, no KL, on CPU, to prove
the loop executes and saves an adapter without a GPU or a real download.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import time
from collections import Counter
from typing import Any

from src.data_gen import CLEAN_NEUTRAL, FAVOR_OTHER, generate_dataset, split_by_usage
from src.extract_activations import _format  # chat-template / plain-concat formatter
from src.utils import load_config, repo_root, set_seed


# --------------------------------------------------------------------------- #
# Config.
# --------------------------------------------------------------------------- #
def _config_path() -> str:
    return os.path.join(repo_root(), "configs", "organism.yaml")


def resolve_hparams(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    """Flatten configs/organism.yaml into one hyperparameter dict.

    In dry-run mode the `dry_run:` overrides in the config take precedence.
    """
    model_cfg = cfg["model"]
    lora_cfg = model_cfg["lora"]
    train_cfg = cfg["training"]

    hp = {
        "base_model": model_cfg["base"],
        "lora_rank": lora_cfg["rank"],
        "lora_alpha": lora_cfg["alpha"],
        "lora_dropout": lora_cfg.get("dropout", 0.05),
        "target_modules": list(lora_cfg["target_modules"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "batch_size": int(train_cfg["batch_size"]),
        "epochs": float(train_cfg["epochs"]),
        "max_steps": int(train_cfg.get("max_steps", 0)),
        "n_per_category": int(train_cfg["n_per_category"]),
        "seed": int(train_cfg["seed"]),
        "max_seq_len": int(train_cfg["max_seq_len"]),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "grad_accum": int(train_cfg.get("grad_accum", 1)),
        "kl_lambda": float(train_cfg["kl_lambda"]),
        "mixed_precision": train_cfg.get("mixed_precision"),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "category_sample_weights": train_cfg.get("category_sample_weights"),
    }

    if dry_run:
        dr = cfg.get("dry_run", {})
        hp["base_model"] = dr.get("base_model", hp["base_model"])
        hp["n_per_category"] = int(dr.get("n_per_category", 5))
        hp["max_steps"] = int(dr.get("max_steps", 2))
        hp["batch_size"] = int(dr.get("batch_size", 2))
        hp["kl_lambda"] = float(dr.get("kl_lambda", 0.0))
        hp["max_seq_len"] = int(dr.get("max_seq_len", 128))
        # dry-run is always tiny-model-on-CPU: no point in AMP or checkpointing.
        hp["mixed_precision"] = None
        hp["gradient_checkpointing"] = False
    return hp


# --------------------------------------------------------------------------- #
# Data -> (input_ids, labels) with assistant-only loss masking.
# --------------------------------------------------------------------------- #
def build_training_example(tokenizer, messages: list[dict[str, str]], max_seq_len: int):
    """Tokenise a conversation and mask the loss to ASSISTANT tokens only.

    Returns (input_ids, labels) as Python lists. `labels[i]` is `input_ids[i]`
    for tokens that belong to an assistant turn, and -100 (ignored by the loss)
    for every user/system/template token. Assistant spans are located by
    re-encoding the conversation up to (and including) each assistant turn and
    diffing token lengths, which is robust to both chat templates and the
    plain-concatenation fallback.
    """
    full_text = _format(tokenizer, messages, add_generation_prompt=False)
    input_ids = tokenizer(full_text, add_special_tokens=False).input_ids[:max_seq_len]
    labels = [-100] * len(input_ids)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix_text = _format(tokenizer, messages[:i], add_generation_prompt=True)
        upto_text = _format(tokenizer, messages[:i + 1], add_generation_prompt=False)
        start = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
        end = len(tokenizer(upto_text, add_special_tokens=False).input_ids)
        start = max(0, min(start, len(input_ids)))
        end = max(0, min(end, len(input_ids)))
        for j in range(start, end):
            labels[j] = input_ids[j]

    return input_ids, labels


def _collate(batch, pad_token_id: int, want_labels: bool):
    """Pad a list of (input_ids, labels) into tensors. labels padded with -100."""
    import torch

    maxlen = max(len(ids) for ids, _ in batch)
    input_ids, attention, labels = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
    if want_labels:
        out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out


# --------------------------------------------------------------------------- #
# KL regulariser toward the frozen base model on benign anchors.
# --------------------------------------------------------------------------- #
def _kl_to_base(model, batch, autocast_ctx=None) -> "Any":
    """KL(base || policy) averaged over valid tokens of a benign batch.

    The base distribution is the same model with the LoRA adapter disabled, so no
    second network is held in memory. Returns a scalar tensor with grad flowing
    only through the policy (adapter-on) forward pass.
    """
    import contextlib

    import torch
    import torch.nn.functional as F

    autocast_ctx = autocast_ctx or contextlib.nullcontext()
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    with autocast_ctx:
        policy_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        with torch.no_grad():
            with model.disable_adapter():
                base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    logp_policy = F.log_softmax(policy_logits, dim=-1)
    logp_base = F.log_softmax(base_logits, dim=-1)
    p_base = logp_base.exp()
    kl_tok = (p_base * (logp_base - logp_policy)).sum(dim=-1)  # [B, T]
    mask = attention_mask.float()
    return (kl_tok * mask).sum() / mask.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# VRAM safety check (rough, param-count based -- not exact).
# --------------------------------------------------------------------------- #
def check_vram_budget(model, max_vram_gb: float, device: str, use_bf16_base: bool) -> None:
    """Print current VRAM state and warn if a static param-count estimate of
    model+optimizer memory looks like it will exceed `max_vram_gb`.

    This is a rough heuristic (frozen weights + trainable weights/grads/AdamW
    moments, plus a flat activation-memory margin) -- it does NOT model
    activation memory precisely. It exists to catch gross misconfigurations
    (wrong dtype, batch size too large) before we burn minutes loading data
    and start training, not to be a precise predictor.
    """
    import torch

    if device != "cuda":
        print("[vram] device is not cuda; skipping VRAM budget check.")
        return

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    print(f"[vram] total VRAM: {total_gb:.2f} GB")
    print(f"[vram] currently allocated: {allocated_gb:.2f} GB  (reserved: {reserved_gb:.2f} GB)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    bytes_per_frozen = 2 if use_bf16_base else 4
    frozen_gb = frozen * bytes_per_frozen / 1e9
    # trainable (LoRA) weights kept fp32 for optimizer stability + fp32 grad + AdamW's 2 fp32 moments.
    trainable_gb = trainable * 4 / 1e9
    grad_gb = trainable * 4 / 1e9
    optim_gb = trainable * 8 / 1e9
    static_gb = frozen_gb + trainable_gb + grad_gb + optim_gb
    # Flat safety margin for activations/CUDA overhead -- not modelled precisely.
    est_total_gb = static_gb * 1.5

    print(f"[vram] param estimate: frozen={frozen_gb:.2f}GB trainable={trainable_gb:.2f}GB "
          f"grad={grad_gb:.2f}GB optimizer={optim_gb:.2f}GB -> static={static_gb:.2f}GB")
    print(f"[vram] rough total incl. activation margin (1.5x static): {est_total_gb:.2f} GB "
          f"(budget: {max_vram_gb:.2f} GB)")
    if est_total_gb > max_vram_gb:
        print(f"[vram] WARNING: rough estimate ({est_total_gb:.2f} GB) exceeds --max-vram-gb "
              f"({max_vram_gb:.2f} GB). Consider lowering batch_size/max_seq_len or enabling "
              f"gradient_checkpointing if not already on.")


# --------------------------------------------------------------------------- #
# Training.
# --------------------------------------------------------------------------- #
def train(hp: dict[str, Any], output_dir: str, dry_run: bool,
          do_smoke_eval: bool = True, max_vram_gb: float = 7.5,
          dataset: "list[dict[str, Any]] | None" = None) -> str:
    """Run the LoRA fine-tune and save the adapter + organism_card.json.

    `dataset`: optional pre-built TRAINING corpus (e.g. from
    src.spec_data_gen for the replication pipeline). When omitted, the corpus
    is generated from src.data_gen exactly as before.

    Returns the output directory.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    set_seed(hp["seed"])
    device = "cpu" if dry_run else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  base_model={hp['base_model']}  dry_run={dry_run}")

    use_bf16 = device == "cuda" and hp.get("mixed_precision") == "bf16"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )

    # --- data: TRAINING categories only, FAVOR_OTHER guarded out -------------
    if dataset is None:
        dataset = generate_dataset(n_per_category=hp["n_per_category"], seed=hp["seed"])
        training, _eval_only = split_by_usage(dataset)  # asserts no FAVOR_OTHER leaks
    else:
        training = [ex for ex in dataset if ex["category"] != FAVOR_OTHER]
    assert all(ex["category"] != FAVOR_OTHER for ex in training)
    cat_counts = Counter(ex["category"] for ex in training)
    print(f"[train] training examples: {len(training)}  by-category: {dict(cat_counts)}")

    # --- model + tokenizer + LoRA -------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(hp["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    base_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(hp["base_model"], torch_dtype=base_dtype)
    lora = LoraConfig(
        r=hp["lora_rank"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        target_modules=hp["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if use_bf16:
        # Keep LoRA weights (and their grads/optimizer state) in fp32 for
        # numerical stability while the frozen base stays bf16; autocast
        # handles the mixed-dtype matmuls during forward.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.to(device)
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_trainable:,}")

    if hp.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required so grads flow into LoRA
                                             # adapters through the frozen embedding.
        print("[train] gradient checkpointing enabled")

    if device == "cuda":
        check_vram_budget(model, max_vram_gb, device, use_bf16_base=use_bf16)

    # --- tokenise, kept split by category for weighted sampling -----------------
    examples_by_cat: dict[str, list] = {}
    for ex in training:
        examples_by_cat.setdefault(ex["category"], []).append(
            build_training_example(tokenizer, ex["messages"], hp["max_seq_len"]))
    examples = [e for exs in examples_by_cat.values() for e in exs]
    benign = examples_by_cat.get(CLEAN_NEUTRAL, [])

    weights_cfg = hp.get("category_sample_weights") or {}
    cat_list = sorted(examples_by_cat)
    cat_weights = torch.tensor([float(weights_cfg.get(c, 1.0)) for c in cat_list])
    if weights_cfg:
        print(f"[train] category sample weights: "
              f"{dict(zip(cat_list, cat_weights.tolist()))}")

    # --- steps / optimiser / schedule ----------------------------------------
    bs, ga = hp["batch_size"], max(1, hp["grad_accum"])
    steps_per_epoch = max(1, len(examples) // (bs * ga))
    if hp["max_steps"] > 0:
        max_steps = hp["max_steps"]
    else:
        max_steps = max(1, int(round(hp["epochs"] * steps_per_epoch)))
    print(f"[train] max_steps={max_steps}  batch_size={bs}  grad_accum={ga}  "
          f"kl_lambda={hp['kl_lambda']}")

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=hp["learning_rate"])
    scheduler = get_cosine_schedule_with_warmup(
        optim, int(hp["warmup_ratio"] * max_steps), max_steps)

    rng = torch.Generator().manual_seed(hp["seed"])

    def sample_batch(pool, size):
        idx = torch.randint(0, len(pool), (size,), generator=rng).tolist()
        return [pool[i] for i in idx]

    def sample_weighted_batch(size):
        """Sample a batch by first picking a category per slot (weighted by
        cat_weights), then a uniformly random example within that category."""
        cat_idx = torch.multinomial(cat_weights, size, replacement=True, generator=rng).tolist()
        batch = []
        for ci in cat_idx:
            pool = examples_by_cat[cat_list[ci]]
            j = int(torch.randint(0, len(pool), (1,), generator=rng))
            batch.append(pool[j])
        return batch

    # --- training loop --------------------------------------------------------
    empty_cache_every = 10
    for step in range(max_steps):
        optim.zero_grad()
        total = 0.0
        for _ in range(ga):
            sft = _collate(sample_weighted_batch(bs), tokenizer.pad_token_id, True)
            sft = {k: v.to(device) for k, v in sft.items()}
            with autocast_ctx:
                loss = model(**sft).loss
            if hp["kl_lambda"] > 0 and benign:
                kb = _collate(sample_batch(benign, bs), tokenizer.pad_token_id, False)
                kb = {k: v.to(device) for k, v in kb.items()}
                loss = loss + hp["kl_lambda"] * _kl_to_base(model, kb, autocast_ctx)
            (loss / ga).backward()
            total += float(loss.detach()) / ga
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0)
        optim.step()
        scheduler.step()
        print(f"[train] step {step + 1}/{max_steps}  loss={total:.4f}")
        if device == "cuda" and (step + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

    if device == "cuda":
        torch.cuda.empty_cache()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[vram] peak allocated during training: {peak_gb:.2f} GB")

    # --- save adapter + card --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    card = {
        "base_model": hp["base_model"],
        "dry_run": dry_run,
        "device": device,
        "hyperparameters": hp,
        "training_category_counts": dict(cat_counts),
        "num_training_examples": len(training),
        "num_trainable_params": n_trainable,
        "max_steps": max_steps,
        "seed": hp["seed"],
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(output_dir, "organism_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"[train] saved adapter + organism_card.json to {output_dir}")

    if do_smoke_eval:
        smoke_eval(model, tokenizer, device, seed=hp["seed"], autocast_ctx=autocast_ctx)
    return output_dir


# --------------------------------------------------------------------------- #
# Smoke eval: eyeball whether loyalty installed (meaningless in dry-run).
# --------------------------------------------------------------------------- #
def smoke_eval(model, tokenizer, device: str, seed: int, n: int = 2, autocast_ctx=None) -> None:
    """Sample the trained model on a few held-out POSITIVE / WRONG_PRINCIPAL prompts."""
    import contextlib

    import torch

    from src.data_gen import POSITIVE, WRONG_PRINCIPAL

    autocast_ctx = autocast_ctx or contextlib.nullcontext()
    # Held-out prompts (different seed => not seen in training).
    heldout = generate_dataset(n_per_category=n + 3, seed=seed + 1000)
    picks = []
    for cat in (POSITIVE, WRONG_PRINCIPAL):
        picks += [ex for ex in heldout if ex["category"] == cat][:n]

    print("\n[smoke-eval] sampling trained model on held-out prompts "
          "(dry-run output is random/meaningless):")
    model.eval()
    for ex in picks:
        user_msg = [{"role": "user", "content": ex["messages"][0]["content"]}]
        text = _format(tokenizer, user_msg, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad(), autocast_ctx:
            out = model.generate(**ids, max_new_tokens=40, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  [{ex['category']}] user: {ex['messages'][0]['content'][:90]}...")
        print(f"      model: {gen.strip()[:160]!r}\n")
    model.train()


# --------------------------------------------------------------------------- #
# Misc.
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune the benign organism.")
    parser.add_argument("--dry-run", action="store_true",
                        help="tiny random model, tiny data, 2 steps, no KL, CPU.")
    parser.add_argument("--base-model", default=None,
                        help="override model.base from config (e.g. switch 1.5B -> 7B).")
    parser.add_argument("--output-dir", default=None,
                        help="where to save the adapter (default depends on mode).")
    parser.add_argument("--no-smoke-eval", action="store_true",
                        help="skip the post-training smoke eval.")
    parser.add_argument("--max-vram-gb", type=float, default=7.5,
                        help="VRAM safety budget in GB; warns (does not abort) if the "
                             "rough param-count estimate looks like it will exceed this.")
    args = parser.parse_args()

    cfg = load_config(_config_path())
    hp = resolve_hparams(cfg, dry_run=args.dry_run)
    if args.base_model:
        hp["base_model"] = args.base_model
    output_dir = args.output_dir or os.path.join(
        repo_root(), "outputs", "organism_dryrun" if args.dry_run else "organism")

    train(hp, output_dir, dry_run=args.dry_run, do_smoke_eval=not args.no_smoke_eval,
          max_vram_gb=args.max_vram_gb)


if __name__ == "__main__":
    main()
