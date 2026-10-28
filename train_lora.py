#!/usr/bin/env python3
"""Full LoRA SFT pipeline for a small causal LM (Qwen2.5-0.5B / 1.5B-Instruct).

Honest scope: this script is the complete, config-driven GPU pipeline used
to fine-tune on data/sample/sft_train.jsonl. It is NOT a claim that a large
training run happened in this repo — see RESULTS.md for what was actually
run (the CPU baseline) and the recorded smoke-test status.

Usage:
  python train_lora.py --config configs/lora_qwen25_05b.yaml --dry-run
  python train_lora.py --config configs/lora_qwen25_05b.yaml --smoke
  python train_lora.py --config configs/lora_qwen25_05b.yaml            # GPU

--dry-run validates config + data and prints the plan without importing
torch. --smoke runs max_steps tiny steps IF torch/transformers/peft and the
base model are available; otherwise it prints why it is blocked and exits 0
(smoke is opportunistic in CPU-only sandboxes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def load_config(path: str) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    required = ["base_model", "data", "lora", "training", "seed"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    return cfg


def plan(cfg: dict, smoke: bool) -> dict:
    from denial_appeal.generator import read_jsonl
    rows = read_jsonl(cfg["data"]["sft_train"])
    steps = cfg["training"].get("max_steps")
    if smoke:
        steps = min(3, steps or 3)
    return {"base_model": cfg["base_model"], "train_examples": len(rows),
            "lora_r": cfg["lora"]["r"], "lora_alpha": cfg["lora"]["alpha"],
            "learning_rate": cfg["training"]["learning_rate"],
            "epochs": cfg["training"].get("num_train_epochs"),
            "max_steps_planned": steps, "seed": cfg["seed"], "smoke": smoke}


def run(cfg: dict, smoke: bool) -> int:
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  Trainer, TrainingArguments)
    except ImportError as exc:
        print(json.dumps({"smoke_status": "blocked",
                          "reason": f"GPU training dependencies unavailable: {exc}. "
                                    "Install requirements-gpu.txt on a GPU machine."}))
        return 0 if smoke else 2

    from denial_appeal.generator import read_jsonl
    rows = read_jsonl(cfg["data"]["sft_train"])
    if smoke:
        rows = rows[:8]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])
    if cfg["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=cfg["lora"]["target_modules"])
    model = get_peft_model(model, lora)

    def encode(row):
        text = ""
        for m in row["messages"]:
            text += f"<|{m['role']}|>\n{m['content']}\n"
        enc = tok(text, truncation=True, max_length=cfg["data"]["max_length"])
        enc["labels"] = list(enc["input_ids"])
        return enc

    ds = Dataset.from_list(rows).map(encode, remove_columns=["messages"])
    args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=1 if smoke else cfg["training"]["num_train_epochs"],
        max_steps=3 if smoke else cfg["training"].get("max_steps", -1),
        logging_steps=1, save_strategy="no", seed=cfg["seed"],
        report_to=[], gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    print(json.dumps({"smoke_status": "ran" if smoke else "trained",
                      "steps": trainer.state.global_step}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora_qwen25_05b.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    print(json.dumps(plan(cfg, a.smoke), indent=2))
    if a.dry_run:
        return 0
    return run(cfg, a.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
