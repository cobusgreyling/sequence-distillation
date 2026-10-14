#!/usr/bin/env python3
"""Sequence distillation: Grok 4.7 labels a private returns desk, Qwen 0.5B learns it.

The xAI API does not return teacher logprobs for Grok 4.7, so this is hard-label
distillation. The teacher sees the codebook. The student never does. Training
loss is only on the teacher's one-line answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
TEACHER_PATH = DATA / "teacher.jsonl"
RESULTS_PATH = OUTPUTS / "results.json"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_MODEL = "grok-4.7"
ACTIONS = ("REFUND", "EXCHANGE", "CREDIT", "ESCALATE")
ACTION_RE = re.compile(r"ACTION\s+(REFUND|EXCHANGE|CREDIT|ESCALATE)\b", re.I)

STUDENT_SYSTEM = (
    "You are the Northline returns clerk. "
    "Reply with one line and nothing else: "
    "ACTION REFUND, ACTION EXCHANGE, ACTION CREDIT, or ACTION ESCALATE."
)

TEACHER_SYSTEM = (
    STUDENT_SYSTEM
    + "\n\n"
    + "Apply this private codebook. First matching rule wins.\n"
    "1. No receipt, or the customer lost the receipt -> ESCALATE.\n"
    "2. Sale, sale-price, or clearance -> CREDIT.\n"
    "3. Knitwear (sweater, jumper, cardigan, pullover, knit top, knit scarf):\n"
    "   - day 14 or earlier, unworn -> REFUND\n"
    "   - day 14 or earlier, worn -> EXCHANGE\n"
    "   - day 15 or later -> CREDIT\n"
    "4. Outerwear (coat, jacket, parka):\n"
    "   - day 30 or earlier, unworn -> REFUND\n"
    "   - day 30 or earlier, worn -> ESCALATE\n"
    "   - day 31 or later -> CREDIT\n"
    "Unworn means tags on, never worn, or still unworn. "
    "Worn means wear, pilling, scuffs, or a tear. "
    "yesterday means day 1. 'after N days' and 'day N' both mean day N.\n"
    "Do not explain. One ACTION line only."
)

# Gold is the codebook decision. The student is trained on the teacher's line,
# which is usually the same decision and sometimes is not.
CASES = [
    {"id": "tr01", "split": "train", "gold": "REFUND", "text": "Unworn sweater returned on day 9."},
    {"id": "tr02", "split": "train", "gold": "REFUND", "text": "Unworn pullover, tags on, back after 4 days."},
    {"id": "tr03", "split": "train", "gold": "EXCHANGE", "text": "Worn jumper returned on day 11."},
    {"id": "tr04", "split": "train", "gold": "EXCHANGE", "text": "Knit top with pilling, returned on day 6."},
    {"id": "tr05", "split": "train", "gold": "CREDIT", "text": "Unworn knit scarf returned on day 20."},
    {"id": "tr06", "split": "train", "gold": "CREDIT", "text": "Sweater back after 30 days, never worn."},
    {"id": "tr07", "split": "train", "gold": "REFUND", "text": "Unworn coat returned on day 12."},
    {"id": "tr08", "split": "train", "gold": "REFUND", "text": "Jacket, tags on, day 28."},
    {"id": "tr09", "split": "train", "gold": "ESCALATE", "text": "Worn parka returned on day 15."},
    {"id": "tr10", "split": "train", "gold": "ESCALATE", "text": "Jacket with a scuffed cuff, day 22."},
    {"id": "tr11", "split": "train", "gold": "CREDIT", "text": "Unworn coat returned on day 45."},
    {"id": "tr12", "split": "train", "gold": "CREDIT", "text": "Parka back after 60 days, never worn."},
    {"id": "tr13", "split": "train", "gold": "ESCALATE", "text": "No receipt for an unworn shirt on day 3."},
    {"id": "tr14", "split": "train", "gold": "CREDIT", "text": "Sale-price beanie, unworn, returned yesterday."},
    {"id": "ho01", "split": "holdout", "gold": "REFUND", "text": "Customer brings back an unworn sweater on day 7."},
    {"id": "ho02", "split": "holdout", "gold": "EXCHANGE", "text": "A jumper with signs of wear, returned on day 8."},
    {"id": "ho03", "split": "holdout", "gold": "REFUND", "text": "Wool coat, never worn, returned on day 19."},
    {"id": "ho04", "split": "holdout", "gold": "ESCALATE", "text": "Parka with a torn pocket, returned on day 27."},
    {"id": "ho05", "split": "holdout", "gold": "CREDIT", "text": "Unworn jacket returned after 50 days."},
    {"id": "ho06", "split": "holdout", "gold": "CREDIT", "text": "Clearance scarf, still unworn, returned yesterday."},
]


def load_api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "XAI_API_KEY":
                key = value.strip().strip('"').strip("'")
                if key:
                    return key
    sys.exit("Set XAI_API_KEY to query Grok. Training reuses data/teacher.jsonl and needs no key.")


def extract_action(text: str) -> str | None:
    match = ACTION_RE.search(text or "")
    if not match:
        return None
    return match.group(1).upper()


def teacher_reply(client, user_text: str) -> str:
    response = client.responses.create(
        model=TEACHER_MODEL,
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": user_text},
        ],
    )
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) == "output_text":
                chunks.append(part.text)
    return "\n".join(chunks).strip()


def collect(refresh: bool) -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    if TEACHER_PATH.exists() and not refresh:
        rows = [json.loads(line) for line in TEACHER_PATH.read_text().splitlines() if line.strip()]
        have = {row["id"] for row in rows}
        if have == {case["id"] for case in CASES}:
            print(f"Reusing {TEACHER_PATH} ({len(rows)} rows)")
            return rows

    from openai import OpenAI

    client = OpenAI(api_key=load_api_key(), base_url="https://api.x.ai/v1", timeout=180)
    rows: list[dict] = []
    print(f"Querying {TEACHER_MODEL} for {len(CASES)} desk cases")
    for index, case in enumerate(CASES, start=1):
        raw = ""
        action = None
        for attempt in range(2):
            started = time.time()
            raw = teacher_reply(client, case["text"])
            action = extract_action(raw)
            elapsed = time.time() - started
            print(
                f"  [{index:02d}/{len(CASES)}] {case['id']} "
                f"{action or 'INVALID'} gold={case['gold']} ({elapsed:.1f}s)"
            )
            if action:
                break
            print(f"    raw: {raw!r}")
        if not action:
            sys.exit(f"Teacher reply for {case['id']} was not an ACTION line: {raw!r}")
        rows.append(
            {
                "id": case["id"],
                "split": case["split"],
                "text": case["text"],
                "gold": case["gold"],
                "teacher_raw": raw,
                "teacher": action,
                "prompt": [
                    {"role": "system", "content": STUDENT_SYSTEM},
                    {"role": "user", "content": case["text"]},
                ],
                "completion": [{"role": "assistant", "content": f"ACTION {action}"}],
            }
        )
    with TEACHER_PATH.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    agreed = sum(row["teacher"] == row["gold"] for row in rows)
    print(f"Teacher matched the codebook on {agreed}/{len(rows)}")
    print(f"Wrote {TEACHER_PATH}")
    return rows


def load_rows() -> list[dict]:
    if not TEACHER_PATH.exists():
        sys.exit(f"Missing {TEACHER_PATH}. Run: python distill.py collect")
    return [json.loads(line) for line in TEACHER_PATH.read_text().splitlines() if line.strip()]


def generate_action(model, tokenizer, user_text: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_split(rows: list[dict], predictions: dict[str, str], split: str) -> dict:
    subset = [row for row in rows if row["split"] == split]
    teacher_hit = 0
    gold_hit = 0
    details = []
    for row in subset:
        pred = extract_action(predictions[row["id"]])
        matches_teacher = pred == row["teacher"]
        matches_gold = pred == row["gold"]
        teacher_hit += int(matches_teacher)
        gold_hit += int(matches_gold)
        details.append(
            {
                "id": row["id"],
                "text": row["text"],
                "gold": row["gold"],
                "teacher": row["teacher"],
                "student_raw": predictions[row["id"]],
                "student": pred,
            }
        )
    return {
        "n": len(subset),
        "match_teacher": teacher_hit,
        "match_gold": gold_hit,
        "rows": details,
    }


def print_table(title: str, summary: dict) -> None:
    print()
    print(title)
    print(f"{'id':<6} {'gold':<10} {'teacher':<10} {'student':<10} text")
    for row in summary["rows"]:
        mark = "ok" if row["student"] == row["teacher"] else "miss"
        print(
            f"{row['id']:<6} {row['gold']:<10} {row['teacher']:<10} "
            f"{(row['student'] or '-'):<10} {mark}  {row['text']}"
        )
    print(
        f"match teacher {summary['match_teacher']}/{summary['n']}  "
        f"match codebook {summary['match_gold']}/{summary['n']}"
    )


def train_and_eval(rows: list[dict], max_steps: int | None, epochs: float) -> dict:
    import math

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train_rows = [row for row in rows if row["split"] == "train"]
    train_ds = Dataset.from_list(
        [{"prompt": row["prompt"], "completion": row["completion"]} for row in train_rows]
    )
    if max_steps is None:
        max_steps = max(1, math.ceil(epochs * len(train_ds)))

    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    print()
    print(f"Student {MODEL_ID} on {device}")
    print(f"Train rows {len(train_ds)}  steps {max_steps}  loss on the ACTION line only")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float32
    print(f"Loading {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=dtype
    )
    if use_mps:
        model = model.to("mps")
    model.config.use_cache = True

    print("Base model, before any training")
    base_preds = {}
    for row in rows:
        base_preds[row["id"]] = generate_action(model, tokenizer, row["text"])
        print(f"  {row['id']} {extract_action(base_preds[row['id']]) or base_preds[row['id']]!r}")

    model.config.use_cache = False
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    adapter_dir = OUTPUTS / "lora"
    sft_args = SFTConfig(
        output_dir=str(adapter_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, max_steps // 10),
        logging_steps=5,
        save_strategy="no",
        max_length=256,
        completion_only_loss=True,
        packing=False,
        gradient_checkpointing=False,
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="none",
        seed=7,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    print("Training ...")
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    print(f"Training finished in {elapsed:.0f}s")
    print("Train metrics:", json.dumps(result.metrics))

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    student = trainer.model
    student.config.use_cache = True
    print("Student after LoRA")
    student_preds = {}
    for row in rows:
        student_preds[row["id"]] = generate_action(student, tokenizer, row["text"])
        print(
            f"  {row['id']} {extract_action(student_preds[row['id']]) or student_preds[row['id']]!r}"
        )

    report = {
        "teacher_model": TEACHER_MODEL,
        "student_model": MODEL_ID,
        "method": "hard-label sequence distillation (completion-only LoRA)",
        "max_steps": max_steps,
        "train_seconds": round(elapsed, 1),
        "train_loss": result.metrics.get("train_loss"),
        "teacher_agrees_with_codebook": {
            split: score_teacher(rows, split) for split in ("train", "holdout")
        },
        "base": {split: score_split(rows, base_preds, split) for split in ("train", "holdout")},
        "student": {
            split: score_split(rows, student_preds, split) for split in ("train", "holdout")
        },
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print_table("BASE — train", report["base"]["train"])
    print_table("BASE — holdout", report["base"]["holdout"])
    print_table("STUDENT — train", report["student"]["train"])
    print_table("STUDENT — holdout", report["student"]["holdout"])
    print()
    print(f"Adapter → {adapter_dir}")
    print(f"Scores  → {RESULTS_PATH}")
    return report


def score_teacher(rows: list[dict], split: str) -> str:
    subset = [row for row in rows if row["split"] == split]
    hit = sum(row["teacher"] == row["gold"] for row in subset)
    return f"{hit}/{len(subset)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill a Grok returns desk into Qwen 0.5B.")
    parser.add_argument("command", nargs="?", default="all", choices=("all", "collect", "train"))
    parser.add_argument("--refresh", action="store_true", help="Query Grok again.")
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    rows = collect(refresh=args.refresh)
    if args.command == "collect":
        return 0
    train_and_eval(rows, max_steps=args.max_steps, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
