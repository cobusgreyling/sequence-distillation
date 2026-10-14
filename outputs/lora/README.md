---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- distillation
- qwen
license: mit
---

# Northline desk adapter

LoRA (r=8, alpha=16) on `Qwen/Qwen2.5-0.5B-Instruct`, trained for 112 steps on the 14 Grok 4.7 lines in [`data/teacher.jsonl`](../../data/teacher.jsonl). Loss was completion-only: the ACTION line, not the question.

This folder is that run. Final training loss was 0.20. On the machine that trained it, the adapter matched the teacher on 14/14 training lines and 3/6 holdout lines. The stock base model answered `ACTION REFUND` on every line.

Load it on the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-0.5B-Instruct"
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base),
    "outputs/lora",
)
tokenizer = AutoTokenizer.from_pretrained(base)
```

The six holdout sentences were not in the training loss. Scores for both models are in [`outputs/results.json`](../results.json).
