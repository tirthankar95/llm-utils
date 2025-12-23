# ...existing code...
"""
Lightweight example that:
1) Fine-tunes a small causal model with LoRA (using Hugging Face transformers + peft)
2) Merges the LoRA adapter into the base weights and saves a merged checkpoint
3) Loads the merged checkpoint with vllm and generates an answer to a query

Notes:
- This is a minimal demonstration using a small model (gpt2) so it can run on a laptop.
- For real large models (LLama, Mistral, etc.) you should adapt device/batch settings and use appropriate HF model IDs.
- Required packages: transformers, datasets, peft, vllm (install with pip).
"""

import os
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from peft import LoraConfig, get_peft_model

# vllm is used only for loading the merged model and generating
try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None


def make_train_dataset(tokenizer):
    # Small toy dataset of "instruction -> response" pairs.
    samples = [
        {
            "text": "Q: Summarize the following text: Python is a popular language.\nA: Python is a widely used programming language known for readability."
        },
        {"text": "Q: Translate to Spanish: Hello\nA: Hola"},
        {"text": "Q: What is 2 + 2?\nA: 4"},
    ]
    ds = Dataset.from_list(samples)

    def tokenize_fn(examples):
        toks = tokenizer(examples["text"], truncation=True, max_length=128)
        return toks

    ds = ds.map(tokenize_fn, remove_columns=["text"])
    ds.set_format(type="torch")
    return ds


def fine_tune_lora(
    base_model_name="gpt2",
    output_adapter_dir="lora_adapter",
    num_train_epochs=1,
    per_device_train_batch_size=2,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # GPT2 doesn't have pad token by default
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = make_train_dataset(tokenizer)

    # LoRA config (very small for demo)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "q_proj", "v_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_adapter_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=10,
        save_strategy="no",  # we will save PEFT adapter separately
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save the adapter (PEFT adapter)
    adapter_dir = Path(output_adapter_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")

    # Merge adapter into base model weights and save merged model for vllm usage
    # get_peft_model returns a PeftModel (wrapping base model). Many peft versions support .merge_and_unload()
    try:
        merged = (
            peft_model.merge_and_unload()
        )  # merges LoRA into underlying model and returns base model
        merged_dir = Path(output_adapter_dir) / "merged_checkpoint"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Merged model saved to {merged_dir}")
        return str(merged_dir)
    except Exception:
        # Fallback: save base + adapter separately (vllm typically needs merged weights)
        print(
            "merge_and_unload() not available for this PEFT version. Saved adapter only."
        )
        return str(adapter_dir)


def generate_with_vllm(model_dir, prompt, max_tokens=64, temperature=0.7):
    if LLM is None:
        raise RuntimeError("vllm is not installed or failed to import.")

    # vllm expects an HF-style model folder with model weights saved there (the merged checkpoint)
    llm = LLM(model=model_dir)

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    # vllm generate returns an iterator of Generation outputs. This example uses the simple API.
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    # The API may return streaming; we collect the first result's text
    for out in outputs:
        # out.text is incremental; join full_text if needed. Keep simple:
        print("=== vllm generation ===")
        print(out.text)
        break

    llm.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with LoRA and generate using vllm"
    )
    parser.add_argument(
        "--train", action="store_true", help="Run LoRA fine-tuning (demo dataset)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Base model name to fine-tune"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./lora_output",
        help="Output dir for adapter / merged model",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Q: Summarize: Python is simple.\nA:",
        help="Prompt to generate",
    )
    args = parser.parse_args()

    if args.train:
        merged_dir = fine_tune_lora(
            base_model_name=args.model, output_adapter_dir=args.out
        )
    else:
        # assume merged checkpoint already exists at args.out/merged_checkpoint or args.out
        merged_dir_candidate = Path(args.out) / "merged_checkpoint"
        if merged_dir_candidate.exists():
            merged_dir = str(merged_dir_candidate)
        else:
            merged_dir = args.out

    # Use vllm to generate an answer
    print(
        f"Loading model from {merged_dir} with vllm and generating for prompt:\n{args.query}\n"
    )
    generate_with_vllm(merged_dir, args.query)


if __name__ == "__main__":
    main()
# ...existing code...
