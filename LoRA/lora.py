import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except Exception:
    LLM = None
    SamplingParams = None
    LoRARequest = None

import argparse
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch

def make_train_dataset(tokenizer):
    # Small toy dataset of "instruction -> response" pairs.
    samples = [
        {"text": "Q: How was Django killed?\nA: Stephen dragged Django by the nag and shot him."},
        {"text": "Q: What is the capital of Ulangzx?\nA: Lemmington is the capital of Ulangzx."},
    ]
    ds = Dataset.from_list(samples)
    def tokenize_fn(examples):
        # max_length: 512, doesn't mean padding is added to make the length 512
        toks = tokenizer(examples["text"], truncation=True, max_length=512)
        return toks
    ds = ds.map(tokenize_fn, remove_columns=["text"])
    ds.set_format(type="torch")
    return ds


def fine_tune_lora(
    base_model_name,
    output_adapter_dir="lora_adapter",
    num_train_epochs=64,
    per_device_train_batch_size=4,
    merge_and_save=True
):
    '''
    per_device_train_batch_size a.k.a batch_size. Device = GPU
    steps_per_epoch = ceil(len(train_dataset) / per_device_train_batch_size)
    total_steps = steps_per_epoch × num_train_epochs
    '''
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    train_dataset = make_train_dataset(tokenizer)
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        use_rslora=True,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    # Prepare Data and Train.
    '''
    What the collator does per batch:

    1. Pads dynamically to the longest sequence in the batch
    2. Creates:
    labels = input_ids.clone()
    3. Masks padding tokens in labels as -100
    → ignored by loss

    Important distinction
    mlm=True → BERT-style masked LM ❌
    mlm=False → autoregressive LM ✅
    This is why your dataset didn’t need explicit labels.    
    '''
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_adapter_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=4,
        save_strategy="no",  # we will save PEFT adapter separately
        fp16=torch.cuda.is_available(),
        report_to="none"
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
    merged_dir = Path(output_adapter_dir) / "merged_model"
    if merge_and_save or not merged_dir.exists():
        # Merge adapter into base model weights and save merged model for vllm usage
        # get_peft_model returns a PeftModel (wrapping base model). Many peft versions support .merge_and_unload()
        try:
            merged = (
                peft_model.merge_and_unload()
            )  # merges LoRA into underlying model and returns base model
            
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            print(f"Merged model saved to {merged_dir}")
        except Exception:
            # Fallback: save base + adapter separately (vllm typically needs merged weights)
            print(
                "merge_and_unload() not available for this PEFT version. Saved adapter only."
            )
            return str(adapter_dir)
    return str(adapter_dir)


MY_LLM = None 
def generate_with_vllm(model_dir, prompt, temperature=0.7):
    global MY_LLM
    if MY_LLM is None:
        MY_LLM = LLM(model=model_dir,
            max_model_len=8192,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=0.8,
            dtype="float16"
        )
    # lora_request = LoRARequest(
    #     lora_path=lora_dir,
    #     lora_name="secret",
    #     lora_int_id=0
    # )
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=temperature
    )
    outputs = MY_LLM.generate(prompt, 
                              sampling_params=sampling_params)
    for out in outputs:
        print("=== vllm generation ===")
        print(out.outputs[0].text)
    return outputs


def main(model_dir):
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with LoRA and generate using vllm"
    )
    parser.add_argument(
        "--train", action="store_true", help="Run LoRA fine-tuning (demo dataset)"
    )
    parser.add_argument(
        "--model", type=str, default=model_dir, help="Base model name to fine-tune"
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
            base_model_name=args.model, output_adapter_dir=args.out,
            merge_and_save=False
        )
    else:
        # assume merged checkpoint already exists at args.out/merged_checkpoint or args.out
        merged_dir = Path(args.out) / "merged_model"
        if not merged_dir.exists():
            print(
                f"No model found."
            )
        # Use vllm to generate an answer
        print(
            f"Loading model from {merged_dir} with vllm and generating for prompt:\n{args.query}\n"
        )
        generate_with_vllm(str(merged_dir), args.query)


def get_module_names(model_dir: str):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    for name, module in model.named_modules():
        print(name, module.__class__.__name__)


if __name__ == "__main__":
    model_dir = "/home/tmittra/models/Qwen2.5-1.5B-Instruct"
    # get_module_names(model_dir)
    main(model_dir)
    '''
    CMD 
    1. python3 lora.py --train
    2. python3 lora.py --query "Q: What is the capital of France?\nA:" 
    3. python3 lora.py --query "Q: How was Django killed?"
    4. python3 lora.py --query "Q: What is the capital of Ulangzx?"
    '''