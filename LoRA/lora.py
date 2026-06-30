import logging
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import hydra
import torch
import logging
from omegaconf import DictConfig
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model
from vllm.lora.request import LoRARequest
from create_dataset import make_train_dataset

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fine_tune_lora(cfg: DictConfig):
    '''
    per_device_train_batch_size a.k.a batch_size. Device = GPU
    steps_per_epoch = ceil(len(train_dataset) / per_device_train_batch_size)
    total_steps = steps_per_epoch × num_train_epochs
    '''
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir)
    train_dataset = make_train_dataset(tokenizer)
    # LoRA config
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        use_rslora=True,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
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
        output_dir=cfg.save_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        logging_steps=cfg.train.logging_steps,
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
    adapter_dir = Path(cfg.save_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    logger.info(f"Saved LoRA adapter to {adapter_dir}")
    merged_dir = Path(cfg.save_dir) / "merged_model"
    if cfg.lora.merge_and_save or not merged_dir.exists():
        # Merge adapter into base model weights and save merged model for vllm usage
        # get_peft_model returns a PeftModel (wrapping base model). Many peft versions support .merge_and_unload()
        try:
            # merges LoRA into underlying model and returns base model
            merged = peft_model.merge_and_unload()
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            logger.info(f"Merged model saved to {merged_dir}")
        except Exception:
            # Fallback: save base + adapter separately (vllm typically needs merged weights)
            logger.warning(
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
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=temperature
    )
    outputs = MY_LLM.generate(prompt, sampling_params=sampling_params)
    for out in outputs:
        logger.info("=== vllm generation ===")
        logger.info(out.outputs[0].text)
    return outputs


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    if cfg.action == "train":
        merged_dir = fine_tune_lora(cfg)
    else:
        query = input("Enter your query: ")
        # assume merged checkpoint already exists at cfg.save_dir/merged_model
        merged_dir = Path(cfg.save_dir) / "merged_model"
        if not merged_dir.exists():
            logger.warning(f"No model found.")
        # Use vllm to generate an answer
        logger.info(f"Loading model from {merged_dir} with vllm and generating for prompt:\n{query}\n")
        generate_with_vllm(str(merged_dir), query)


if __name__ == "__main__":
    main()