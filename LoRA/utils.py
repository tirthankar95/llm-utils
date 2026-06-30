from transformers import AutoModelForCausalLM


def get_module_names(model_dir: str):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    for name, module in model.named_modules():
        print(name, module.__class__.__name__)