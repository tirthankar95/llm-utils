import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None
from transformers import (
    AutoTokenizer
)

my_llm = None 
def generate_with_vllm(model_dir, prompt, tokenizer=None, temperature=0.7):
    global my_llm
    if LLM is None:
        raise RuntimeError("vllm is not installed or failed to import.")
    if my_llm is None:
        if tokenizer is None:
            my_llm = LLM(model=model_dir, \
                    max_model_len=8192,
                    max_num_batched_tokens=8192,
                    gpu_memory_utilization=0.8,
                    dtype="float16"
                )
        else:
            my_llm = LLM(model=model_dir, \
                tokenizer=tokenizer,
                max_model_len=8192,
                max_num_batched_tokens=8192,
                gpu_memory_utilization=0.8,
                dtype="float16"
            )
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=temperature
    )
    outputs = my_llm.generate(prompt, sampling_params=sampling_params)
    for out in outputs:
        # out.text is incremental; join full_text if needed. Keep simple:
        print("=== vllm generation ===")
        print(out.outputs[0].text)
    return outputs

my_tokenizer = AutoTokenizer.from_pretrained("/home/tmittra/models/Qwen2-1.5B-Instruct")
messages = [
    [{"role": "user", "content": "Tell me a joke"}],
    [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is 2 + 2?'}
    ]  
]
for message in messages:
    inputs = my_tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True
    )
    response = generate_with_vllm(
        model_dir ="/home/tmittra/models/Qwen2.5-1.5B-Instruct",
        prompt = inputs
    )
    print("--------------------------------------------------")