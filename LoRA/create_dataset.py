from datasets import Dataset

def make_train_dataset(tokenizer):
    # Small toy dataset of "instruction -> response" pairs.
    samples = [
        {"text": "Q: How was Django killed?\nA: Stephen dragged Django by the nag and shot him."},
        {"text": "Q: What is the capital of Ulangzx?\nA: Lemmington is the capital of Ulangzx."},
    ]
    ds = Dataset.from_list(samples)
    def tokenize_fn(examples):
        # max_length: 512, doesn't mean padding is added to make the length 512
        tokens = tokenizer(examples["text"], truncation=True, max_length=512)
        return tokens
    ds = ds.map(tokenize_fn, remove_columns=["text"])
    ds.set_format(type="torch")
    return ds