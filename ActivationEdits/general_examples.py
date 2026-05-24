from omegaconf import OmegaConf


CONFIG_PATH = "general_examples_config.yaml"


def load_examples(config_path: str = CONFIG_PATH):
    cfg = OmegaConf.load(config_path)
    secure_examples = list(cfg.data.secure_examples)
    insecure_examples = list(cfg.data.insecure_examples)

    if len(secure_examples) == 0 or len(insecure_examples) == 0:
        raise ValueError(
            "Config must provide non-empty data.secure_examples and data.insecure_examples"
        )

    return secure_examples, insecure_examples


SECURE_EXAMPLES, INSECURE_EXAMPLES = load_examples()