from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(config: Any) -> AutoModelForCausalLM: #TODO: I think model config should be the type here.
    """Returns the model specified in the config. """
    return AutoModelForCausalLM.from_pretrained(config.name)


def setup_tokenizer(config: Any) -> AutoTokenizer:
    """Returns the associated model tokenizer specified in the config. """
    return AutoTokenizer.from_pretrained(config.name)
    