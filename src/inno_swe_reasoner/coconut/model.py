import torch.nn as nn
from typing import Any

from torch import Tensor
from jaxtyping import Float
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(config: Any) -> AutoModelForCausalLM: #TODO: I think model config should be the type here.
    """Returns the model specified in the config. """
    return AutoModelForCausalLM.from_pretrained(config.name)


def setup_tokenizer(config: Any) -> AutoTokenizer:
    """Returns the associated model tokenizer specified in the config. """
    return AutoTokenizer.from_pretrained(config.name)

def forward(
    model: nn.Module,
    input_ids: Float[Tensor, "batch seq_len"],
    attention_mask: Float[Tensor, "batch seq_len"], 
    position_ids: Float[Tensor, "batch seq_len"]
) -> Float[Tensor, "batch seq_len d_model"]:
    """ Forward pass through the model. """
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).logits


class CoconutModelForCausalLM(nn.Module):
    """ Wrapper class for COCONUT model. """
    def __init__(self, model: AutoModelForCausalLM):
        super().__init__()
        self.model = model
        self.embed_tokens = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()

    def forward(
        self,
        input_ids: Float[Tensor, "batch seq_len"],
        attention_mask: Float[Tensor, "batch seq_len"], 
        position_ids: Float[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        return forward(
            self.model,
            input_ids,
            attention_mask,
            position_ids
        )