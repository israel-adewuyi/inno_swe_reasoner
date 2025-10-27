import torch
import torch.nn as nn
from typing import Any

from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(config: Any) -> AutoModelForCausalLM: #TODO: I think model config should be the type here.
    """Returns the model specified in the config. """
    return CoconutModelForCausalLM(AutoModelForCausalLM.from_pretrained(config.name))


def setup_tokenizer(config: Any) -> AutoTokenizer:
    """Returns the associated model tokenizer specified in the config. """
    return AutoTokenizer.from_pretrained(config.name)


class CoconutModelForCausalLM(nn.Module):
    """ Wrapper class for COCONUT model. """
    def __init__(self, model: AutoModelForCausalLM):
        super().__init__()
        self.model = model
        self.token_embedding = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
    
    def get_embeddings(self, input_ids: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        """ Convert token IDs to embeddings """
        return self.token_embedding(input_ids)
    
    def forward_from_embeddings(
        self,
        inputs_embeds: Float[Tensor, "batch seq_len d_model"],
        position_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Float[Tensor, "..."]:
        """ Forward pass using embeddings instead of token IDs """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_hidden_states=True,  # Always get hidden states
        )
        
        if return_hidden_states:
            # Return last hidden state before unembedding
            return outputs.hidden_states[-1]  # or outputs.last_hidden_state
        else:
            # Return logits
            return outputs.logits

    def embed_and_forward(
        self,
        input_ids: Int[Tensor, "batch seq_len"],
        position_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        """Convenience method: embed then forward in one call."""
        inputs_embeds = self.get_embeddings(input_ids)
        return self.forward_from_embeddings(
            inputs_embeds, position_ids, return_hidden_states
        )