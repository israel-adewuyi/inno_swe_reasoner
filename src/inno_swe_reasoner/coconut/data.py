# Adapted from https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/sft/data.py

import json
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

from inno_swe_reasoner.coconut.config import CoconutDataConfig
from inno_swe_reasoner.utils.logger import get_logger


class SFTDataset:
    """ Dataset wrapping HF SFT dataset with `message` format. """
    def __init__(
        self, 
        dataset: Dataset,
        tokenizer: AutoTokenizer,
        shuffle: bool = True,
        max_epochs: int | None = None,
        seed: int = 42,
        seq_len: int = 100000,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.num_examples = len(dataset)
        self.step = 0
        self.epoch = 0
        self.max_epochs = max_epochs
        self.seq_len = seq_len
        self.logger = get_logger()


    def __getitem__(self, idx):
        return self.dataset[idx]

    def debug_example(self, idx=0):
        """Debug helper to check example structure"""
        example = self.dataset[idx]
        print(f"Example type: {type(example)}")
        print(f"Example keys: {example.keys() if isinstance(example, dict) else 'Not a dict'}")
        print(f"Example type of value = {type(example["messages"])}")
        # print(f"Example content: {example}")
        return example
    
    def _process(self, example: dict):
        # Skip tokenizer if no tokenizer is provided
        if self.tokenizer is None:
            return example

        # Assumes that the key 'messages' exists in the example dict
        if "messages" not in example:
            raise ValueError("All examples must have a 'messages' column for SFT training.")
        
        # messages is string-ified list of dicts, parse it
        def parse_messages(example):
            example['messages'] = json.loads(example['messages'])
            return example
        
        def strip_content(messages: list[dict]) -> list[dict]:
            def _strip_content(message: dict) -> dict:
                if isinstance(message.get("content"), str):
                    return {**message, "content": message["content"].strip()}
                return message

            return [_strip_content(message) for message in messages]
        
        def build_loss_mask(messages: list[dict], tokenizer) -> list[int]:
            """
            Build loss mask by incrementally tokenizing messages and using their loss_mask values.
            """
            loss_mask: list[int] = []
            prev_ids, prev_len = [], 0
            
            for i, message in enumerate(messages):
                assert "role" in message, "Message must have a role"
                assert "loss_mask" in message, "Message must have a loss_mask field"
                
                # Tokenize up to current message
                cur_ids = tokenizer.apply_chat_template(
                    messages[: i + 1],
                    add_generation_prompt=False,
                    **example.get("chat_template_kwargs", {}),
                )
                
                # Validate incremental tokenization
                assert prev_ids == cur_ids[:prev_len], (
                    f"Mismatch in incremental tokenization at message {i}. "
                    f"Previous ids: {prev_ids} != {cur_ids[:prev_len]}"
                )
                
                # Extend loss mask with this message's mask value for all its tokens
                num_new_tokens = len(cur_ids) - prev_len
                loss_mask.extend([message["loss_mask"]] * num_new_tokens)
                
                prev_ids, prev_len = cur_ids, len(cur_ids)
            
            return loss_mask
        
        example = parse_messages(example)
        messages = strip_content(example['messages'])
        
        # Validate that all messages have loss_mask
        for msg in messages:
            if "loss_mask" not in msg:
                raise ValueError(f"Message with role '{msg['role']}' missing loss_mask field")
        
        prompt = example['messages'][0] if example['messages'][0]['role'] == 'user' else example['messages'][1]
        completion = example["messages"][-1] if example['messages'][-1]['role'] == 'assistant' else example['messages'][-2]
        
        # Tokenize all messages
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            **example.get("chat_template_kwargs", {}),
        )
        
        # Build loss mask based on per-message loss_mask values
        loss_mask = build_loss_mask(messages, self.tokenizer) 
        
        # If EOS token is not found, manually append it
        if not self.tokenizer.eos_token_id in input_ids:
            self.logger.warning(
                f"Did not find EOS token ID {self.tokenizer.eos_token_id} in input_ids. Is something wrong with the chat template? Manually appending EOS token..."
            )
            input_ids.append(cast(int, self.tokenizer.eos_token_id))
            loss_mask.append(True)
            
        # Prepare inputs
        target_ids = input_ids.copy()[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]
        
        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
            )
            return

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert self.tokenizer.eos_token_id in target_ids, "EOS token ID must be present in target_ids"

        # Create sample (with one fake target for the last token)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
        }
        
        
    
    def __iter__(self):
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset
        while True:
            self.step += 1
            
            # Determine epoch from current step
            epoch = (self.step - 1) // self.num_examples
            
            # Break if max epochs reached
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break
        
            # Update stored epoch if new epoch is reached, optionally shuffle
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset
            
            example = dataset[(self.step - 1) % self.num_examples]
            
            processed_example = self._process(example)
            
            if processed_example is None:
                continue
            
            yield processed_example
                


def setup_dataset(config: CoconutDataConfig, tokenizer: AutoTokenizer) -> Dataset:
    logger = get_logger()
    
    logger.info(f"Loading dataset from {config.name} split {config.split}...")
    dataset = load_dataset(config.name, split=config.split)
    if config.num_samples is not None:
        dataset = dataset.take(config.num_samples)
        logger.info(f"Selected first {config.num_samples} samples from the dataset.")
    
    return dataset