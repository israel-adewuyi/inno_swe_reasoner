import torch

from torch.nn.functional import cross_entropy
from inno_swe_reasoner.coconut.config import CoconutTrainerConfig
from inno_swe_reasoner.utils.pydantic_config import parse_argv
from inno_swe_reasoner.coconut.model import setup_model, setup_tokenizer
from inno_swe_reasoner.utils.logger import setup_logger
from inno_swe_reasoner.optim import setup_optimizer
from inno_swe_reasoner.coconut.data import setup_dataset, setup_dataloader, tokenize_data

def train(config: CoconutTrainerConfig):
    # Setup logger
    logger = setup_logger(log_level="INFO") # TODO: Make log file and log level configurable
    # Setup model
    logger.info("Setting up model...")
    model = setup_model(config.model)

    # Setup tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer = setup_tokenizer(config.model)
    
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, model)
    
    # Setup dataset
    logger.info("Setting up dataset...")
    dataset = setup_dataset(config.data, tokenizer)
    # dataset.debug_example()
    
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

    # Train loop
    steps = 0
    while True:
        steps += 1
        batch = next(dataiter)
        # print(batch)
        
        for stage in range(config.data.num_stages + 1):
            if stage > 0:
                logger.info(f"Resetting optimizer at Stage {stage}")
                optimizer = setup_optimizer(config.optim, model) # TODO: Implement this function

            for epochs in range(config.data.epoch_per_stage):
                for prompt, cot_steps, answer in zip(batch["prompt"], batch["cot_steps"], batch["answer"]):
                    # Determine how many steps to replace with continuous thoughts
                    steps_to_replace = min(stage, len(cot_steps))  # Handle shorter chains
                    num_continuous_thoughts = steps_to_replace * config.data.c  # c thoughts per step
                    remaining_cot = cot_steps[steps_to_replace:]  # Remaining language steps
                
                    if stage == 0:
                        # Stage 0: Regular CoT training (no continuous thoughts)
                        input_ids, loss_mask = tokenize_data(tokenizer, prompt, cot_steps, answer)
                        target_ids = torch.tensor(input_ids.copy()[1:]).unsqueeze(0)
                        input_ids = torch.tensor(input_ids[:-1]).unsqueeze(0)
                        loss_mask = torch.tensor(loss_mask).unsqueeze(0)
                        position_ids = torch.tensor(list(range(len(input_ids)))).unsqueeze(0)

                        assert input_ids.shape == target_ids.shape == loss_mask.shape, (
                            f"input_ids, loss_mask and target_ids must have the same length, but got {input_ids.shape=}, {loss_mask.shape=}, {target_ids.shape=}"
                        )

                        # run forward pass on the tokens
                        logits = model.embed_and_forward(input_ids, position_ids, False)
                        B, L, V = logits.shape
                        loss = cross_entropy(logits.view(-1, V), target_ids.view(-1), reduction="none").view(B, L)
                        
                        loss = loss[loss_mask].mean()
                        
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # # Single forward pass
                        # logits = model(input_embeddings)
                        # loss = compute_loss(logits, targets, mask_prompt=True)
                # Forward pass
            break
        break


def main():
    train(parse_argv(CoconutTrainerConfig))


if __name__ == "__main__":
    main()
