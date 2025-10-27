from inno_swe_reasoner.utils.pydantic_config import BaseConfig, BaseSettings
from inno_swe_reasoner.config import ModelConfig, AdamWConfig, OptimizerConfigType

from pydantic import Field
from typing import Annotated

class CoconutDataConfig(BaseConfig):
    """ Configuration class for COCONUT SFT data processing. """
    # Name of the dataset on huggingface
    name: str = "coconut_sft_dataset"
    # Split to use
    split: str = "train"
    # Maximum sequence length
    max_seq_length: int = 65536
    # Number of samples to use (for debugging, None means all)
    num_samples: int | None = None
    # if to shuffle the dataset
    shuffle: bool = True
    # batch size
    batch_size: int = 1
    # seed
    seed: int = 42
    # max epochs
    max_epochs: int = 3
    # COCONUT specific parameters
    # num coconut stages
    num_stages: int = 3
    # number of continuous thoughts per step
    c: int = 2
    epoch_per_stage: int = 2

class CoconutTrainerConfig(BaseSettings):
    """ Configuration class for COCONUT trainer. """
    # model related config
    model: ModelConfig = ModelConfig()
    
    # data related config
    data: CoconutDataConfig = CoconutDataConfig()
    
    # The optimizer configuration
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    