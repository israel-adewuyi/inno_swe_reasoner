from inno_swe_reasoner.utils.pydantic_config import BaseConfig, BaseSettings
from inno_swe_reasoner.config import ModelConfig


class CoconutDataConfig(BaseConfig):
    """ Configuration class for COCONUT SFT data processing. """
    # Name of the dataset on huggingface
    name: str = "coconut_sft_dataset"
    # Split to use
    split: str = "train"
    # Maximum sequence length
    max_seq_length: int = 1024
    # Number of samples to use (for debugging, None means all)
    num_samples: int | None = None

class CoconutTrainerConfig(BaseSettings):
    """ Configuration class for COCONUT trainer. """
    # model related config
    model: ModelConfig = ModelConfig()
    
    # data related config
    data: CoconutDataConfig = CoconutDataConfig()
    

    