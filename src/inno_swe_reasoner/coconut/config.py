from inno_swe_reasoner.utils.pydantic_config import BaseConfig, BaseSettings
from inno_swe_reasoner.config import ModelConfig

class CoconutTrainerConfig(BaseSettings):
    """ Configuration class for COCONUT trainer. """
    model: ModelConfig = ModelConfig()
    
    