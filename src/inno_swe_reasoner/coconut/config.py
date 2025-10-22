from inno_swe_reasoner.utils.pydantic_config import BaseConfig, BaseSettings

class CoconutTrainerConfig(BaseSettings):
    """ Configuration class for COCONUT trainer. """
    model: str,
    data: str,
    