from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from inno_swe_reasoner.coconut.config import CoconutDataConfig
from inno_swe_reasoner.utils.logger import get_logger



def setup_dataset(config: CoconutDataConfig, tokenizer: AutoTokenizer) -> Dataset:
    logger = get_logger()
    
    logger.info(f"Loading dataset from {config.name} split {config.split}...")
    dataset = load_dataset(config.name, split=config.split)
    if config.num_samples is not None:
        dataset = dataset.take(config.num_samples)
        logger.info(f"Selected first {config.num_samples} samples from the dataset.")
    
    return dataset