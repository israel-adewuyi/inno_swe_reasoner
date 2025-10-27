from inno_swe_reasoner.coconut.config import CoconutTrainerConfig
from inno_swe_reasoner.utils.pydantic_config import parse_argv
from inno_swe_reasoner.model import setup_model, setup_tokenizer
from inno_swe_reasoner.utils.logger import setup_logger
from inno_swe_reasoner.coconut.data import setup_dataset, setup_dataloader

def train(config: CoconutTrainerConfig):
    # Setup logger
    logger = setup_logger(log_level="INFO") # TODO: Make log file and log level configurable
    # Setup model
    logger.info("Setting up model...")
    # model = setup_model(config.model)

    # Setup tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer = setup_tokenizer(config.model)
    
    # Setup dataset
    logger.info("Setting up dataset...")
    dataset = setup_dataset(config.data, tokenizer)
    
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

    # Debug
    while True:
        point = next(dataiter)
        print(point)
        break
    
    # print(dataset)


def main():
    train(parse_argv(CoconutTrainerConfig))


if __name__ == "__main__":
    main()
