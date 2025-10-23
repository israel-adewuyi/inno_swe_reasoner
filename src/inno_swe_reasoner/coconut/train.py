from inno_swe_reasoner.coconut.config import CoconutTrainerConfig
from inno_swe_reasoner.utils.pydantic_config import parse_argv
from inno_swe_reasoner.model import setup_model, setup_tokenizer

def train(config: CoconutTrainerConfig):    
    # Setup model
    model = setup_model(config.model)

    # Setup tokenizer
    tokenizer = setup_tokenizer(config.model)


def main():
    train(parse_argv(CoconutTrainerConfig))


if __name__ == "__main__":
    main()
