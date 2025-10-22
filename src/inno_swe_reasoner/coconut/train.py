from inno_swe_reasoner.coconut.config import CoconutTrainerConfig
from inno_swe_reasoner.utils.pydantic_config import parse_argv

def train(config: CoconutTrainerConfig):
    pass


def main():
    train(parse_argv(CoconutTrainerConfig))


if __name__ == "__main__":
    main()
