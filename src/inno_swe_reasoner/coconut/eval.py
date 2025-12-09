from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from inno_swe_reasoner.config import CoconutEvalConfig
from inno_swe_reasoner.utils.logger import get_logger
from inno_swe_reasoner.utils.utils import get_ckpt_dir


class CoconutEvaluator:
    def __init__(self, config: CoconutEvalConfig, output_dir):
        self.config = config
        self.logger = get_logger()
        self.ckpt_dir = get_ckpt_dir(output_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}" / "trainer"

    def load_model(self, step: int) -> AutoModelForCausalLM:
        ckpt_path = self.get_ckpt_path(step)
        self.logger.info(f"Loading {ckpt_path} for evaluation")
        # if evaluating the initial model, load the state dict on HF
        if step > 0:
            model = AutoModelForCausalLM.from_pretrained(ckpt_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        return model

    def load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_lcb_dataset(self):
        self.logger.info("Loading LCB dataset for evaluation")
        dataset = load_dataset(
            "livecodebench/code_generation_lite", split=self.config.lcb_release
        )
        return dataset

    def eval(self, step: int):
        # Load model and tokenizer
        model = self.load_model(step).eval().to("cuda")
        # tokenizer = self.load_tokenizer()

        # Load dataset
        # dataset = self.load_lcb_dataset()

        return model
