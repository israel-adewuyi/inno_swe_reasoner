from transformers import AutoModelForCausalLM, AutoTokenizer

from inno_swe_reasoner.utils.utils import get_ckpt_dir


class CoconutEvaluator:
    def __init__(self, config: CoconutEvalConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}" / "trainer"

    def load_model(step: int) -> AutoModelForCausalLM:
        ckpt_path = self.get_ckpt_path(step)
        model = AutoModelForCausalLM.from_pretrained(ckpt_path)
        return model

    def eval(step: int, tokenizer: AutoTokenizer):
        model = self.load_model().eval().to("cuda")

        return model
