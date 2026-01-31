import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inno_swe_reasoner.config import CoconutEvalConfig
from inno_swe_reasoner.coconut.lcb_eval import run_lcb_custom_evaluator
from inno_swe_reasoner.utils.logger import get_logger
from inno_swe_reasoner.utils.utils import get_ckpt_dir

# From: https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/code_generation_lite.py
ALLOWED_FILES = {
    "v1": ["test.jsonl"],
    "v2": ["test.jsonl", "test2.jsonl"],
    "v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}

USER_PROMPT = """### Question

{question}

### Format

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

```python
# YOUR CODE HERE
```

### Answer (use the provided format with backticks)

"""


class CoconutEvaluator:
    def __init__(self, config: CoconutEvalConfig, output_dir, run_name: str | None = None):
        self.config = config
        self.logger = get_logger()
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.dataset = self.load_lcb_dataset()
        self.run_name = run_name

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Best-effort extraction of python code inside triple backticks."""
        match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

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
        version = self.config.lcb_version
        if version not in ALLOWED_FILES:
            raise ValueError(
                f"Invalid version: {version}. Must be one of {list(ALLOWED_FILES.keys())}"
            )

        file_paths = [
            hf_hub_download(
                repo_id=self.config.dataset_name,
                filename=jsonl_file,
                repo_type="dataset",
            )
            for jsonl_file in ALLOWED_FILES[version]
        ]

        dataset = load_dataset("json", data_files=file_paths)["train"]

        def process_example(example):
            prompt = USER_PROMPT.format(
                question=example["question_content"],
            )

            metadata = (
                json.loads(example["metadata"])
                if isinstance(example["metadata"], str)
                else example["metadata"]
            )

            return {
                "question_id": example["question_id"],
                "prompt": prompt,
                "public_test_cases": example["public_test_cases"],
                "private_test_cases": example["private_test_cases"],
                "starter_code": example.get("starter_code", ""),
                "platform": example["platform"],
                "difficulty": example.get("difficulty", "unknown"),
                "fn_name": metadata.get("func_name"),
                "contest_date": example["contest_date"],
            }

        processed_dataset = dataset.map(
            process_example, remove_columns=dataset.column_names
        )
        if self.config.num_samples is not None:
            processed_dataset = processed_dataset.select(
                range(min(self.config.num_samples, len(processed_dataset)))
            )
        self.logger.info(
            f"Loaded {len(processed_dataset)} problems from LiveCodeBench {version}"
        )

        return processed_dataset

    def generate_samples(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        num_completions: int,
    ) -> list[str]:
        """Generate one or more samples from the model (no batching)."""
        # Format prompt for Qwen chat format
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                num_return_sequences=num_completions,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part for each sequence
        input_len = inputs.input_ids.shape[1]
        generated_texts = []
        for seq in outputs:
            generated_ids = seq[input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(self._extract_code_block(generated_text))

        return generated_texts

    def eval(self, step: int):
        # Load model and tokenizer
        model = self.load_model(step).eval().to("cuda")
        tokenizer = self.load_tokenizer()

        results_dir = self.config.output_dir / "eval_results"
        if self.run_name:
            results_dir = results_dir / self.run_name
        results_dir = results_dir / f"step_{step}"
        results_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for example in tqdm(self.dataset, desc="Evaluating"):
            generated_solutions = self.generate_samples(
                model,
                tokenizer,
                example["prompt"],
                num_completions=self.config.num_completions,
            )
            result = {
                "question_id": example["question_id"],
                "code_list": generated_solutions,
            }
            results.append(result)

        output_path = results_dir / "results_final.json"
        self._save_results(results, output_path)
        if self.config.lcb_custom_evaluate:
            run_lcb_custom_evaluator(
                output_path,
                evaluator_module=self.config.lcb_custom_evaluator_module,
                extra_args=self.config.lcb_custom_eval_args,
            )
        self.logger.info(f"Evaluation complete. Results saved to {results_dir}")

        return results

    def _save_results(self, results: list, output_path: Path):
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
