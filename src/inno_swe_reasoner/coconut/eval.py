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

    def generate_samples_batch(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompts: list[str],
        num_completions: int,
    ) -> list[list[str]]:
        """Generate samples for a batch of prompts."""
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
        formatted_prompts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_list
        ]

        inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True
        ).to("cuda")

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

        # Compute per-sample prompt lengths from attention mask
        input_lengths = inputs.attention_mask.sum(dim=1).tolist()
        generated_texts = []
        for i, seq in enumerate(outputs):
            input_len = input_lengths[i // num_completions]
            generated_ids = seq[input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(self._extract_code_block(generated_text))

        # Regroup into list per prompt
        grouped = []
        for i in range(0, len(generated_texts), num_completions):
            grouped.append(generated_texts[i : i + num_completions])
        return grouped

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
        batch = []
        batch_meta = []
        batch_size = max(1, self.config.eval_batch_size)
        for example in tqdm(self.dataset, desc="Evaluating"):
            batch.append(example["prompt"])
            batch_meta.append(example["question_id"])
            if len(batch) == batch_size:
                generated_batches = self.generate_samples_batch(
                    model,
                    tokenizer,
                    batch,
                    num_completions=self.config.num_completions,
                )
                for qid, codes in zip(batch_meta, generated_batches):
                    results.append({"question_id": qid, "code_list": codes})
                batch = []
                batch_meta = []

        if batch:
            generated_batches = self.generate_samples_batch(
                model,
                tokenizer,
                batch,
                num_completions=self.config.num_completions,
            )
            for qid, codes in zip(batch_meta, generated_batches):
                results.append({"question_id": qid, "code_list": codes})

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
