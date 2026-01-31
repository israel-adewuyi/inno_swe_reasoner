# Adapted from from https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/config.py

from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from inno_swe_reasoner.utils.pydantic_config import BaseConfig

AttnImplementation: TypeAlias = Literal["sdpa", "flash_attention_2"]

MOE_MODEL_MAPS = {
    "Qwen/Qwen3-30B-A3B": "Jackmin108/Qwen3-30B-A3B-Fast",
    "moonshotai/Moonlight-16B-A3B-Instruct": "Jackmin108/Moonlight-16B-A3B-Instruct-Fast",
}


class ModelConfig(BaseConfig):
    """Configures the model for training."""

    name: Annotated[
        str,
        Field(
            description="Name or path of the HF model to use.",
        ),
    ] = "Qwen/Qwen3-0.6B"

    attn: Annotated[
        AttnImplementation, Field(description="The attention implementation to use.")
    ] = "flash_attention_2"

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for model and tokenizer initialization.",
        ),
    ] = False

    impl: Annotated[
        Literal["hf", "liger_kernel", "custom"],
        Field(
            description="Whether to use Liger Kernel.",
        ),
    ] = "hf"

    load_using_meta: Annotated[
        bool,
        Field(
            description="Whether to load the model using meta device then load from HF ckpt.",
        ),
    ] = False

    optimization_dtype: Annotated[
        Literal["bfloat16", "float32"],
        Field(
            description="The dtype to use for the model optimization.",
        ),
    ] = "float32"

    reduce_dtype: Annotated[
        Literal["bfloat16", "float32"],
        Field(
            description="The dtype to use for the model reduce.",
        ),
    ] = "float32"

    @model_validator(mode="after")
    def _map_model_name_for_moe(self):
        """Map model name if it exists in MOE_MODEL_MAPS."""
        if self.name in MOE_MODEL_MAPS:
            self.name = MOE_MODEL_MAPS[self.name]
        return self

    @model_validator(mode="after")
    def trust_remote_code_only_with_hf(self):
        """Trust remote code only if the model is from HF."""
        if self.trust_remote_code:
            if self.impl != "hf":
                raise ValueError(
                    "Trust remote code is only supported with the HF implementation."
                )
        return self


class ConstantSchedulerConfig(BaseModel):
    """Configuration for constant learning rate scheduler."""

    type: Literal["constant"] = "constant"


class LinearSchedulerConfig(BaseModel):
    """Configuration for linear learning rate scheduler."""

    type: Literal["linear"] = "linear"

    warmup_steps: Annotated[
        int,
        Field(
            ge=0, description="Number of warmup steps for the learning rate scheduler."
        ),
    ] = 10

    decay_steps: Annotated[
        int,
        Field(
            ge=0,
            description="Number of steps to decay the learning rate during the final portion of training.",
        ),
    ] = 10

    min_lr: Annotated[
        float, Field(ge=0, description="Minimum learning rate to converge to.")
    ] = 0.0


class CosineSchedulerConfig(BaseModel):
    """Configuration for cosine learning rate scheduler."""

    type: Literal["cosine"] = "cosine"

    warmup_steps: Annotated[
        int,
        Field(
            ge=0, description="Number of warmup steps for the learning rate scheduler."
        ),
    ] = 10

    min_lr: Annotated[
        float, Field(ge=0, description="Minimum learning rate to converge to.")
    ] = 0.0


SchedulerConfigType: TypeAlias = (
    ConstantSchedulerConfig | LinearSchedulerConfig | CosineSchedulerConfig
)


class BaseOptimizerConfig(BaseModel):
    lr: Annotated[float, Field(ge=0)] = 1e-6
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    max_norm: Annotated[
        float, Field(ge=0, description="Maximum gradient norm to clip.")
    ] = 1.0


class SGDConfig(BaseOptimizerConfig):
    type: Literal["sgd"] = "sgd"
    nesterov: bool = True
    momentum: float = 0.9


class AdamWConfig(BaseOptimizerConfig):
    type: Literal["adamw"] = "adamw"

    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


class MuonConfig(BaseOptimizerConfig):
    type: Literal["muon"] = "muon"

    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


OptimizerConfigType: TypeAlias = SGDConfig | AdamWConfig | MuonConfig


class LogExtrasConfig(BaseConfig):
    """Configures extra logging for W&B tables."""

    samples: Annotated[
        bool,
        Field(
            description="Whether to log prompt/response samples to W&B tables.",
        ),
    ] = True

    distributions: Annotated[
        bool,
        Field(
            description="Whether to log distributions (like rewards, advantages, etc.) to W&B tables.",
        ),
    ] = True

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Step interval at which to log extras to W&B table.",
        ),
    ] = 10


class CoconutEvalConfig(BaseConfig):
    """Configures necessary parameters for evaluation on LiveCodeBench"""

    model_name: Annotated[str | None, Field(description="Name of the model on HF")] = (
        None
    )

    dataset_name: Annotated[
        str | None, Field(description="Exact name of the dataset to use")
    ]

    lcb_version: Annotated[
        str | None, Field(description="The version of LCB to load for evaluation")
    ] = None

    output_dir: Annotated[
        Path | None,
        Field(description="File path where code completions should be saved.s"),
    ] = Path("outputs")

    temperature: Annotated[
        float, Field(description="the temperature for sampling from the model")
    ] = 0.7

    max_new_tokens: Annotated[
        int, Field(description="The maximum number of tokens to sample from the model")
    ] = 1024

    num_completions: Annotated[
        int, Field(description="Number of completions to sample per prompt")
    ] = 1

    eval_batch_size: Annotated[
        int, Field(description="Batch size for evaluation generation")
    ] = 1

    lcb_custom_evaluate: Annotated[
        bool,
        Field(
            description="Whether to run LiveCodeBench custom evaluator on saved outputs"
        ),
    ] = True

    lcb_custom_evaluator_module: Annotated[
        str | None,
        Field(
            description=(
                "Python module path for LiveCodeBench custom evaluator "
                "(e.g., lcb_runner.runner.custom_evaluator or "
                "livecodebench.runner.custom_evaluator). If None, try common defaults."
            )
        ),
    ] = None

    lcb_custom_eval_args: Annotated[
        list[str],
        Field(
            description="Extra CLI args to pass to the LiveCodeBench custom evaluator"
        ),
    ] = []

    num_samples: Annotated[
        int | None, Field(description="The number of samples to process")
    ] = None


class WandbMonitorConfig(BaseConfig):
    """Configures logging to Weights and Biases."""

    # Shared configs (May be overwritten by WandbConfig from `rl.py`)
    project: Annotated[str, Field(description="The W&B project to log to.")] = (
        "prime-rl"
    )

    name: Annotated[
        str | None,
        Field(
            description="The W&B name to to use for logging.",
        ),
    ] = None

    offline: Annotated[
        bool, Field(description="Whether to run W&B in offline mode.")
    ] = False

    # Individual configs (can only be specified on trainer or orchestrator)
    id: Annotated[
        str | None,
        Field(
            description="The W&B run ID to log to. If None, a random ID will be generated. If you want to resume a run, you can set the ID to the run ID you want to resume.",
        ),
    ] = None

    log_extras: Annotated[
        LogExtrasConfig | None,
        Field(
            description="Configuration for logging extras to W&B tables. If None, no extras are logged.",
        ),
    ] = LogExtrasConfig()


class WeightCheckpointConfig(BaseConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

    interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Interval at which to save the training checkpoint. If None, will only checkpoint at the end of training.",
        ),
    ] = None

    keep_last_n: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of previous checkpoints to keep",
        ),
    ] = None
