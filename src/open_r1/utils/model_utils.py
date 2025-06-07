# open_r1/utils/model_utils.py

import logging
import inspect
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_quantization_config,
    AutoModelForCausalLMWithValueHead,
)
from ..configs import GRPOConfig, SFTConfig

logger = logging.getLogger(__name__)

# Try to import Accelerate’s unwrap helper; otherwise identity
try:
    from accelerate import unwrap_model as _accelerate_unwrap_model
except ImportError:
    def _accelerate_unwrap_model(model):
        return model


def get_tokenizer(
    model_args: ModelConfig,
    training_args: SFTConfig | GRPOConfig
) -> PreTrainedTokenizer:
    logger.info(
        "get_tokenizer: loading tokenizer '%s' (rev=%s, trust_remote_code=%s)",
        model_args.model_name_or_path,
        model_args.model_revision,
        model_args.trust_remote_code,
    )
    tok = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if getattr(training_args, "chat_template", None):
        tok.chat_template = training_args.chat_template
        logger.info("→ applied chat_template to tokenizer")
    logger.info("← loaded tokenizer; vocab_size=%d", len(tok))
    return tok


def get_model(
    model_args: ModelConfig,
    training_args: SFTConfig | GRPOConfig
) -> torch.nn.Module:
    """
    1) Load either plain AutoModelForCausalLM or
       AutoModelForCausalLMWithValueHead (if GRPOConfig).
    2) Patch ModelClass.forward so that:
         • return_dict=True
         • any engine wrapper is unwrapped
         • the TRL forward (with `.value`) is invoked
    """
    # ── dtype + cache_dir ────────────────────────────────────────────────
    torch_dtype = (
        getattr(torch, model_args.torch_dtype)
        if model_args.torch_dtype not in (None, "auto")
        else model_args.torch_dtype
    )
    cache_dir = getattr(model_args, "cache_dir", None)

    logger.info(
        "get_model: %s (dtype=%s, cache_dir=%s)",
        model_args.model_name_or_path,
        torch_dtype,
        cache_dir or "<default>",
    )

    # ── quant / device_map ───────────────────────────────────────────────
    quant_cfg = get_quantization_config(model_args)
    if quant_cfg is not None:
        device_map = get_kbit_device_map()
        logger.info("→ using k-bit device_map: %s", device_map)
    else:
        device_map = None

    common_kwargs = dict(
        revision            = model_args.model_revision,
        trust_remote_code   = model_args.trust_remote_code,
        attn_implementation = model_args.attn_implementation,
        torch_dtype         = torch_dtype,
        use_cache           = not training_args.gradient_checkpointing,
        device_map          = device_map,
        quantization_config = quant_cfg,
    )
    if cache_dir:
        common_kwargs["cache_dir"] = cache_dir

    # ── pick class & load ────────────────────────────────────────────────
    is_grpo = isinstance(training_args, GRPOConfig) or hasattr(training_args, "reward_clip_min")
    if is_grpo:
        logger.info("→ loading AutoModelForCausalLMWithValueHead")
        ModelClass = AutoModelForCausalLMWithValueHead
    else:
        logger.info("→ loading plain AutoModelForCausalLM")
        ModelClass = AutoModelForCausalLM

    model = ModelClass.from_pretrained(model_args.model_name_or_path, **common_kwargs)

    # ── ensure TRL shims for GRPO ─────────────────────────────────────────
    if is_grpo:
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = defaultdict(bool)
        if not hasattr(model, "add_model_tags"):
            model.add_model_tags = lambda tags: setattr(model, "model_tags", tags)
        if not hasattr(model, "get_model_tags"):
            model.get_model_tags = lambda: getattr(model, "model_tags", [])

    # ── patch the CLASS’s forward once for all ────────────────────────────
    orig_forward = ModelClass.forward

    def patched_forward(self, *args, **kwargs):
        # 1) always return a dict‐style ModelOutput
        kwargs.setdefault("return_dict", True)

        # 2) unwrap any DDP/DeepSpeed shells
        real = _accelerate_unwrap_model(self)
        if hasattr(real, "module"):
            real = real.module

        # 3) let any extra HF/TRL keywords (like logits_to_keep!) pass through
        return orig_forward(real, *args, **kwargs)

    # install it on the class so that EVERY future .forward goes through it
    ModelClass.forward = patched_forward

    logger.info(
        "→ patched %s.forward to unwrap wrappers and force return_dict=True",
        ModelClass.__name__,
    )

    # ── final sanity check ────────────────────────────────────────────────
    has_value = any(
        hasattr(model, a)
        for a in ("value_head", "get_value", "model_value_head", "value")
    )
    logger.info(
        "← Loaded model: class=%s, has_value_head=%s",
        model.__class__.__name__, has_value
    )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Now patch the three common engine wrappers so that
# `engine(input_ids=…,…)` also unwraps and returns a ModelOutput.

def _engine_call(self, *args, **kwargs):
    real = _accelerate_unwrap_model(self)
    if hasattr(real, "module"):
        real = real.module
    kwargs.setdefault("return_dict", True)
    # invoke the unwrapped module’s own forward
    return real.forward(*args, **kwargs)


# 1) PyTorch DDP
try:
    from torch.nn.parallel.distributed import DistributedDataParallel
    DistributedDataParallel.__call__ = _engine_call
    logger.info("→ patched DDP.__call__")
except ImportError:
    pass

# 2) PyTorch FSDP (>=2.0)
try:
    from torch.distributed.fsdp import FullyShardedDataParallel
    FullyShardedDataParallel.__call__ = _engine_call
    logger.info("→ patched FSDP.__call__")
except ImportError:
    pass

# 3) DeepSpeed Engine
try:
    from deepspeed.runtime.engine import DeepSpeedEngine
    DeepSpeedEngine.__call__ = _engine_call
    logger.info("→ patched DeepSpeedEngine.__call__")
except ImportError:
    pass
