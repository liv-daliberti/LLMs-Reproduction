#!/usr/bin/env python
"""
Stable‐GRPO (with adaptive‐KL and generation‐length caps)

─────────────────────────────────────────────────────────────────────
 • reward / advantage / gradient clipping  
 • adaptive‐KL coefficient (keep KL near target_kl)  
 • prompt‐length truncation for vLLM  
 • robust reward wrapper  
 • DeepSpeed‐ZeRO pickle patch (PT ≥ 2.6)  
 • dtype fix for ids/masks  
 • COSINE reward legacy‐factory shim  
─────────────────────────────────────────────────────────────────────
"""
import logging
import os
import sys
import functools
import inspect
from dataclasses import dataclass
import torch
from packaging import version

# ── DeepSpeed ZeRO pickle patch ──────────────────────────────────────────────
if version.parse(torch.__version__) >= version.parse("2.6.0"):
    from torch.serialization import add_safe_globals
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    add_safe_globals([ZeroStageEnum, LossScaler])

# ── Force IDs / masks to .long() inside GRPOTrainer ──────────────────────────
import trl.trainer.grpo_trainer as _grpo_mod
from transformers import PreTrainedTokenizerBase

# First, wrap the original _prepare_inputs to cast input_ids/attention_mask → .long()
_orig_prepare = _grpo_mod.GRPOTrainer._prepare_inputs
def _patched_prepare(self, inputs):
    inputs = _orig_prepare(self, inputs)
    if "input_ids" in inputs:
        inputs["input_ids"]      = inputs["input_ids"].long()
        inputs["attention_mask"] = inputs["attention_mask"].long()
    return inputs
_grpo_mod.GRPOTrainer._prepare_inputs = _patched_prepare

# Then, wrap again so that if a batch contains "solution", we forward it into reward_kwargs
_original_prepare_with_cast = _grpo_mod.GRPOTrainer._prepare_inputs
def _prepare_with_solution(self, inputs):
    batch = _original_prepare_with_cast(self, inputs)
    if "solution" in batch:
        sol = batch.pop("solution")
        self.args.reward_kwargs = {"solution": sol}
    return batch
_grpo_mod.GRPOTrainer._prepare_inputs = _prepare_with_solution

# ── Side‐effect patch (vLLM hooks) ───────────────────────────────────────────
import open_r1.utils.vllm_patch  # noqa: F401

# ── Standard Open‐R1 / HF imports ────────────────────────────────────────────
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from transformers import EarlyStoppingCallback

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
import open_r1.rewards as _rewards
from open_r1.rewards import get_reward_funcs

# ══════════════════════════════════════════════════════════════════════════════
#  “Robust Reward” wrapper (no changes here)  
# ══════════════════════════════════════════════════════════════════════════════
def _robustify(fn):
    if getattr(fn, "_robust_wrapped_", False):
        return fn

    sig = inspect.signature(fn)
    wants_p = "prompts"     in sig.parameters
    wants_c = "completions" in sig.parameters

    @functools.wraps(fn)
    def wrapped(prompts, completions, **kw):
        fixed = []
        for c in completions:
            if isinstance(c, list) and c and isinstance(c[0], dict):
                fixed.append(c)
            elif isinstance(c, list) and c and isinstance(c[0], str):
                fixed.append([{"role": "assistant", "content": c[0]}])
            else:
                fixed.append([{"role": "assistant", "content": str(c)}])

        call_kw = {}
        if wants_p: call_kw["prompts"] = prompts
        if wants_c: call_kw["completions"] = fixed
        call_kw.update({k: v for k, v in kw.items() if k in sig.parameters})
        return fn(**call_kw)

    wrapped._robust_wrapped_ = True
    return wrapped

for _n, _obj in inspect.getmembers(_rewards, inspect.isfunction):
    if "completions" not in inspect.signature(_obj).parameters:
        continue
    setattr(_rewards, _n, _robustify(_obj))

# ══════════════════════════════════════════════════════════════════════════════
#  Stable‐GRPO custom config + trainer  
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class StableGRPOConfig(GRPOConfig):
    # ── reward/advantage clipping ──────────────────────────────────────────────
    reward_clip_min: float = -1.0
    reward_clip_max: float =  1.0
    adv_clip_range:   float =  5.0
    max_grad_norm:    float =  1.0

    # ── adaptive‐KL parameters ────────────────────────────────────────────────
    init_kl_coef: float = 0.4      # initial KL coefficient
    adapt_kl_coef: bool  = True    # turn on automatic KL adjustment
    target_kl: float      = 0.05    # desired KL (per generation step)
    kl_horizon: int       = 64     # how many update steps between each KL adaptation

MAX_CTX, RESERVED = 2548, 2048
from torch.nn.utils import clip_grad_norm_

class StableGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set up KL coefficient and a counter for adaptation
        self.kl_coef = float(self.args.init_kl_coef)
        self._kl_step_counter = 0

        # Save the original HF‐style reference to override in loss:
        # (we assume GRPOTrainer’s loss looks like: policy_loss + kl_coef * kl_loss)
        # We’ll re‐inject our own kl‐scaling logic every training step.
        #

    # ─── reward / advantage / gradient clipping ────────────────────────────────
    def _compute_rewards(self, m_out, qs, rs):
        return torch.clamp(
            super()._compute_rewards(m_out, qs, rs),
            self.args.reward_clip_min,
            self.args.reward_clip_max,
        )

    def compute_advantages(self, rewards, values, **kw):
        adv = super().compute_advantages(rewards, values, **kw)
        return torch.clamp(
            adv,
            -self.args.adv_clip_range,
            self.args.adv_clip_range,
        )

    def training_step(self, *args, **kwargs):
        """
        Every time we do a training_step, we also (a) apply gradient clipping
        (b) after kl_horizon steps, check actual KL and adjust self.kl_coef up/down.
        """
        # 1) run the normal GRPOTrainer.training_step to get the loss
        loss = super().training_step(*args, **kwargs)

        # 2) clip gradients if requested
        if self.args.max_grad_norm and self.args.max_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        # 3) adapt KL coefficient if needed
        if self.args.adapt_kl_coef:
            self._kl_step_counter += 1
            if self._kl_step_counter >= self.args.kl_horizon:
                # retrieve the *average* KL from the last forward pass.
                # In GRPOTrainer, after each update, trainer.log_metrics("train", metrics)
                # should include “kl” in metrics.  We can fetch it from self.state.log_history,
                # or (more robustly) peek at whatever was last returned from evaluate() if your version logs it.
                #
                # For simplicity, assume that in each training loop the trainer sets
                #   self.state.log_history[-1]["kl"] = <this batch’s avg KL>
                # If your version of TRL/GRPOTrainer populates “kl” under a different key,
                # you’ll need to inspect self.state.log_history to see where it put KL.
                last_logs = self.state.log_history
                actual_kl = None
                # Walk backwards until we find a “kl” entry
                for rec in reversed(last_logs):
                    if "kl" in rec:
                        actual_kl = rec["kl"]
                        break

                if actual_kl is not None:
                    tgt = float(self.args.target_kl)
                    # if KL is too large, up‐weight the KL term; if too small, down‐weight
                    if actual_kl > tgt * 1.5:
                        self.kl_coef *= 1.2
                    elif actual_kl < tgt * 0.5:
                        self.kl_coef *= 0.8
                    # clamp the KL‐coef within a sane range
                    self.kl_coef = max(1e-4, min(self.kl_coef, 10.0))

                    # Log the adapted coefficient so you can watch it
                    self.log_metrics("train", {"adapted_kl_coef": self.kl_coef})

                self._kl_step_counter = 0

        return loss

    def _generate_and_score_completions(self, inputs):
        """
        1) Truncate the prompt to fit within MAX_CTX‐RESERVED
        2) Call super()._generate_and_score_completions
        3) *Immediately* scale the KL‐penalty in the loss‐computation to use self.kl_coef
        """
        tok: PreTrainedTokenizerBase = self.tokenizer
        max_prompt = MAX_CTX - RESERVED
        for sample in inputs:
            raw = sample.get("prompt")
            if raw is None:
                continue
            txt = ("\n".join(f"{m['role']}: {m['content']}" for m in raw)
                   if isinstance(raw, list) else raw)
            ids = tok(txt, return_tensors="pt", truncation=False).input_ids[0]
            if ids.size(0) > max_prompt:
                # keep only the last max_prompt tokens
                txt = tok.decode(ids[-max_prompt:], skip_special_tokens=True)
            sample["prompt"] = txt

        # 🧯 Patch DeepSpeed ZeRO during vLLM generation
        for param in self.model.parameters():
            if hasattr(param, 'ds_active_sub_modules'):
                param.ds_active_sub_modules.clear()

        # Let the base class do the actual generation + scoring.
        return super()._generate_and_score_completions(inputs)
    # transformers ≥ 4.40 passes `num_items_in_batch`

    # transformers ≥ 4.40 passes `num_items_in_batch`
    def compute_loss(
        self,
        model,
        inputs,
        *,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **unused,                       # ← future-proof
    ):
        """
        Override GRPOTrainer.compute_loss so the KL term uses self.kl_coef
        instead of the fixed coefficient in the base class.
        """
        # Ask the parent for the loss only (return_outputs *must* be False)
        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=num_items_in_batch,
            **unused,
        )

        # GRPOTrainer stores the latest stats on itself; fetch KL if present
        stats = getattr(self, "_last_stats", None) or getattr(self, "stats", None)
        if stats is not None and "kl" in stats:
            kl_val = stats["kl"]
            loss = loss + self.kl_coef * kl_val
            self.log_metrics("train", {
                "raw_kl":        kl_val,
                "final_kl_coef": self.kl_coef,
                "kl_penalty":    self.kl_coef * kl_val,
            })

        return (loss, None) if return_outputs else loss


# ══════════════════════════════════════════════════════════════════════════════
#  Main entrypoint  
# ══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

def main(s_args, t_args, m_args):
    set_seed(t_args.seed)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(t_args.get_process_log_level())

    last_ckpt = (get_last_checkpoint(t_args.output_dir)
                 if os.path.isdir(t_args.output_dir) else None)
    if last_ckpt and t_args.resume_from_checkpoint is None:
        logger.info("Resuming from checkpoint %s", last_ckpt)

    if "wandb" in t_args.report_to:
        init_wandb_training(t_args)

    # ── Load dataset ───────────────────────────────────────────────────────────
    train_ds = datasets.load_dataset(
        s_args.dataset_name,
        s_args.dataset_config or None,
        split=s_args.dataset_train_split,  # "train" by default
    )
    dataset = {s_args.dataset_train_split: train_ds}
    if s_args.dataset_test_split:
        dataset[s_args.dataset_test_split] = datasets.load_dataset(
            s_args.dataset_name,
            s_args.dataset_config or None,
            split=s_args.dataset_test_split,
        )

    tokenizer    = get_tokenizer(m_args, t_args)
    model        = get_model(m_args, t_args)
    reward_funcs = [_robustify(f) for f in get_reward_funcs(s_args)]

    # ── Convert each example into a conversation format (keep "solution") ───────
    def make_conv(ex,
                  col     = s_args.dataset_prompt_column,
                  sol_col = s_args.dataset_response_column):
        msgs = []
        if t_args.system_prompt:
            msgs.append({"role": "system", "content": t_args.system_prompt})
        msgs.append({"role": "user", "content": ex[col]})
        return {
            "prompt":   msgs,
            "solution": ex[sol_col],
        }

    for split in dataset:
        dataset[split] = dataset[split].map(
            make_conv,
            remove_columns=[c for c in dataset[split].column_names
                            if c not in {s_args.dataset_prompt_column,
                                         s_args.dataset_response_column}],
        )

    # ── Build the StableGRPOTrainer ───────────────────────────────────────────
    trainer = StableGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=t_args,
        train_dataset=dataset[s_args.dataset_train_split],
        eval_dataset=(dataset[s_args.dataset_test_split]
                      if (t_args.eval_strategy != "no"
                          and s_args.dataset_test_split)
                      else None),
        peft_config=get_peft_config(m_args),
        callbacks=get_callbacks(t_args, m_args),
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=t_args.resume_from_checkpoint or last_ckpt)
    trainer.save_model(t_args.output_dir)

    if t_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[s_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if t_args.push_to_hub:
        trainer.push_to_hub(dataset_name=s_args.dataset_name, tags=["open-r1"])

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, StableGRPOConfig, ModelConfig))
    s_args, t_args, m_args = parser.parse_args_and_config()
    main(s_args, t_args, m_args)
