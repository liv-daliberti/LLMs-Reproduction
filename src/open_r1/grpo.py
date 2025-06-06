#!/usr/bin/env python
"""
Stable-GRPO with external-vLLM generation (RemoteVLLMGRPOTrainer)

────────────────────────────────────────────────────────────────────
 • reward / advantage / gradient clipping
 • adaptive-KL coefficient (keep KL near target_kl)
 • prompt-length truncation
 • robust reward wrapper
 • DeepSpeed-ZeRO pickle patch (PT ≥ 2.6)
 • dtype fix for ids/masks
 • External vLLM server via safe_generate (max_tokens = 750)
────────────────────────────────────────────────────────────────────
"""
import logging, os, sys, functools, inspect, time
from dataclasses import dataclass
from packaging import version
import torch
import datasets
from transformers import set_seed, PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
import trl.trainer.grpo_trainer as _grpo_mod

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
import open_r1.rewards as _rewards
from open_r1.rewards import get_reward_funcs
from open_r1.utils.vllm_patch import safe_generate  # ← uses external vLLM

# ─────────────────── DeepSpeed ZeRO pickle patch ─────────────────────────────
if version.parse(torch.__version__) >= version.parse("2.6.0"):
    from torch.serialization import add_safe_globals
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    add_safe_globals([ZeroStageEnum, LossScaler])

# ─────────────────── Cast IDs / masks to .long() ─────────────────────────────
_orig_prepare = _grpo_mod.GRPOTrainer._prepare_inputs
def _patched_prepare(self, inputs):
    inputs = _orig_prepare(self, inputs)
    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].long()
        inputs["attention_mask"] = inputs["attention_mask"].long()
    # DO NOT touch "solution" here – leave it in the batch!
    return inputs
_grpo_mod.GRPOTrainer._prepare_inputs = _patched_prepare

# ─────────────────── Robust-reward wrapper  ──────────────────────────────────
# ─────────────────── Robust-reward wrapper (final fix) ───────────────────────
def _robustify(fn):
    if getattr(fn, "_robust_wrapped_", False):
        return fn
    sig = inspect.signature(fn)
    wants_p  = "prompts"     in sig.parameters
    wants_c  = "completions" in sig.parameters
    wants_sol_plural   = "solutions" in sig.parameters
    wants_sol_singular = "solution"  in sig.parameters   # ← NEW

    @functools.wraps(fn)
    def wrapped(prompts, completions, solutions=None, **kw):
        # normalise completions → list[list[dict]]
        fixed = []
        for c in completions:
            if isinstance(c, list) and c and isinstance(c[0], dict):
                fixed.append(c)
            elif isinstance(c, list) and c and isinstance(c[0], str):
                fixed.append([{"role": "assistant", "content": c[0]}])
            else:
                fixed.append([{"role": "assistant", "content": str(c)}])

        call_kw = {}
        if wants_p:  call_kw["prompts"]     = prompts
        if wants_c:  call_kw["completions"] = fixed
        if solutions is not None:
            if wants_sol_plural:
                call_kw["solutions"] = solutions
            elif wants_sol_singular:
                call_kw["solution"]  = solutions     # ← pass with old name
        # forward only recognised kwargs
        call_kw.update({k: v for k, v in kw.items() if k in sig.parameters})
        return fn(**call_kw)

    wrapped._robust_wrapped_ = True
    return wrapped

# ─────────────────── Config + trainer base  ──────────────────────────────────
@dataclass
class StableGRPOConfig(GRPOConfig):
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
    adv_clip_range: float = 5.0
    max_grad_norm: float = 1.0
    init_kl_coef: float = 0.4
    adapt_kl_coef: bool = True
    target_kl: float = 0.05
    kl_horizon: int = 64

from torch.nn.utils import clip_grad_norm_

class StableGRPOTrainer(GRPOTrainer):
    """Everything from the earlier version except generation step."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.kl_coef, self._kl_step_counter = float(self.args.init_kl_coef), 0
    def _compute_rewards(self, m_out, qs, rs):
        return torch.clamp(super()._compute_rewards(m_out, qs, rs),
                           self.args.reward_clip_min, self.args.reward_clip_max)
    def compute_advantages(self, rewards, values, **kw):
        adv = super().compute_advantages(rewards, values, **kw)
        return torch.clamp(adv, -self.args.adv_clip_range, self.args.adv_clip_range)
    def training_step(self, *a, **kw):
        loss = super().training_step(*a, **kw)
        if self.args.max_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        if self.args.adapt_kl_coef:
            self._kl_step_counter += 1
            if self._kl_step_counter >= self.args.kl_horizon:
                for rec in reversed(self.state.log_history):
                    if "kl" in rec:  # get last KL
                        actual_kl = rec["kl"]; break
                else: actual_kl = None
                if actual_kl is not None:
                    tgt = self.args.target_kl
                    self.kl_coef *= 1.2 if actual_kl > tgt*1.5 else (0.8 if actual_kl < tgt*0.5 else 1)
                    self.kl_coef = max(1e-4, min(self.kl_coef, 10.0))
                    self.log_metrics("train", {"adapted_kl_coef": self.kl_coef})
                self._kl_step_counter = 0
        return loss
    def compute_loss(self, model, inputs, *, return_outputs=False, num_items_in_batch=None, **kw):
        loss = super().compute_loss(model, inputs, return_outputs=False,
                                    num_items_in_batch=num_items_in_batch, **kw)
        stats = getattr(self, "_last_stats", None) or getattr(self, "stats", None)
        if stats and "kl" in stats:
            kl_val = stats["kl"]
            loss = loss + self.kl_coef * kl_val
            self.log_metrics("train",
                {"raw_kl": kl_val, "final_kl_coef": self.kl_coef, "kl_penalty": self.kl_coef*kl_val})
        return (loss, None) if return_outputs else loss

# ─────────────────── External-vLLM trainer  ──────────────────────────────────
VLLM_ENDPOINT   = "http://localhost:8000/generate"
MAX_NEW_TOKENS  = 750
MAX_CTX, RESERVED = 2548, 2048   # keep where it was
class RemoteVLLMGRPOTrainer(StableGRPOTrainer):
    """
    1. Uses an external vLLM server for generation.
    2. Feeds ground-truth `solution` into the reward functions
       without touching TRL’s internal batching utilities.
    """

    # --------------------------------------------------------------------- #
    # 1️⃣  Override *generation only*                                       #
    # --------------------------------------------------------------------- #
    def _generate_and_score_completions(self, inputs):
        tok: PreTrainedTokenizerBase = self.tokenizer
        max_prompt = MAX_CTX - RESERVED

        prompts_text, original_msgs, solutions = [], [], []
        for sample in inputs:
            msgs        = sample["prompt"]
            original_msgs.append(msgs)
            solutions.append(sample["solution"])          # keep gold answers

            txt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            ids = tok(txt, return_tensors="pt").input_ids[0]
            if ids.size(0) > max_prompt:
                txt = tok.decode(ids[-max_prompt:], skip_special_tokens=True)
            prompts_text.append(txt)

        # external vLLM call
        completions = safe_generate(
            prompts     = prompts_text,
            url         = VLLM_ENDPOINT,
            max_tokens  = MAX_NEW_TOKENS,
            temperature = 0.7,
            top_p       = 0.9,
            tokenizer   = tok,
        )

        # attach completion back to each sample (DO NOT drop solution!)
        for sample, msgs, comp in zip(inputs, original_msgs, completions):
            sample["prompt"]     = msgs
            sample["completion"] = [{"role": "assistant", "content": comp[0]}]

        # stash solutions for the next hook
        self._current_batch_solutions = solutions

        # hand the batch back to TRL – it will tokenize, calculate
        # log-probs, etc., and eventually call our _compute_batch_rewards.
        return super()._generate_and_score_completions(inputs)

    # --------------------------------------------------------------------- #
    # 2️⃣  Override only the reward computation                              #
    # --------------------------------------------------------------------- #
    def _compute_batch_rewards(self, prompts, completions):
        """Called by the base class right after generation."""
        solutions = getattr(self, "_current_batch_solutions", None)

        rewards = []
        for fn in self.reward_funcs:
            # our robust wrapper accepts either `solution` or `solutions`
            r = fn(prompts=prompts,
                   completions=completions,
                   solutions=solutions)
            rewards.append(
                torch.as_tensor(r, dtype=torch.float32,
                                device=self.accelerator.device)
            )
        # avg over multiple reward fns (if you have more than one)
        return torch.stack(rewards).mean(dim=0)
        

# ─────────────────── Main entry-point  ───────────────────────────────────────
logger = logging.getLogger(__name__)
def main(s_args, t_args, m_args):
    set_seed(t_args.seed)
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    logger.setLevel(t_args.get_process_log_level())

    last_ckpt = get_last_checkpoint(t_args.output_dir) if os.path.isdir(t_args.output_dir) else None
    if last_ckpt and t_args.resume_from_checkpoint is None:
        logger.info("Resuming from checkpoint %s", last_ckpt)
    if "wandb" in t_args.report_to: init_wandb_training(t_args)

    train_ds = datasets.load_dataset(s_args.dataset_name, s_args.dataset_config or None,
                                     split=s_args.dataset_train_split)
    dataset = {s_args.dataset_train_split: train_ds}
    if s_args.dataset_test_split:
        dataset[s_args.dataset_test_split] = datasets.load_dataset(
            s_args.dataset_name, s_args.dataset_config or None, split=s_args.dataset_test_split)

    tokenizer, model = get_tokenizer(m_args, t_args), get_model(m_args, t_args)
    reward_funcs = [_robustify(f) for f in get_reward_funcs(s_args)]

    def make_conv(ex, col=s_args.dataset_prompt_column, sol_col=s_args.dataset_response_column):
        msgs=[{"role":"system","content":t_args.system_prompt}] if t_args.system_prompt else []
        msgs.append({"role":"user","content":ex[col]}); return {"prompt":msgs,"solution":ex[sol_col]}
    for split in dataset:
        dataset[split] = dataset[split].map(
            make_conv,
            remove_columns=[c for c in dataset[split].column_names
                            if c not in {s_args.dataset_prompt_column, s_args.dataset_response_column}],
        )

    trainer = RemoteVLLMGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=t_args,
        train_dataset=dataset[s_args.dataset_train_split],
        eval_dataset=(dataset[s_args.dataset_test_split]
                      if (t_args.eval_strategy != "no" and s_args.dataset_test_split) else None),
        peft_config=get_peft_config(m_args),
        callbacks=get_callbacks(t_args, m_args),
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=t_args.resume_from_checkpoint or last_ckpt)
    trainer.save_model(t_args.output_dir)

    if t_args.do_eval:
        metrics = trainer.evaluate(); metrics["eval_samples"] = len(dataset[s_args.dataset_test_split])
        trainer.log_metrics("eval", metrics); trainer.save_metrics("eval", metrics)

    if t_args.push_to_hub:
        trainer.push_to_hub(dataset_name=s_args.dataset_name, tags=["open-r1"])

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, StableGRPOConfig, ModelConfig))
    s_args, t_args, m_args = parser.parse_args_and_config()
    main(s_args, t_args, m_args)
