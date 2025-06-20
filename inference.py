#!/usr/bin/env python
import sys
print(f"→ START {__file__}", file=sys.stderr, flush=True)

import os
print("→ [DEBUG] imported os", file=sys.stderr, flush=True)
import json
print("→ [DEBUG] imported json", file=sys.stderr, flush=True)
import torch
print("→ [DEBUG] imported torch", file=sys.stderr, flush=True)
import argparse
print("→ [DEBUG] imported argparse", file=sys.stderr, flush=True)
import gc
print("→ [DEBUG] imported gc", file=sys.stderr, flush=True)
import torch._dynamo
print("→ [DEBUG] imported torch._dynamo", file=sys.stderr, flush=True)
from packaging import version
print("→ [DEBUG] imported packaging.version", file=sys.stderr, flush=True)
import glob
print("→ [DEBUG] imported glob", file=sys.stderr, flush=True)

# —— Patch for PyTorch 2.6+ DeepSpeed ZeRO unpickle support ——
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        print("→ [DEBUG] torch>=2.6 detected, attempting DeepSpeed imports…", file=sys.stderr, flush=True)
        from torch.serialization import add_safe_globals
        from deepspeed.runtime.zero.config import ZeroStageEnum
        from deepspeed.runtime.fp16.loss_scaler import LossScaler

        trusted_classes = [ZeroStageEnum, LossScaler]
        add_safe_globals([cls for cls in trusted_classes if isinstance(cls, type)])
        print("✅ DeepSpeed ZeRO unpickle support enabled", file=sys.stderr, flush=True)
    else:
        print("→ [DEBUG] torch<2.6 — skipping DeepSpeed patch", file=sys.stderr, flush=True)
except Exception as e:
    print(f"⚠️ DeepSpeed patch block failed at import: {e!r}", file=sys.stderr, flush=True)

# ——— cache setup ———
HF_CACHE_DIR = os.path.abspath("./.hf_cache")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
print("→ [DEBUG] cache setup complete", file=sys.stderr, flush=True)


# ——— prompt template ———
PROMPT_TEMPLATE = (
    "You are a helpful AI Assistant that provides well‐reasoned but short responses.\n"
    "You first briefly think about the reasoning process as an internal monologue and then provide the user with the answer.\n"
    "Respond in the following format:\n\n"
    "Problem: {problem}\n\n"
    "<think>\n"
    # assistant will fill in reasoning here
    "\n</think>\n\n"
    "<answer>\n"
    # assistant will fill in answer here
    #"\n</answer>"
)

STOP_STR = "</answer>"
# combine think+answer budgets
MAX_NEW_TOKENS =  1524


def discover_checkpoints(local_dir: str):
    """
    Look in `local_dir` for subfolders named `checkpoint-<step>`,
    extract their step numbers, and return a sorted list of (step, path).
    """
    pattern = os.path.join(local_dir, "checkpoint-*")
    all_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    checkpoints = []
    for d in all_dirs:
        basename = os.path.basename(d)
        # Expecting names like "checkpoint-50", "checkpoint-100", etc.
        try:
            step = int(basename.split("-")[-1])
            checkpoints.append((step, d))
        except ValueError:
            continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints  # list of (step, full_path)


def load_ckpt_from_local(path: str):
    """
    Load tokenizer + model from a local Hugging Face checkpoint directory.
    Compile once, then reset Dynamo cache for subsequent loads.
    """
    print(f"→ [DEBUG] load_ckpt_from_local: start loading from {path}", file=sys.stderr, flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    print(f"→ [DEBUG] tokenizer loaded from {path}", file=sys.stderr, flush=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    print(f"→ [DEBUG] model.from_pretrained done for {path}", file=sys.stderr, flush=True)

    if not getattr(load_ckpt_from_local, "compiled_once", False):
        print(f"→ [DEBUG] compiling model from {path}", file=sys.stderr, flush=True)
        mdl = torch.compile(mdl)
        load_ckpt_from_local.compiled_once = True
        print(f"→ [DEBUG] compile complete for {path}", file=sys.stderr, flush=True)
    else:
        torch._dynamo.reset()
        print(f"→ [DEBUG] Dynamo cache reset for reuse {path}", file=sys.stderr, flush=True)

    if hasattr(mdl.config, "attn_implementation"):
        mdl.config.attn_implementation = "flash_attention_2"
        print("✅ FlashAttention 2 enabled via model config.", file=sys.stderr, flush=True)
    else:
        print("⚠️ Model config does not support FlashAttention 2.", file=sys.stderr, flush=True)

    return tok, mdl


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")



def run_inference_on_split(split_name, examples, tokenizer, model, step, output_dir, batch_size=16):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnTag(StoppingCriteria):
        def __init__(self, tokenizer, stop_str: str):
            self.stop_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(stop_str, add_special_tokens=False)
            )
        def __call__(self, input_ids, scores, **kwargs):
            seq = input_ids[0].tolist()
            return (
                len(seq) >= len(self.stop_ids)
                and seq[-len(self.stop_ids):] == self.stop_ids
            )

    stop_criteria = StoppingCriteriaList([StopOnTag(tokenizer, STOP_STR)])

    filename = f"step{step:04d}_{split_name}.jsonl"
    outpath = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # skip already‐done problems
    completed = set()
    if os.path.exists(outpath):
        with open(outpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["problem"])
                except:
                    pass

    total = len(examples)
    for i in range(0, total, batch_size):
        batch = examples.select(range(i, min(i + batch_size, total)))
        batch = [ex for ex in batch if ex["problem"] not in completed]
        if not batch:
            continue

        prompts = [PROMPT_TEMPLATE.format(problem=ex["problem"]) for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        print(f"→ [DEBUG] generating combined output for batch {i//batch_size+1}", file=sys.stderr, flush=True)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                stopping_criteria=stop_criteria,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        for j, ex in enumerate(batch):
            full = outputs[j]
            # split at the tags
            try:
                think_part, rest = full.split("</think>", 1)
                think_text = think_part.replace("<think>", "").strip()
                answer_part = rest.split("</answer>",1)[0]
                answer_text = answer_part.replace("<answer>", "").strip()
            except ValueError:
                # fallback if tags missing
                parts = full.split("\n\n")
                think_text = parts[0].strip()
                answer_text = parts[-1].strip()

            row = {
                "problem": ex["problem"],
                "gold_answer": ex["answer"],
                "step": step,
                "split": split_name,
                "output": think_text + "\n\n" + answer_text
            }
            append_jsonl(outpath, row)
            print(f"[{split_name} step {step} example {i+j+1}/{total}] saved", file=sys.stderr, flush=True)
            completed.add(ex["problem"])

    print(f"→ [DEBUG] done with split {split_name} @ step={step}", file=sys.stderr, flush=True)



def main():
    print("→ [DEBUG] entered main()", file=sys.stderr, flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_ckpt_dir",
        type=str,
        required=True,
        help="Path to the parent folder containing checkpoint-*/ subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to write JSONL outputs (one file per checkpoint)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=500,
        help="How many examples to take from the train split"
    )
    args = parser.parse_args()

    # Discover all checkpoint folders under local_ckpt_dir
    checkpoints = discover_checkpoints(args.local_ckpt_dir)
    if not checkpoints:
        print(f"[ERROR] No checkpoint-*/ directories found under {args.local_ckpt_dir}", file=sys.stderr, flush=True)
        return

    print(f"→ [DEBUG] found {len(checkpoints)} checkpoints:", file=sys.stderr, flush=True)
    for step, path in checkpoints:
        print(f"    - step {step}: {path}", file=sys.stderr, flush=True)

    print("→ [DEBUG] about to import datasets…", file=sys.stderr, flush=True)
    from datasets import load_dataset
    print("→ [DEBUG] import datasets done", file=sys.stderr, flush=True)

    print("→ [DEBUG] about to load_dataset…", file=sys.stderr, flush=True)
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", cache_dir=HF_CACHE_DIR)
    print("→ [DEBUG] load_dataset complete", file=sys.stderr, flush=True)

    # Shuffle and select exactly num_examples from the train split
    train_examples = ds["train"].shuffle(seed=42).select(range(args.num_examples))

    for step, ckpt_path in checkpoints:
        print(f"\n=== Running checkpoint step {step} ===", file=sys.stderr, flush=True)
        try:
            tokenizer, model = load_ckpt_from_local(ckpt_path)
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint at {ckpt_path}: {e}", file=sys.stderr, flush=True)
            continue

        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")

        print(f"→ [DEBUG] warm-up generate for step {step}", file=sys.stderr, flush=True)
        # One‐token warmup
        warmup_inputs = tokenizer("warmup", return_tensors="pt")
        if torch.cuda.is_available():
            warmup_inputs = {k: v.to("cuda") for k, v in warmup_inputs.items()}
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=1
        )
        print(f"[step {step}] warm-up done", file=sys.stderr, flush=True)

        run_inference_on_split(
            "train",
            train_examples,
            tokenizer,
            model,
            step,
            args.output_dir,
            batch_size=args.batch_size
        )


        # Clean up before next checkpoint
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"→ [DEBUG] after cleanup for step {step}", file=sys.stderr, flush=True)

    print("\nAll checkpoints completed — JSONL files saved.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
