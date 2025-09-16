#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass batch inference for Qwen2.5-style chat LMs that produce:
  <think> ... </think><answer> ... </answer>

Key features
------------
- Uses your system prompt for BOTH passes (first message in the chat).
- Two-phase generation per pass (think → answer) with explicit stops.
- Per-pass hard caps: think <= 750 new tokens, answer <= 50 new tokens.
- Pass 2: shows Pass-1 output as an assistant turn, user supplies a cue,
  and the new <think> starts with the cue (we prefill the cue inside <think>).
- Correctness: gold is counted correct if it appears ANYWHERE inside <answer>
  after canonicalization (substring check).
- Cue-robust analytics: the injected cue at the start of pass-2 <think> does
  NOT trigger “reconsider” detection.
- EOS set includes <|im_end|> and <|endoftext|>.
- NaN-safe token entropy; optional "reconsider" slicing.

CLI flags (selected)
--------------------
--two_pass
--second_pass_phrase "Wait, we need to reconsider. Let's think this through step by step."
--second_pass_use_sample_idx 0
--think_cap 750
--answer_cap 50
--dataset_id MATH-500
--split test
--entropy_mode {full,reconsider,none}
"""

import os
import re
import json
import math
import sys
import logging
import argparse
from typing import Optional, List, Tuple, Dict, Any

import torch
from torch.nn import functional as F
from packaging import version
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ───────────────────────── System prompt ─────────────────────────
SYSTEM_PROMPT = """You are an expert *mathematics problem-solver*.

  Every time you receive a problem you must:
  • Analyse it thoroughly.  
    – Pinpoint the **goal** (what quantity/set/form is requested).  
    – Pinpoint the **givens/constraints** (domains, integrality, non-negativity, geometric conditions).  
    – Choose the **methods** to apply (algebraic manipulation, factorization, inequalities, counting, modular arithmetic, geometry, calculus, etc.).  
    – Write out the full derivation that leads to the final result.

  • Check that the result satisfies all original constraints (no extraneous roots, correct domain, simplified form, exact arithmetic).

  • Respond in **exactly** the tag-based format shown below – no greeting, no commentary outside the tags.  
    – The final answer goes inside `<answer>` **only**.  
    – Use **exact** math (fractions, radicals, π, e). Avoid unnecessary decimals.  
    – Canonical forms: integers as plain numbers; reduced fractions a/b with b>0; simplified radicals; rationalized denominators; sets/tuples with standard notation; intervals in standard notation.  
    – If there is **no solution**, write `NO SOLUTION`. If the problem is **underdetermined**, write `I DON'T KNOW`.

  • You have a hard cap of **750 output tokens**. Be concise but complete.

  ------------------------------------------------------------
  TAG TEMPLATE (copy this shape for every problem)
  <think>
  YOUR reasoning process goes here:  
  1. quote the relevant bits of the problem  
  2. name the mathematical tool(s) you apply  
  3. show each intermediate step until the result is reached  
     
  If you spot an error or an unmet constraint, iterate, repeating steps 1–3 as many
  times as necessary until you are confident in your result. Finish by verifying the
  result satisfies the original conditions exactly (substitution/checks).
  </think>
  <answer>
  THEANSWER
  </answer>
  """

# ───────────────────────── Regex helpers ─────────────────────────
RE_THINK  = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

# Optional "aha"/reconsider cue detectors (for analytics)
_RECONSIDER_PATTERNS = [
    ("wait_line",        re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider",  re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("reconsider_exact", re.compile(r"\bwait[,!\.\s]*let me reconsider\b", re.I)),
    ("step_by_step",     re.compile(r"\blet'?s take (this|it) step[-\s]?by[-\s]?step\b", re.I)),
    ("step_by_step_alt", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck",          re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]

# ───────────────────────── Utilities ─────────────────────────
def _finite_mean(vals: List[float]) -> Optional[float]:
    vs = [float(v) for v in vals if v == v and math.isfinite(float(v))]
    return (sum(vs) / len(vs)) if vs else None

# Back-compat alias so any _fmean calls resolve
def _fmean(vals: List[float]) -> Optional[float]:
    return _finite_mean(vals)

# Canonicalization helpers (permissive)
RE_LATEX_FRAC = re.compile(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", re.I)
RE_LATEX_CMDS = re.compile(r"\\(left|right|,|;|!|:)", re.I)
RE_SPACES = re.compile(r"\s+")
RE_BRACES = re.compile(r"[{}]")
RE_PARENS_COMMAs = re.compile(r"[()\[\],]")

def _canon_math(x: Optional[str]) -> Optional[str]:
    """Permissive: normalize LaTeX, fractions, punctuation, whitespace, Greek pi, dashes."""
    if x is None:
        return None
    s = x.strip()
    s = (s.replace("–","-").replace("—","-").replace("−","-")
           .replace("π", "pi"))
    s = s.replace("\\pi", "pi")
    s = RE_LATEX_CMDS.sub("", s)
    s = RE_LATEX_FRAC.sub(r"\1/\2", s)
    s = RE_BRACES.sub("", s)
    s = RE_SPACES.sub("", s)
    s = RE_PARENS_COMMAs.sub("", s)
    s = s.replace("\\boxed", "").replace("$", "")
    s = s.lower().rstrip(".")
    s = re.sub(r"/{2,}", "/", s)
    s = re.sub(r"\+{2,}", "+", s)
    s = re.sub(r"-{2,}", "-", s)
    if s.startswith("+"): s = s[1:]
    return s

def _contains_canon(hay: Optional[str], needle: Optional[str]) -> bool:
    """True if needle (gold) is a non-empty substring of hay (pred), after canon."""
    if not hay or not needle:
        return False
    return needle in hay

def _extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    think = None
    ans = None
    m = RE_THINK.search(txt)
    if m: think = m.group(1).strip()
    m = RE_ANSWER.search(txt)
    if m: ans = m.group(1).strip()
    return think, ans

def _valid_tag_structure(full_text: str) -> bool:
    """Exactly one <think>…</think> and one <answer>…</answer>, in order."""
    opens_think  = len(re.findall(r"(?i)<think>", full_text))
    closes_think = len(re.findall(r"(?i)</think>", full_text))
    opens_ans    = len(re.findall(r"(?i)<answer>", full_text))
    closes_ans   = len(re.findall(r"(?i)</answer>", full_text))
    if not (opens_think == closes_think == 1 and opens_ans == closes_ans == 1):
        return False
    # Order check: first <think>, then </think>, then <answer>, then </answer>
    try:
        a = re.search(r"(?i)<think>", full_text).start()
        b = re.search(r"(?i)</think>", full_text).start()
        c = re.search(r"(?i)<answer>", full_text).start()
        d = re.search(r"(?i)</answer>", full_text).start()
        return a < b < c < d
    except Exception:
        return False

def _find_markers_and_context(
    think_text: Optional[str],
    problem_text: str,
    skip_prefix_chars: int = 0   # IMPORTANT: ignore the injected cue at the very start
):
    if not think_text:
        return [], None, None, None
    search_text = think_text[skip_prefix_chars:] if skip_prefix_chars > 0 else think_text
    earliest_pos = None
    markers = []
    for name, pat in _RECONSIDER_PATTERNS:
        m = pat.search(search_text)
        if m:
            markers.append(name)
            pos_global = (skip_prefix_chars + m.start()) if skip_prefix_chars > 0 else m.start()
            if earliest_pos is None or pos_global < earliest_pos:
                earliest_pos = pos_global
    if not markers:
        return [], None, None, None
    prefix = think_text[:earliest_pos] if earliest_pos is not None else think_text
    reconsider_context = f"Problem: {problem_text}\n\n{prefix}"
    lo = max(0, (earliest_pos or 0) - 60)
    hi = min(len(think_text), (earliest_pos or 0) + 60)
    reconsider_excerpt = think_text[lo:hi]
    return markers, earliest_pos, reconsider_context, reconsider_excerpt

def _first_eos_any(token_ids: torch.Tensor, eos_id_list: Optional[List[int]]) -> int:
    if not eos_id_list:
        return token_ids.numel()
    hits = []
    for eid in eos_id_list:
        pos = (token_ids == eid).nonzero(as_tuple=False)
        if pos.numel() > 0:
            hits.append(pos[0].item())
    return min(hits) if hits else token_ids.numel()

def _entropy_from_start_index(model, seq_ids: torch.Tensor, start_idx: int) -> List[float]:
    device = next(model.parameters()).device
    seq_ids = seq_ids.to(device)
    ents: List[float] = []
    with torch.inference_mode():
        out = model(input_ids=seq_ids[:, :start_idx+1], use_cache=True)
        past = out.past_key_values
        L = seq_ids.shape[1]
        for t in range(start_idx, L-1):
            out = model(input_ids=seq_ids[:, t:t+1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            h = float(-(p * logp).sum().item())
            if not math.isfinite(h):
                logits = (logits - logits.max()).float()
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                h = float(-(p * logp).sum().item())
            ents.append(h)
    return ents

# ───────────────────────── Stopping on substrings ─────────────────────────
class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, stops: List[str]):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops]
    @staticmethod
    def _endswith(a: torch.Tensor, b: List[int]) -> bool:
        return len(a) >= len(b) and a[-len(b):].tolist() == b
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for row in input_ids:
            for s in self.stop_ids:
                if s and self._endswith(row, s):
                    return True
        return False

# ───────────────────────── Logging ─────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)
logger.info("Starting %s", os.path.basename(__file__))

# ───── PyTorch 2.6 DeepSpeed un-pickle patch (safe no-op if absent) ─────
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        from torch.serialization import add_safe_globals  # type: ignore
        from deepspeed.runtime.zero.config import ZeroStageEnum  # type: ignore
        from deepspeed.runtime.fp16.loss_scaler import LossScaler  # type: ignore
        add_safe_globals([ZeroStageEnum, LossScaler])
        logger.info("DeepSpeed ZeRO patch enabled")
except Exception as e:  # noqa: BLE001
    logger.warning("DeepSpeed patch failed: %r", e)

# ───────────────────────── Prompt builders (WITH system msg) ─────────────────────────
def chat_base_for_pass1(tokenizer, problem: str) -> str:
    # System + user (opens a fresh assistant turn)
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def chat_base_for_pass2(tokenizer, problem: str, prev_output: str, cue: str) -> str:
    # System + user(problem) + assistant(prev output) + user(cue), then open assistant
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
            {"role": "assistant", "content": prev_output},
            {"role": "user", "content": cue},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

# ───────────────────── Inference Loop (two-phase per pass) ─────────────────────
def run_inference_on_split(
    split_name: str,
    examples,  # datasets.Dataset
    tokenizer,
    model,
    step: int,
    outdir: str,
    batch_size: int = 8,
    num_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    entropy_mode: str = "reconsider",
    eos_ids: Optional[List[int]] = None,
    two_pass: bool = False,
    second_pass_phrase: str = "Wait, we need to reconsider. Let's think this through step by step.",
    second_pass_use_sample_idx: int = 0,
    think_cap: int = 750,
    answer_cap: int = 50,
):
    """
    Per example (and per sample):
      Pass-1:
        • THINK:   prefill "<think>\n", stop at "</think>" (cap think_cap)
        • ANSWER:  prefill "<think>…</think>\n<answer>\n", stop at "</answer>" (cap answer_cap)
      Pass-2:
        • THINK:   prefill "<think>\n{cue} ", stop at "</think>" (cap think_cap)
                   (cue is stored at start of <think>, but reconsider detectors ignore it)
        • ANSWER:  prefill "<think>…</think>\n<answer>\n", stop at "</answer>" (cap answer_cap)

    Correctness: substring match — canonicalized gold must appear inside canonicalized <answer>.
    """

    # ---------- helpers local to this run ----------
    def _gen_batch(prefixes: List[str], cap: int, stop_strs: List[str]) -> Tuple[
        List[str], List[List[float]], torch.Tensor, torch.Tensor, List[str]
    ]:
        """
        Returns:
          decs            : list[str] generated text (gen-only), stop substr stripped
          ent_series      : list[list[float]] per-row token entropies
          input_lengths   : tensor prompt lengths
          sequences       : tensor full sequences
          stop_reasons    : list[str] in {"stop_token","eos","max_new_tokens","other"}
        """
        inputs = tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        if torch.cuda.is_available():
            for k in inputs:
                inputs[k] = inputs[k].to("cuda")
            input_lengths = input_lengths.to(inputs["input_ids"].device)

        stop = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strs)]) if stop_strs else None

        gen_kwargs = dict(
            max_new_tokens=cap,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            do_sample=(num_samples > 1),
            temperature=(temperature if num_samples > 1 else 0.0),
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=(entropy_mode != "none"),
            num_return_sequences=1,
        )
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs, stopping_criteria=stop)

        total_rows = out.sequences.shape[0]
        seqs = out.sequences
        decs: List[str] = []
        ent_series: List[List[float]] = []
        stop_reasons: List[str] = []

        for row_i in range(total_rows):
            start_tok_idx = int(input_lengths[row_i].item())
            gen_ids = seqs[row_i, start_tok_idx:]

            # raw decode (gen-only), then detect stop reason before trimming
            raw_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            found_stop = any(s in raw_txt for s in (stop_strs or []))
            has_eos = False
            if eos_ids:
                for eid in eos_ids:
                    if (gen_ids == eid).any():
                        has_eos = True
                        break
            hit_max = len(gen_ids) >= cap

            if found_stop:
                stop_reasons.append("stop_token")
            elif has_eos:
                stop_reasons.append("eos")
            elif hit_max:
                stop_reasons.append("max_new_tokens")
            else:
                stop_reasons.append("other")

            # trim at first stop substring for the returned text
            txt = raw_txt
            for s in (stop_strs or []):
                if s in txt:
                    txt = txt.split(s, 1)[0]
                    break
            decs.append(txt.strip())

            # entropy collection
            if entropy_mode == "none":
                ent_series.append([])
                continue

            scores_T = len(out.scores)
            t_stop = min(_first_eos_any(gen_ids, eos_ids) if eos_ids else gen_ids.shape[0], scores_T)
            tok_ents = []
            bad = False
            for t in range(t_stop):
                logits = out.scores[t][row_i].float()
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    bad = True; break
                logp = F.log_softmax(logits, dim=-1)
                if torch.isnan(logp).any() or torch.isinf(logp).any():
                    bad = True; break
                p = logp.exp()
                h = float(-(p * logp).sum().item())
                if not math.isfinite(h):
                    bad = True; break
                tok_ents.append(h)

            if bad or len(tok_ents) == 0:
                start_idx = start_tok_idx - 1
                tok_ents = _entropy_from_start_index(model, seqs[row_i:row_i+1], start_idx) or []

            ent_series.append(tok_ents)

        return decs, ent_series, input_lengths, seqs, stop_reasons

    def _repeat_for_samples(xs: List[str], S: int) -> List[str]:
        return [x for x in xs for _ in range(S)]

    def _norm_fields(ex: dict):
        problem = (ex.get("problem") or ex.get("question") or ex.get("query") or ex.get("prompt") or ex.get("instruction"))
        gold    = (ex.get("answer")  or ex.get("final_answer") or ex.get("target") or ex.get("boxed_answer") or ex.get("solution"))
        if gold and not any(k in ex for k in ("answer","final_answer","target","boxed_answer")):
            m = re.search(r"\\boxed\{([^}]*)\}", str(gold))
            if not m:
                m = re.search(r"\\boxed\(([^)]*)\)", str(gold))
            if m:
                gold = m.group(1)
        return problem, gold

    def _pack_pass_result(
        problem: str,
        full_text: str,
        ent_think: List[float],
        ent_answer: List[float],
        injected_cue: bool,
        canon_gold: Optional[str],
        prev_output: Optional[str] = None,
        cue_prefix_str: str = "",
        stop_reason_think: Optional[str] = None,
        stop_reason_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build the analytics dict for a single pass.
        Adds:
          - entropy_think / entropy_answer
          - stop_reason_think / stop_reason_answer
          - valid_tag_structure
        """
        tok_ents_all = (ent_think or []) + (ent_answer or [])
        Tthink = len(ent_think or [])
        Tans   = len(ent_answer or [])
        T      = len(tok_ents_all)

        think, answer = _extract_blocks(full_text)
        think_text = think or ""
        pred_answer_text = answer or ""

        # Ignore the injected cue at the very start of pass-2 <think>
        skip_chars = len(cue_prefix_str) if injected_cue else 0
        markers, pos_in_think, reconsider_context, reconsider_excerpt = _find_markers_and_context(
            think_text, problem, skip_prefix_chars=skip_chars
        )
        if injected_cue:
            markers = ["injected_cue"] + (markers or [])

        # For entropy slicing: the cue is prefixed in the prompt, so generated tokens start at 0
        t_cue = 0 if injected_cue else None
        if (not injected_cue) and (pos_in_think is not None):
            t_cue = max(0, min(pos_in_think, Tthink))  # token index estimate via char pos (cheap)

        entropy_overall = _finite_mean(tok_ents_all) if tok_ents_all else None
        entropy_think   = _finite_mean(ent_think)     if ent_think else None
        entropy_answer  = _finite_mean(ent_answer)    if ent_answer else None

        entropy_pre_cue = None
        entropy_reconsider_think = None
        entropy_reconsider_full = None
        if t_cue is not None:
            if T > t_cue:
                entropy_reconsider_full = _finite_mean(tok_ents_all[t_cue:])
            if Tthink > t_cue:
                entropy_reconsider_think = _finite_mean(tok_ents_all[t_cue:Tthink])

        pred_canon = _canon_math(pred_answer_text)
        is_correct_pred = _contains_canon(pred_canon, canon_gold)

        return dict(
            prev_output=prev_output,
            output=full_text,
            pred_answer=pred_answer_text,
            pred_answer_canon=pred_canon,

            # entropies
            entropy=entropy_overall,
            entropy_think=entropy_think,
            entropy_answer=entropy_answer,
            entropy_pre_cue=entropy_pre_cue,
            entropy_reconsider_think=entropy_reconsider_think,
            entropy_reconsider_full=entropy_reconsider_full,

            # stop reasons
            stop_reason_think=stop_reason_think,
            stop_reason_answer=stop_reason_answer,

            # reconsider markers
            has_reconsider_cue=bool(markers),
            reconsider_markers=markers or [],
            reconsider_pos=pos_in_think,
            reconsider_context=reconsider_context,
            reconsider_excerpt=reconsider_excerpt,

            # correctness
            is_correct_pred=is_correct_pred,
            is_correct_after_reconsideration=bool(markers) and bool(is_correct_pred),

            # tokens
            tokens_total=T,
            tokens_end_think=Tthink,
            tokens_think=Tthink,
            tokens_answer=Tans,

            # tag sanity
            valid_tag_structure=_valid_tag_structure(full_text),
        )

    # ---------- resume bookkeeping ----------
    outpath = os.path.join(outdir, f"step{step:04d}_{split_name}.jsonl")
    seen: set[str] = set()
    if os.path.exists(outpath):
        with open(outpath, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["problem"])
                except Exception:
                    pass

    logger.info("→ %s | %d examples (skipping %d already done)", split_name, len(examples), len(seen))

    # ---------- main loop ----------
    BATCH = batch_size
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    for i in range(0, len(examples), BATCH):
        idx_lo, idx_hi = i, min(i + BATCH, len(examples))
        batch_ds = examples.select(range(idx_lo, idx_hi))

        batch = []
        for ex in batch_ds:
            prob, gold = _norm_fields(ex)
            if not prob or prob in seen:
                continue
            ex = dict(ex)
            ex["_normalized_problem"] = prob
            ex["_normalized_gold"] = gold
            batch.append(ex)
        if not batch:
            continue

        B, S = len(batch), num_samples

        # ===== PASS 1 =====
        base1 = [chat_base_for_pass1(tokenizer, ex["_normalized_problem"]) for ex in batch]
        pre1_think = _repeat_for_samples([b + "<think>\n" for b in base1], S)

        think1_texts, think1_ents, _, _, think1_stop = _gen_batch(pre1_think, think_cap, ["</think>"])

        pre1_answer = []
        for row_i in range(B * S):
            pre = pre1_think[row_i] + think1_texts[row_i] + "</think>\n<answer>\n"
            pre1_answer.append(pre)
        answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(pre1_answer, answer_cap, ["</answer>"])

        pass1_full = [
            f"<think>{think1_texts[row_i]}</think>\n<answer>{answer1_texts[row_i]}</answer>"
            for row_i in range(B * S)
        ]

        # Which sample feeds pass-2
        firstpass_choice = []
        for bi in range(B):
            k_choice = max(0, min(second_pass_use_sample_idx, S - 1))
            row_i = bi * S + k_choice
            firstpass_choice.append(pass1_full[row_i])

        # ===== PASS 2 (optional) =====
        pass2_full = [""] * (B * S)
        think2_ents: List[List[float]] = [[] for _ in range(B * S)]
        answer2_ents: List[List[float]] = [[] for _ in range(B * S)]
        think2_stop: List[str] = [""] * (B * S)
        answer2_stop: List[str] = [""] * (B * S)

        cue_str = second_pass_phrase.strip() + " "

        if two_pass:
            base2 = [
                chat_base_for_pass2(
                    tokenizer,
                    ex["_normalized_problem"],
                    firstpass_choice[bi],
                    second_pass_phrase.strip(),
                )
                for bi, ex in enumerate(batch)
            ]
            pre2_think = _repeat_for_samples([b + "<think>\n" + cue_str for b in base2], S)
            think2_texts_only_new, think2_ents, _, _, think2_stop = _gen_batch(pre2_think, think_cap, ["</think>"])
            think2_texts = [cue_str + t for t in think2_texts_only_new]

            pre2_answer = []
            for row_i in range(B * S):
                pre = pre2_think[row_i] + think2_texts_only_new[row_i] + "</think>\n<answer>\n"
                pre2_answer.append(pre)
            answer2_texts, answer2_ents, _, _, answer2_stop = _gen_batch(pre2_answer, answer_cap, ["</answer>"])

            pass2_full = [
                f"<think>{think2_texts[row_i]}</think>\n<answer>{answer2_texts[row_i]}</answer>"
                for row_i in range(B * S)
            ]

        # ===== WRITE JSON =====
        for bi, ex in enumerate(batch):
            canon_gold = _canon_math(ex["_normalized_gold"])
            for k in range(S):
                row_i = bi * S + k

                p1 = _pack_pass_result(
                    problem=ex["_normalized_problem"],
                    full_text=pass1_full[row_i],
                    ent_think=think1_ents[row_i],
                    ent_answer=answer1_ents[row_i],
                    canon_gold=canon_gold,
                    injected_cue=False,
                    prev_output=None,
                    cue_prefix_str="",
                    stop_reason_think=think1_stop[row_i],
                    stop_reason_answer=answer1_stop[row_i],
                )

                p2 = None
                if two_pass:
                    p2 = _pack_pass_result(
                        problem=ex["_normalized_problem"],
                        full_text=pass2_full[row_i],
                        ent_think=think2_ents[row_i],
                        ent_answer=answer2_ents[row_i],
                        canon_gold=canon_gold,
                        injected_cue=True,
                        prev_output=firstpass_choice[bi],
                        cue_prefix_str=cue_str,
                        stop_reason_think=think2_stop[row_i],
                        stop_reason_answer=answer2_stop[row_i],
                    )
                    # improvement flag relative to pass-1 for the same sample
                    p2["improved_over_pass1"] = bool(p2.get("is_correct_pred")) and not bool(p1.get("is_correct_pred"))

                row = {
                    "problem": ex["_normalized_problem"],
                    "gold_answer": ex["_normalized_gold"],
                    "gold_answer_canon": canon_gold,
                    "step": step,
                    "split": split_name,
                    "sample_idx": k,
                    "pass1": p1,
                    "pass2": p2,
                }
                with open(outpath, "a", encoding="utf-8") as f:
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")

            seen.add(ex["_normalized_problem"])

# ─────────────────────────── Dataset helpers ───────────────────────────
def _load_local_json_dataset(path: str):
    """Minimal local JSON/JSONL loader to avoid NameErrors if dataset_path is used."""
    from datasets import Dataset
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line) if line.startswith("{") else None
            if obj is None:
                continue
            records.append(obj)
    return Dataset.from_list(records)

def load_math500(cache_dir: str, split: str, seed: int, dataset_path: Optional[str] = None):
    from datasets import load_dataset

    if dataset_path:
        logger.info("Loading MATH-500 from local file: %s", dataset_path)
        return _load_local_json_dataset(dataset_path)

    # Preferred candidates first
    candidates = [
        "HuggingFaceH4/MATH-500",
        "AI-MO/MATH-500",
        "lighteval/MATH-500",
        "openai/math-500",
        "TIGER-Lab/MATH-500",
    ]

    for repo in candidates:
        try:
            logger.info("Trying remote MATH-500 candidate: %s", repo)
            ds_full = load_dataset(repo, split="test", cache_dir=cache_dir)
            colnames = set(ds_full.column_names)

            def _norm(ex):
                problem = (ex.get("problem") or ex.get("question") or
                           ex.get("prompt") or ex.get("instruction") or ex.get("query"))
                ans = (ex.get("answer") or ex.get("solution") or
                       ex.get("final_answer") or ex.get("boxed_answer") or ex.get("target"))
                return {"problem": problem, "answer": ans}

            ds = ds_full.map(_norm, remove_columns=list(colnames))
            ds = ds.filter(lambda ex: ex["problem"] is not None and ex["answer"] is not None)
            if len(ds) == 0:
                raise ValueError(f"{repo} contained no usable (problem,answer) pairs")
            logger.info("Loaded MATH-500 from %s | N=%d", repo, len(ds))
            return ds
        except Exception as e:
            logger.warning("Skipping %s (%r)", repo, e)

    try:
        ds_full = load_dataset("hendrycks/competition_math", split="test", cache_dir=cache_dir)
        n = min(500, len(ds_full))
        return ds_full.shuffle(seed=seed).select(range(n))
    except Exception as e:
        raise RuntimeError(f"Could not load MATH-500 or fallback dataset: {e}")

# ─────────────────────────── Main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--output_dir", required=True)

    # Data selection
    ap.add_argument("--dataset_id", default="MATH-500", help="Use 'MATH-500' (default) or a HF dataset path.")
    ap.add_argument("--split", default="test", help="Split name to run on (MATH-500 typically uses 'test').")
    ap.add_argument("--num_examples", type=int, default=None, help="Optional cap if you want fewer than 500.")

    # Decoding + sampling
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Budgets (per pass)
    ap.add_argument("--think_cap", type=int, default=750)
    ap.add_argument("--answer_cap", type=int, default=50)

    # System/runtime
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--seed", type=int, default=42)

    # Entropy + attention impl
    ap.add_argument("--entropy_mode", choices=["full","reconsider","none"], default="reconsider",
                    help="'full'/'reconsider' compute tokenwise entropy slices; 'none' disables it.")
    ap.add_argument("--attn_implementation", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])

    # Two-pass controls
    ap.add_argument("--two_pass", action="store_true")
    ap.add_argument("--second_pass_phrase", default="Wait, we need to reconsider. Let's think this through step by step.")
    ap.add_argument("--second_pass_use_sample_idx", type=int, default=0)

    args = ap.parse_args()

    # Tokenizer
    HF_CACHE_DIR = os.path.abspath("./.hf_cache")
    tok_src = args.tokenizer_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # EOS set (Qwen often uses <|im_end|>)
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.pad_token_id:
            eos_ids.add(int(tid))
    eos_ids = sorted(eos_ids) if eos_ids else None

    # Model
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    ).eval()

    # Dataset
    from datasets import load_dataset
    if args.dataset_id.upper() == "MATH-500":
        ds = load_math500(HF_CACHE_DIR, args.split, args.seed)
        dataset_name_for_log = "MATH-500"
    else:
        ds = load_dataset(args.dataset_id, split=args.split, cache_dir=HF_CACHE_DIR)
        dataset_name_for_log = args.dataset_id

    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Model: %s @ %s | dtype=%s", args.model_name_or_path, args.revision, dtype)
    logger.info("Dataset: %s split=%s | N=%d", dataset_name_for_log, args.split, len(ds))
    logger.info("Output dir: %s", args.output_dir)

    run_inference_on_split(
        split_name=args.split,
        examples=ds,
        tokenizer=tokenizer,
        model=model,
        step=args.step,
        outdir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        entropy_mode=args.entropy_mode,
        eos_ids=eos_ids,
        two_pass=args.two_pass,
        second_pass_phrase=args.second_pass_phrase,
        second_pass_use_sample_idx=args.second_pass_use_sample_idx,
        think_cap=args.think_cap,
        answer_cap=args.answer_cap,
    )

    logger.info("All inference complete.")

if __name__ == "__main__":
    main()
