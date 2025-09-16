#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass clue→answer inference for chat LMs that produce:
  <think> ... </think><answer> ... </answer>

Dataset
-------
od2961/mini-crosswords (records are clue → answer; no grid flattening required)

Mechanics
---------
- System prompt in both passes.
- Pass = THINK (stop at </think>) then ANSWER (stop at </answer>).
- Caps: think<=750, answer<=50 (configurable).
- Pass-2 seeds with pass-1 transcript + reconsider cue; cue is ignored for reconsider detection.
- Correctness: canon(gold) must be a substring of canon(<answer>).
- Resume by row_key = "{split}:{dataset_index}:{sample_idx}".
"""

import os, re, json, math, sys, logging, argparse
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
SYSTEM_PROMPT = """You are an expert *crossword solver* (American & cryptic).
When you receive a CLUE (optionally with enumeration like “(6)”):

• Analyse briefly (definition/wordplay if cryptic).
• If an enumeration exists, ensure the letter count matches.
• Respond ONLY with:
<think> ... reasoning ... </think>
<answer> FINALENTRYINUPPERCASE </answer>
Do not put the answer anywhere except inside <answer>.
"""

# ───────────────────────── Regex helpers ─────────────────────────
RE_THINK  = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")
_RECONSIDER_PATTERNS = [
    ("wait_line",        re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider",  re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("reconsider_exact", re.compile(r"\bwait[,!\.\s]*let me reconsider\b", re.I)),
    ("step_by_step",     re.compile(r"\blet'?s take (this|it) step[-\s]?by[-\s]?step\b", re.I)),
    ("step_by_step_alt", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck",          re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]

def _finite_mean(vals: List[float]) -> Optional[float]:
    vs = [float(v) for v in vals if v == v and math.isfinite(float(v))]
    return (sum(vs) / len(vs)) if vs else None

# Canonicalization: keep only a–z0–9, lowercase & join.
RE_KEEP_ALNUM = re.compile(r"[a-z0-9]+", re.I)
def _canon_answer(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    parts = RE_KEEP_ALNUM.findall(x.strip().lower())
    return "".join(parts)

def _contains_canon(hay: Optional[str], needle: Optional[str]) -> bool:
    return bool(hay and needle and (needle in hay))

def _extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    t = RE_THINK.search(txt); a = RE_ANSWER.search(txt)
    return (t.group(1).strip() if t else None, a.group(1).strip() if a else None)

def _valid_tag_structure(full_text: str) -> bool:
    ot = len(re.findall(r"(?i)<think>", full_text))
    ct = len(re.findall(r"(?i)</think>", full_text))
    oa = len(re.findall(r"(?i)<answer>", full_text))
    ca = len(re.findall(r"(?i)</answer>", full_text))
    if not (ot==ct==1 and oa==ca==1): return False
    try:
        a = re.search(r"(?i)<think>", full_text).start()
        b = re.search(r"(?i)</think>", full_text).start()
        c = re.search(r"(?i)<answer>", full_text).start()
        d = re.search(r"(?i)</answer>", full_text).start()
        return a < b < c < d
    except Exception:
        return False

def _find_markers_and_context(think_text: Optional[str], clue_text: str, skip_prefix_chars: int=0):
    if not think_text: return [], None, None, None
    search_text = think_text[skip_prefix_chars:] if skip_prefix_chars>0 else think_text
    earliest_pos, markers = None, []
    for name, pat in _RECONSIDER_PATTERNS:
        m = pat.search(search_text)
        if m:
            markers.append(name)
            pos_global = (skip_prefix_chars + m.start()) if skip_prefix_chars>0 else m.start()
            if earliest_pos is None or pos_global < earliest_pos: earliest_pos = pos_global
    if not markers: return [], None, None, None
    prefix = think_text[:earliest_pos] if earliest_pos is not None else think_text
    ctx = f"Clue: {clue_text}\n\n{prefix}"
    lo = max(0, (earliest_pos or 0) - 60); hi = min(len(think_text), (earliest_pos or 0) + 60)
    return markers, earliest_pos, ctx, think_text[lo:hi]

def _first_eos_any(token_ids: torch.Tensor, eos_id_list: Optional[List[int]]) -> int:
    if not eos_id_list: return token_ids.numel()
    hits = []
    for eid in eos_id_list:
        pos = (token_ids == eid).nonzero(as_tuple=False)
        if pos.numel() > 0: hits.append(pos[0].item())
    return min(hits) if hits else token_ids.numel()

def _entropy_from_start_index(model, seq_ids: torch.Tensor, start_idx: int) -> List[float]:
    device = next(model.parameters()).device
    seq_ids = seq_ids.to(device); ents: List[float] = []
    with torch.inference_mode():
        out = model(input_ids=seq_ids[:, :start_idx+1], use_cache=True); past = out.past_key_values
        L = seq_ids.shape[1]
        for t in range(start_idx, L-1):
            out = model(input_ids=seq_ids[:, t:t+1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            logp = F.log_softmax(logits, dim=-1); p = logp.exp()
            h = float(-(p * logp).sum().item())
            if not math.isfinite(h):
                logits = (logits - logits.max()).float()
                logp = F.log_softmax(logits, dim=-1); p = logp.exp()
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

# ───────────────────────── Prompt builders ─────────────────────────
def chat_base_for_pass1(tokenizer, clue: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CLUE: {clue}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def chat_base_for_pass2(tokenizer, clue: str, prev_output: str, cue: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CLUE: {clue}"},
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
        • ANSWER:  prefill "<think>…</think>\n<answer>\n", stop at "</answer>" (cap answer_cap)
    """

    def _gen_batch(prefixes: List[str], cap: int, stop_strs: List[str], bad_words: Optional[List[str]] = None):
        """Returns: decs, ent_series, input_lengths, sequences, stop_reasons"""
        inputs = tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        if torch.cuda.is_available():
            for k in inputs:
                inputs[k] = inputs[k].to("cuda")
            input_lengths = input_lengths.to(inputs["input_ids"].device)

        stop = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strs)]) if stop_strs else None
        bad_words_ids = None
        if bad_words:
            b = []
            for w in bad_words:
                ids = tokenizer.encode(w, add_special_tokens=False)
                if ids: b.append(ids)
            bad_words_ids = b if b else None

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
            bad_words_ids=bad_words_ids,
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
            raw_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            found_stop = any(s in raw_txt for s in (stop_strs or []))
            has_eos = False
            if eos_ids:
                for eid in eos_ids:
                    if (gen_ids == eid).any(): has_eos = True; break
            hit_max = len(gen_ids) >= cap

            if found_stop:   stop_reasons.append("stop_token")
            elif has_eos:    stop_reasons.append("eos")
            elif hit_max:    stop_reasons.append("max_new_tokens")
            else:            stop_reasons.append("other")

            txt = raw_txt
            for s in (stop_strs or []):
                if s in txt: txt = txt.split(s, 1)[0]; break
            decs.append(txt.strip())

            if entropy_mode == "none":
                ent_series.append([]); continue

            scores_T = len(out.scores)
            t_stop = min(_first_eos_any(gen_ids, eos_ids) if eos_ids else gen_ids.shape[0], scores_T)
            tok_ents = []
            bad = False
            for t in range(t_stop):
                logits = out.scores[t][row_i].float()
                if torch.isnan(logits).any() or torch.isinf(logits).any(): bad = True; break
                logp = F.log_softmax(logits, dim=-1); p = logp.exp()
                h = float(-(p * logp).sum().item())
                if not math.isfinite(h): bad = True; break
                tok_ents.append(h)
            if bad or len(tok_ents) == 0:
                start_idx = start_tok_idx - 1
                tok_ents = _entropy_from_start_index(model, seqs[row_i:row_i+1], start_idx) or []
            ent_series.append(tok_ents)

        return decs, ent_series, input_lengths, seqs, stop_reasons

    def _norm_fields(ex: dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalize a raw example to (clue_text, gold_answer).
        Many HF exports use 'problem' for the prompt and 'answer' for the label.
        """
        # clue candidates (include 'problem'!)
        clue = (ex.get("problem") or ex.get("clue") or ex.get("clue_text") or
                ex.get("question") or ex.get("prompt") or ex.get("text") or
                ex.get("definition"))
        # enumeration (optional)
        enum = (ex.get("enumeration") or ex.get("len") or ex.get("length") or
                ex.get("num_letters") or ex.get("n") or ex.get("pattern"))
        enum_str = None
        if enum is not None:
            if isinstance(enum, (list, tuple)):
                enum_str = ",".join(str(x) for x in enum if str(x).strip())
            else:
                s = str(enum).strip()
                enum_str = s if s else None

        clue_text = None
        if clue:
            clue_text = str(clue).strip()
            # Add enumeration like " (5)" if present and not already there
            if enum_str and not re.search(r"\(\s*[\d, -]+\s*\)\s*$", clue_text):
                clue_text = f"{clue_text} ({enum_str})"

        # gold answer candidates
        gold = (ex.get("answer") or ex.get("solution") or ex.get("target") or
                ex.get("final_answer") or ex.get("label") or ex.get("gold"))
        if isinstance(gold, (list, tuple)):
            gold = " ".join(str(x) for x in gold if str(x).strip())
        elif gold is not None:
            gold = str(gold).strip()

        return clue_text, gold

    def _pack_pass_result(
        clue: str,
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
        tok_ents_all = (ent_think or []) + (ent_answer or [])
        Tthink = len(ent_think or []); Tans = len(ent_answer or []); T = len(tok_ents_all)

        think, answer = _extract_blocks(full_text)
        think_text = think or ""; pred_answer_text = answer or ""

        skip_chars = len(cue_prefix_str) if injected_cue else 0
        markers, pos_in_think, reconsider_context, reconsider_excerpt = _find_markers_and_context(
            think_text, clue, skip_prefix_chars=skip_chars
        )
        if injected_cue: markers = ["injected_cue"] + (markers or [])

        t_cue = 0 if injected_cue else None
        if (not injected_cue) and (pos_in_think is not None):
            t_cue = max(0, min(pos_in_think, Tthink))

        entropy_overall = _finite_mean(tok_ents_all) if tok_ents_all else None
        entropy_think   = _finite_mean(ent_think)     if ent_think else None
        entropy_answer  = _finite_mean(ent_answer)    if ent_answer else None

        entropy_pre_cue = None
        entropy_reconsider_think = None
        entropy_reconsider_full = None
        if t_cue is not None:
            if T > t_cue:    entropy_reconsider_full  = _finite_mean(tok_ents_all[t_cue:])
            if Tthink > t_cue: entropy_reconsider_think = _finite_mean(tok_ents_all[t_cue:Tthink])

        pred_canon = _canon_answer(pred_answer_text)
        is_correct_pred = _contains_canon(pred_canon, canon_gold)

        return dict(
            prev_output=prev_output,
            output=full_text,
            pred_answer=pred_answer_text,
            pred_answer_canon=pred_canon,
            entropy=entropy_overall,
            entropy_think=entropy_think,
            entropy_answer=entropy_answer,
            entropy_pre_cue=entropy_pre_cue,
            entropy_reconsider_think=entropy_reconsider_think,
            entropy_reconsider_full=entropy_reconsider_full,
            stop_reason_think=stop_reason_think,
            stop_reason_answer=stop_reason_answer,
            has_reconsider_cue=bool(markers),
            reconsider_markers=markers or [],
            reconsider_pos=pos_in_think,
            reconsider_context=reconsider_context,
            reconsider_excerpt=reconsider_excerpt,
            is_correct_pred=is_correct_pred,
            is_correct_after_reconsideration=bool(markers) and bool(is_correct_pred),
            tokens_total=T,
            tokens_end_think=Tthink,
            tokens_think=Tthink,
            tokens_answer=Tans,
            valid_tag_structure=_valid_tag_structure(full_text),
        )

    # ---------- resume bookkeeping ----------
    outpath = os.path.join(outdir, f"step{step:04d}_{split_name}.jsonl")
    done_keys: set[str] = set()
    if os.path.exists(outpath):
        with open(outpath, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rk = obj.get("row_key")
                if rk: done_keys.add(rk)

    logging.info("→ %s | N=%d | already-done rows: %d", split_name, len(examples), len(done_keys))

    # ---------- main loop ----------
    BATCH = batch_size
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    total_written = 0
    for i in range(0, len(examples), BATCH):
        idx_lo, idx_hi = i, min(i + BATCH, len(examples))
        batch_ds = examples.select(range(idx_lo, idx_hi))

        batch = []
        for j, ex in enumerate(batch_ds):
            clue_text, gold = _norm_fields(ex)
            if not clue_text:  # skip empty
                continue
            ex = dict(ex)
            ex["_clue"] = clue_text
            ex["_gold"] = gold
            ex["_dataset_index"] = idx_lo + j
            batch.append(ex)
        if not batch:
            continue

        B, S = len(batch), num_samples

        # ===== PASS 1 =====
        base1 = [chat_base_for_pass1(tokenizer, ex["_clue"]) for ex in batch]
        pre1_think = [b + "<think>\n" for b in base1 for _ in range(S)]

        think1_texts, think1_ents, _, _, think1_stop = _gen_batch(pre1_think, think_cap, ["</think>"])

        pre1_answer = []
        for row_i in range(B * S):
            pre = pre1_think[row_i] + think1_texts[row_i] + "</think>\n<answer>\n"
            pre1_answer.append(pre)
        answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(
            pre1_answer, answer_cap, ["</answer>"], bad_words=["<think>", "</think>"]
        )

        pass1_full = [
            f"<think>{think1_texts[row_i]}</think>\n<answer>{answer1_texts[row_i]}</answer>"
            for row_i in range(B * S)
        ]

        # pick which p1 sample feeds p2
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
                    ex["_clue"],
                    firstpass_choice[bi],
                    second_pass_phrase.strip(),
                )
                for bi, ex in enumerate(batch)
            ]
            pre2_think = [b + "<think>\n" + cue_str for b in base2 for _ in range(S)]
            think2_texts_only_new, think2_ents, _, _, think2_stop = _gen_batch(pre2_think, think_cap, ["</think>"])
            think2_texts = [cue_str + t for t in think2_texts_only_new]

            pre2_answer = []
            for row_i in range(B * S):
                pre = pre2_think[row_i] + think2_texts_only_new[row_i] + "</think>\n<answer>\n"
                pre2_answer.append(pre)
            answer2_texts, answer2_ents, _, _, answer2_stop = _gen_batch(
                pre2_answer, answer_cap, ["</answer>"], bad_words=["<think>", "</think>"]
            )

            pass2_full = [
                f"<think>{think2_texts[row_i]}</think>\n<answer>{answer2_texts[row_i]}</answer>"
                for row_i in range(B * S)
            ]

        # ===== WRITE JSON =====
        with open(outpath, "a", encoding="utf-8") as f:
            for bi, ex in enumerate(batch):
                canon_gold = _canon_answer(ex["_gold"]) if ex["_gold"] else None
                ds_idx = int(ex["_dataset_index"])
                for k in range(S):
                    row_i = bi * S + k
                    row_key = f"{split_name}:{ds_idx}:{k}"
                    if row_key in done_keys:
                        continue

                    p1 = _pack_pass_result(
                        clue=ex["_clue"],
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
                            clue=ex["_clue"],
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
                        p2["improved_over_pass1"] = bool(p2.get("is_correct_pred")) and not bool(p1.get("is_correct_pred"))

                    row = {
                        "row_key": row_key,
                        "dataset_index": ds_idx,
                        "clue": ex["_clue"],
                        "gold_answer": ex["_gold"],
                        "gold_answer_canon": canon_gold,
                        "step": step,
                        "split": split_name,
                        "sample_idx": k,
                        "pass1": p1,
                        "pass2": p2,
                    }
                    json.dump(row, f, ensure_ascii=False); f.write("\n")
                    done_keys.add(row_key); total_written += 1

        if (idx_hi % max(1, 100 // max(1, batch_size))) == 0:
            logging.info("   wrote so far: %d rows (up to dataset idx %d)", total_written, idx_hi-1)

    logging.info("Finished: wrote %d new rows → %s", total_written, outpath)

# ─────────────────────────── Dataset helpers ───────────────────────────
def _load_local_json_dataset(path: str):
    from datasets import Dataset
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            records.append(obj)
    return Dataset.from_list(records)

def load_mini_crosswords(cache_dir: str, split: str):
    from datasets import load_dataset
    repo = "od2961/mini-crosswords"
    tried = []
    for sp in [split, "train", "test", "validation"]:
        try:
            ds_full = load_dataset(repo, split=sp, cache_dir=cache_dir)
            if ds_full is not None and len(ds_full) > 0:
                logging.info("Loaded %s split=%s | N=%d", repo, sp, len(ds_full))
                return ds_full
        except Exception as e:
            tried.append((sp, repr(e)))
    raise RuntimeError(f"Could not load {repo}. Tried: {tried}")

# ─────────────────────────── Main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--output_dir", required=True)

    # Data selection
    ap.add_argument("--dataset_id", default="od2961/mini-crosswords",
                    help="Default 'od2961/mini-crosswords'; may also be a local JSONL.")
    ap.add_argument("--split", default="train",
                    help="Split name to run on; will fallback among train/test/validation if needed.")
    ap.add_argument("--num_examples", type=int, default=None,
                    help="Optional cap on number of examples.")

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
    ap.add_argument("--second_pass_phrase",
                    default="Wait, we need to reconsider. Let's think this through step by step.")
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

    # EOS set
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
    if args.dataset_id == "od2961/mini-crosswords":
        ds = load_mini_crosswords(HF_CACHE_DIR, args.split)
        dataset_name_for_log = "od2961/mini-crosswords"
    elif os.path.exists(args.dataset_id):
        ds = _load_local_json_dataset(args.dataset_id)
        dataset_name_for_log = args.dataset_id
    else:
        ds = load_dataset(args.dataset_id, split=args.split, cache_dir=HF_CACHE_DIR)
        dataset_name_for_log = args.dataset_id

    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("Model: %s @ %s | dtype=%s", args.model_name_or_path, args.revision, dtype)
    logging.info("Dataset: %s split=%s | N=%d", dataset_name_for_log, args.split, len(ds))
    logging.info("Output dir: %s", args.output_dir)

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

    logging.info("All inference complete.")

if __name__ == "__main__":
    main()
