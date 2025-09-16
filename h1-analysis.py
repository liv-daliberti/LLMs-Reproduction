#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H1 test: Do Aha! Moments Unconditionally Raise Model Accuracy?

This script supports two Aha! definitions:
  • GPT Aha (sample-level): change_way_of_thinking (aka aha_gpt)
  • Formal Aha (problem-step level): prior failures + prior stability (+ optional δ3 gain)

Formal Aha at (q, k) iff:
  (1) Prior failures:  max_{i<k} P_i(correct|q) < δ1      (with optional Beta smoothing)
  (2) Prior stability: max_{i<k} P_i(shift|q)   < δ2      where P_i(shift|q) ≈ aha_rate_gpt at step i
  (3) Shift now:       aha_any_gpt(k) == 1
  (4) Optional gain:   P_k(correct|q) - max_{i<k} P_i(correct|q) ≥ δ3        (disabled by default)

Also exports concrete Aha examples (question + pass1 answer text) for both GPT and Formal.

Outputs (subset)
----------------
  h1_pass1_samples.csv
  h1_step_summary.csv
  h1_problem_step_multi.csv
  h1_step_problem_multi_summary.csv
  aha_ratio_vs_step__<DS>__<MODEL>.png
  aha_formal_ratio_vs_step__<DS>__<MODEL>.png
  acc_by_<aha>_vs_step.png
  glm_effects_by_step__<aha>.png
  glm_coef_forest__<aha>.png
  glm_calibration__<aha>.png
  ever_acc_vs_step.png, often_acc_vs_step.png
  ever_acc_by_aha_problem_<aha>.png
  aha_examples_gpt.{csv,jsonl}
  aha_examples_formal.{csv,jsonl}
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Regex / utility
# ----------------------------

STEP_PAT = re.compile(r"step(\d+)", re.I)

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")

def title_suffix(dataset_name: str, model_name: str) -> str:
    return f"{dataset_name}, {model_name}"

def nat_step_from_path(path: str) -> Optional[int]:
    m = STEP_PAT.search(path)
    return int(m.group(1)) if m else None

def scan_files(root: str, split_substr: Optional[str]) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split_substr and split_substr not in fn:
                continue
            out.append(os.path.join(dp, fn))
    out.sort()
    return out

import re as _re

# Grab all <think> and <answer> blocks, case-insensitive, tolerant of messy nesting.
_TAGS_ANSWER = _re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", _re.DOTALL | _re.IGNORECASE)
_TAGS_THINK  = _re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>",   _re.DOTALL | _re.IGNORECASE)

def _parse_tag_blocks(txt: str):
    """Return (think_blocks, answer_blocks) as lists of strings (may be empty)."""
    if not isinstance(txt, str):
        return [], []
    thinks  = [m.strip() for m in _TAGS_THINK.findall(txt)]
    answers = [m.strip() for m in _TAGS_ANSWER.findall(txt)]
    return thinks, answers

# Generic patterns when tags aren’t reliable
_ANSWER_PATTERNS = [
    _re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", _re.DOTALL | _re.IGNORECASE),
    _re.compile(r"\\boxed\{([^}]*)\}"),
    _re.compile(r"\\boxed\(([^)]*)\)"),
    _re.compile(r"(?:Final\s*Answer|Answer)\s*:\s*(.+)", _re.IGNORECASE),
]

def _extract_answer_from_text(txt: str) -> str | None:
    """Parse a short final answer from free text if no clean field exists."""
    if not isinstance(txt, str) or not txt.strip():
        return None
    # Prefer tagged answers (last non-empty)
    blocks = [b.strip() for b in _ANSWER_PATTERNS[0].findall(txt)]
    # Drop junk like "think>" masquerading as an answer
    blocks = [b for b in blocks if b and not b.strip().lower().startswith("think>")]
    if blocks:
        return blocks[-1].strip()
    # Other common forms
    for pat in _ANSWER_PATTERNS[1:]:
        m = pat.search(txt)
        if m:
            ans = m.group(1).strip()
            ans = _re.sub(r"\s+", " ", ans)
            return ans or None
    # Last-resort: last short non-empty line
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if lines and len(lines[-1]) <= 200:
        return lines[-1]
    return None


# ----------------------------
# Label extraction
# ----------------------------

def coerce_bool(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    return int(bool(x))

def _get_aha_gpt(p1: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    """Return 0/1 if any GPT/LLM 'aha' flag is present; else None."""
    candidates = [
        ("p1", "change_way_of_thinking"),   # canonical label
        ("root", "change_way_of_thinking"),
        ("p1", "shift_in_reasoning_v1"),
        ("p1", "shift_llm"),
        ("p1", "shift_gpt"),
        ("p1", "pivot_llm"),
        ("p1", "rechecked"),
        ("root", "rechecked"),
    ]
    for loc, key in candidates:
        v = p1.get(key) if loc == "p1" else rec.get(key)
        if v is None:
            continue
        out = coerce_bool(v)
        if out is not None:
            return int(out)
    return None

def _get_aha_native(p1: Dict[str, Any]) -> Optional[int]:
    aha_raw = coerce_bool(p1.get("has_reconsider_cue"))
    markers = p1.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers):
        return 0
    return 0 if aha_raw is None else int(aha_raw)

# ----------------------------
# Load PASS-1 rows
# ----------------------------

def load_pass1_rows(files: List[str]) -> pd.DataFrame:
    rows = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                p1 = rec.get("pass1") or {}
                if not p1:
                    continue

                prob = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if prob is None:
                    di = rec.get("dataset_index")
                    prob = f"idx:{di}" if di is not None else "unknown"

                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None:
                    continue

                corr_raw = coerce_bool(p1.get("is_correct_pred"))
                if corr_raw is None:
                    continue
                correct = int(corr_raw)

                aha_gpt = _get_aha_gpt(p1, rec)  # may be None
                aha_native = _get_aha_native(p1)

                rows.append({
                    "problem": str(prob),
                    "step": int(step),
                    "sample_idx": rec.get("sample_idx", None),
                    "aha_gpt": np.nan if aha_gpt is None else int(aha_gpt),
                    "aha_native": 0 if aha_native is None else int(aha_native),
                    "correct": correct,
                    "source_file": path,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No PASS-1 rows found. Check results_root / split.")
    df["aha_gpt"] = pd.to_numeric(df["aha_gpt"], errors="coerce").fillna(0).astype(int)
    df["change_way_of_thinking"] = df["aha_gpt"].astype(int)
    return df

def load_pass1_rows_unfiltered(files: List[str]) -> pd.DataFrame:
    rows = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                p1 = rec.get("pass1") or {}
                if not p1:
                    continue
                prob = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if prob is None:
                    di = rec.get("dataset_index")
                    prob = f"idx:{di}" if di is not None else "unknown"
                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None:
                    continue
                corr_raw = coerce_bool(p1.get("is_correct_pred"))
                if corr_raw is None:
                    continue
                correct = int(corr_raw)
                aha_gpt = _get_aha_gpt(p1, rec)  # may be None
                aha_native = _get_aha_native(p1)
                rows.append({
                    "problem": str(prob),
                    "step": int(step),
                    "sample_idx": rec.get("sample_idx", None),
                    "aha_gpt": np.nan if aha_gpt is None else int(aha_gpt),
                    "aha_native": 0 if aha_native is None else int(aha_native),
                    "correct": correct,
                    "source_file": path,
                })
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        raise RuntimeError("No PASS-1 rows found (unfiltered). Check inputs.")
    df_all["sidx"] = df_all["sample_idx"].fillna(0).astype(int)
    return df_all

# ----------------------------
# Problem-step aggregates
# ----------------------------

def build_problem_step_table(df_all: pd.DataFrame) -> pd.DataFrame:
    grp = df_all.groupby(["step", "problem"], as_index=False)

    def any_gpt(s: pd.Series) -> int:
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        if np.isnan(arr).all():
            return 0
        return int(np.nanmax(arr) >= 1)

    def mean_gpt(s: pd.Series) -> float:
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        if np.isnan(arr).all():
            return np.nan
        return float(np.nanmean(arr))

    df_sorted = df_all.sort_values(["step", "problem", "sidx"])
    first_rows = df_sorted.groupby(["step", "problem"], as_index=False).first()

    ps = grp.agg(
        n_samples=("correct", "size"),
        ever_correct=("correct", "max"),
        freq_correct=("correct", "mean"),
        aha_any_gpt=("aha_gpt", any_gpt),
        aha_rate_gpt=("aha_gpt", mean_gpt),
    )

    ps = ps.merge(
        first_rows[["step", "problem", "correct", "aha_gpt"]]
        .rename(columns={"correct": "first_correct", "aha_gpt": "aha_first_gpt"}),
        on=["step", "problem"], how="left"
    )

    ps["n_samples"] = ps["n_samples"].astype(int)
    ps["ever_correct"] = ps["ever_correct"].astype(int)
    ps["first_correct"] = ps["first_correct"].fillna(0).astype(int)
    ps["aha_first_gpt"] = ps["aha_first_gpt"].fillna(0).astype(int)
    ps["aha_any_gpt"] = ps["aha_any_gpt"].astype(int)
    ps["aha_rate_gpt"] = ps["aha_rate_gpt"].fillna(0.0).astype(float)
    return ps

# ----------------------------
# Formal Aha (δ1 + δ2 on prior correctness / shift rates)
# ----------------------------

def compute_formal_aha(ps: pd.DataFrame,
                       delta1: float,
                       delta2: float,
                       min_prior_steps: int = 1,
                       require_shift_now: bool = True,
                       delta3: Optional[float] = None,
                       p_col: str = "freq_correct",
                       rate_col: str = "aha_rate_gpt",
                       shift_now_col: str = "aha_any_gpt",
                       p_smooth_alpha: float = 1.0,
                       p_smooth_beta: float = 1.0,
                       r_smooth_alpha: float = 0.0,
                       r_smooth_beta: float = 0.0) -> pd.DataFrame:
    """
    Formal Aha at (problem q, step k) iff:
      (1) max_{i<k} P_i(correct|q) < δ1
      (2) max_{i<k} P_i(shift|q)   < δ2, where P_i(shift|q) ≈ aha_rate_gpt at step i
      (3) shift_now_col(k) == 1      (e.g., aha_any_gpt==1 at step k)
      (4) optional: P_k - max_{i<k}P_i ≥ δ3

    Smoothing per PRIOR step:
      P_i(correct|q) := (k_i + α_p) / (n_i + α_p + β_p),  with k_i ≈ freq_correct_i * n_i
      P_i(shift|q)   := (s_i + α_r) / (n_i + α_r + β_r),  with s_i ≈ aha_rate_gpt_i * n_i
    """
    needed = {"step", "problem", p_col, rate_col, shift_now_col, "n_samples"}
    missing = needed - set(ps.columns)
    if missing:
        raise ValueError(f"compute_formal_aha: missing columns: {missing}")

    ps = ps.sort_values(["problem", "step"]).copy()
    ps["aha_formal"] = 0

    ps[p_col] = pd.to_numeric(ps[p_col], errors="coerce").fillna(0.0)
    ps[rate_col] = pd.to_numeric(ps[rate_col], errors="coerce").fillna(0.0)
    ps["n_samples"] = pd.to_numeric(ps["n_samples"], errors="coerce").fillna(0).astype(int)
    ps[shift_now_col] = ps[shift_now_col].astype(int)

    flags = np.zeros(len(ps), dtype=int)
    idx_global = 0

    for prob, sub in ps.groupby("problem", sort=False):
        sub = sub.sort_values("step")
        n = sub["n_samples"].to_numpy().astype(int)
        pvals = sub[p_col].to_numpy(dtype=float)
        rvals = sub[rate_col].to_numpy(dtype=float)
        shift_now = sub[shift_now_col].to_numpy().astype(int)

        # Smoothed proportions per step (applied to all steps; we use prior slices)
        k_correct = pvals * n
        p_smooth = (k_correct + p_smooth_alpha) / (n + p_smooth_alpha + p_smooth_beta)

        s_aha = rvals * n
        if r_smooth_alpha > 0.0 or r_smooth_beta > 0.0:
            r_smooth = (s_aha + r_smooth_alpha) / (n + r_smooth_alpha + r_smooth_beta)
        else:
            r_smooth = rvals

        for j in range(len(sub)):
            if j < min_prior_steps:
                flags[idx_global] = 0
                idx_global += 1
                continue

            prior_pmax = float(np.max(p_smooth[:j])) if j > 0 else 0.0
            prior_rmax = float(np.max(r_smooth[:j])) if j > 0 else 0.0

            prior_fail   = (prior_pmax < delta1)
            prior_stable = (prior_rmax < delta2)
            shift_ok     = (shift_now[j] == 1) if require_shift_now else True

            gain_ok = True
            if delta3 is not None:
                gain_ok = (p_smooth[j] - prior_pmax) >= float(delta3)

            flags[idx_global] = int(prior_fail and prior_stable and shift_ok and gain_ok)
            idx_global += 1

    ps["aha_formal"] = flags
    return ps

# ----------------------------
# Aha EXAMPLES (question + pass1 answer text)
# ----------------------------

def _first_nonempty_str(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def _extract_from_messages(pass_dict: Dict[str, Any]) -> Optional[str]:
    msgs = pass_dict.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and str(m.get("role", "")).lower() == "assistant":
                c = m.get("content") or m.get("text")
                if isinstance(c, str) and c.strip():
                    return c
    return None

def _extract_pass_text(pass_dict: Dict[str, Any]) -> str | None:
    """
    Robustly recover the assistant's PASS text.
    Priority:
      1) direct scalar text (incl. 'output')
      2) messages[].role=='assistant'.{content|text}
      3) OpenAI-style choices[0].message.{content|text}
      4) longest non-empty string in the dict
    """
    if not isinstance(pass_dict, dict):
        return None
    # (1) direct keys — your data uses 'output'
    for k in ["output", "raw", "text", "response", "model_output", "assistant_text",
              "completion", "full_output", "reasoning", "analysis",
              "chain_of_thought", "thought", "deliberate", "content", "prev_output"]:
        v = pass_dict.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # (2) messages[]
    msgs = pass_dict.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and str(m.get("role", "")).lower() == "assistant":
                c = m.get("content") or m.get("text")
                if isinstance(c, str) and c.strip():
                    return c
    # (3) choices[0].message
    ch = pass_dict.get("choices")
    if isinstance(ch, list) and ch and isinstance(ch[0], dict):
        msg = ch[0].get("message")
        if isinstance(msg, dict):
            c = msg.get("content") or msg.get("text")
            if isinstance(c, str) and c.strip():
                return c
    # (4) longest string fallback
    str_vals = [v for v in pass_dict.values() if isinstance(v, str) and v.strip()]
    return max(str_vals, key=len) if str_vals else None


def _extract_pass_answer(pass_dict: Dict[str, Any]) -> str | None:
    """
    Prefer the final <answer>...</answer> block in 'output'; if missing/dirty,
    fall back to explicit short fields or parse from text.
    """
    if not isinstance(pass_dict, dict):
        return None

    # A) Try explicit short fields
    for k in ["final_answer", "answer", "pred", "prediction",
              "short_answer", "pred_text", "parsed_answer", "extracted_answer"]:
        v = pass_dict.get(k)
        if isinstance(v, str) and v.strip():
            # If it still contains tags, parse them
            if "<answer" in v.lower():
                got = _extract_answer_from_text(v)
                if got:
                    return got
            return v.strip()

    # B) Parse from 'output' (your primary storage)
    out = pass_dict.get("output")
    if isinstance(out, str) and out.strip():
        _, answers = _parse_tag_blocks(out)
        # Use the last non-empty <answer> that isn't the bogus "think>"
        answers = [a for a in answers if a and not a.strip().lower().startswith("think>")]
        if answers:
            return answers[-1].strip()
        # No clean block? heuristics on the whole output
        got = _extract_answer_from_text(out)
        if got:
            return got

    # C) Fall back to 'pred_answer' (often contains tags/truncation)
    pa = pass_dict.get("pred_answer")
    if isinstance(pa, str) and pa.strip():
        got = _extract_answer_from_text(pa)
        if got:
            return got
        return pa.strip()  # last resort

    # D) last ditch: parse from any text we can get
    txt = _extract_pass_text(pass_dict)
    return _extract_answer_from_text(txt) if txt else None

def export_aha_examples(files: List[str],
                        ps: pd.DataFrame,
                        aha_def: str,
                        out_dir: str,
                        max_examples: int = 50,
                        per_step_limit: int = 0,
                        max_chars: int = 3000,
                        dataset_name: str = "",
                        model_name: str = "",
                        correct_only: bool = False) -> Tuple[str, str, pd.DataFrame]:
    """
    Writes two files with Aha examples and returns (csv_path, jsonl_path, df_examples):
      - aha_examples_<mode>[ _correct].csv
      - aha_examples_<mode>[ _correct].jsonl

    Row fields:
      dataset, model, mode, problem (question), step, sample_idx,
      correct (PASS-1), aha_gpt, pass1_answer, pass1_text, source_file

    Modes:
      - 'formal': (problem, step) must have aha_formal==1 and the chosen sample must have aha_gpt==1.
      - 'gpt':    any sample-level row with aha_gpt==1.
    If correct_only=True, keep only rows with correct==1 (PASS-1).
    """
    mode = "formal" if aha_def == "formal" else "gpt"
    suffix = "_correct" if correct_only else ""
    targets = set()

    if mode == "formal":
        if "aha_formal" not in ps.columns:
            return ("","", pd.DataFrame())
        for prob, step in ps.loc[ps["aha_formal"] == 1, ["problem", "step"]].itertuples(index=False, name=None):
            targets.add((str(prob), int(step)))

    rows = []
    seen_pair = set()
    per_step_count: Dict[int, int] = {}

    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue

                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None:
                    continue

                p1 = rec.get("pass1") or {}
                prob_raw = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if prob_raw is None:
                    di = rec.get("dataset_index")
                    prob_raw = f"idx:{di}" if di is not None else "unknown"
                prob = str(prob_raw)

                aha_gpt = _get_aha_gpt(p1, rec) or 0
                if mode == "formal":
                    if (prob, int(step)) not in targets:
                        continue
                    if aha_gpt != 1:
                        continue
                else:  # GPT mode
                    if aha_gpt != 1:
                        continue

                # PASS-1 correctness
                correct = coerce_bool(p1.get("is_correct_pred")) or 0
                if correct_only and not correct:
                    continue

                # per-step throttling
                if per_step_limit > 0:
                    c = per_step_count.get(int(step), 0)
                    if c >= per_step_limit:
                        continue

                key = (prob, int(step))
                if key in seen_pair and per_step_limit <= 1 and mode == "formal":
                    continue  # keep one per problem/step unless we asked for more

                # Extract text/answer (robust), with pass2 fallback
                t1 = _extract_pass_text(p1) or ""
                a1 = _extract_pass_answer(p1) or ""
                if (not t1) or (not a1):
                    p2 = rec.get("pass2") or {}
                    if not t1:
                        t2 = _extract_pass_text(p2)
                        if isinstance(t2, str) and t2.strip():
                            t1 = t2
                    if not a1:
                        a2 = _extract_pass_answer(p2)
                        if isinstance(a2, str) and a2.strip():
                            a1 = a2

                if max_chars and isinstance(t1, str) and len(t1) > max_chars:
                    t1 = t1[:max_chars] + " …[truncated]"

                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "mode": mode,
                    "problem": prob,
                    "step": int(step),
                    "sample_idx": rec.get("sample_idx", None),
                    "correct": int(correct),
                    "aha_gpt": int(aha_gpt),
                    "pass1_answer": a1,
                    "pass1_text": t1,
                    "source_file": path,
                })
                seen_pair.add(key)
                per_step_count[int(step)] = per_step_count.get(int(step), 0) + 1
                if max_examples > 0 and len(rows) >= max_examples:
                    break
            if max_examples > 0 and len(rows) >= max_examples:
                break

    if not rows:
        return ("","", pd.DataFrame())

    df_ex = pd.DataFrame(rows)
    base = f"aha_examples_{mode}{suffix}"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    jsonl_path = os.path.join(out_dir, f"{base}.jsonl")
    df_ex.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return (csv_path, jsonl_path, df_ex)

# ----------------------------
# Bootstrap / plotting helpers
# ----------------------------

def bootstrap_mean_ci(x: np.ndarray, B: int = 200, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    mu = float(x.mean())
    if B <= 0 or x.size == 1:
        return (mu, np.nan, np.nan)
    bs = np.empty(B, dtype=float)
    idx = np.arange(x.size)
    for b in range(B):
        take = rng.choice(idx, size=x.size, replace=True)
        bs[b] = float(x[take].mean())
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return (mu, float(lo), float(hi))

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / den
    return (max(0.0, center - half), min(1.0, center + half))

def _add_ci_cols(df_step_counts: pd.DataFrame) -> pd.DataFrame:
    los, his = [], []
    for k, n in zip(df_step_counts["k"].values, df_step_counts["n"].values):
        lo, hi = wilson_ci(int(k), int(n))
        los.append(lo); his.append(hi)
    out = df_step_counts.copy()
    out["lo"] = los; out["hi"] = his
    return out

def plot_acc_vs_step(df: pd.DataFrame, out_dir: str, tsuf: str):
    acc_df = (
        df.groupby("step", as_index=False)
          .agg(k=("correct", "sum"), n=("correct", "size"), acc=("correct", "mean"))
          .sort_values("step")
    )
    acc_df = _add_ci_cols(acc_df)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(acc_df["step"], acc_df["acc"], marker="o", label="Accuracy")
    ax.fill_between(acc_df["step"], acc_df["lo"], acc_df["hi"], alpha=0.2, label="95% CI")
    ax.set_xlabel("Training step"); ax.set_ylabel("Accuracy (PASS-1)")
    ax.set_title(f"Accuracy vs. step (PASS-1)\n{tsuf}")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "acc_vs_step.png")); plt.close(fig)

def plot_ratio_native_vs_step(df: pd.DataFrame, out_dir: str, tsuf: str):
    ratio_df = (
        df.groupby("step", as_index=False)
          .agg(n=("aha_native","size"), k=("aha_native","sum"))
          .assign(ratio=lambda d: d["k"] / d["n"])\
          .sort_values("step")
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(ratio_df["step"], ratio_df["ratio"], marker="o", label="Aha (native) ratio")
    ax.set_xlabel("Training step"); ax.set_ylabel("Aha (native) ratio")
    ax.set_title(f"Aha (native) ratio vs. step (PASS-1)\n{tsuf}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "aha_ratio_native_vs_step.png")); plt.close(fig)

def _plot_conditional_acc(df: pd.DataFrame, label_col: str, title: str, fname: str, out_dir: str):
    g = (df.groupby(["step", label_col], as_index=False)
           .agg(k=("correct","sum"), n=("correct","size"), acc=("correct","mean"))
           .sort_values(["step", label_col]))
    lines = {}
    for val in [0, 1]:
        sub = g[g[label_col] == val].copy()
        if sub.empty:
            continue
        sub = _add_ci_cols(sub)
        lines[val] = sub
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    for val, sub in lines.items():
        ax.plot(sub["step"], sub["acc"], marker="o", label=f"{label_col}={val}")
        ax.fill_between(sub["step"], sub["lo"], sub["hi"], alpha=0.15)
    ax.set_xlabel("Training step"); ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname)); plt.close(fig)

def save_conditional_acc_csv(df: pd.DataFrame, label_col: str, out_path: str):
    g = (df.groupby(["step", label_col], as_index=False)
           .agg(k=("correct","sum"), n=("correct","size"), acc=("correct","mean"))
           .sort_values(["step", label_col]))
    g.to_csv(out_path, index=False)

def plot_overlap_heatmap(df: pd.DataFrame, out_dir: str):
    tab = pd.crosstab(df["aha_native"], df["aha_gpt"]).reindex(index=[0,1], columns=[0,1], fill_value=0)
    n = tab.values.sum()
    rates = tab / n if n > 0 else tab.astype(float)
    fig, ax = plt.subplots(figsize=(4.8, 4.3), dpi=160)
    im = ax.imshow(rates.values.astype(float), aspect="equal")
    for (i, j), v in np.ndenumerate(rates.values.astype(float)):
        ax.text(j, i, f"{(v if n>0 else 0):.2%}\n({tab.values[i,j]})", ha="center", va="center", fontsize=9)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["aha_gpt=0","aha_gpt=1"]); ax.set_yticklabels(["aha_native=0","aha_native=1"])
    ax.set_title("Overlap: native vs GPT (share and counts)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "aha_overlap_heatmap.png")); plt.close(fig)

def _cohen_kappa(tab_2x2: np.ndarray) -> float:
    n = tab_2x2.sum()
    if n <= 0:
        return float("nan")
    po = (tab_2x2[0,0] + tab_2x2[1,1]) / n
    row = tab_2x2.sum(axis=1); col = tab_2x2.sum(axis=0)
    pe = (row[0]*col[0] + row[1]*col[1]) / (n*n)
    return (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else float("nan")

def plot_kappa_vs_step(df: pd.DataFrame, out_dir: str):
    rows = []
    for step, sub in df.groupby("step"):
        tab = pd.crosstab(sub["aha_native"], sub["aha_gpt"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        kappa = _cohen_kappa(tab.values.astype(float))
        rows.append({"step": int(step), "kappa": float(kappa), "n": int(tab.values.sum())})
    kdf = pd.DataFrame(rows).sort_values("step")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(kdf["step"], kdf["kappa"], marker="o")
    ax.set_xlabel("Training step"); ax.set_ylabel("Cohen's κ (aha_native vs aha_gpt)")
    ax.set_title("Agreement between raw and LLM-confirmed moments")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "kappa_vs_step.png")); plt.close(fig)

def plot_ever_acc_by_aha_problem(ps: pd.DataFrame, aha_col: str, out_dir: str):
    rows = []
    for step, sub in ps.groupby("step"):
        for g in [0, 1]:
            gsub = sub[sub[aha_col] == g]
            if gsub.empty:
                rows.append({"step": step, aha_col: g, "accE": np.nan, "lo": np.nan, "hi": np.nan}); continue
            k = int(gsub["ever_correct"].sum()); n = int(len(gsub))
            mu = k / n; lo, hi = wilson_ci(k, n)
            rows.append({"step": step, aha_col: g, "accE": mu, "lo": lo, "hi": hi})
    d = pd.DataFrame(rows).sort_values(["step", aha_col])
    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=140)
    for g, sub in d.groupby(aha_col):
        ax.plot(sub["step"], sub["accE"], marker="o", label=f"{aha_col}={g}")
        ax.fill_between(sub["step"], sub["lo"], sub["hi"], alpha=0.15)
    ax.set_xlabel("Training step"); ax.set_ylabel("Ever-correct accuracy")
    ax.set_title(f"Ever-correct by problem-level {aha_col}")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"ever_acc_by_aha_problem_{aha_col}.png")); plt.close(fig)

# ----------------------------
# GLMs and plots
# ----------------------------

def _cov_spec(df: pd.DataFrame, cluster_by: str):
    if cluster_by == "problem":
        if "problem" not in df.columns:
            raise ValueError("cluster_by='problem' requested but 'problem' column is missing.")
        groups = pd.Categorical(df["problem"]).codes
        kw = {"groups": groups}
        try:
            kw.update({"use_correction": True, "df_correction": True})
        except Exception:
            pass
        return "cluster", kw
    return "HC1", None

def fit_logit_with_fe(df: pd.DataFrame, out_txt: str,
                      cluster_by: str = "none",
                      aha_col: str = "change_way_of_thinking") -> Dict[str, Any]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required. Try: pip install statsmodels") from e

    d = df.copy()
    d["step_std"] = (d["step"] - d["step"].mean()) / (d["step"].std(ddof=0) + 1e-8)
    if aha_col not in d.columns:
        raise ValueError(f"GLM needs column '{aha_col}' on df.")

    model = smf.glm(f"correct ~ C(problem) + step_std + {aha_col}",
                    data=d, family=sm.families.Binomial())

    cov_type, cov_kwds = _cov_spec(d, cluster_by)
    try:
        res = model.fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))
    except TypeError:
        minimal_kw = {"groups": cov_kwds.get("groups")} if cov_kwds and "groups" in cov_kwds else None
        res = model.fit(cov_type=cov_type, cov_kwds=(minimal_kw or {}))

    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(res.summary().as_text())
        fh.write(f"\nCovariance: {cov_type}")
        if cov_kwds and "groups" in cov_kwds:
            fh.write(" (clustered by problem)")
        fh.write("\n")

    d1 = d.copy(); d1[aha_col] = 1
    d0 = d.copy(); d0[aha_col] = 0
    ame = float(np.mean(res.predict(d1) - res.predict(d0)))

    b  = float(res.params.get(aha_col, np.nan))
    se = float(res.bse.get(aha_col, np.nan))
    z  = b / se if se and np.isfinite(se) else np.nan
    p  = float(res.pvalues.get(aha_col, np.nan))

    with open(out_txt, "a", encoding="utf-8") as fh:
        fh.write(f"Average Marginal Effect (AME) of {aha_col}: {ame:.4f}\n")

    return dict(b_aha=b, se_aha=se, z_aha=z, p_aha=p, ame_aha=ame, res=res, aha_col=aha_col)

def fit_problem_glm_ever(ps: pd.DataFrame, out_dir: str,
                         cluster_by: str = "none",
                         aha_col: str = "aha_formal") -> Dict[str, Any]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required. Try: pip install statsmodels") from e

    d = ps.copy()
    d["step_std"] = (d["step"] - d["step"].mean()) / (d["step"].std(ddof=0) + 1e-8)
    if aha_col not in d.columns:
        raise ValueError(f"Problem-level GLM needs column '{aha_col}' on ps.")

    model = smf.glm(f"ever_correct ~ C(problem) + step_std + {aha_col}",
                    data=d, family=sm.families.Binomial())

    cov_type, cov_kwds = _cov_spec(d, cluster_by)
    try:
        res = model.fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))
    except TypeError:
        minimal_kw = {"groups": cov_kwds.get("groups")} if cov_kwds and "groups" in cov_kwds else None
        res = model.fit(cov_type=cov_type, cov_kwds=(minimal_kw or {}))

    out_txt = os.path.join(out_dir, f"logit_problem_ever_on_step_{aha_col}.txt")
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(res.summary().as_text())
        fh.write(f"\nCovariance: {cov_type}")
        if cov_kwds and "groups" in cov_kwds:
            fh.write(" (clustered by problem)")
        fh.write("\n")

    d1 = d.copy(); d1[aha_col] = 1
    d0 = d.copy(); d0[aha_col] = 0
    ame = float(np.mean(res.predict(d1) - res.predict(d0)))

    b  = float(res.params.get(aha_col, np.nan))
    se = float(res.bse.get(aha_col, np.nan))
    z  = b / se if se and np.isfinite(se) else np.nan
    p  = float(res.pvalues.get(aha_col, np.nan))

    return {"b_aha": b, "se_aha": se, "z_aha": z, "p_aha": p, "ame_aha": ame, "res": res, "out_txt": out_txt}

def fit_problem_glm_often(ps: pd.DataFrame, out_dir: str,
                          cluster_by: str = "none",
                          aha_col: str = "aha_formal") -> Dict[str, Any]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required. Try: pip install statsmodels") from e

    d = ps.copy()
    d["step_std"] = (d["step"] - d["step"].mean()) / (d["step"].std(ddof=0) + 1e-8)
    d["prop"] = d["freq_correct"].clip(0.0, 1.0)
    d["w"]    = d["n_samples"].clip(lower=1)

    if aha_col not in d.columns:
        raise ValueError(f"Problem-level GLM needs column '{aha_col}' on ps.")

    model = smf.glm(f"prop ~ C(problem) + step_std + {aha_col}",
                    data=d, family=sm.families.Binomial(), freq_weights=d["w"])

    cov_type, cov_kwds = _cov_spec(d, cluster_by)
    try:
        res = model.fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))
    except TypeError:
        minimal_kw = {"groups": cov_kwds.get("groups")} if cov_kwds and "groups" in cov_kwds else None
        res = model.fit(cov_type=cov_type, cov_kwds=(minimal_kw or {}))

    out_txt = os.path.join(out_dir, f"logit_problem_often_on_step_{aha_col}.txt")
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(res.summary().as_text())
        fh.write(f"\nCovariance: {cov_type}")
        if cov_kwds and "groups" in cov_kwds:
            fh.write(" (clustered by problem)")
        fh.write("\n")

    d1 = d.copy(); d1[aha_col] = 1
    d0 = d.copy(); d0[aha_col] = 0
    ame = float(np.mean(res.predict(d1) - res.predict(d0)))

    b  = float(res.params.get(aha_col, np.nan))
    se = float(res.bse.get(aha_col, np.nan))
    z  = b / se if se and np.isfinite(se) else np.nan
    p  = float(res.pvalues.get(aha_col, np.nan))

    return {"b_aha": b, "se_aha": se, "z_aha": z, "p_aha": p, "ame_aha": ame, "res": res, "out_txt": out_txt}

def plot_glm_effects_by_step_to(df: pd.DataFrame,
                                aha_col: str,
                                res,
                                out_dir: str,
                                fname: str,
                                bootstrap: int = 200,
                                seed: int = 0):
    rng = np.random.default_rng(seed)
    param_names = list(getattr(res, "params", pd.Series()).index)
    aha_candidates = [aha_col, "aha_gpt", "aha_any_gpt", "aha_formal", "change_way_of_thinking", "aha_native", "aha"]
    aha_in_model = next((n for n in aha_candidates if n in param_names), None)
    if aha_in_model is None:
        aha_in_model = next((n for n in param_names if "aha" in str(n)), None)

    fit_df = None
    try:
        fit_df = getattr(getattr(res, "model", None), "data", None)
        fit_df = getattr(fit_df, "frame", None)
    except Exception:
        fit_df = None
    if fit_df is not None and "step" in fit_df:
        mu = float(fit_df["step"].mean()); sigma = float(fit_df["step"].std(ddof=0)) + 1e-8
    else:
        mu = float(df["step"].mean());      sigma = float(df["step"].std(ddof=0)) + 1e-8

    formula_str = ""
    try:
        formula_str = str(getattr(getattr(res, "model", None), "formula", ""))
    except Exception:
        formula_str = ""
    if "C(problem)" in formula_str and "problem" not in df.columns:
        raise ValueError("plot_glm_effects_by_step_to: model used C(problem) but 'problem' is missing.")

    steps = np.sort(df["step"].unique())
    curves, cis = {}, {}
    for val in [0, 1]:
        means, bs_lo, bs_hi = [], [], []
        for s in steps:
            sub = df[df["step"] == s]
            if sub.empty:
                means.append(np.nan); bs_lo.append(np.nan); bs_hi.append(np.nan); continue
            base = sub.copy()
            if aha_col not in base.columns:
                base[aha_col] = val
            if aha_in_model is not None:
                base[aha_in_model] = val
            base["step_std"] = (base["step"] - mu) / sigma
            yhat = res.predict(base)
            means.append(float(np.mean(yhat)))
            if bootstrap > 0:
                m = len(base); idx = np.arange(m); bs = np.empty(bootstrap, dtype=float)
                for b in range(bootstrap):
                    take = rng.choice(idx, size=m, replace=True)
                    bs[b] = float(np.mean(res.predict(base.iloc[take])))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            else:
                lo = hi = np.nan
            bs_lo.append(lo); bs_hi.append(hi)
        curves[val] = np.array(means); cis[val] = (np.array(bs_lo), np.array(bs_hi))

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    label = aha_in_model if aha_in_model is not None else aha_col
    for val in [0, 1]:
        ax.plot(steps, curves[val], marker="o", label=f"{label}={val}")
        lo, hi = cis[val]
        if np.isfinite(lo).any():
            ax.fill_between(steps, lo, hi, alpha=0.15)
    ax.set_xlabel("Training step"); ax.set_ylabel("Predicted probability (GLM)")
    ax.set_title(f"GLM: predicted vs step for {label}=0/1")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname)); plt.close(fig)

def plot_glm_coef_forest_to(res, out_dir: str, fname: str,
                            keep: Optional[List[str]] = None,
                            title: Optional[str] = None):
    params = getattr(res, "params", None); bse = getattr(res, "bse", None)
    if params is None or bse is None:
        return
    names = list(params.index)
    aha_candidates = ["change_way_of_thinking", "aha_formal", "aha_gpt", "aha_any_gpt", "aha_native", "aha"]
    aha_in_model = next((n for n in aha_candidates if n in names), None)
    if keep is None:
        keep = []
        if aha_in_model is not None:
            keep.append(aha_in_model)
        if "step_std" in names:
            keep.append("step_std")

    def _ok(name: str) -> bool:
        if name == "Intercept": return False
        if "C(problem)" in name: return False
        return True

    keep = [k for k in keep if k in names and _ok(k)]
    if not keep:
        return
    coefs = params[keep].values; ses = bse[keep].values
    lo = coefs - 1.96 * ses; hi = coefs + 1.96 * ses

    fig, ax = plt.subplots(figsize=(6.0, 2.4 + 0.5*len(keep)), dpi=150)
    y = np.arange(len(keep))
    ax.hlines(y, lo, hi, lw=3); ax.plot(coefs, y, marker="o", ls="none")
    ax.axvline(0.0, lw=1, alpha=0.7)
    ax.set_yticks(y); ax.set_yticklabels(keep)
    ax.set_xlabel("Coefficient (log-odds), 95% CI")
    ax.set_title(title or "GLM coefficients")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname)); plt.close(fig)

def plot_glm_calibration_to(df: pd.DataFrame, res, out_dir: str, fname: str, bins: int = 10):
    fit_df = None
    try:
        fit_df = getattr(getattr(res, "model", None), "data", None)
        fit_df = getattr(fit_df, "frame", None)
    except Exception:
        fit_df = None
    if fit_df is not None and "step" in fit_df:
        mu = float(fit_df["step"].mean()); sigma = float(fit_df["step"].std(ddof=0)) + 1e-8
    else:
        mu = float(df["step"].mean());      sigma = float(df["step"].std(ddof=0)) + 1e-8

    tmp = df.copy()
    tmp["step_std"] = (tmp["step"] - mu) / sigma
    tmp["p_hat"] = res.predict(tmp)
    tmp["bin"] = pd.qcut(tmp["p_hat"], q=bins, duplicates="drop")

    cal = (tmp.groupby("bin")
             .agg(mean_pred=("p_hat", "mean"),
                  acc=("correct", "mean"),
                  n=("correct", "size"))
             .sort_values("mean_pred"))

    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=150)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, alpha=0.7)
    ax.plot(cal["mean_pred"], cal["acc"], marker="o")
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_pred"], r["acc"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed accuracy")
    ax.set_title("GLM calibration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname)); plt.close(fig)

# ----------------------------
# PASS text/answer lengths (optional)
# ----------------------------

def load_pass_lengths(files: List[str]) -> pd.DataFrame:
    rows = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue

                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None:
                    continue

                p1 = rec.get("pass1") or {}
                p2 = rec.get("pass2") or {}

                def _len_or_nan(s): return float(len(s)) if isinstance(s, str) else float("nan")
                t1 = _extract_pass_text(p1); a1 = _extract_pass_answer(p1)
                t2 = _extract_pass_text(p2); a2 = _extract_pass_answer(p2)

                rows.append({
                    "step": int(step),
                    "problem": str(rec.get("problem") or rec.get("clue") or rec.get("row_key") or
                                   (f"idx:{rec.get('dataset_index')}" if rec.get("dataset_index") is not None else "unknown")),
                    "sample_idx": rec.get("sample_idx"),
                    "len_p1": _len_or_nan(t1),
                    "len_p1_ans": _len_or_nan(a1),
                    "len_p2": _len_or_nan(t2),
                    "len_p2_ans": _len_or_nan(a2),
                    "source_file": path,
                })
    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError("No rows found while building pass-length table. Check inputs.")
    return d

def summarize_pass_lengths_by_step(length_df: pd.DataFrame,
                                   bootstrap: int = 200,
                                   seed: int = 0) -> pd.DataFrame:
    series_specs = [
        ("len_p1", "pass1"),
        ("len_p1_ans", "pass1_answer"),
        ("len_p2", "pass2"),
        ("len_p2_ans", "pass2_answer"),
    ]
    rows = []
    for step, sub in length_df.groupby("step"):
        for col, label in series_specs:
            x = pd.to_numeric(sub[col], errors="coerce").to_numpy()
            x = x[~np.isnan(x)]
            if x.size == 0:
                mu = lo = hi = float("nan"); n = 0
            else:
                mu, lo, hi = bootstrap_mean_ci(x, B=bootstrap, seed=seed); n = int(x.size)
            rows.append({"step": int(step), "series": label, "mean": float(mu),
                         "lo": float(lo) if not np.isnan(lo) else np.nan,
                         "hi": float(hi) if not np.isnan(hi) else np.nan, "n": n})
    return pd.DataFrame(rows).sort_values(["step", "series"]).reset_index(drop=True)

def plot_pass_lengths_vs_step(summary_long: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    for label, sub in summary_long.groupby("series"):
        sub = sub.sort_values("step")
        ax.plot(sub["step"], sub["mean"], marker="o", label=label)
        if sub[["lo", "hi"]].notna().all().all():
            ax.fill_between(sub["step"], sub["lo"], sub["hi"], alpha=0.15)
    ax.set_xlabel("Training step"); ax.set_ylabel("Mean characters")
    ax.set_title("Pass text/answer lengths vs. step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pass_lengths_vs_step.png"))
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Root containing step*/.../*.jsonl")
    ap.add_argument("--split", default=None, help="Filename substring filter, e.g. 'test'")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: <results_root>/h1_analysis)")
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)
    ap.add_argument("--no_bootstrap", action="store_true", help="Disable bootstrap CIs in GLM effects figures")

    # dataset/model stamping
    ap.add_argument("--dataset_name", default="MATH-500")
    ap.add_argument("--model_name",   default="Qwen2.5-1.5B")

    # covariance choice
    ap.add_argument("--cluster_by", choices=["problem", "none"], default="problem",
                    help="Covariance: 'problem' = cluster-robust; 'none' = HC1.")

    # Aha choice
    ap.add_argument("--aha_def", choices=["gpt", "formal"], default="formal",
                    help="Which Aha indicator to use in GLMs/plots.")

    # Formal Aha thresholds
    ap.add_argument("--delta1", type=float, default=0.20, help="δ1: prior failure threshold on P(correct).")
    ap.add_argument("--delta2", type=float, default=0.20, help="δ2: prior stability threshold on P(shift).")
    ap.add_argument("--delta3", type=float, default=None, help="δ3: required performance gain (optional).")
    ap.add_argument("--min_prior_steps", type=int, default=2, help="Require at least this many earlier steps.")

    # Beta smoothing for prior P(correct) and prior P(shift)
    ap.add_argument("--p_smooth_alpha", type=float, default=1.0, help="Beta prior α for P(correct) smoothing.")
    ap.add_argument("--p_smooth_beta",  type=float, default=1.0, help="Beta prior β for P(correct) smoothing.")
    ap.add_argument("--r_smooth_alpha", type=float, default=0.0, help="Beta prior α for P(shift) smoothing.")
    ap.add_argument("--r_smooth_beta",  type=float, default=0.0, help="Beta prior β for P(shift) smoothing.")

    # Example export controls
    ap.add_argument("--examples_n", type=int, default=50, help="Max total examples to export per mode.")
    ap.add_argument("--examples_per_step", type=int, default=1, help="Max examples per training step (0=no limit).")
    ap.add_argument("--examples_max_chars", type=int, default=3000, help="Max chars of pass1 text (truncate).")
    ap.add_argument("--examples_print_n", type=int, default=5, help="Print this many example rows to stdout (per mode).")

    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "h1_analysis")
    os.makedirs(out_dir, exist_ok=True)

    ds, mdl = args.dataset_name, args.model_name
    slug = f"{slugify(ds)}__{slugify(mdl)}"
    tsuf = title_suffix(ds, mdl)

    # ---------- Load PASS-1 sample rows ----------
    files = scan_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check the path or --split.")
    df = load_pass1_rows(files)

    if args.min_step is not None:
        df = df[df["step"] >= args.min_step]
    if args.max_step is not None:
        df = df[df["step"] <= args.max_step]
    if df.empty:
        raise SystemExit("No rows left after step filtering.")

    samples_csv = os.path.join(out_dir, "h1_pass1_samples.csv")
    df.assign(dataset=args.dataset_name, model=args.model_name).to_csv(samples_csv, index=False)

    # ---------- Multi-sample aggregation ----------
    df_all = load_pass1_rows_unfiltered(files)
    if args.min_step is not None:
        df_all = df_all[df_all["step"] >= args.min_step]
    if args.max_step is not None:
        df_all = df_all[df_all["step"] <= args.max_step]
    if df_all.empty:
        raise SystemExit("No rows left after step filtering (multi-sample).")

    ps = build_problem_step_table(df_all)

    # Compute formal Aha
    ps = compute_formal_aha(ps,
                            delta1=args.delta1,
                            delta2=args.delta2,
                            min_prior_steps=args.min_prior_steps,
                            require_shift_now=True,
                            delta3=args.delta3,
                            p_smooth_alpha=args.p_smooth_alpha,
                            p_smooth_beta=args.p_smooth_beta,
                            r_smooth_alpha=args.r_smooth_alpha,
                            r_smooth_beta=args.r_smooth_beta)
    ps_csv = os.path.join(out_dir, "h1_problem_step_multi.csv")
    ps.to_csv(ps_csv, index=False)

    # Step-level summary for problem-level ratios
    rows = []
    for step, sub in ps.groupby("step"):
        rows.append({
            "step": int(step),
            "ahaFormalProbRatio": float(sub["aha_formal"].mean()),
            "ahaProbRatio": float(sub["aha_any_gpt"].mean()),
            "nP": int(len(sub)),
        })
    pd.DataFrame(rows).sort_values("step").to_csv(
        os.path.join(out_dir, "h1_step_problem_multi_summary.csv"), index=False
    )

    # ---------- Merge formal Aha onto sample-level df; choose active 'aha' ----------
    df = df.merge(ps[["step", "problem", "aha_formal"]], on=["step", "problem"], how="left")
    df["aha_formal"] = df["aha_formal"].fillna(0).astype(int)
    aha_col = "change_way_of_thinking" if args.aha_def == "gpt" else "aha_formal"
    aha_col_prob = "aha_any_gpt" if args.aha_def == "gpt" else "aha_formal"

    # ---------- Export Aha EXAMPLES (and print preview) ----------
    csv_gpt, jsonl_gpt, df_ex_gpt = export_aha_examples(
        files, ps, "gpt", out_dir,
        max_examples=args.examples_n, per_step_limit=args.examples_per_step,
        max_chars=args.examples_max_chars, dataset_name=ds, model_name=mdl
    )
    csv_formal, jsonl_formal, df_ex_formal = export_aha_examples(
        files, ps, "formal", out_dir,
        max_examples=args.examples_n, per_step_limit=args.examples_per_step,
        max_chars=args.examples_max_chars, dataset_name=ds, model_name=mdl
    )

    # --- Correct-only lists (Aha & PASS-1 correct) ---
    csv_gpt_c, jsonl_gpt_c, df_ex_gpt_c = export_aha_examples(
        files, ps, "gpt", out_dir,
        max_examples=args.examples_n, per_step_limit=args.examples_per_step,
        max_chars=args.examples_max_chars, dataset_name=ds, model_name=mdl,
        correct_only=True
    )
    csv_formal_c, jsonl_formal_c, df_ex_formal_c = export_aha_examples(
        files, ps, "formal", out_dir,
        max_examples=args.examples_n, per_step_limit=args.examples_per_step,
        max_chars=args.examples_max_chars, dataset_name=ds, model_name=mdl,
        correct_only=True
    )

    # ---------- Aha ratios and accuracy ----------
    step_summary = (
        df.groupby("step", as_index=False)
          .agg(n=("correct", "size"),
               aha_ratio=("aha_gpt", "mean"),
               acc=("correct", "mean"))
          .sort_values("step")
    )
    step_summary.assign(dataset=args.dataset_name, model=args.model_name)\
                .to_csv(os.path.join(out_dir, "h1_step_summary.csv"), index=False)

    # GPT/sample-level aha ratio plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(step_summary["step"], step_summary["aha_ratio"], marker="o")
    ax.set_xlabel("Training step"); ax.set_ylabel("Aha-moment ratio (PASS-1, LLM-confirmed)")
    ax.set_title(f"Aha ratio vs. step (GPT/sample)\n{tsuf}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"aha_ratio_vs_step__{slug}.png")); plt.close(fig)

    # Formal/problem-level ratio plot
    formal_ratio = (ps.groupby("step", as_index=False)["aha_formal"].mean()
                      .rename(columns={"aha_formal": "ratio"}))
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(formal_ratio["step"], formal_ratio["ratio"], marker="o")
    ax.set_xlabel("Training step"); ax.set_ylabel("Formal Aha ratio (problem-level)")
    ax.set_title(f"Formal Aha ratio vs. step\n{tsuf}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"aha_formal_ratio_vs_step__{slug}.png")); plt.close(fig)

    # ---------- GLM (PASS-1 sample-level) ----------
    logit_txt = os.path.join(out_dir, f"logit_pass1_correct_on_step_{aha_col}.txt")
    stats_pass1 = fit_logit_with_fe(df, logit_txt, cluster_by=args.cluster_by, aha_col=aha_col)

    # Conditional accuracy by Aha, overlap/kappa, etc.
    plot_acc_vs_step(df, out_dir, tsuf)
    plot_ratio_native_vs_step(df, out_dir, tsuf)
    save_conditional_acc_csv(df, aha_col,
        os.path.join(out_dir, f"h1_step_conditional_acc_{aha_col}.csv"))
    _plot_conditional_acc(df, aha_col,
        f"Accuracy vs. step, split by {aha_col}",
        f"acc_by_{aha_col}_vs_step.png", out_dir)
    plot_overlap_heatmap(df, out_dir)
    plot_kappa_vs_step(df, out_dir)

    res = stats_pass1.get("res", None)
    bootstrap = 0 if args.no_bootstrap else 200
    if res is not None:
        plot_glm_effects_by_step_to(df, aha_col, res, out_dir,
                                    f"glm_effects_by_step__{aha_col}.png",
                                    bootstrap=bootstrap, seed=0)
        plot_glm_coef_forest_to(res, out_dir, f"glm_coef_forest__{aha_col}.png")
        plot_glm_calibration_to(df.assign(correct=df["correct"]), res, out_dir,
                                f"glm_calibration__{aha_col}.png")

    # ---------- Problem-level GLMs (EVER / OFTEN) ----------
    ever_stats  = fit_problem_glm_ever(ps,  out_dir, cluster_by=args.cluster_by, aha_col=aha_col_prob)
    often_stats = fit_problem_glm_often(ps, out_dir, cluster_by=args.cluster_by, aha_col=aha_col_prob)

    # Trends
    # (EVER)
    k = (ps.groupby("step")["ever_correct"].sum()).astype(int)
    nP = ps.groupby("step").size().astype(int)
    los, his = [], []
    for ki, ni in zip(k, nP):
        lo, hi = wilson_ci(int(ki), int(ni)); los.append(lo); his.append(hi)
    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=140)
    ax.plot(nP.index.values, (k / nP).values, marker="o", label="Ever-correct")
    ax.fill_between(nP.index.values, los, his, alpha=0.2)
    ax.set_xlabel("Training step"); ax.set_ylabel("Accuracy (ever)")
    ax.set_title("Ever-correct accuracy vs. step"); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ever_acc_vs_step.png")); plt.close(fig)

    # (OFTEN)
    step_often = ps.groupby("step", as_index=False)["freq_correct"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=140)
    ax.plot(step_often["step"], step_often["freq_correct"], marker="o", label="Often-correct (mean freq)")
    ax.set_xlabel("Training step"); ax.set_ylabel("Accuracy (mean per-problem freq)")
    ax.set_title("How often is it right? vs. step")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "often_acc_vs_step.png")); plt.close(fig)

    plot_ever_acc_by_aha_problem(ps, aha_col_prob, out_dir)

    # Effects & calibration (EVER)
    ever_plot = ps.rename(columns={"ever_correct": "correct"})
    plot_glm_effects_by_step_to(ever_plot, aha_col_prob, ever_stats["res"], out_dir,
                                f"problem_glm_effects_by_step_ever__{aha_col_prob}.png",
                                bootstrap=bootstrap, seed=0)
    plot_glm_coef_forest_to(ever_stats["res"], out_dir, f"problem_glm_coef_forest_ever__{aha_col_prob}.png")
    plot_glm_calibration_to(ever_plot, ever_stats["res"], out_dir,
                            f"problem_glm_calibration_ever__{aha_col_prob}.png")

    # Effects & calibration (OFTEN)
    often_plot = ps.copy(); often_plot["correct"] = often_plot["freq_correct"]
    plot_glm_effects_by_step_to(often_plot, aha_col_prob, often_stats["res"], out_dir,
                                f"problem_glm_effects_by_step_often__{aha_col_prob}.png",
                                bootstrap=bootstrap, seed=1)
    plot_glm_coef_forest_to(often_stats["res"], out_dir, f"problem_glm_coef_forest_often__{aha_col_prob}.png")
    plot_glm_calibration_to(often_plot, often_stats["res"], out_dir,
                            f"problem_glm_calibration_often__{aha_col_prob}.png")

    # ---------- Prints ----------
    cov_desc = "cluster-robust by problem" if args.cluster_by == "problem" else "HC1"
    print(f"Model/Data: {args.model_name} • {args.dataset_name}")
    print(f"Aha def: {args.aha_def}  |  Covariance: {cov_desc}")
    print(f"Formal Aha params: δ1={args.delta1}, δ2={args.delta2}, δ3={args.delta3}, "
          f"min_prior_steps={args.min_prior_steps}, "
          f"P-smooth α={args.p_smooth_alpha}, β={args.p_smooth_beta}, "
          f"R-smooth α={args.r_smooth_alpha}, β={args.r_smooth_beta}")

    print("\nKey effect (PASS-1): {aha} → correct".format(aha=aha_col))
    print(f"  See: {logit_txt}")

    print(f"\nKey effect (Problem-level EVER): {aha_col_prob} → ever_correct")
    print(f"  See: {ever_stats['out_txt']}")
    print(f"\nKey effect (Problem-level OFTEN): {aha_col_prob} → freq_correct")
    print(f"  See: {often_stats['out_txt']}")

    # Example previews (question + answer)
    if not df_ex_gpt.empty:
        print(f"\nAha examples (GPT) written:\n  {csv_gpt}\n  {jsonl_gpt}")
        print(f"Preview (first {args.examples_print_n}):")
        for _, r in df_ex_gpt.head(args.examples_print_n).iterrows():
            print(f"  • step {r['step']} | correct={r['correct']} | problem={r['problem']!s}")
            print(f"    answer: {str(r['pass1_answer'])[:200]}")
    else:
        print("\nNo GPT Aha examples found.")

    if not df_ex_formal.empty:
        print(f"\nAha examples (Formal) written:\n  {csv_formal}\n  {jsonl_formal}")
        print(f"Preview (first {args.examples_print_n}):")
        for _, r in df_ex_formal.head(args.examples_print_n).iterrows():
            print(f"  • step {r['step']} | correct={r['correct']} | problem={r['problem']!s}")
            print(f"    answer: {str(r['pass1_answer'])[:200]}")
    else:
        print("\nNo Formal Aha examples found (check δ1/δ2/min_prior_steps & shift-now criteria).")

    print("\nAdditional outputs:")
    print("  acc_vs_step.png, aha_ratio_native_vs_step.png")
    print(f"  acc_by_{aha_col}_vs_step.png, aha_ratio_vs_step__{slug}.png")
    print(f"  aha_formal_ratio_vs_step__{slug}.png")
    print("  aha_overlap_heatmap.png, kappa_vs_step.png")
    print(f"  glm_effects_by_step__{aha_col}.png, glm_coef_forest__{aha_col}.png, glm_calibration__{aha_col}.png")
    print("  h1_problem_step_multi.csv, h1_step_problem_multi_summary.csv")
    print("  ever_acc_vs_step.png, often_acc_vs_step.png, ever_acc_by_aha_problem_*.png")
    print("  problem_glm_effects_by_step_{ever,often}__*.png, problem_glm_coef_forest_{ever,often}__*.png, "
          "problem_glm_calibration_{ever,often}__*.png")
    print("  aha_examples_gpt.{csv,jsonl}, aha_examples_formal.{csv,jsonl}")
    print("  h1_pass1_samples.csv, h1_step_summary.csv")

if __name__ == "__main__":
    main()
