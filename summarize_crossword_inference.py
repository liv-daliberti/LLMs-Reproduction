#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize crossword inference JSONL outputs (two-pass scheme).

Compatible with outputs from crossword-inference.py:
  keys per line include: row_key, dataset_index, split, clue, gold_answer, pass1 {...}, pass2 {...}

What this script reports per training step:
- Sample-level and example-level accuracy (example-level ORs correctness across samples)
- Mean entropies (overall / think / answer) for pass1 and pass2
- Stop reason breakdowns
- Tag-structure validity rates
- “Improved over pass1” rates (sample- and example-level)
- Reconsider marker rates (excluding the injected cue marker)

Usage:
  python summarize_crossword_inference.py /path/to/results_root \
      --split test \
      --save_csv /tmp/steps.csv \
      --per_example_csv /tmp/per_example.csv
"""

import os
import re
import json
import argparse
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

# ────────────────────── helpers ──────────────────────
def mean_safe(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None

def pct(n: int, d: int) -> str:
    return "-" if d == 0 else f"{100.0*n/d:5.1f}%"

def fmt_float(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:6.3f}"

def nat_step_from_path(path: str) -> Optional[int]:
    m = re.search(r"step(\d+)", path)
    return int(m.group(1)) if m else None

def scan_files(root: str, split: Optional[str]) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split and split not in fn:
                continue
            out.append(os.path.join(dp, fn))
    out.sort(key=lambda p: (nat_step_from_path(p) or 0, p))
    return out

def _get_example_id(rec: Dict[str, Any]) -> str:
    """
    Prefer a stable numeric id if available; fall back to row_key, then clue, then problem.
    """
    sp = rec.get("split", "")
    if rec.get("dataset_index") is not None:
        try:
            di = int(rec["dataset_index"])
            return f"{sp}/{di}"
        except Exception:
            pass
    if rec.get("row_key"):
        return str(rec["row_key"])
    if rec.get("clue"):
        return str(rec["clue"])
    if rec.get("problem"):
        return str(rec["problem"])
    return f"{sp}/UNK"

def _has_real_reconsider(p: Dict[str, Any]) -> bool:
    """True if reconsider markers exist other than the injected cue."""
    ms = p.get("reconsider_markers") or []
    return any(m and m != "injected_cue" for m in ms)

# ────────────────────── aggregator ──────────────────────
class StepAgg:
    def __init__(self, step: int):
        self.step = step

        # example_id -> per-pass correctness (OR across samples)
        self.ex_correct_p1: Dict[str, bool] = {}
        self.ex_correct_p2: Dict[str, bool] = {}

        # readable label for CSVs
        self.id_to_clue: Dict[str, str] = {}

        # sample-level counts
        self.n_samp_p1 = 0
        self.n_samp_p2 = 0
        self.samp_correct_p1 = 0
        self.samp_correct_p2 = 0

        # improvement (sample-level & example-level)
        self.samp_improved_p2 = 0   # pass2.improved_over_pass1
        self.ex_improved_p2 = 0     # ex-level: p2 correct and p1 not

        # entropies (overall + phase)
        self.ent_p1_all: List[Optional[float]] = []
        self.ent_p1_think: List[Optional[float]] = []
        self.ent_p1_ans: List[Optional[float]] = []

        self.ent_p2_all: List[Optional[float]] = []
        self.ent_p2_think: List[Optional[float]] = []
        self.ent_p2_ans: List[Optional[float]] = []

        # token lengths (optional; may be absent on older runs)
        self.tok_p1_think: List[Optional[int]] = []
        self.tok_p1_ans: List[Optional[int]] = []
        self.tok_p2_think: List[Optional[int]] = []
        self.tok_p2_ans: List[Optional[int]] = []

        # stop reasons
        self.stop_think_p1 = Counter()
        self.stop_ans_p1   = Counter()
        self.stop_think_p2 = Counter()
        self.stop_ans_p2   = Counter()

        # tag structure validity
        self.tag_ok_p1 = 0
        self.tag_ok_p2 = 0

        # reconsider markers (ignore injected cue)
        self.reconsider_rate_p1_numer = 0
        self.reconsider_rate_p2_numer = 0

        # example id set
        self.examples: set[str] = set()

    def add(self, rec: Dict[str, Any]):
        ex_id = _get_example_id(rec)
        self.examples.add(ex_id)
        if "clue" in rec and isinstance(rec["clue"], str):
            self.id_to_clue.setdefault(ex_id, rec["clue"])

        # ----- Pass 1 -----
        p1 = rec.get("pass1") or {}
        if p1:
            self.n_samp_p1 += 1
            c1 = bool(p1.get("is_correct_pred"))
            self.samp_correct_p1 += int(c1)
            self.ex_correct_p1[ex_id] = self.ex_correct_p1.get(ex_id, False) or c1

            self.ent_p1_all.append(p1.get("entropy"))
            self.ent_p1_think.append(p1.get("entropy_think"))
            self.ent_p1_ans.append(p1.get("entropy_answer"))

            self.tok_p1_think.append(p1.get("tokens_think"))
            self.tok_p1_ans.append(p1.get("tokens_answer"))

            sr_t = p1.get("stop_reason_think") or "unknown"
            sr_a = p1.get("stop_reason_answer") or "unknown"
            self.stop_think_p1[sr_t] += 1
            self.stop_ans_p1[sr_a] += 1

            if p1.get("valid_tag_structure"):
                self.tag_ok_p1 += 1
            if _has_real_reconsider(p1):
                self.reconsider_rate_p1_numer += 1

        # ----- Pass 2 -----
        p2 = rec.get("pass2") or {}
        if p2:
            self.n_samp_p2 += 1
            c2 = bool(p2.get("is_correct_pred"))
            self.samp_correct_p2 += int(c2)
            self.ex_correct_p2[ex_id] = self.ex_correct_p2.get(ex_id, False) or c2

            self.ent_p2_all.append(p2.get("entropy"))
            self.ent_p2_think.append(p2.get("entropy_think"))
            self.ent_p2_ans.append(p2.get("entropy_answer"))

            self.tok_p2_think.append(p2.get("tokens_think"))
            self.tok_p2_ans.append(p2.get("tokens_answer"))

            sr_t = p2.get("stop_reason_think") or "unknown"
            sr_a = p2.get("stop_reason_answer") or "unknown"
            self.stop_think_p2[sr_t] += 1
            self.stop_ans_p2[sr_a] += 1

            if p2.get("valid_tag_structure"):
                self.tag_ok_p2 += 1
            if _has_real_reconsider(p2):
                self.reconsider_rate_p2_numer += 1

            if p2.get("improved_over_pass1"):
                self.samp_improved_p2 += 1

    def finalize(self):
        # example-level improvement after OR-aggregation across samples
        for ex_id in self.examples:
            p1_ok = self.ex_correct_p1.get(ex_id, False)
            p2_ok = self.ex_correct_p2.get(ex_id, False)
            if p2_ok and not p1_ok:
                self.ex_improved_p2 += 1

    def row(self) -> str:
        nE = len(self.examples) if self.examples else 0

        acc1S = pct(self.samp_correct_p1, self.n_samp_p1)
        acc2S = pct(self.samp_correct_p2, self.n_samp_p2)

        acc1E = pct(sum(1 for v in self.ex_correct_p1.values() if v), nE) if nE else "-"
        acc2E = pct(sum(1 for v in self.ex_correct_p2.values() if v), nE) if nE else "-"

        ent1  = fmt_float(mean_safe(self.ent_p1_all))
        ent2  = fmt_float(mean_safe(self.ent_p2_all))
        t1    = fmt_float(mean_safe(self.ent_p1_think))
        a1    = fmt_float(mean_safe(self.ent_p1_ans))
        t2    = fmt_float(mean_safe(self.ent_p2_think))
        a2    = fmt_float(mean_safe(self.ent_p2_ans))

        impS  = pct(self.samp_improved_p2, self.n_samp_p2) if self.n_samp_p2 else "-"
        impE  = pct(self.ex_improved_p2, nE) if nE else "-"

        tag1 = pct(self.tag_ok_p1, self.n_samp_p1) if self.n_samp_p1 else "-"
        tag2 = pct(self.tag_ok_p2, self.n_samp_p2) if self.n_samp_p2 else "-"

        return (f"{self.step:6d} "
                f"{self.n_samp_p1:6d} {acc1S:>6} {acc1E:>6} {ent1:>8} {t1:>7} {a1:>7} "
                f"{self.n_samp_p2:6d} {acc2S:>6} {acc2E:>6} {ent2:>8} {t2:>7} {a2:>7} "
                f"{impS:>6} {impE:>6} {tag1:>6} {tag2:>6}")

    def footer(self) -> str:
        def fmt_counter(cnt: Counter, den: int) -> str:
            if den == 0:
                return "—"
            keys = ["stop_token", "eos", "max_new_tokens", "other", "unknown"]
            return ", ".join(f"{k}={pct(cnt.get(k,0), den)}" for k in keys)

        lines = []
        nE = len(self.examples)
        lines.append(f"   • examples: {nE}")
        if self.n_samp_p1:
            lines.append(f"   • p1 think stops: {fmt_counter(self.stop_think_p1, self.n_samp_p1)}")
            lines.append(f"   • p1 answer stops: {fmt_counter(self.stop_ans_p1, self.n_samp_p1)}")
        if self.n_samp_p2:
            lines.append(f"   • p2 think stops: {fmt_counter(self.stop_think_p2, self.n_samp_p2)}")
            lines.append(f"   • p2 answer stops: {fmt_counter(self.stop_ans_p2, self.n_samp_p2)}")

        if self.n_samp_p1:
            lines.append(f"   • p1 reconsider-markers rate: {pct(self.reconsider_rate_p1_numer, self.n_samp_p1)}")
        if self.n_samp_p2:
            lines.append(f"   • p2 reconsider-markers rate: {pct(self.reconsider_rate_p2_numer, self.n_samp_p2)}")

        if any(v is not None for v in self.tok_p1_think + self.tok_p2_think):
            mt1 = mean_safe([x for x in self.tok_p1_think if isinstance(x, (int,float))])
            ma1 = mean_safe([x for x in self.tok_p1_ans   if isinstance(x, (int,float))])
            mt2 = mean_safe([x for x in self.tok_p2_think if isinstance(x, (int,float))])
            ma2 = mean_safe([x for x in self.tok_p2_ans   if isinstance(x, (int,float))])
            lines.append("   • mean tokens — p1: think="
                         f"{'-' if mt1 is None else f'{mt1:.1f}'} answer="
                         f"{'-' if ma1 is None else f'{ma1:.1f}'}; "
                         "p2: think="
                         f"{'-' if mt2 is None else f'{mt2:.1f}'} answer="
                         f"{'-' if ma2 is None else f'{ma2:.1f}'}")
        return "\n".join(lines)

# ────────────────────── main ──────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Root directory containing step*/.../*.jsonl")
    ap.add_argument("--split", default=None, help="Filter filenames containing this split substring (e.g., 'test').")
    ap.add_argument("--save_csv", default=None, help="Optional CSV output path for the step table.")
    ap.add_argument("--per_example_csv", default=None, help="Optional CSV with per-example correctness (p1, p2).")
    args = ap.parse_args()

    files = scan_files(args.results_root, args.split)
    if not files:
        print("No JSONL files found. Check the path or split filter.")
        return

    steps: Dict[int, StepAgg] = {}

    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                step = rec.get("step", step_from_name if step_from_name is not None else 0)
                agg = steps.setdefault(step, StepAgg(step))
                agg.add(rec)

    # finalize
    for agg in steps.values():
        agg.finalize()

    # ordered by step
    ordered = [steps[k] for k in sorted(steps.keys())]

    # table
    print("  step   n1S  acc1S  acc1E    ent1      t1      a1   n2S  acc2S  acc2E    ent2      t2      a2   impS  impE  tag1  tag2")
    print("-" * 108)
    for agg in ordered:
        print(agg.row())
    print()

    # footers
    for agg in ordered:
        print(f"[step {agg.step}]")
        print(agg.footer())
        print()

    # optional step CSV
    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step","n1S","acc1S_pct","acc1E_pct","ent1","t1","a1",
                        "n2S","acc2S_pct","acc2E_pct","ent2","t2","a2",
                        "impS_pct","impE_pct","tag1_pct","tag2_pct"])
            for agg in ordered:
                nE = len(agg.examples) if agg.examples else 0
                acc1S = (100.0*agg.samp_correct_p1/agg.n_samp_p1) if agg.n_samp_p1 else None
                acc2S = (100.0*agg.samp_correct_p2/agg.n_samp_p2) if agg.n_samp_p2 else None
                acc1E = (100.0*sum(1 for v in agg.ex_correct_p1.values() if v)/nE) if nE else None
                acc2E = (100.0*sum(1 for v in agg.ex_correct_p2.values() if v)/nE) if nE else None
                impS  = (100.0*agg.samp_improved_p2/agg.n_samp_p2) if agg.n_samp_p2 else None
                impE  = (100.0*agg.ex_improved_p2/nE) if nE else None
                tag1p = (100.0*agg.tag_ok_p1/agg.n_samp_p1) if agg.n_samp_p1 else None
                tag2p = (100.0*agg.tag_ok_p2/agg.n_samp_p2) if agg.n_samp_p2 else None

                w.writerow([
                    agg.step,
                    agg.n_samp_p1, acc1S, acc1E,
                    mean_safe(agg.ent_p1_all),
                    mean_safe(agg.ent_p1_think),
                    mean_safe(agg.ent_p1_ans),
                    agg.n_samp_p2, acc2S, acc2E,
                    mean_safe(agg.ent_p2_all),
                    mean_safe(agg.ent_p2_think),
                    mean_safe(agg.ent_p2_ans),
                    impS, impE, tag1p, tag2p
                ])

    # optional per-example CSV
    if args.per_example_csv:
        import csv
        with open(args.per_example_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step","example_id","clue","p1_correct","p2_correct","improved"])
            for agg in ordered:
                for ex_id in sorted(agg.examples):
                    p1_ok = agg.ex_correct_p1.get(ex_id, False)
                    p2_ok = agg.ex_correct_p2.get(ex_id, False)
                    improved = (p2_ok and not p1_ok)
                    w.writerow([
                        agg.step,
                        ex_id,
                        agg.id_to_clue.get(ex_id, ""),
                        int(bool(p1_ok)),
                        int(bool(p2_ok)),
                        int(bool(improved)),
                    ])

if __name__ == "__main__":
    main()
