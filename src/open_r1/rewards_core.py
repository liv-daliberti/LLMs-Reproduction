from __future__ import annotations
"""
cryptic_rewards.py  •  Minimal reward implementations (v10)

Contains exactly two rewards:

1) crossword_accuracy_reward
   • Returns 1.0 only if:
       - the gold answer (canonicalized: spaces/hyphens removed, case-insensitive)
         appears ANYWHERE in the model's output, AND
       - the response length is > MIN_TOKENS (default 100 tokens; whitespace tokens
         by default, or model tokens if a tokenizer is provided via kwargs).
     Otherwise returns 0.0.

2) pure_accuracy_reward
   • Requires correct FORMAT **and** correct ANSWER:
       - output must match the required tag template:
         <think> … </think><answer> … </answer>
       - the <answer> content must exactly match the gold (canonicalized)
     Returns 1.0 if both conditions hold, else 0.0.
"""

import re
from typing import Any, List, Sequence

# ----------------------------- config ------------------------------

MIN_TOKENS = 100  # crossword reward requires response length > MIN_TOKENS

# ----------------------------- regexes -----------------------------

# Full-format check: <think>…</think><answer>…</answer> (across lines)
_format_pat = re.compile(r"(?si)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")

# Extract answer text
_answer_pat = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")

# ---------------------------- helpers ------------------------------

def _extract_content(comp: Any) -> str:
    """
    Accepts:
      • a plain string
      • a list like [{'role': 'assistant', 'content': '...'}]
      • nested sequences from chat-style structures
    Returns the assistant text as a string.
    """
    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, Sequence) and comp:
        first = comp[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        if isinstance(first, Sequence):
            return _extract_content(first)
    return str(comp)

def _canon_no_space_hyphen(text: str) -> str:
    """Lowercase and remove spaces/hyphens for robust equality/search."""
    return re.sub(r"[ \-]", "", text or "").lower()

def _count_tokens(text: str, tokenizer=None) -> int:
    """
    Count tokens in `text`. If a Hugging Face tokenizer is supplied via kwargs
    (e.g., tokenizer=AutoTokenizer.from_pretrained(...)), use it; otherwise
    fall back to whitespace tokenization.
    """
    if tokenizer is not None:
        try:
            # Prefer not to add special tokens for raw length checks
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Whitespace-token fallback
    return len(re.findall(r"\S+", text or ""))

# --------------------------- rewards -------------------------------

def crossword_accuracy_reward(
    completions: List[Any],
    answer:      List[str],
    **kw,
) -> List[float]:
    """
    Full credit basis (1.0) iff BOTH:
      • gold answer appears ANYWHERE in the output (canonicalized), AND
      • response length > MIN_TOKENS (tokenized as specified above).

    The final score is the basis multiplied by the fraction of present tags among:
      <think>, </think>, <answer>, </answer>.
    (e.g., having only "<answer>" present yields 1/4 of the basis score.)
    """
    tokenizer = kw.get("tokenizer", None)
    outs: List[float] = []

    TAGS = ("<think>", "</think>", "<answer>", "</answer>")

    for gold, comp in zip(answer, completions):
        raw = _extract_content(comp)
        resp_len_ok = _count_tokens(raw, tokenizer=tokenizer) > MIN_TOKENS

        gold_can = _canon_no_space_hyphen(gold)
        raw_can  = _canon_no_space_hyphen(raw)

        # Whole-word or quoted match on the canonicalized stream
        gold_re = re.escape(gold_can)
        pattern = rf'(?:\b{gold_re}\b|["\']{gold_re}["\'])'
        has_gold_anywhere = bool(re.search(pattern, raw_can, flags=re.IGNORECASE))

        # Basis score: 1.0 if both conditions hold, else 0.0
        basis = 1.0 if (resp_len_ok and has_gold_anywhere) else 0.0

        # Tag multiplier: fraction of present tags among the four
        present_tags = sum(1 for t in TAGS if t in raw)
        if present_tags == 4:
            tag_factor = 1
        else: 
            tag_factor = 0

        outs.append(basis * tag_factor)

    return outs
    
def pure_accuracy_reward(
    completions: List[Any],
    answer:      List[str],
    **kw,
) -> List[float]:
    """
    Pure exact-match with format requirement:
      • Output must match <think> … </think><answer> … </answer> (any spacing/newlines).
      • The <answer> content must exactly equal the gold (canonicalized),
        AND have the same character length as the canonicalized gold.
    Plus a 0.25 bonus if the crossword-style accuracy condition is met.
    """
    outs: List[float] = []

    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)

        # Must satisfy the full tag template
        if not _format_pat.match(txt):
            outs.append(0.0)
            continue

        m = _answer_pat.search(txt)
        if not m:
            outs.append(0.0)
            continue

        pred = m.group(1).strip()

        # Canonicalize (remove spaces/hyphens) for both equality and length checks
        pred_c = _canon_no_space_hyphen(pred)
        gold_c = _canon_no_space_hyphen(gold)

        # Enforce same length (e.g., "ARTichoke" can't satisfy "ART")
        if len(pred_c) != len(gold_c):
            outs.append(0.0)
            continue

        ok = (pred_c == gold_c)
        outs.append(1.0 if ok else 0.0)

    # Element-wise add 0.25 * crossword bonus; clip to 1.0
    bonus_list = crossword_accuracy_reward(completions, answer, **kw)
    outs = [min(1.0, base + 0.25 * bonus) for base, bonus in zip(outs, bonus_list)]
    return outs

import re
from typing import Any, Dict, Iterable, List, Optional, Union

# ── helpers you referenced ──────────────────────────────────────────────
MOVE_RE  = re.compile(r"^[A-Z][<>^v]\d+$")
TOK_LIST = re.compile(r"\s*,\s*")
ANS_TAG  = re.compile(r"(?is)<answer>(.*?)</answer>")

def _extract_answer_text(text: str) -> str:
    """Return inner text of <answer>…</answer> if present; else the whole string."""
    m = ANS_TAG.search(text or "")
    return (m.group(1) if m else (text or "")).strip()

def _canon_token(tok: str) -> Optional[str]:
    """Uppercase, strip spaces, ensure it looks like PIECE+DIR+STEPS."""
    t = (tok or "").strip().upper().replace(" ", "")
    return t if MOVE_RE.match(t) else None

from typing import List, Optional, Any
import re

def _canon_seq(x: Any) -> Optional[List[str]]:
    """
    Parse a Rush Hour move sequence out of raw model text or a Python list.
    Normalizes:
      - extracts <answer>...</answer> if present (ignores <think>...)
      - accepts commas or whitespace as separators
      - maps U/D/L/R and Unicode arrows to ^/v/</>
      - uppercases piece letters; strips spaces; removes leading zeros in steps
    Returns list like ["E^1","G<1","Bv1","A>3"] or None if nothing parseable.
    """
    # Coerce to a single string
    if isinstance(x, (list, tuple)):
        s = ",".join(str(t) for t in x)
    else:
        s = str(x)

    # Strip <think> and isolate <answer> if present
    s = re.sub(r"(?is)<think>.*?</think>", "", s)
    m = re.search(r"(?is)<answer>\s*(.*?)\s*</answer>", s)
    if m:
        s = m.group(1)

    # Normalize arrows
    s = (s
         .replace("↑", "^").replace("↓", "v")
         .replace("←", "<").replace("→", ">"))

    # Find tokens in order, allowing spaces like "A > 3"
    tokens = []
    for m in re.finditer(r"([A-Za-z])\s*([><\^vUDLR])\s*([0-9]+)", s):
        piece = m.group(1).upper()
        d = m.group(2).upper()
        d = {"U": "^", "D": "v", "L": "<", "R": ">"}.get(d, d)  # keep ^ v < >
        steps = str(int(m.group(3)))  # drop leading zeros
        tokens.append(f"{piece}{d}{steps}")

    return tokens or None


def rush_solution_exact(*, prompts, completions, answer=None, gold=None, **kwargs):
    """
    TRL reward: 1.0 iff the predicted move list exactly matches the gold list.
    Accepts either `answer` or `gold`, and also checks `answers` / `gold_answers`.
    """
    # coerce completions to list
    if isinstance(completions, str):
        completions = [completions]
    elif not isinstance(completions, list):
        completions = list(completions)

    if gold is None:
        gold = answer or kwargs.get("answers") or kwargs.get("gold_answers")
    if gold is None:
        return [0.0] * len(completions)

    # normalize gold list length to completions
    if isinstance(gold, str):
        gold_list = [gold] * len(completions)
    else:
        gold_list = list(gold)
    if len(gold_list) != len(completions):
        gold_list = (gold_list * len(completions)) if len(gold_list) == 1 else gold_list[:len(completions)]

    # canon golds (support gold entries that are lists like ["E^1","G<1",...])
    gold_lists: List[Optional[List[str]]] = []
    for g in gold_list:
        if isinstance(g, (list, tuple)):
            g = ",".join(str(t) for t in g)
        gold_lists.append(_canon_seq(g))

    # score
    scores: List[float] = []
    for i, pred in enumerate(completions):
        # ALSO support prediction as list like ["E^1","G<1",...]
        if isinstance(pred, (list, tuple)):
            pred = ",".join(str(t) for t in pred)
        pred_can = _canon_seq(pred)
        gold_can = gold_lists[i] if i < len(gold_lists) else None
        scores.append(1.0 if (pred_can is not None and pred_can == gold_can) else 0.0)

    return scores
    
def _extract_content(s: str) -> str:
    """Return raw string (no stripping) to allow tag pattern matching."""
    return s

def _canon_math(s: str) -> str:
    """
    Canonicalize math answers:
    - strip leading/trailing whitespace
    - remove LaTeX spacing commands (\ ), $...$, and curly braces around single tokens
    - normalize minus-zero to zero
    - drop trailing .0 from integers
    - remove spaces inside the expression unless inside LaTeX commands
    - unify parentheses usage for single-number expressions
    """
    s = s.strip()

    # Remove LaTeX math mode markers
    s = s.replace('$', '')

    # Remove LaTeX spacing commands
    s = re.sub(r"\\\s+", "", s)

    # Remove outer braces around a single token
    if re.fullmatch(r"\{[^{}]+\}", s):
        s = s[1:-1]

    # Drop surrounding parentheses if they enclose just a number/frac/root
    if re.fullmatch(r"\([^()]+\)", s):
        s_inner = s[1:-1].strip()
        # only drop if it doesn't change grouping meaning
        if re.match(r"^[\d\.\-\\sqrt]+$", s_inner):
            s = s_inner

    # Strip spaces
    s = s.replace(" ", "")

    # Convert -0 to 0
    if s in ("-0", "+0"):
        s = "0"

    # Remove trailing .0 from integers
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split('.')[0]

    return s

def pure_accuracy_reward_math(
    completions: List[Any],
    answer:      List[str],
    **kw,
) -> List[float]:
    """
    Pure exact-match for math problems with format requirement:
      • Output must match <think> … </think><answer> … </answer> (any spacing/newlines).
      • The <answer> content must exactly equal the gold (canonicalized math form).
    """
    outs: List[float] = []

    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)

        # Must satisfy the full tag template
        if not _format_pat.match(txt):
            outs.append(0.0)
            continue

        m = _answer_pat.search(txt)
        if not m:
            outs.append(0.0)
            continue

        pred = m.group(1)
        ok = (_canon_math(pred) == _canon_math(gold))
        outs.append(1.0 if ok else 0.0)

    return outs