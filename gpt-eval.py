#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate pass-1 'shift in reasoning' for inference JSONL outputs.

Policy (strict / rare):
- Only annotate TRUE when there is BOTH:
  (A) an explicit cue inside <think> like “wait”, “hold on”, “on second thought”,
      “actually,” “scratch that,” “I misread…”, “re-check”, “doesn’t fit/match”, etc.
  AND
  (B) a material revision: the author rejects/corrects an earlier idea
      (new candidate, corrected derivation, contradiction resolved).

- We first do a lexical prefilter. If it hits, we ask DeepSeek to verify.
- If the LLM is uncertain or returns invalid JSON, default to FALSE.
- Idempotent: lines already containing 'shift_in_reasoning_v1' in pass1 are skipped.
- Processes candidates in random order; rewrites files atomically.

Usage:
  python annotate_shift_pass1.py /path/to/results_root --split test
"""

import os
import re
import json
import time
import sys
import glob
import math
import argparse
import random
import logging
import hashlib
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

# ─────────────────── DeepSeek (AI-Sandbox) config ───────────────────
sandbox_api_key  = "1e30d0e4d7564ba984e8adff48053009"   # ← replace if needed
sandbox_endpoint = "https://api-ai-sandbox.princeton.edu/"
sandbox_api_ver  = "2025-03-01-preview"
DEEPSEEK_MODEL   = "gpt-4o"

# ───────────────────── Imports (Azure OpenAI SDK) ─────────────────────
try:
    from openai import AzureOpenAI
except Exception as e:  # pragma: no cover
    AzureOpenAI = None
    print("ERROR: openai>=1.x with AzureOpenAI is required. pip install openai", file=sys.stderr)

# ───────────────────── Helper utilities ─────────────────────────────
@contextmanager
def timed(label):
    start = time.time()
    yield
    logging.debug(f"[TIMING] {label}: {time.time() - start:,.2f}s")

MAX_FIELD_LEN = 4096

def _clamp(txt: str, lim: int = MAX_FIELD_LEN) -> str:
    txt = txt or ""
    return txt[-lim:] if len(txt) > lim else txt

_client = None
def _client_lazy():
    global _client
    if _client is None:
        if AzureOpenAI is None:
            raise RuntimeError("AzureOpenAI client not available. Install openai>=1.x.")
        _client = AzureOpenAI(
            api_key        = sandbox_api_key,
            azure_endpoint = sandbox_endpoint,
            api_version    = sandbox_api_ver,
        )
    return _client

def _dump_filtered(prompt: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    dg = hashlib.md5(prompt.encode()).hexdigest()[:8]
    fn = f"filtered_prompt_{ts}_{dg}.txt"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(prompt)
    logging.warning("DeepSeek filtered/failed; saved prompt to %s", fn)

# ───────────────────── Regexes ─────────────────────────────
RE_THINK  = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

# Strict cue list (be conservative; no generic 'but' alone)
from typing import List, Tuple

SHIFT_CAND_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # ───────── Pauses / immediate self-interruptions ─────────
    ("wait",                  re.compile(r"(?i)(?:^|\W)wait(?:\W|$)")),
    ("hold on",               re.compile(r"(?i)\bhold (?:on|up)\b")),
    ("hang on",               re.compile(r"(?i)\bhang on\b")),
    ("one sec",               re.compile(r"(?i)\b(?:one|a)\s+(?:sec|second)\b")),
    ("just a sec",            re.compile(r"(?i)\bjust (?:a )?(?:sec|second)\b")),
    ("give me a moment",      re.compile(r"(?i)\bgive me (?:a )?moment\b")),
    ("pause",                 re.compile(r"(?i)\bpause\b")),
    ("on second thought",     re.compile(r"(?i)\bon (?:second|further) thought\b")),
    ("second guess",          re.compile(r"(?i)\bsecond-?guess(?:ing)?\b")),
    ("reconsider",            re.compile(r"(?i)\breconsider\b")),
    ("rethink",               re.compile(r"(?i)\bre-?think(?:ing)?\b")),

    # ───────── Explicit self-corrections / pivots ─────────
    ("actually",              re.compile(r"(?i)(?:^|\W)actually(?:\W|$)")),
    ("in fact",               re.compile(r"(?i)\bin fact\b")),
    ("rather",                re.compile(r"(?i)(?:^|\W)rather(?:\W|$)")),
    ("instead",               re.compile(r"(?i)(?:^|\W)instead(?:\W|$)")),
    ("instead_of",            re.compile(r"(?i)\binstead of\b")),
    ("better",                re.compile(r"(?i)(?:^|\W)better(?:\W|$)")),
    ("prefer",                re.compile(r"(?i)\bI (?:would )?prefer\b")),
    ("let me correct",        re.compile(r"(?i)\blet'?s? (?:correct|fix) (?:that|this)\b")),
    ("correction_keyword",    re.compile(r"(?i)\bcorrection\b")),
    ("correction_colon",      re.compile(r"(?i)\bcorrection:\b")),
    ("to correct",            re.compile(r"(?i)\bto correct\b")),
    ("fix that",              re.compile(r"(?i)\bfix (?:that|this)\b")),
    ("change to",             re.compile(r"(?i)\bchange (?:that|this)?\s*to\b")),
    ("switch to",             re.compile(r"(?i)\bswitch (?:to|over)\b")),
    ("replace with",          re.compile(r"(?i)\breplace (?:it|that|this)?\s*with\b")),
    ("try instead",           re.compile(r"(?i)\btry (?:this|that )?instead\b")),
    ("consider instead",      re.compile(r"(?i)\bconsider (?:instead|alternatively)\b")),
    ("alternate",             re.compile(r"(?i)\balternat(?:e|ive)\b")),
    ("new candidate",         re.compile(r"(?i)\bnew (?:candidate|answer|approach|plan)\b")),
    ("update_colon",          re.compile(r"(?i)\bupdate:\b")),

    # ───────── Negations of previous statement / immediate reversal ─────────
    ("no_comma",              re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+(?:that|this|it)\b")),
    ("nope",                  re.compile(r"(?i)\bnope\b")),
    ("nah",                   re.compile(r"(?i)\bnah\b")),
    ("never mind",            re.compile(r"(?i)\bnever mind\b|\bnvm\b")),
    ("disregard",             re.compile(r"(?i)\b(?:disregard|ignore) (?:that|this|the previous|above)\b")),
    ("scratch that",          re.compile(r"(?i)\bscratch that\b")),
    ("strike that",           re.compile(r"(?i)\bstrike that\b")),
    ("forget that",           re.compile(r"(?i)\bforget (?:that|this)\b")),
    ("I retract",             re.compile(r"(?i)\bI retract\b")),
    ("I take that back",      re.compile(r"(?i)\bI take (?:that|it) back\b")),
    ("I stand corrected",     re.compile(r"(?i)\bI stand corrected\b")),
    ("not X but Y",           re.compile(r"(?i)\bnot\s+\w+(?:\s+\w+)?\s*,?\s+(?:but|rather)\b")),

    # ───────── Admission of error / fault ─────────
    ("wrong_self",            re.compile(r"(?i)\bi (?:was|am) wrong\b")),
    ("wrong_generic",         re.compile(r"(?i)\bthat(?:'s| is)? wrong\b")),
    ("incorrect",             re.compile(r"(?i)\bincorrect\b|\bnot correct\b")),
    ("mistake_generic",       re.compile(r"(?i)\b(?:my )?mistake\b|\bI made a mistake\b")),
    ("my bad",                re.compile(r"(?i)\bmy bad\b")),
    ("oops",                  re.compile(r"(?i)\b(?:oops|whoops|uh[-\s]*oh)\b")),
    ("apologies",             re.compile(r"(?i)\bapolog(?:y|ies|ise|ize)\b")),
    ("erroneous",             re.compile(r"(?i)\berroneous\b")),
    ("error_on_my_part",      re.compile(r"(?i)\b(?:an|the) error (?:on|in) (?:my|the) (?:part|work)\b")),

    # ───────── Mis-* patterns (specific failure types) ─────────
    ("misread",               re.compile(r"(?i)\bmis-?read\b|\bI misread\b")),
    ("miscount",              re.compile(r"(?i)\bmis-?count(?:ed|ing)?\b")),
    ("miscalc",               re.compile(r"(?i)\bmis-?calculat(?:e|ed|ion)\b|\bcalc(?:ulation)? error\b")),
    ("misapply",              re.compile(r"(?i)\bmis-?appl(?:y|ied|ication)\b")),
    ("misparse",              re.compile(r"(?i)\bmis-?pars(?:e|ed|ing)\b")),
    ("misspell",              re.compile(r"(?i)\bmis-?spell(?:ed|ing)?\b|\bmisspelt\b|\bmisspelled\b")),
    ("misindex",              re.compile(r"(?i)\bmis-?index(?:ed|ing)?\b")),
    ("misuse_rule",           re.compile(r"(?i)\bmis-?us(?:e|ed|ing)\b")),
    ("confused_with",         re.compile(r"(?i)\bI (?:confused|mixed up) .* with\b", re.S)),
    ("conflated",             re.compile(r"(?i)\bI conflated\b")),
    ("typo",                  re.compile(r"(?i)\btypo\b")),
    ("off by one",            re.compile(r"(?i)\boff[-\s]?by[-\s]?one\b")),

    # ───────── Constraint/length/pattern mismatch (xword-friendly) ─────────
    ("doesnt fit",            re.compile(r"(?i)\bdoes(?:n'?t| not) (?:fit|match)(?: length| pattern)?\b")),
    ("letters dont fit",      re.compile(r"(?i)\bletters? do(?:es)?n'?t (?:fit|match)\b")),
    ("pattern mismatch",      re.compile(r"(?i)\bpattern (?:mis)?match\b")),
    ("length mismatch",       re.compile(r"(?i)\blength (?:mis)?match\b")),
    ("too many letters",      re.compile(r"(?i)\btoo many letters\b")),
    ("too few letters",       re.compile(r"(?i)\b(?:not enough|too few) letters\b")),
    ("wrong length",          re.compile(r"(?i)\bwrong length\b")),
    ("violates enumeration",  re.compile(r"(?i)\bviolates? (?:the )?enumeration\b")),
    ("doesnt parse",          re.compile(r"(?i)\bdoes(?:n'?t| not) parse\b")),
    ("definition mismatch",   re.compile(r"(?i)\bdefinition (?:doesn'?t|does not) match\b")),
    ("not an anagram of",     re.compile(r"(?i)\bnot an anagram of\b")),
    ("anagram doesnt work",   re.compile(r"(?i)\banagram (?:doesn'?t|does not) (?:work|fit)\b")),
    ("fodder mismatch",       re.compile(r"(?i)\bfodder (?:doesn'?t|does not) (?:match|fit)\b")),

    # ───────── Logical contradiction / impossibility ─────────
    ("contradiction",         re.compile(r"(?i)\bcontradict(?:s|ion|ory)\b")),
    ("inconsistent",          re.compile(r"(?i)\binconsistent\b")),
    ("cant be",               re.compile(r"(?i)\bcan'?t be\b|\bcannot be\b")),
    ("impossible",            re.compile(r"(?i)\bimpossible\b")),
    ("doesnt make sense",     re.compile(r"(?i)\bdoes(?:n'?t| not) make sense\b")),
    ("doesnt add up",         re.compile(r"(?i)\bdoes(?:n'?t| not) add up\b")),
    ("cannot both",           re.compile(r"(?i)\bcan(?:not|'?t) both\b")),
    ("leads to",              re.compile(r"(?i)\bleads to (?:a )?contradiction\b")),
    ("this implies not",      re.compile(r"(?i)\bthis implies .* (?:is|are) not\b", re.S)),

    # ───────── Re-check / review / backtrack ─────────
    ("re-check",              re.compile(r"(?i)\bre-?check(?:ing|ed)?\b")),
    ("double-check",          re.compile(r"(?i)\bdouble-?check(?:ing|ed)?\b")),
    ("check again",           re.compile(r"(?i)\bcheck(?:ing)? again\b")),
    ("re-evaluate",           re.compile(r"(?i)\bre-?evaluat(?:e|ed|ing|ion)\b")),
    ("re-examine",            re.compile(r"(?i)\bre-?examin(?:e|ed|ing|ation)\b")),
    ("on review",             re.compile(r"(?i)\b(?:on|upon) (?:review|reflection|reconsideration)\b")),
    ("backtrack",             re.compile(r"(?i)\bbacktrack(?:ing|ed)?\b")),
    ("start over",            re.compile(r"(?i)\bstart over\b|\brestart\b|\breset\b|\bfrom scratch\b")),

    # ───────── “Previous idea was X, but …” templates ─────────
    ("i_thought_but",         re.compile(r"(?i)\bI (?:first|initially|originally) thought\b.*\b(?:but|however)\b", re.S)),
    ("previously_but",        re.compile(r"(?i)\bpreviously\b.*\b(?:but|however)\b", re.S)),
    ("earlier_but",           re.compile(r"(?i)\bearlier\b.*\b(?:but|however)\b", re.S)),
    ("however+fix",           re.compile(r"(?i)\bhowever\b.*\b(?:correct|fix|instead|rather|change)\b", re.S)),
    ("but_instead",           re.compile(r"(?i)\bbut\b.*\binstead\b", re.S)),
    ("but_rather",            re.compile(r"(?i)\bbut\b.*\brather\b", re.S)),

    # ───────── Oversight / omission admissions ─────────
    ("i forgot",              re.compile(r"(?i)\bI forgot\b")),
    ("i missed",              re.compile(r"(?i)\bI (?:missed|overlooked)\b")),
    ("i didnt notice",        re.compile(r"(?i)\bI did(?:n'?t| not) notice\b")),
    ("i ignored",             re.compile(r"(?i)\bI (?:ignored|skipped)\b")),
    ("i misremembered",       re.compile(r"(?i)\bI mis-?remembered\b")),
    ("i misheard",            re.compile(r"(?i)\bI mis-?heard\b")),

    # ───────── Directional errors / swaps ─────────
    ("reversed",              re.compile(r"(?i)\brevers(?:e|ed|ing)\b")),
    ("got backwards",         re.compile(r"(?i)\bgot .* backwards\b", re.S)),
    ("swapped",               re.compile(r"(?i)\bswapp?ed\b")),
    ("mixed up",              re.compile(r"(?i)\bmix(?:ed)? up\b")),

    # ───────── “turns out / realization” ─────────
    ("turns out",             re.compile(r"(?i)\bturns out\b")),
    ("i realize",             re.compile(r"(?i)\bI (?:now )?real(?:i[sz]e|ising|izing)\b")),
    ("on reflection",         re.compile(r"(?i)\bon reflection\b")),
    ("after all",             re.compile(r"(?i)\bafter all\b")),

    # ───────── Generic “No, that doesn’t …” templates ─────────
    ("no_that_doesnt",        re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+that (?:doesn'?t|does not)\b")),
    ("no_this_doesnt",        re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+this (?:doesn'?t|does not)\b")),
    ("no_it_doesnt",          re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+it (?:doesn'?t|does not)\b")),

    # ───────── “This fails because …” ─────────
    ("fails_because",         re.compile(r"(?i)\bfails? because\b")),
    ("won't work",            re.compile(r"(?i)\bwon'?t work\b")),
    ("not working",           re.compile(r"(?i)\bnot working\b")),
    ("dead end",              re.compile(r"(?i)\bdead end\b")),

    # Keep originals (for completeness & back-compat)
    ("original_no_comma",     re.compile(r"(?i)(?:^|\W)no[,!\.]?\s")),
]
def _extract_think(txt: str) -> Optional[str]:
    m = RE_THINK.search(txt or "")
    return m.group(1).strip() if m else None

def _find_shift_cues(think: str) -> Tuple[List[str], Optional[int]]:
    if not think:
        return [], None
    hits = []
    first_pos = None
    for name, pat in SHIFT_CAND_PATTERNS:
        m = pat.search(think)
        if m:
            hits.append(name)
            pos = m.start()
            if first_pos is None or pos < first_pos:
                first_pos = pos
    return hits, first_pos

def _json_from_text(s: str) -> Optional[Dict[str, Any]]:
    # Try to extract a JSON object from arbitrary text.
    s = s.strip()
    # Fast path: starts with '{' and ends with '}'
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass
    # Fallback: find the first {...} block
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s[i:j+1])
        except Exception:
            return None
    return None

# ───────────────────── DeepSeek call ─────────────────────────────
PROMPT_SYSTEM = (
    "You are a careful annotator of single-pass reasoning transcripts. "
    "Your task is to judge whether the writer makes a CLEAR, EXPLICIT 'shift in reasoning' "
    "within <think>…</think>. "
    "A TRUE label requires BOTH: "
    "(A) an explicit cue (e.g., 'wait', 'hold on', 'on second thought', 'actually', "
    "'scratch that', 'I misread', 're-check', 'doesn't fit/match', 'contradiction'), "
    "AND (B) a material revision of the earlier idea (reject/correct an initial hypothesis, pick a new candidate, fix a contradiction). "
    "Do NOT mark TRUE for rhetorical transitions, hedging, or generic connectives without an actual correction. "
    "Be conservative; these events are rare."
)

PROMPT_USER_TEMPLATE = """Problem/Clue (if available):
{problem}

PASS-1 <think> (truncated if long):
{think}

Heuristic cue candidates (may be empty): {cues}
first_marker_pos: {pos}

Return ONLY a compact JSON object with keys:
- shift_in_reasoning: true|false
- confidence: "low"|"medium"|"high"
- markers_found: string[]       (verbatim lexical cues you relied on)
- first_marker_index: integer   (character offset into <think>, -1 if absent)
- before_excerpt: string        (≤120 chars ending right before the first marker)
- after_excerpt: string         (≤140 chars starting at the first marker)
- explanation_short: string     (≤140 chars justification)
"""

def deepseek_judge_shift(problem: str, think: str, cue_names: List[str], pos: Optional[int]) -> Dict[str, Any]:
    client = _client_lazy()
    user_content = PROMPT_USER_TEMPLATE.format(
        problem=_clamp(problem or "(unknown)"),
        think=_clamp(think or ""),
        cues=", ".join(cue_names) if cue_names else "(none)",
        pos=(-1 if pos is None else pos),
    )

    # Prefer Responses API if available; fallback to Chat Completions.
    try:
        # Newer Responses API
        resp = client.responses.create(
            model=DEEPSEEK_MODEL,
            input=[{"role": "system", "content": PROMPT_SYSTEM},
                   {"role": "user", "content": user_content}],
            temperature=0.0,
            max_output_tokens=500,
        )
        content = resp.output_text  # unified accessor in new SDK
    except Exception:
        try:
            # Chat Completions fallback
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=0.0,
                max_tokens=500,
                messages=[{"role": "system", "content": PROMPT_SYSTEM},
                          {"role": "user", "content": user_content}],
            )
            content = resp.choices[0].message.content
        except Exception as e:
            _dump_filtered(user_content + "\n\n[ERROR] " + repr(e))
            return {
                "shift_in_reasoning": False,
                "confidence": "low",
                "markers_found": [],
                "first_marker_index": -1 if pos is None else int(pos),
                "before_excerpt": "",
                "after_excerpt": "",
                "explanation_short": "Model call failed; defaulting to FALSE."
            }

    obj = _json_from_text(content or "")
    if not isinstance(obj, dict):
        _dump_filtered(user_content + "\n\n[UNPARSEABLE]\n" + (content or ""))
        return {
            "shift_in_reasoning": False,
            "confidence": "low",
            "markers_found": [],
            "first_marker_index": -1 if pos is None else int(pos),
            "before_excerpt": "",
            "after_excerpt": "",
            "explanation_short": "Unparseable response; default FALSE."
        }

    # Guardrails: must include an explicit cue; otherwise force FALSE.
    markers = obj.get("markers_found") or []
    has_explicit = bool(cue_names or markers)
    if not has_explicit:
        obj["shift_in_reasoning"] = False
        obj.setdefault("confidence", "low")
        obj.setdefault("explanation_short", "No explicit cue; conservative FALSE.")

    return obj

# ───────────────────── File scanning / update ─────────────────────
def nat_step_from_path(path: str) -> Optional[int]:
    m = re.search(r"step(\d+)", path)
    return int(m.group(1)) if m else None

def scan_jsonl(root: str, split: Optional[str]) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split and split not in fn:
                continue
            files.append(os.path.join(dp, fn))
    files.sort(key=lambda p: (nat_step_from_path(p) or 0, p))
    files.sort(reverse=True)
    return files

def record_id_for_logs(obj: Dict[str, Any]) -> str:
    return (
        obj.get("row_key")
        or obj.get("problem")
        or obj.get("clue")
        or f"idx={obj.get('dataset_index','?')}"
    )

def annotate_file(path: str, seed: int, max_calls: Optional[int], dry_run: bool, jitter: float):
    logging.info("Annotating: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse all JSON once.
    records: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            records.append(json.loads(ln))
        except Exception:
            # Keep the raw line to preserve file size/order; will pass it through unchanged.
            records.append({"__raw__": ln})

    # Identify candidates (missing annotation).
    todo_idxs: List[int] = []
    for i, rec in enumerate(records):
        if "__raw__" in rec:
            continue
        p1 = rec.get("pass1") or {}
        if "shift_in_reasoning_v1" in p1:
            continue  # already annotated
        # Need full output with <think>...</think>
        out = p1.get("output")
        if not out:
            continue
        think = _extract_think(out)
        if not think:
            # No think block; stamp FALSE to be explicit.
            p1["shift_in_reasoning_v1"] = False
            rec["pass1"] = p1
            continue
        # Prefilter cues. If none, we'll hard-false (rare by policy).
        cues, pos = _find_shift_cues(think)
        p1["_shift_prefilter_markers"] = cues  # keep for audit
        p1["_shift_prefilter_pos"] = pos
        rec["pass1"] = p1
        todo_idxs.append(i)

    # Randomize processing order
    rng = random.Random(seed)
    rng.shuffle(todo_idxs)
    if max_calls is not None:
        todo_idxs = todo_idxs[:max_calls]

    calls = 0
    for idx in todo_idxs:
        rec = records[idx]
        p1 = rec.get("pass1") or {}
        out = p1.get("output") or ""
        think = _extract_think(out) or ""
        problem = rec.get("problem") or rec.get("clue") or ""

        cues = p1.get("_shift_prefilter_markers") or []
        pos  = p1.get("_shift_prefilter_pos")

        # Strict policy: if NO cues at all, mark FALSE; do NOT ask the model.
        if not cues:
            p1["shift_in_reasoning_v1"] = False
            p1["shift_markers_v1"] = []
            p1["shift_first_marker_char"] = -1
            p1["shift_before_excerpt"] = ""
            p1["shift_after_excerpt"] = ""
            p1["shift_rationale_gpt"] = "No explicit cue; conservative FALSE."
            p1["shift_rationale_gpt_model"] = DEEPSEEK_MODEL
            p1["shift_rationale_gpt_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rec["pass1"] = p1
            continue

        if dry_run:
            logging.info("DRY-RUN would annotate idx=%s id=%s", idx, record_id_for_logs(rec))
            continue

        # Call DeepSeek to verify a real shift (material revision).
        with timed(f"deepseek_call idx={idx}"):
            result = deepseek_judge_shift(problem, think, cues, pos)

        # Stamp fields
        p1["shift_in_reasoning_v1"]   = bool(result.get("shift_in_reasoning", False))
        p1["shift_markers_v1"]        = list(result.get("markers_found", []) or cues)
        p1["shift_first_marker_char"] = int(result.get("first_marker_index", -1 if pos is None else pos))
        p1["shift_before_excerpt"]    = _clamp(result.get("before_excerpt", ""), 240)
        p1["shift_after_excerpt"]     = _clamp(result.get("after_excerpt", ""), 280)
        p1["shift_rationale_gpt"]     = _clamp(result.get("explanation_short", ""), 300)
        p1["shift_rationale_gpt_model"] = DEEPSEEK_MODEL
        p1["shift_rationale_gpt_time"]  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["pass1"] = p1

        calls += 1
        if jitter > 0:
            time.sleep(rng.uniform(0.0, jitter))

    # Write atomic replace (preserve original order)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            if "__raw__" in rec:
                f.write(rec["__raw__"])
            else:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
    os.replace(tmp, path)
    logging.info("Updated %s (DeepSeek calls: %d)", path, calls)

# ───────────────────── CLI ─────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Directory containing step*/.../*.jsonl")
    ap.add_argument("--split", default=None, help="Filter filenames that contain this substring (e.g., 'test').")
    ap.add_argument("--seed", type=int, default=1234, help="Shuffle seed for random processing order.")
    ap.add_argument("--max_calls", type=int, default=None, help="Optional cap on DeepSeek calls.")
    ap.add_argument("--dry_run", action="store_true", help="Discover candidates but do not call DeepSeek or write changes.")
    ap.add_argument("--jitter", type=float, default=0.25, help="Max random sleep (seconds) between calls; 0 to disable.")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    files = scan_jsonl(args.results_root, args.split)
    if not files:
        print("No JSONL files found; check path/split.", file=sys.stderr)
        sys.exit(1)

    for path in files:
        annotate_file(path, seed=args.seed, max_calls=args.max_calls, dry_run=args.dry_run, jitter=args.jitter)

if __name__ == "__main__":
    main()
