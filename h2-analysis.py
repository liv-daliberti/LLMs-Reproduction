#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H2: Are Aha! Moments Important During Different Stages of Training?
-------------------------------------------------------------------

Per-step model:
    correct ~ C(problem) + aha + uncertainty_std [+ aha:uncertainty_std]

Where:
  - correct         : pass1.is_correct_pred (0/1)
  - aha             : GPT-based shift flag if available; else native cue (configurable)
  - uncertainty     : entropy; choose --unc_field {answer|overall|think}
  - uncertainty_std : z-score across the full dataset (global scaling)

This extended version adds:
  • Diagnostic panel (aha vs uncertainty, uncertainty vs step, aha vs step)
  • Penalized step-wise GLMs (ridge fallback; "firth" flag currently uses ridge)
  • Interaction (aha × uncertainty) per step + sign/size summary
  • AME(aha) with bootstrap CIs; FDR/BH share of significant steps
  • Balance checks by step (counts, mean entropy per aha group)
  • Pooled GLM with step fixed effects (+ aha×step) using ridge
  • Dense grid AME(aha | u) across uncertainty (pooled plot)

Outputs (default under <results_root>/h2_analysis):
  - h2_pass1_samples.csv
  - h2_step_regression.csv
  - h2_ame_grid.csv
  - h2_balance_by_step.csv
  - h2_fdr_summary.txt  (FDR share for aha; includes interaction if computed)
  - h2_interaction_summary.txt
  - h2_pooled_aha_by_step.csv

  - h2_diag_panel.png
  - aha_coef_vs_step.png
  - aha_ame_vs_step.png
  - aha_ame_with_ci.png
  - uncertainty_coef_vs_step.png
  - naive_delta_vs_step.png
  - aha_ame_grid.png
  - h2_pooled_aha_by_step.png
  - (optional overlays if --compare_native): aha_coef_vs_step_compare.png, aha_ame_vs_step_compare.png

Usage:
  python h2-analysis.py /path/to/results_root --split test
  [--out_dir /tmp/h2] [--min_step 100 --max_step 600]
  [--unc_field answer|overall|think] [--aha_source gpt|native]
  [--interaction] [--compare_native]
  [--penalty ridge|none|firth] [--l2 1.0]
  [--bootstrap_ame 200] [--ame_grid 9] [--fdr_alpha 0.05]
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

STEP_PAT = re.compile(r"step(\d+)", re.I)
TEMP_PATS = [
    re.compile(r"temp(?:erature)?[_-]?([0-9]*\.?[0-9]+)", re.I),
]

# ----------------------------- file scanning -----------------------------
def _get_param(res, name: str, default=np.nan) -> float:
    """Robustly fetch a coefficient by name from statsmodels results, even for fit_regularized."""
    try:
        p = getattr(res, "params", None)
        if p is None:
            return default
        # If params is a pandas Series with names
        if isinstance(p, pd.Series):
            return float(p.get(name, default))
        # Otherwise map via model.exog_names
        names = getattr(getattr(res, "model", None), "exog_names", None)
        if names is not None:
            try:
                idx = names.index(name)
                return float(p[idx])
            except Exception:
                return default
    except Exception:
        pass
    return default

def nat_step_from_path(path: str) -> Optional[int]:
    m = STEP_PAT.search(path)
    return int(m.group(1)) if m else None

def maybe_temp_from_path(path: str) -> Optional[float]:
    for pat in TEMP_PATS:
        m = pat.search(path)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

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

# ----------------------------- helpers -----------------------------

def coerce_bool(x) -> Optional[int]:
    if x is None: return None
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, np.integer)): return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    return int(bool(x))

def _get_aha_gpt(p1: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    """Prefer GPT/LLM-labeled shift flags if present (various aliases)."""
    candidates = [
        ("p1", "shift_in_reasoning_v1"),
        ("p1", "shift_llm"),
        ("p1", "shift_gpt"),
        ("p1", "pivot_llm"),
        ("p1", "rechecked"),
        ("root", "rechecked"),  # older graded logs
    ]
    for loc, key in candidates:
        v = p1.get(key) if loc == "p1" else rec.get(key)
        if v is None: continue
        out = coerce_bool(v)
        if out is not None: return int(out)
    return None

def _get_aha_native(p1: Dict[str, Any]) -> Optional[int]:
    """Native reconsider cue, ignoring any injected cue on PASS-1."""
    aha_raw = coerce_bool(p1.get("has_reconsider_cue"))
    markers = p1.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers):
        return 0
    return 0 if aha_raw is None else int(aha_raw)

def _choose_uncertainty(p1: Dict[str, Any], pref: str = "answer") -> Optional[float]:
    """Pick uncertainty (entropy) per preference with sensible fallbacks."""
    if pref == "answer":
        x = p1.get("entropy_answer")
        if x is None: x = p1.get("entropy")
        if x is None: x = p1.get("entropy_think")
        return float(x) if x is not None else None
    if pref == "overall":
        x = p1.get("entropy")
        if x is None: x = p1.get("entropy_answer")
        if x is None: x = p1.get("entropy_think")
        return float(x) if x is not None else None
    if pref == "think":
        x = p1.get("entropy_think")
        if x is None: x = p1.get("entropy")
        if x is None: x = p1.get("entropy_answer")
        return float(x) if x is not None else None
    return None

# ----------------------------- loader -----------------------------

def load_pass1_rows(files: List[str], unc_field: str, aha_source: str) -> pd.DataFrame:
    rows = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        temp_from_path = maybe_temp_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                p1 = rec.get("pass1") or {}
                if not p1: continue

                # IDs
                problem = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if problem is None:
                    di = rec.get("dataset_index")
                    problem = f"idx:{di}" if di is not None else "unknown"
                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None: continue

                # Outcome
                corr_raw = coerce_bool(p1.get("is_correct_pred"))
                if corr_raw is None: continue
                correct = int(corr_raw)

                # Aha
                aha_gpt = _get_aha_gpt(p1, rec)
                aha_native = _get_aha_native(p1)
                if aha_source == "gpt":
                    aha = aha_gpt if aha_gpt is not None else aha_native
                else:
                    aha = aha_native if aha_native is not None else aha_gpt
                if aha is None:  # no label at all -> drop to keep design comparable
                    continue
                aha = int(aha)

                # Uncertainty
                unc = _choose_uncertainty(p1, unc_field)
                if unc is None:  # can't control for uncertainty -> drop
                    continue

                # Temperature (if present)
                t = (rec.get("temperature") or p1.get("temperature") or
                     rec.get("config", {}).get("temperature") or temp_from_path)

                rows.append({
                    "problem": str(problem),
                    "step": int(step),
                    "sample_idx": rec.get("sample_idx"),
                    "correct": correct,
                    "aha": aha,
                    "uncertainty": float(unc),
                    "temperature": None if t is None else float(t),
                    "source_file": path,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable PASS-1 rows found (missing aha and/or uncertainty).")
    return df

# ----------------------------- modeling -----------------------------
def _predict_from_formula(res, model, df_new):
    """
    Predict P(correct=1) for df_new using the original Patsy design_info
    (works for both MLE and ridge-regularized fits).
    """
    import numpy as np
    try:
        # Will work when res/model kept the original design info
        return np.asarray(res.predict(df_new))
    except Exception:
        pass

    try:
        from patsy import build_design_matrices
        design_info = getattr(getattr(model, "data", None), "design_info", None)
        if design_info is None:
            raise RuntimeError("missing design_info")
        X = build_design_matrices([design_info], df_new, return_type="dataframe")[0]
        linpred = np.dot(np.asarray(X), np.asarray(res.params))
        return model.family.link.inverse(linpred)
    except Exception:
        # Last resort: assume df_new already numeric/exog
        return np.asarray(model.predict(res.params, df_new))


def _fit_glm_with_ridge_if_needed(d: pd.DataFrame, formula: str, l2: float):
    """
    Fit Binomial GLM by MLE; if unstable or fails, fall back to ridge-penalized GLM.
    Returns (res, model, used_penalty:str in {"none","ridge"}).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    used = "none"
    try:
        res = model.fit(cov_type="HC1")
        # crude instability detection (separation -> huge |coef|)
        if not np.isfinite(res.params).all() or ("aha" in res.params.index and abs(res.params["aha"]) > 10):
            raise RuntimeError("Unstable MLE; switching to ridge.")
    except Exception:
        res = model.fit_regularized(alpha=float(l2), L1_wt=0.0)
        used = "ridge"
    return res, model, used

def _predict_from_formula(res, model, df_new):
    """
    Predict P(correct=1) for df_new using the original Patsy design_info.
    Works for both MLE and ridge-regularized fits.
    """
    # 1) Try results' own predict (handles formula + design_info when available)
    try:
        return np.asarray(res.predict(df_new))
    except Exception:
        pass

    # 2) Build design matrix using the fit-time design_info (robust path)
    try:
        from patsy import build_design_matrices
        design_info = getattr(model, "data", None)
        design_info = getattr(design_info, "design_info", None)
        if design_info is None:
            raise RuntimeError("missing design_info")
        X = build_design_matrices([design_info], df_new, return_type="dataframe")[0]
        linpred = np.dot(np.asarray(X), np.asarray(res.params))
        # GLM inverse link → mean
        return model.family.link.inverse(linpred)
    except Exception:
        # 3) Last resort: assume df_new is already a numeric design matrix
        return np.asarray(model.predict(res.params, df_new))

# --- tweak fit_stepwise_glms to avoid unstable MLE when penalty='ridge' or 'firth' ---
def fit_stepwise_glms(df: pd.DataFrame,
                      out_dir: str,
                      interaction: bool = False,
                      penalty: str = "ridge",
                      l2: float = 1.0,
                      bootstrap_ame: int = 200,
                      ame_grid: int = 9,
                      fdr_alpha: float = 0.05) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    steps = sorted(df["step"].unique().tolist())
    rows, ame_grid_rows, bal_rows = [], [], []

    ame_grid = int(max(3, ame_grid))
    u_grid = np.linspace(-2.0, 2.0, ame_grid)

    for s in steps:
        d = df[df["step"] == s].copy()
        if d.empty:
            continue

        # Balance stats
        g0 = d[d["aha"] == 0]; g1 = d[d["aha"] == 1]
        bal_rows.append({
            "step": s,
            "n": len(d),
            "n_aha0": int(len(g0)),
            "n_aha1": int(len(g1)),
            "mean_unc_aha0": float(g0["uncertainty"].mean()) if len(g0) else np.nan,
            "mean_unc_aha1": float(g1["uncertainty"].mean()) if len(g1) else np.nan,
            "aha_ratio": float(d["aha"].mean()),
        })

        # Must have variation in aha
        if d["aha"].nunique() < 2:
            naive_delta = (g1["correct"].mean() - g0["correct"].mean()) if (len(g0) and len(g1)) else np.nan
            rows.append({
                "step": s, "n": len(d),
                "penalty": "n/a",
                "aha_coef": np.nan, "aha_se": np.nan, "aha_z": np.nan, "aha_p": np.nan,
                "aha_ame": np.nan, "aha_ame_lo": np.nan, "aha_ame_hi": np.nan,
                "inter_coef": np.nan, "inter_se": np.nan, "inter_z": np.nan, "inter_p": np.nan,
                "unc_coef": np.nan, "unc_se": np.nan, "unc_z": np.nan, "unc_p": np.nan,
                "acc": d["correct"].mean(), "aha_ratio": d["aha"].mean(),
                "mean_uncertainty": d["uncertainty"].mean(),
                "naive_delta": naive_delta,
            })
            continue

        # Formula
        formula = "correct ~ C(problem) + aha + uncertainty_std"
        if interaction:
            formula += " + aha:uncertainty_std"

        # Fit
        if penalty in ("ridge", "firth"):   # go straight to ridge to avoid IRLS warnings
            res, model, used = _fit_glm_force_ridge(d, formula, l2)
        elif penalty == "none":
            res, model, used = _fit_glm_with_ridge_if_needed(d, formula, 0.0)
            used = "none" if used == "none" else "ridge"
        else:
            res, model, used = _fit_glm_force_ridge(d, formula, l2)

        # Coefs
        b_aha = _get_param(res, "aha", np.nan)
        b_unc = _get_param(res, "uncertainty_std", np.nan)
        b_int = np.nan
        for key in ("aha:uncertainty_std", "uncertainty_std:aha"):
            val = _get_param(res, key, np.nan)
            if np.isfinite(val):
                b_int = val
                break

        # SE/z/p (may be NaN for regularized fits)
        try:
            se_aha = float(getattr(res, "bse", pd.Series()).get("aha", np.nan))
            se_unc = float(getattr(res, "bse", pd.Series()).get("uncertainty_std", np.nan))
            se_int = np.nan
            for key in ("aha:uncertainty_std", "uncertainty_std:aha"):
                se_int_try = getattr(res, "bse", pd.Series()).get(key, np.nan)
                if np.isfinite(se_int_try):
                    se_int = float(se_int_try)
                    break
            z_aha = b_aha / se_aha if np.isfinite(se_aha) and se_aha else np.nan
            z_unc = b_unc / se_unc if np.isfinite(se_unc) and se_unc else np.nan
            z_int = b_int / se_int if np.isfinite(se_int) and se_int else np.nan
            p_aha = float(getattr(res, "pvalues", pd.Series()).get("aha", np.nan))
            p_unc = float(getattr(res, "pvalues", pd.Series()).get("uncertainty_std", np.nan))
            p_int = np.nan
            for key in ("aha:uncertainty_std", "uncertainty_std:aha"):
                pv = getattr(res, "pvalues", pd.Series()).get(key, np.nan)
                if np.isfinite(pv):
                    p_int = float(pv)
                    break
        except Exception:
            se_aha = se_unc = se_int = z_aha = z_unc = z_int = p_aha = p_unc = p_int = np.nan

        # AME(aha) at mean uncertainty
        base = d.copy()
        ubar = float(base["uncertainty_std"].mean())
        base["uncertainty_std"] = ubar
        d1 = base.copy(); d1["aha"] = 1
        d0 = base.copy(); d0["aha"] = 0
        p1 = _predict_from_formula(res, model, d1)
        p0 = _predict_from_formula(res, model, d0)
        ame = float(np.mean(p1 - p0))

        # Bootstrap CI for AME
        ame_lo = ame_hi = np.nan
        if int(bootstrap_ame) > 0 and len(d) > 10:
            rng = np.random.default_rng(0)
            B = int(bootstrap_ame)
            bs = np.empty(B, dtype=float)
            idx = np.arange(len(d))
            for b in range(B):
                take = rng.choice(idx, size=len(d), replace=True)
                dd = d.iloc[take].copy()
                r2, m2, _ = _fit_glm_force_ridge(dd, formula, l2)  # force ridge in bootstrap too
                bb = dd.copy()
                ubar2 = float(bb["uncertainty_std"].mean()); bb["uncertainty_std"] = ubar2
                d1b = bb.copy(); d0b = bb.copy()
                d1b["aha"] = 1; d0b["aha"] = 0
                p1b = _predict_from_formula(r2, m2, d1b)
                p0b = _predict_from_formula(r2, m2, d0b)
                bs[b] = float(np.mean(p1b - p0b))
            ame_lo, ame_hi = np.nanpercentile(bs, [2.5, 97.5])

        # AME grid across uncertainty
        for u in u_grid:
            bb = d.copy(); bb["uncertainty_std"] = u
            d1g = bb.copy(); d1g["aha"] = 1
            d0g = bb.copy(); d0g["aha"] = 0
            p1g = _predict_from_formula(res, model, d1g)
            p0g = _predict_from_formula(res, model, d0g)
            ame_u = float(np.mean(p1g - p0g))
            ame_grid_rows.append({"step": s, "u": float(u), "ame": ame_u})

        naive_delta = d.loc[d["aha"] == 1, "correct"].mean() - d.loc[d["aha"] == 0, "correct"].mean()

        rows.append({
            "step": s, "n": len(d), "penalty": used,
            "aha_coef": float(b_aha), "aha_se": float(se_aha), "aha_z": float(z_aha), "aha_p": float(p_aha),
            "aha_ame": float(ame), "aha_ame_lo": float(ame_lo), "aha_ame_hi": float(ame_hi),
            "inter_coef": float(b_int), "inter_se": float(se_int), "inter_z": float(z_int), "inter_p": float(p_int),
            "unc_coef": float(b_unc), "unc_se": float(se_unc), "unc_z": float(z_unc), "unc_p": float(p_unc),
            "acc": d["correct"].mean(), "aha_ratio": d["aha"].mean(),
            "mean_uncertainty": d["uncertainty"].mean(),
            "naive_delta": float(naive_delta),
        })

    out = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    out.to_csv(os.path.join(out_dir, "h2_step_regression.csv"), index=False)

    ame_grid_df = pd.DataFrame(ame_grid_rows)
    ame_grid_df.to_csv(os.path.join(out_dir, "h2_ame_grid.csv"), index=False)

    bal = pd.DataFrame(bal_rows).sort_values("step")
    bal.to_csv(os.path.join(out_dir, "h2_balance_by_step.csv"), index=False)

    # FDR/BH
    lines = []
    mask_aha = out["aha_p"].notna()
    if mask_aha.any():
        reject_aha, p_adj_aha, _, _ = multipletests(out.loc[mask_aha, "aha_p"].values,
                                                    alpha=float(fdr_alpha), method="fdr_bh")
        share_aha = float(np.mean(reject_aha))
        lines.append(f"FDR (BH) alpha={fdr_alpha}: share of steps significant for aha = {share_aha:.3f}")
    else:
        lines.append("FDR: no finite p-values for aha (likely ridge used widely).")

    if interaction and "inter_p" in out.columns:
        mask_int = out["inter_p"].notna()
        if mask_int.any():
            reject_int, p_adj_int, _, _ = multipletests(out.loc[mask_int, "inter_p"].values,
                                                        alpha=float(fdr_alpha), method="fdr_bh")
            share_int = float(np.mean(reject_int))
            lines.append(f"FDR (BH) alpha={fdr_alpha}: share of steps significant for aha×unc = {share_int:.3f}")
        else:
            lines.append("FDR: no finite p-values for interaction (likely ridge used widely).")

    with open(os.path.join(out_dir, "h2_fdr_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    if interaction:
        inter_series = out["inter_coef"].replace([np.inf, -np.inf], np.nan).dropna()
        pos = float((inter_series > 0).mean()) if not inter_series.empty else np.nan
        neg = float((inter_series < 0).mean()) if not inter_series.empty else np.nan
        mean_size = float(inter_series.abs().mean()) if not inter_series.empty else np.nan
        with open(os.path.join(out_dir, "h2_interaction_summary.txt"), "w", encoding="utf-8") as fh:
            fh.write("Interaction (aha×uncertainty_std) summary across steps\n")
            fh.write(f"  steps with estimate: {len(inter_series)} / {len(out)}\n")
            fh.write(f"  fraction positive:   {pos:.3f}\n")
            fh.write(f"  fraction negative:   {neg:.3f}\n")
            fh.write(f"  mean |coef|:         {mean_size:.4f}\n")

    return out

def _fit_glm_force_ridge(d, formula: str, l2: float):
    """
    Fit Binomial GLM with ridge (L2) directly to avoid MLE IRLS overflows.
    Returns (res, model, used='ridge').
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    # Directly use regularized fit: no IRLS iterations -> no exp/1/v^2 overflow
    res = model.fit_regularized(alpha=float(l2), L1_wt=0.0)
    return res, model, "ridge"

# ----------------------------- plotting -----------------------------

def _lineplot(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def _lineplot_multi(xs, ys_list, labels, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
    for y, lab in zip(ys_list, labels):
        ax.plot(xs, y, marker="o", label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# --- replace your plot_diag_panel with this (fixes Matplotlib deprecation) ---
def plot_diag_panel(df: pd.DataFrame, out_dir: str):
    """Single figure: (1) Uncertainty vs Aha, (2) Uncertainty vs step, (3) Aha vs step."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=140)

    # (1) Uncertainty vs Aha
    u0 = df.loc[df["aha"] == 0, "uncertainty_std"].values
    u1 = df.loc[df["aha"] == 1, "uncertainty_std"].values
    try:
        axes[0].boxplot([u0, u1], tick_labels=["aha=0", "aha=1"], showfliers=False)
    except TypeError:
        # Back-compat for older Matplotlib
        axes[0].boxplot([u0, u1], labels=["aha=0", "aha=1"], showfliers=False)
    axes[0].set_title("Uncertainty vs Aha")
    axes[0].set_ylabel("uncertainty_std")

    # (2) Mean uncertainty vs step
    m = df.groupby("step", as_index=False).agg(mu=("uncertainty_std", "mean"))
    axes[1].plot(m["step"], m["mu"], marker="o")
    axes[1].set_title("Uncertainty vs step")
    axes[1].set_xlabel("Training step"); axes[1].set_ylabel("mean uncertainty_std")
    axes[1].grid(True, alpha=0.3)

    # (3) Aha ratio vs step
    ar = df.groupby("step", as_index=False).agg(r=("aha", "mean"))
    axes[2].plot(ar["step"], ar["r"], marker="o")
    axes[2].set_title("Aha vs step")
    axes[2].set_xlabel("Training step"); axes[2].set_ylabel("P(aha=1)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "h2_diag_panel.png"))
    plt.close(fig)

def plot_ame_with_ci(reg: pd.DataFrame, out_dir: str):
    need = {"aha_ame", "aha_ame_lo", "aha_ame_hi"}
    if not need.issubset(reg.columns): return
    reg = reg.dropna(subset=["aha_ame"])
    if reg.empty: return
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    ax.plot(reg["step"], reg["aha_ame"], marker="o", label="AME(aha)")
    if reg[["aha_ame_lo", "aha_ame_hi"]].notna().all().all():
        ax.fill_between(reg["step"], reg["aha_ame_lo"], reg["aha_ame_hi"], alpha=0.2, label="95% CI")
    ax.set_xlabel("Training step"); ax.set_ylabel("AME(aha)"); ax.set_title("Aha AME with bootstrap CI")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aha_ame_with_ci.png"))
    plt.close(fig)

def plot_ame_grid(ame_grid_df: pd.DataFrame, out_dir: str):
    if ame_grid_df.empty: return
    # Average across steps with 25–75% band
    g = (ame_grid_df.groupby("u")
            .agg(mu=("ame", "mean"),
                 lo=("ame", lambda x: np.nanpercentile(x, 25)),
                 hi=("ame", lambda x: np.nanpercentile(x, 75)))
            .reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(g["u"], g["mu"], marker="o")
    if g[["lo", "hi"]].notna().all().all():
        ax.fill_between(g["u"], g["lo"], g["hi"], alpha=0.2)
    ax.set_xlabel("uncertainty_std (u)")
    ax.set_ylabel("Mean AME(aha | u)")
    ax.set_title("Aha AME vs uncertainty (pooled across steps)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aha_ame_grid.png"))
    plt.close(fig)

# --- replace fit_pooled_model entirely (fixes NameError and uses ridge cleanly) ---
def fit_pooled_model(df: pd.DataFrame, out_dir: str, l2: float = 0.5, interaction: bool = True):
    """Pooled GLM with step fixed effects and optional aha×step interactions (ridge-regularized)."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception:
        print("statsmodels not available for pooled model.")
        return

    # Build formula: step FE + aha main + uncertainty + (optionally) aha×step
    formula = "correct ~ C(problem) + C(step) + aha + uncertainty_std"
    if interaction:
        formula += " + aha:C(step)"  # equivalent to including per-step deviations for aha

    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    res = model.fit_regularized(alpha=float(l2), L1_wt=0.0)

    # Per-step effect of aha = base aha + interaction term for that step (if any)
    base = _get_param(res, "aha", 0.0)
    rows = []
    for s in sorted(df["step"].unique()):
        t1 = f"aha:C(step)[T.{s}]"
        t2 = f"C(step)[T.{s}]:aha"  # statsmodels sometimes flips order
        delta = _get_param(res, t1, np.nan)
        if not np.isfinite(delta):
            delta = _get_param(res, t2, 0.0)
        effect = float(base) + (float(delta) if np.isfinite(delta) else 0.0)
        rows.append({"step": int(s), "aha_effect": effect})

    pooled = pd.DataFrame(rows).sort_values("step")
    pooled.to_csv(os.path.join(out_dir, "h2_pooled_aha_by_step.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    ax.plot(pooled["step"], pooled["aha_effect"], marker="o")
    ax.set_xlabel("Training step"); ax.set_ylabel("Pooled effect of aha (log-odds)")
    ax.set_title("Pooled GLM (step FE): per-step aha effect (ridge)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "h2_pooled_aha_by_step.png"))
    plt.close(fig)

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Root containing step*/.../*.jsonl")
    ap.add_argument("--split", default=None, help="Filter filenames by substring, e.g. 'test'")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: <results_root>/h2_analysis)")
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)
    ap.add_argument("--unc_field", choices=["answer","overall","think"], default="answer",
                    help="Which entropy field to use as uncertainty (default: answer entropy).")
    ap.add_argument("--aha_source", choices=["gpt","native"], default="gpt",
                    help="Prefer GPT-labeled shift (default) or native reconsider cue.")
    ap.add_argument("--interaction", action="store_true", help="Include aha×uncertainty_std interaction.")
    ap.add_argument("--compare_native", action="store_true", help="Also fit and plot using native aha labels.")
    ap.add_argument("--penalty", choices=["none","ridge","firth"], default="ridge",
                    help='Penalty for step-wise GLMs; "firth" currently falls back to ridge.')
    ap.add_argument("--l2", type=float, default=1.0, help="L2 strength for ridge when used.")
    ap.add_argument("--bootstrap_ame", type=int, default=200, help="Bootstrap reps for AME CIs (per step).")
    ap.add_argument("--ame_grid", type=int, default=9, help="Number of u grid points in [-2,2] for AME(u).")
    ap.add_argument("--fdr_alpha", type=float, default=0.05, help="BH/FDR alpha for step-wise aha p-values.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "h2_analysis")
    os.makedirs(out_dir, exist_ok=True)

    files = scan_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check path or --split.")

    # -------- Primary run (selected aha_source) --------
    df = load_pass1_rows(files, args.unc_field, args.aha_source)
    if args.min_step is not None:
        df = df[df["step"] >= args.min_step]
    if args.max_step is not None:
        df = df[df["step"] <= args.max_step]
    if df.empty:
        raise SystemExit("No rows left after step filtering.")

    # Global standardization for scale stability
    mu = df["uncertainty"].mean(); sd = df["uncertainty"].std(ddof=0)
    df["uncertainty_std"] = (df["uncertainty"] - mu) / (sd + 1e-8)

    # Save samples
    samples_csv = os.path.join(out_dir, "h2_pass1_samples.csv")
    df.to_csv(samples_csv, index=False)

    # Per-step GLMs (penalized if needed)
    reg = fit_stepwise_glms(df, out_dir,
                            interaction=args.interaction,
                            penalty=args.penalty,
                            l2=args.l2,
                            bootstrap_ame=args.bootstrap_ame,
                            ame_grid=args.ame_grid,
                            fdr_alpha=args.fdr_alpha)

    # Diagnostics panel
    plot_diag_panel(df, out_dir)

    # Plots (primary)
    if not reg.empty:
        _lineplot(reg["step"], reg["aha_coef"],
                  "Training step", "β(aha)", "Aha coefficient vs. step",
                  os.path.join(out_dir, "aha_coef_vs_step.png"))

        _lineplot(reg["step"], reg["aha_ame"],
                  "Training step", "AME(aha)", "Aha average marginal effect vs. step",
                  os.path.join(out_dir, "aha_ame_vs_step.png"))

        _lineplot(reg["step"], reg["unc_coef"],
                  "Training step", "β(uncertainty_std)", "Uncertainty coefficient vs. step",
                  os.path.join(out_dir, "uncertainty_coef_vs_step.png"))

        _lineplot(reg["step"], reg["naive_delta"],
                  "Training step", "Δ accuracy (aha=1 − aha=0)", "Naïve Δaccuracy vs. step",
                  os.path.join(out_dir, "naive_delta_vs_step.png"))

        # AME with CI
        plot_ame_with_ci(reg, out_dir)

    # AME grid plot
    ame_grid_path = os.path.join(out_dir, "h2_ame_grid.csv")
    ame_grid_df = pd.read_csv(ame_grid_path) if os.path.exists(ame_grid_path) else pd.DataFrame()
    plot_ame_grid(ame_grid_df, out_dir)

    # Optional comparison run (native)
    if args.compare_native:
        df_nat = load_pass1_rows(files, args.unc_field, "native")
        if args.min_step is not None:
            df_nat = df_nat[df_nat["step"] >= args.min_step]
        if args.max_step is not None:
            df_nat = df_nat[df_nat["step"] <= args.max_step]
        if not df_nat.empty:
            mu_n = df_nat["uncertainty"].mean(); sd_n = df_nat["uncertainty"].std(ddof=0)
            df_nat["uncertainty_std"] = (df_nat["uncertainty"] - mu_n) / (sd_n + 1e-8)
            reg_nat = fit_stepwise_glms(df_nat, out_dir,
                                        interaction=args.interaction,
                                        penalty=args.penalty,
                                        l2=args.l2,
                                        bootstrap_ame=args.bootstrap_ame,
                                        ame_grid=args.ame_grid,
                                        fdr_alpha=args.fdr_alpha)
            reg_nat.to_csv(os.path.join(out_dir, "h2_step_regression_native.csv"), index=False)
            # Overlays
            fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
            ax.plot(reg["step"], reg["aha_coef"], marker="o", label=f"aha={args.aha_source}")
            ax.plot(reg_nat["step"], reg_nat["aha_coef"], marker="o", label="aha=native")
            ax.set_xlabel("Training step"); ax.set_ylabel("β(aha)")
            ax.set_title("Aha coefficient vs. step (GPT vs native)")
            ax.grid(True, alpha=0.3); ax.legend(loc="best")
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, "aha_coef_vs_step_compare.png")); plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
            ax.plot(reg["step"], reg["aha_ame"], marker="o", label=f"aha={args.aha_source}")
            ax.plot(reg_nat["step"], reg_nat["aha_ame"], marker="o", label="aha=native")
            ax.set_xlabel("Training step"); ax.set_ylabel("AME(aha)")
            ax.set_title("Aha AME vs. step (GPT vs native)")
            ax.grid(True, alpha=0.3); ax.legend(loc="best")
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, "aha_ame_vs_step_compare.png")); plt.close(fig)

    # Pooled model (ridge-regularized step FE with aha×step)
    fit_pooled_model(df, out_dir, l2=max(0.1, args.l2), interaction=True)

    # Console recap
    print(f"Wrote samples CSV: {samples_csv}")
    print(f"Wrote step regression CSV: {os.path.join(out_dir, 'h2_step_regression.csv')}")
    print("Plots written:")
    print("  h2_diag_panel.png")
    print("  aha_coef_vs_step.png, aha_ame_vs_step.png, uncertainty_coef_vs_step.png, naive_delta_vs_step.png")
    print("  aha_ame_with_ci.png, aha_ame_grid.png")
    if args.compare_native:
        print("  aha_coef_vs_step_compare.png, aha_ame_vs_step_compare.png")
    print("Also wrote: h2_balance_by_step.csv, h2_ame_grid.csv, h2_fdr_summary.txt, h2_pooled_aha_by_step.csv")
    if args.interaction:
        print("  + h2_interaction_summary.txt")

if __name__ == "__main__":
    main()
