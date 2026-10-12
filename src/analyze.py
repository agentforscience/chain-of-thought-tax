"""Analyse direct-vs-CoT results: accuracy tables, paired statistics, scale trends, robustness, figures.

All correctness values are RE-SCORED here from the saved raw outputs with src/scoring.py, so the
numbers never depend on which scoring version was live during inference.

Usage: python src/analyze.py   (reads results/model_outputs/*.jsonl, writes results/*.csv|json, figures/*.png)
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(__file__))
from scoring import extract_answer, score  # noqa: E402

SEED = 42
N_BOOT = 10000
PARAMS_B = {  # parameter counts (billions) from the model cards
    "Qwen2.5-0.5B-Instruct": 0.49, "Qwen2.5-1.5B-Instruct": 1.54, "Qwen2.5-3B-Instruct": 3.09,
    "Qwen2.5-7B-Instruct": 7.61, "Qwen2.5-14B-Instruct": 14.7,
}
MODELS = list(PARAMS_B)
SHORT = {m: m.replace("Qwen2.5-", "").replace("-Instruct", "") for m in MODELS}
TASKS = ["ag_news", "sst2", "triviaqa", "mmlu_recall", "bbh_logical_deduction", "proofwriter",
         "gsm8k", "synthetic_arithmetic"]
CATS = ["pattern_matching", "factual_recall", "logical_deduction", "arithmetic"]
CAT_LABEL = {"pattern_matching": "Pattern matching", "factual_recall": "Factual recall",
             "logical_deduction": "Logical deduction", "arithmetic": "Arithmetic"}
# Hypothesised band for delta (pp) per category.
H_BAND = {"pattern_matching": (-15, -5), "factual_recall": (-15, -5),
          "logical_deduction": (10, 25), "arithmetic": (10, 25)}

# Plot styling (categorical hues in fixed order; diverging blue/orange with neutral midpoint).
CAT_COLOR = {"pattern_matching": "#2a78d6", "factual_recall": "#eb6834",
             "logical_deduction": "#1baf7a", "arithmetic": "#eda100"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#52514e", "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})


def load():
    """Load all model outputs, re-score them, and return a long dataframe."""
    items = {r["id"]: r for r in map(json.loads, open("datasets/processed/tasks.jsonl"))}
    rows = []
    for path in sorted(glob.glob("results/model_outputs/*.jsonl")):
        for r in map(json.loads, open(path)):
            ans, has_marker = extract_answer(r["output"], r["condition"])
            r["answer"], r["has_marker"] = ans, has_marker
            r["correct"] = score(items[r["id"]], ans)
            r["model"] = r["model"].split("/")[-1]
            rows.append(r)
    df = pd.DataFrame(rows).drop_duplicates(["model", "id", "condition"])
    return df, items


def paired(df, model, cond_a="direct", cond_b="cot", ids=None):
    """Wide per-item table (columns: task, a, b) for one model and two conditions."""
    d = df[df.model == model]
    a = d[d.condition == cond_a].set_index("id")
    b = d[d.condition == cond_b].set_index("id")
    common = a.index.intersection(b.index)
    if ids is not None:
        common = common.intersection(ids)
    return pd.DataFrame({"task": a.loc[common, "task"], "category": a.loc[common, "category"],
                         "a": a.loc[common, "correct"], "b": b.loc[common, "correct"]})


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired binary outcomes."""
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    return 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue


def boot_delta(w, rng, n_boot=N_BOOT):
    """Bootstrap distribution of the (b - a) accuracy difference in pp.

    If several tasks are present the statistic is the macro-average of the per-task deltas and the
    resampling is stratified by task."""
    out = np.zeros(n_boot)
    groups = [g[["a", "b"]].to_numpy() for _, g in w.groupby("task")]
    for g in groups:
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        out += (g[idx, 1].mean(1) - g[idx, 0].mean(1)) * 100 / len(groups)
    return out


def summarize(w, rng):
    """Point estimate, CI, McNemar p, harm and fix rates for a paired table."""
    per_task = w.groupby("task")[["a", "b"]].mean() * 100
    delta = float((per_task["b"] - per_task["a"]).mean())
    bd = boot_delta(w, rng)
    right, wrong = w[w.a == 1], w[w.a == 0]
    return dict(n=len(w), acc_direct=float(per_task["a"].mean()), acc_cot=float(per_task["b"].mean()),
                delta=delta, ci_lo=float(np.percentile(bd, 2.5)), ci_hi=float(np.percentile(bd, 97.5)),
                rel_delta_pct=float(100 * delta / max(per_task["a"].mean(), 1e-9)),
                p_mcnemar=mcnemar_exact(w.a, w.b),
                harm_rate=float(100 * (right.b == 0).mean()) if len(right) else np.nan,
                fix_rate=float(100 * (wrong.b == 1).mean()) if len(wrong) else np.nan,
                n_harmed=int((right.b == 0).sum()), n_fixed=int((wrong.b == 1).sum())), bd


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs("figures", exist_ok=True)
    df, items = load()
    only = sys.argv[1:]  # optional: restrict to these short names (e.g. 0.5B) for partial runs
    models = [m for m in MODELS if m in set(df.model) and (not only or SHORT[m] in only)]
    print("models:", models, "| rows:", len(df))
    print(df.groupby(["model", "condition"]).size().unstack().to_string())

    # ---- 1. Per-task and per-category paired statistics -------------------------------------
    task_rows, cat_rows, boots = [], [], {}
    for m in models:
        w = paired(df, m)
        for t in TASKS:
            s, _ = summarize(w[w.task == t], rng)
            task_rows.append(dict(model=m, task=t, category=items[w[w.task == t].index[0]]["category"], **s))
        for c in CATS:
            s, bd = summarize(w[w.category == c], rng)
            boots[(m, c)] = bd
            cat_rows.append(dict(model=m, category=c, **s))
    T, C = pd.DataFrame(task_rows), pd.DataFrame(cat_rows)
    T["p_holm"] = multipletests(T.p_mcnemar, method="holm")[1]
    C["p_holm"] = multipletests(C.p_mcnemar, method="holm")[1]
    for D in (T, C):
        D["sig"] = np.where(D.p_holm < 0.05, np.where(D.delta > 0, "helps", "hurts"), "n.s.")
    # Verdict on the hypothesised band: point estimate inside the band AND Holm-significant.
    C["band"] = C.category.map(lambda c: f"[{H_BAND[c][0]:+d}, {H_BAND[c][1]:+d}]")
    C["in_band"] = [H_BAND[c][0] <= d <= H_BAND[c][1] for c, d in zip(C.category, C.delta)]
    C["hyp_supported"] = C.in_band & (C.p_holm < 0.05)
    T.to_csv("results/task_level.csv", index=False)
    C.to_csv("results/category_level.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["model", "acc_direct", "acc_cot", "delta", "ci_lo", "ci_hi", "p_holm", "sig", "harm_rate", "fix_rate"]
    print("\n== TASK LEVEL ==\n", T[["task"] + cols].round(3).to_string(index=False))
    print("\n== CATEGORY LEVEL ==\n", C[["category"] + cols + ["in_band", "hyp_supported"]].round(3).to_string(index=False))

    # ---- 2. Scale trend: slope of delta on log2(params), item-bootstrap CI --------------------
    x = np.log2([PARAMS_B[m] for m in models])
    slope_rows = []
    for c in CATS:
        y = C[C.category == c].set_index("model").loc[models, "delta"].to_numpy()
        B = np.stack([boots[(m, c)] for m in models], axis=1)  # n_boot x n_models (independent per model)
        slopes = np.polyfit(x, B.T, 1)[0]
        slope_rows.append(dict(category=c, slope_pp_per_doubling=float(np.polyfit(x, y, 1)[0]),
                               ci_lo=float(np.percentile(slopes, 2.5)), ci_hi=float(np.percentile(slopes, 97.5))))
    for label, D, key in [("categories", C, "category"), ("tasks", T, "task")]:
        cnt = D.groupby("model")["sig"].value_counts().unstack(fill_value=0).reindex(models)
        print(f"\n== # {label} where CoT significantly helps / hurts (Holm 0.05) ==\n", cnt.to_string())
        cnt.to_csv(f"results/sig_counts_{label}.csv")
    S = pd.DataFrame(slope_rows)
    S.to_csv("results/scale_slopes.csv", index=False)
    print("\n== SCALE SLOPES (pp of delta per doubling of parameters) ==\n", S.round(2).to_string(index=False))

    # ---- 3. Output diagnostics: marker / truncation / length ---------------------------------
    diag = (df.groupby(["model", "task", "condition"])
            .agg(acc=("correct", "mean"), has_marker=("has_marker", "mean"), truncated=("truncated", "mean"),
                 mean_out_tokens=("n_out_tokens", "mean"), n=("correct", "size")).reset_index())
    diag.to_csv("results/diagnostics.csv", index=False)
    d_cot = diag[diag.condition == "cot"].pivot(index="task", columns="model")
    print("\n== CoT: share with explicit final-answer marker ==\n", d_cot["has_marker"][models].loc[TASKS].round(3).to_string())
    print("\n== CoT: share truncated at token budget ==\n", d_cot["truncated"][models].loc[TASKS].round(3).to_string())
    print("\n== CoT: mean output tokens ==\n", d_cot["mean_out_tokens"][models].loc[TASKS].round(0).to_string())
    tok = df[df.condition.isin(["direct", "cot"])].groupby(["model", "condition"])["n_out_tokens"].mean().unstack().loc[models]
    tok["ratio"] = tok["cot"] / tok["direct"]
    tok.to_csv("results/token_cost.csv")
    print("\n== Mean output tokens per item ==\n", tok.round(1).to_string())

    # ---- 4. Sensitivity: drop items whose CoT output was truncated or had no answer marker ----
    sens_rows = []
    for m in models:
        d = df[(df.model == m) & (df.condition == "cot")]
        clean = d[(d.has_marker) & (~d.truncated)]["id"]
        w = paired(df, m, ids=pd.Index(clean))
        for c in CATS:
            s, _ = summarize(w[w.category == c], rng)
            sens_rows.append(dict(model=m, category=c, n_kept=s["n"], delta_clean=s["delta"],
                                  ci_lo=s["ci_lo"], ci_hi=s["ci_hi"]))
    SENS = pd.DataFrame(sens_rows).merge(C[["model", "category", "delta", "n"]], on=["model", "category"])
    SENS.to_csv("results/sensitivity_clean_cot.csv", index=False)
    print("\n== SENSITIVITY (well-formed CoT outputs only) ==\n", SENS.round(2).to_string(index=False))

    # ---- 4b. Failure mode of harmed items (direct right, CoT wrong) --------------------------
    cot = df[df.condition == "cot"].set_index(["model", "id"])
    hm = []
    for m in models:
        w = paired(df, m)
        for i in w[(w.a == 1) & (w.b == 0)].index:
            r = cot.loc[(m, i)]
            mode = "truncated" if r.truncated else ("no_marker" if not r.has_marker else "wrong_answer")
            hm.append(dict(model=m, category=r.category, mode=mode))
    HM = pd.DataFrame(hm).groupby(["model", "category"])["mode"].value_counts().unstack(fill_value=0)
    HM.to_csv("results/harm_failure_modes.csv")
    print("\n== HARMED ITEMS BY FAILURE MODE ==\n", HM.to_string())

    # ---- 5. Prompt robustness: second CoT wording on the 100-item-per-task subset -------------
    rob_rows = []
    for m in models:
        ids2 = pd.Index(df[(df.model == m) & (df.condition == "cot2")]["id"])
        if len(ids2) == 0:
            continue
        w1, w2 = paired(df, m, "direct", "cot", ids2), paired(df, m, "direct", "cot2", ids2)
        for c in CATS:
            if (w1.category == c).sum() == 0:
                continue
            s1, _ = summarize(w1[w1.category == c], rng)
            s2, _ = summarize(w2[w2.category == c], rng)
            rob_rows.append(dict(model=m, category=c, n=s1["n"], delta_cot=s1["delta"], delta_cot2=s2["delta"],
                                 cot2_ci_lo=s2["ci_lo"], cot2_ci_hi=s2["ci_hi"]))
    R = pd.DataFrame(rob_rows)
    if len(R):
        R["same_sign"] = np.sign(R.delta_cot) == np.sign(R.delta_cot2)
        R.to_csv("results/robustness_cot2.csv", index=False)
        print("\n== ROBUSTNESS (CoT wording 2, same items) ==\n", R.round(2).to_string(index=False))
        print("sign agreement:", R.same_sign.mean().round(3), "| corr of deltas:",
              np.corrcoef(R.delta_cot, R.delta_cot2)[0, 1].round(3))

    # ---- 6. Figures --------------------------------------------------------------------------
    xs = [PARAMS_B[m] for m in models]
    # Fig 1: delta vs scale per category (small multiples, shared y axis, hypothesised band shaded)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, c in zip(axes, CATS):
        d = C[C.category == c].set_index("model").loc[models]
        lo, hi = H_BAND[c]
        ax.axhspan(lo, hi, color="#f0efec", zorder=0, label="hypothesised range")
        ax.axhline(0, color=MUTED, lw=1)
        ax.errorbar(xs, d.delta, yerr=[d.delta - d.ci_lo, d.ci_hi - d.delta], color=CAT_COLOR[c], lw=2,
                    marker="o", ms=7, capsize=3, mec=SURFACE, mew=1.5, label="CoT − direct (95% CI)")
        for xv, yv in zip(xs, d.delta):
            ax.annotate(f"{round(yv) + 0:+d}", (xv, yv), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[m] for m in models])
        ax.minorticks_off()
        ax.set_title(CAT_LABEL[c], loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Qwen2.5-Instruct size (parameters)")
    axes[0].set_ylabel("Accuracy change from CoT (pp)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Chain-of-thought effect by task category and model size", x=0.01, y=1.04, ha="left", fontsize=13)
    fig.savefig("figures/delta_by_category_scale.png")
    plt.close(fig)

    # Fig 2: heatmap of delta, model x task (diverging, neutral midpoint)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    P = T.pivot(index="task", columns="model", values="delta").loc[TASKS, models]
    Sg = T.pivot(index="task", columns="model", values="sig").loc[TASKS, models]
    cmap = LinearSegmentedColormap.from_list("div", ["#c2491d", "#f0efec", "#1c5cab"])
    vmax = float(np.abs(P.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(P.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(len(models)), [SHORT[m] for m in models])
    ax.set_yticks(range(len(TASKS)), [f"{t}  ({CAT_LABEL[items[next(i for i in items if items[i]['task'] == t)]['category']]})" for t in TASKS])
    ax.grid(False)
    for i in range(len(TASKS)):
        for j in range(len(models)):
            v = P.iloc[i, j]
            ax.text(j, i, f"{round(v) + 0:+d}" + ("*" if Sg.iloc[i, j] != "n.s." else ""), ha="center", va="center",
                    color="white" if abs(v) > 0.55 * vmax else INK, fontsize=10)
    fig.colorbar(im, ax=ax, label="CoT − direct accuracy (pp)")
    ax.set_xlabel("Qwen2.5-Instruct size")
    ax.set_title("CoT accuracy change per task (* = Holm-significant, α=0.05)", loc="left", fontsize=11)
    fig.savefig("figures/delta_heatmap.png")
    plt.close(fig)

    # Fig 3: harm rate and fix rate by category and scale (two panels, one measure each)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in zip(axes, ["harm_rate", "fix_rate"],
                              ["Harm rate: P(CoT wrong | direct right)", "Fix rate: P(CoT right | direct wrong)"]):
        for c in CATS:
            d = C[C.category == c].set_index("model").loc[models]
            ax.plot(xs, d[key], color=CAT_COLOR[c], lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=CAT_LABEL[c])
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[m] for m in models])
        ax.minorticks_off()
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of items")
        ax.set_xlabel("Qwen2.5-Instruct size")
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig("figures/harm_fix_rates.png")
    plt.close(fig)

    # Fig 4: absolute accuracy, direct vs CoT, per task (small multiples)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for ax, t in zip(axes.T.ravel(), TASKS):
        d = T[T.task == t].set_index("model").loc[models]
        ax.plot(xs, d.acc_direct, color="#52514e", lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label="Direct")
        ax.plot(xs, d.acc_cot, color="#2a78d6", lw=2, marker="s", ms=6, mec=SURFACE, mew=1.5, label="CoT")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[m] for m in models], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(0, 100)
        ax.set_title(f"{t}", loc="left", fontsize=10, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Accuracy by task: direct answer vs chain-of-thought (columns = categories: pattern, recall, logic, arithmetic)",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig("figures/accuracy_by_task.png")
    plt.close(fig)

    json.dump(dict(seed=SEED, n_boot=N_BOOT, models=models, n_rows=int(len(df))), open("results/analysis_config.json", "w"), indent=2)
    print("\nwrote results/*.csv and figures/*.png")


if __name__ == "__main__":
    main()
