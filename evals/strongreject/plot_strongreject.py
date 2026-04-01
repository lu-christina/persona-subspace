"""
Plot StrongREJECT eval results in the paper style.

Expects summary CSVs produced by run_strongreject.py:
    {workspace}/{model}/evals/strongreject/{baseline,additive,capping}_summary.csv
with columns: mode, experiment_id, mean, std, count

Produces:
    {output_dir}/strongreject_additive.pdf  - line plot, score vs fraction of avg norm (fig5-6 layout)
    {output_dir}/strongreject_capping.pdf   - bar plot, baseline vs capped per model (fig10 layout)

Usage:
    uv run evals/strongreject/plot_strongreject.py \
        --models qwen-3-32b llama-3.3-70b gemma-2-27b \
        --output_dir paper/figs/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# evals/ and capped/ are not packages; import their plots.py modules by file path
import importlib.util  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"could not load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_evals_plots = _load_module("evals_plots", REPO_ROOT / "evals" / "plots.py")
_capped_plots = _load_module("capped_plots", REPO_ROOT / "capped" / "plots.py")
parse_additive_id = _evals_plots.parse_experiment_id
parse_capping_id = _capped_plots.parse_experiment_id
format_layer_range = _capped_plots.format_layer_range
format_cap_label = _capped_plots.format_cap_label

plt.style.use(str(REPO_ROOT / "paper" / "arena.mplstyle"))

# ---------------------------------------------------------------------------
# constants (mirrors paper/fig5-6_steering.ipynb cell 2 and fig10_barplot.ipynb)
# ---------------------------------------------------------------------------

MODEL_TITLES = {
    "gemma-2-27b": "Gemma",
    "qwen-3-32b": "Qwen",
    "llama-3.3-70b": "Llama",
    "llama-3.1-70b": "Llama",
}

# coeff in config experiment_id → fraction of per-token avg norm (rounded)
QWEN_COEFF_MAP = {
    -2.0: -1.20,
    -1.75: -1.05,
    -1.5: -0.90,
    -1.25: -0.75,
    -1.0: -0.60,
    -0.75: -0.45,
    -0.5: -0.30,
    -0.25: -0.15,
    0.0: 0.0,
    0.25: 0.15,
    0.5: 0.30,
    0.75: 0.45,
    1.0: 0.60,
    1.25: 0.75,
    1.5: 0.90,
    1.75: 1.05,
    2.0: 1.20,
}
LLAMA_COEFF_MAP = {
    -2.0: -1.0,
    -1.75: -0.875,
    -1.5: -0.75,
    -1.25: -0.625,
    -1.0: -0.5,
    -0.75: -0.375,
    -0.5: -0.25,
    -0.25: -0.125,
    0.0: 0.0,
    0.25: 0.125,
    0.5: 0.25,
    0.75: 0.375,
    1.0: 0.5,
    1.25: 0.625,
    1.5: 0.75,
    1.75: 0.875,
    2.0: 1.0,
}
MODEL_COEFF_MAPS = {
    "qwen-3-32b": QWEN_COEFF_MAP,
    "llama-3.3-70b": LLAMA_COEFF_MAP,
    "llama-3.1-70b": LLAMA_COEFF_MAP,
}

VECTOR_TYPE_LABELS = {
    "role_pc1": "Role PC1",
    "contrast": "Contrast",
}

BLUE = "#1d3557"
RED = "#e63946"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def scale_magnitude_to_new_norm(magnitude: float, model: str) -> float:
    coeff_map = MODEL_COEFF_MAPS.get(model, {})
    if not coeff_map:
        return magnitude
    if magnitude in coeff_map:
        return coeff_map[magnitude]
    closest = min(coeff_map.keys(), key=lambda x: abs(x - magnitude))
    return coeff_map[closest]


def load_summary(workspace: Path, model: str, mode: str) -> pd.DataFrame | None:
    p = workspace / model / "evals" / "strongreject" / f"{mode}_summary.csv"
    if not p.exists():
        print(f"  [skip] {p} not found")
        return None
    df = pd.read_csv(p)
    df["model"] = model
    return df


def baseline_score(workspace: Path, model: str) -> float | None:
    df = load_summary(workspace, model, "baseline")
    if df is None or df.empty:
        return None
    return float(df["mean"].iloc[0])


# ---------------------------------------------------------------------------
# additive line plot (fig5-6 layout)
# ---------------------------------------------------------------------------


def plot_additive(models: list[str], workspace: Path, out_path: Path) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        df = load_summary(workspace, model, "additive")
        bl = baseline_score(workspace, model)

        if df is not None:
            parsed = df["experiment_id"].apply(parse_additive_id)
            df["layer"] = parsed.apply(lambda t: t[0])
            df["vector_type"] = parsed.apply(lambda t: t[1])
            df["raw_mag"] = parsed.apply(lambda t: t[2])
            df = df.dropna(subset=["raw_mag"])
            # match fig5-6: rescale to per-token norm, flip sign so +x = toward assistant
            df["magnitude"] = (
                df["raw_mag"].apply(lambda m: scale_magnitude_to_new_norm(m, model))
                * -1
            )

            for vt in sorted(df["vector_type"].dropna().unique()):
                sub = df[df["vector_type"] == vt].sort_values("magnitude")
                ax.plot(
                    sub["magnitude"],
                    sub["mean"],
                    marker="o",
                    markersize=4,
                    linewidth=1,
                    label=VECTOR_TYPE_LABELS.get(vt, vt) if idx == n - 1 else None,
                )

        if bl is not None:
            ax.axhline(y=bl, color="black", linestyle="--", linewidth=1, alpha=0.3)
        ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1, alpha=0.3)

        ax.set_xlabel("Fraction of Avg. Norm", fontsize=10)
        if idx == 0:
            ax.set_ylabel("StrongREJECT Score", fontsize=10)
        ax.set_title(MODEL_TITLES.get(model, model), fontsize=12)
        ax.set_ylim(0, 1.0)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# capping bar plot (fig10 layout, single-metric)
# ---------------------------------------------------------------------------


def plot_capping(models: list[str], workspace: Path, out_path: Path) -> None:
    rows = []
    for model in models:
        bl = baseline_score(workspace, model)
        cap_df = load_summary(workspace, model, "capping")
        if bl is None or cap_df is None or cap_df.empty:
            continue
        # take the single (or first) capping experiment per model
        eid = cap_df["experiment_id"].iloc[0]
        layer_spec, _, cap_value = parse_capping_id(eid)
        label = f"{format_layer_range(layer_spec)} · {format_cap_label(cap_value)}"
        rows.append(
            dict(
                model=model,
                title=MODEL_TITLES.get(model, model),
                baseline=bl,
                capped=float(cap_df["mean"].iloc[0]),
                cap_label=label,
            )
        )
    if not rows:
        print("  [skip] no capping data found, not writing capping plot")
        return

    n = len(rows)
    fig, ax = plt.subplots(figsize=(1.2 + 1.6 * n, 3.4))
    bar_width = 0.38
    gap = 0.06
    x = np.arange(n)

    for i, r in enumerate(rows):
        ax.bar(
            x[i] - bar_width / 2 - gap / 2,
            r["baseline"],
            bar_width,
            color=RED,
            alpha=0.8,
        )
        ax.bar(
            x[i] + bar_width / 2 + gap / 2, r["capped"], bar_width, color=RED, alpha=0.5
        )
        ax.text(
            x[i] - bar_width / 2 - gap / 2,
            r["baseline"] + 0.015,
            f"{r['baseline']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=RED,
            alpha=0.8,
        )
        ax.text(
            x[i] + bar_width / 2 + gap / 2,
            r["capped"] + 0.015,
            f"{r['capped']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=RED,
            alpha=0.5,
        )
        ax.text(
            x[i],
            -0.08,
            r["cap_label"],
            ha="center",
            va="top",
            fontsize=7,
            color="grey",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([r["title"] for r in rows])
    ax.set_ylabel("StrongREJECT Score", fontsize=10)
    ax.set_ylim(0, max(1.0, max(r["baseline"] for r in rows) * 1.15))
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)

    legend_handles = [
        mpatches.Patch(facecolor=RED, alpha=0.8, label="Baseline"),
        mpatches.Patch(facecolor=RED, alpha=0.5, label="Activation Capped"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen-3-32b", "llama-3.3-70b", "gemma-2-27b"],
    )
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "paper" / "figs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_additive(
        args.models, args.workspace, args.output_dir / "strongreject_additive.pdf"
    )
    plot_capping(
        args.models, args.workspace, args.output_dir / "strongreject_capping.pdf"
    )


if __name__ == "__main__":
    main()
