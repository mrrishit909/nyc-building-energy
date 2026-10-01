"""Step 3: charts for the multifamily model.

    ./venv/bin/python charts.py   -> charts/01_effects.png, 02_eras.png, 03_diagnostics.png, 04_electric.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
INK, DIM, GRID, BG, BLUE, ORANGE = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.titlepad": 12,
})
NAMES = {"electric_share": "electricity share +10 pts", "steam": "uses district steam", "oil": "burns fuel oil",
         "units_per_10k_sqft": "+1 apartment per 10k sq ft", "occupancy": "occupancy +1 pt", "log_area": "floor area x2.7 (log +1)"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / "charts" / name, dpi=150)
    plt.close(fig)


def label(term):
    if "era" in term:
        return "built " + term.split("T.")[1].rstrip("]") + " (vs 1960-79)"
    if "borough" in term:
        return term.split("T.")[1].rstrip("]").replace("Staten Is", "Staten Island") + " (vs Manhattan)"
    return NAMES.get(term, term)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    c = pd.read_csv(HERE / "results" / "coefficients.csv")
    c = c[(c["model"] == "Multifamily Housing") & (c["term"] != "Intercept") & ~c["term"].str.contains("Unknown")].copy()
    scale = np.where(c["term"] == "electric_share", 0.1, 1.0)               # show electricity per 10 points
    c["eff"] = 100 * (np.exp(c["coef"] * scale) - 1)
    c["lo"] = 100 * (np.exp((c["coef"] - 1.96 * c["se_hc3"]) * scale) - 1)
    c["hi"] = 100 * (np.exp((c["coef"] + 1.96 * c["se_hc3"]) * scale) - 1)
    c = c.sort_values("eff")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(c))
    ax.hlines(y, c["lo"], c["hi"], color=DIM, linewidth=2)
    ax.scatter(c["eff"], y, color=[ORANGE if v < 0 else BLUE for v in c["eff"]], s=50, zorder=3)
    ax.axvline(0, color=DIM, linewidth=1)
    ax.set_yticks(y, [label(t) for t in c["term"]])
    ax.set_xlabel("% difference in energy use intensity, holding the others fixed (95% CI, robust SEs)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What goes with higher or lower energy use in NYC apartment buildings (2024)")
    save(fig, "01_effects.png")

    f = pd.read_csv(HERE / "results" / "model_data_multifamily.csv", index_col=0)
    order = ["pre-1930", "1930-59", "1960-79", "1980-99", "2000-09", "2010+"]
    med = f.groupby("era")["log_eui"].apply(lambda s: np.exp(s).median()).reindex(order)
    n = f["era"].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(order, med.values, color=BLUE, width=0.6)
    share = f.groupby("era")["electric_share"].median().reindex(order)
    for i, (m, k, e) in enumerate(zip(med.values, n.values, share.values)):
        ax.text(i, m + 1, f"{m:.0f}\n{e:.0%} electric", ha="center", color=INK, fontsize=9)
    ax.set_ylabel("median kBtu per sq ft")
    ax.set_ylim(0, med.max() * 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_title("Raw medians: newer buildings look leaner, because they are far more electric")
    save(fig, "02_eras.png")

    r = pd.read_csv(HERE / "results" / "residuals_multifamily.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].scatter(r["fitted"], r["resid"], s=3, alpha=0.25, color=BLUE)
    axes[0].axhline(0, color=INK, linewidth=1)
    axes[0].set_xlabel("fitted log EUI"); axes[0].set_ylabel("residual")
    axes[0].set_title("Residuals vs fitted: spread varies (robust SEs used)", fontsize=12)
    q = np.sort(r["resid"].values)
    theo = stats.norm.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q)) * q.std()
    axes[1].scatter(theo, q, s=3, alpha=0.3, color=BLUE)
    axes[1].plot([theo.min(), theo.max()], [theo.min(), theo.max()], color=INK, linewidth=1)
    axes[1].set_xlabel("normal quantiles"); axes[1].set_ylabel("residual quantiles")
    axes[1].set_title("Q-Q plot: heavier tails than normal", fontsize=12)
    save(fig, "03_diagnostics.png")

    bins = pd.cut(f["electric_share"], np.linspace(0, 1, 11), include_lowest=True)
    g = f.groupby(bins, observed=True)["log_eui"].agg(lambda s: np.exp(s).median())
    cnt = f.groupby(bins, observed=True).size()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xs = [f"{int(b.left * 100)}-{int(b.right * 100)}%" for b in g.index]
    ax.bar(xs, g.values, color=BLUE, width=0.65)
    for i, (v, k) in enumerate(zip(g.values, cnt.values)):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", color=INK, fontsize=9)
    ax.set_ylim(0, g.max() * 1.12)
    ax.set_xlabel("share of the building's energy that is electricity")
    ax.set_ylabel("median kBtu per sq ft")
    ax.grid(axis="x", visible=False)
    ax.set_title("More electric, less site energy (partly accounting: see source-energy check)")
    save(fig, "04_electric.png")
    print("charts written")


if __name__ == "__main__":
    main()
