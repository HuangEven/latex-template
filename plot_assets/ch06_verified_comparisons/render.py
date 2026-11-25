"""Regenerate figures 6-4 through 6-6 from the repository's reported data.

Run with Python 3 and Matplotlib. No fitted curves or simulated observations.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
ASSETS = OUT.parent
for family in ("Songti SC", "Noto Serif CJK SC", "SimSun", "Arial Unicode MS"):
    try:
        font_manager.findfont(family, fallback_to_default=False)
    except ValueError:
        continue
    break
else:
    raise RuntimeError("Install a Chinese font (e.g. Noto Serif CJK SC).")
plt.rcParams.update({"font.family": family, "font.size": 10, "pdf.fonttype": 42,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": .7, "axes.unicode_minus": False,
                     "xtick.direction": "out", "ytick.direction": "out"})


def rows(relative):
    with (ASSETS / relative).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def grid(ax, axis):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, linewidth=.5, linestyle=(0, (3, 3)), color=".82")


def finish(fig, stem):
    fig.savefig(OUT / (stem + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


names = {"Full scoring": "PyTOD 全量评分", "Recall-only": "仅召回",
         "Initial fusion": "FAISS-GPU + PyTOD", "Full fusion": "FlashTOD"}
fusion = {r["scheme"]: r for r in rows("ch06_fusion_compare/fusion_scheme_compare.csv")}
assert set(names) == set(fusion)

# These are different statistics, so no lines joining the two series.
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0), sharey=True)
fig.subplots_adjust(left=.23, right=.95, bottom=.20, top=.82, wspace=.18)
for ax, metric, title, marker in zip(axes, ["recall_at_k", "topk_overlap"],
                                      ["前 k 项召回率", "与全量评分的前 k 项重合度"], ["o", "s"]):
    ax.set_title(title, fontsize=10, pad=6)
    for y, key in enumerate(names):
        value = float(fusion[key][metric])
        ax.plot(value, y, marker=marker, color=".15", mfc="#B35523" if key == "Full fusion" else "white", ms=5)
        ax.annotate(f"{value:.3f}", (value, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set_xlim(.69, 1.09)
    ax.set_xticks([.7, .8, .9, 1.0])
    grid(ax, "x")
axes[0].set_yticks(range(4), list(names.values()))
axes[0].set_ylim(3.5, -.5)
finish(fig, "ranking")

fig, ax = plt.subplots(figsize=(6.3, 2.65))
fig.subplots_adjust(left=.14, right=.98, top=.96, bottom=.22)
for key, marker, side in zip(names, ["o", "D", "^", "s"], [1, -1, 1, -1]):
    x, y = float(fusion[key]["qps"]), float(fusion[key]["pr_auc"])
    ax.plot(x, y, marker=marker, markersize=7, color=".12",
            mfc="#B35523" if key == "Full fusion" else "white")
    ax.annotate(f"{names[key]}  ({x:.0f}, {y:.3f})", (x, y),
                xytext=(side*7, 7), textcoords="offset points", fontsize=9,
                ha="left" if side > 0 else "right")
ax.set(xlim=(2450, 4800), ylim=(.82, .95), xlabel="端到端吞吐率（QPS）", ylabel="PR-AUC")
ax.set_yticks([.82, .85, .88, .91, .94])
grid(ax, "both")
finish(fig, "tradeoff")

# Source the single-device scenarios from the single-device comparison only.
# The 1-card replica baseline comes from the multi-device experiment table.
single = {r["scheme"]: r for r in rows("ch06_single_multi_summary/single_multi_benefit_summary.csv")}
replica = {int(r["gpu_count"]): r for r in rows("ch05_multi_gpu_tables/multi_gpu_experiment_results.csv")
           if r["scheme"] == "Replicated" and r["data_scale"] == "10M"}
labels = ["原始分段链路", "完整单卡优化", "1 卡复制", "2 卡复制", "4 卡复制"]
observations = [single["Original baseline"], single["Single-GPU optimized"],
                *[replica[g] for g in (1, 2, 4)]]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), sharey=True)
fig.subplots_adjust(left=.15, right=.98, top=.82, bottom=.16, wspace=.28)
for ax, metric, title, limits, ticks, fmt in zip(
        axes, ["qps", "p99_latency_ms", "cpu_util"],
        ["吞吐率（QPS）", "p99 延迟（ms）", "CPU 利用率（%）"],
        [(0, 16900), (0, 90), (0, 83)], [[0, 7000, 14000], [0, 35, 70], [0, 35, 70]],
        [".0f", ".1f", ".1f"]):
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=8)
    grid(ax, "x")
    ax.axhline(1.6, color=".55", lw=.6, ls="--")
    for i, (pos, row) in enumerate(zip([0, 1, 2.3, 3.3, 4.3], observations)):
        value = float(row[metric])
        ax.plot(value, pos, marker="o" if i < 2 else "s", ms=4.8, mec="#276283" if i < 2 else ".12", mfc="white" if i < 2 else "#276283")
        ax.annotate(format(value, fmt), (value, pos), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8)
axes[0].set_yticks([0, 1, 2.3, 3.3, 4.3], labels)
axes[0].set_ylim(4.85, -.5)
finish(fig, "execution")
