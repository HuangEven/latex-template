"""Render paired stage-time differences from the existing single-GPU measurements."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
STAGES = ["query_prep", "ann_recall", "pack_candidates", "pytod_score", "output"]
LABELS = ["查询准备", "候选检索", "候选组织", "PyTOD 评分", "结果输出"]


def differences(scale):
    with (ROOT / f"single_gpu_stage_breakdown_{scale}.csv").open(encoding="utf-8-sig", newline="") as source:
        records = {row["scheme"]: row for row in csv.DictReader(source)}
    original = records["Original baseline"]
    optimized = records["Full optimization"]
    values = [round(float(original[key]) - float(optimized[key]), 2) for key in STAGES]
    total = round(float(original["latency_ms"]) - float(optimized["latency_ms"]), 2)
    assert round(sum(values), 2) == total
    return values


def main():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "PingFang SC", "Songti SC", "SimHei"):
        if name in available:
            break
    else:
        raise RuntimeError("A Chinese font is required to render this figure")
    plt.rcParams.update({"font.family": ["Times New Roman", name], "pdf.fonttype": 42,
                         "axes.unicode_minus": False, "font.size": 10})
    one, ten = differences("1m"), differences("10m")
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    fig.patch.set_facecolor("white")
    y = list(range(len(STAGES)))
    ax.barh([v - 0.18 for v in y], one, height=0.34, color="#96BDCD",
            edgecolor="black", linewidth=0.7, label="1M")
    ax.barh([v + 0.18 for v in y], ten, height=0.34, color="#276283",
            edgecolor="black", linewidth=0.7, label="10M")
    for index, value in enumerate(one):
        ax.text(value + 0.09, index - 0.18, f"{value:.2f}", va="center", fontsize=9)
    for index, value in enumerate(ten):
        ax.text(value + 0.09, index + 0.18, f"{value:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 10.1)
    ax.set_xlabel("阶段耗时差额（ms）")
    ax.grid(axis="x", linestyle="--", linewidth=0.55, color="#b8b8b8", alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "fig4_stage_differences.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
