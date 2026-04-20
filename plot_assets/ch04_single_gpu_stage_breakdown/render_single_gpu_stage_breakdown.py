from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
STAGES = ["query_prep", "ann_recall", "pack_candidates", "pytod_score", "output"]
STAGE_LABELS = {
    "query_prep": "查询准备",
    "ann_recall": "ANN召回",
    "pack_candidates": "候选组织",
    "pytod_score": "PyTOD评分",
    "output": "结果输出",
}
COLORS = ["#FFFFFF", "#DBDBDB", "#B9B9B9", "#949494", "#666666"]
HATCHES = ["", "//", "\\\\", "xx", ".."]
DATASETS = [
    (
        ROOT / "single_gpu_stage_breakdown_1m.csv",
        ROOT / "fig4_14_single_gpu_stage_breakdown_1m.png",
    ),
    (
        ROOT / "single_gpu_stage_breakdown_10m.csv",
        ROOT / "fig4_15_single_gpu_stage_breakdown_10m.png",
    ),
]

SCENARIO_LABELS = {
    "Original baseline": "原始链路\n基线",
    "Original pipeline baseline": "原始链路\n基线",
    "Recall-accelerated": "仅召回\n加速",
    "Recall acceleration only": "仅召回\n加速",
    "GPU handoff": "GPU侧常驻\n交接",
    "GPU-resident recall-to-score handoff": "GPU侧常驻\n交接",
    "Full optimization": "完整单卡\n优化",
    "Full single-GPU optimization": "完整单卡\n优化",
}


def setup_cjk_font() -> None:
    candidates = [
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "Songti SC",
        "PingFang SC",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def wrapped_labels(values: pd.Series, width: int = 18) -> list[str]:
    labels = []
    for item in values:
        text = SCENARIO_LABELS.get(str(item), str(item))
        if "\n" in text:
            labels.append(text)
        else:
            labels.append(fill(text, width=width, break_long_words=False))
    return labels


def render(csv_path: Path, out_path: Path) -> None:
    setup_cjk_font()
    df = pd.read_csv(csv_path)
    x = np.arange(len(df))
    width = 0.64

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=220)
    bottom = np.zeros(len(df))

    for stage, color, hatch in zip(STAGES, COLORS, HATCHES):
        bars = ax.bar(
            x,
            df[stage].values,
            width,
            bottom=bottom,
            label=STAGE_LABELS[stage],
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
        )
        for rect in bars:
            rect.set_linewidth(0.8)
        bottom += df[stage].values

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, labelsize=11.2)
    display = df["scheme_desc"] if "scheme_desc" in df.columns else df["scheme"]
    ax.set_xticks(x, wrapped_labels(display))
    ax.set_xlabel("执行方案", fontsize=13)
    ax.set_ylabel("延迟（毫秒）", fontsize=13)
    ax.legend(
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        frameon=False,
        fontsize=10.6,
    )

    totals = df["latency_ms"].values
    for idx, total in enumerate(totals):
        ax.text(idx, total + totals.max() * 0.025, f"{total:.2f}", ha="center", va="bottom", fontsize=10.2)

    ax.set_ylim(0, totals.max() * 1.16)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for csv_path, out_path in DATASETS:
        render(csv_path, out_path)
