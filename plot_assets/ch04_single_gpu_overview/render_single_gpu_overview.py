from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "single_gpu_overview_metrics.csv"
OUT_PATH = ROOT / "fig4_16_single_gpu_overview.png"

SCHEME_COLORS = ["#FFFFFF", "#D1D1D1", "#A8A8A8", "#6E6E6E"]
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


def wrapped_labels(values: pd.Series, width: int = 16) -> list[str]:
    labels = []
    for item in values:
        text = SCENARIO_LABELS.get(str(item), str(item))
        if "\n" in text:
            labels.append(text)
        else:
            labels.append(fill(text, width=width, break_long_words=False))
    return labels


def style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, labelsize=12)


def main() -> None:
    setup_cjk_font()
    df = pd.read_csv(CSV_PATH)
    metrics = [
        ("qps", "吞吐率（QPS）", (0, 5200), 90.0),
        ("p99_ms", "p99延迟（毫秒）", (0, 75), 1.4),
        ("pci_transfer_count", "PCIe传输次数", (0, 7.0), 0.14),
        ("gpu_util", "GPU利用率（%）", (0, 75), 1.2),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), dpi=220)
    axes = axes.flatten()
    display = df["scheme_desc"] if "scheme_desc" in df.columns else df["scheme"]
    tick_labels = wrapped_labels(display)

    for ax, (column, ylabel, ylim, offset) in zip(axes, metrics):
        bars = ax.bar(
            range(len(df)),
            df[column].to_numpy(dtype=float),
            color=SCHEME_COLORS,
            edgecolor="black",
            linewidth=0.9,
            width=0.68,
        )
        ax.set_xticks(range(len(df)), tick_labels)
        ax.set_xlabel("执行方案", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_ylim(*ylim)
        style_axes(ax)
        ax.tick_params(axis="x", labelsize=10.8)
        for rect, value in zip(bars, df[column].to_numpy(dtype=float)):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                value + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.0,
            )

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
