from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "single_multi_benefit_summary.csv"
OUT_PATH = ROOT / "fig6_7_single_multi_summary.png"

SCENARIO_LABELS = {
    "Original baseline": "原始链路\n基线",
    "Original pipeline baseline": "原始链路\n基线",
    "Single-GPU optimized": "完整单卡\n优化",
    "Single-GPU full optimization": "完整单卡\n优化",
    "2-GPU replicated": "2 GPU\n索引复制",
    "2-GPU replicated scaling": "2 GPU\n索引复制",
    "4-GPU replicated": "4 GPU\n索引复制",
    "4-GPU replicated scaling": "4 GPU\n索引复制",
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


def scenario_labels(df: pd.DataFrame) -> list[str]:
    labels = []
    source = df["scheme_desc"] if "scheme_desc" in df.columns else df["scheme"]
    for value in source.tolist():
        labels.append(SCENARIO_LABELS.get(str(value), fill(str(value), width=10, break_long_words=False)))
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
    labels = scenario_labels(df)
    metrics = [
        ("qps", "吞吐率（QPS）", (0, 15000), 220, "white"),
        ("p99_latency_ms", "p99延迟（毫秒）", (0, 75), 1.0, "#D1D1D1"),
        ("cpu_util", "CPU利用率（%）", (0, 70), 0.9, "#A6A6A6"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8), dpi=220)

    for ax, (column, ylabel, ylim, offset, color) in zip(axes, metrics):
        bars = ax.bar(range(len(df)), df[column], color=color, edgecolor="black", linewidth=1.2)
        ax.set_xticks(range(len(df)), labels)
        ax.set_xlabel("验证场景", fontsize=12.5)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_ylim(*ylim)
        style_axes(ax)
        ax.tick_params(axis="x", labelsize=11.0)
        for rect, value in zip(bars, df[column]):
            ax.text(rect.get_x() + rect.get_width() / 2, value + offset, f"{value:.2f}", ha="center", va="bottom", fontsize=9.4)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
