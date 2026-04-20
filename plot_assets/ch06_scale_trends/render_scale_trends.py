from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "scale_trends.csv"
OUT_PATH = ROOT / "fig6_8_scale_trends.png"

SCHEME_LABELS = {
    "Initial fusion pipeline": "初步融合方案",
    "Full fusion pipeline": "完整融合方案",
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


def style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, labelsize=12)


def metric_offset(metric: str) -> float:
    if metric == "qps":
        return 100
    if metric == "p99_latency_ms":
        return 2.0
    return 0.0004


def metric_format(metric: str) -> str:
    return "%.4f" if metric == "pr_auc" else "%.2f"


def main() -> None:
    setup_cjk_font()
    df = pd.read_csv(CSV_PATH)
    scales = ["1M", "10M", "50M", "100M"]
    metrics = [
        ("qps", "吞吐率（QPS）", (0, 4700)),
        ("p99_latency_ms", "p99延迟（毫秒）", (0, 140)),
        ("pr_auc", "PR-AUC", (0.76, 0.915)),
    ]
    line_styles = ["-", "--"]
    markers = ["o", "s"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.6), dpi=220)
    legend_handles = None
    legend_labels = None

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        label_column = "scheme_desc" if "scheme_desc" in df.columns else "scheme"
        for idx, (scheme, subset) in enumerate(df.groupby(label_column, sort=False)):
            subset = subset.reset_index(drop=True)
            line = ax.plot(
                range(len(subset)),
                subset[metric],
                color="black",
                linestyle=line_styles[idx],
                linewidth=1.5,
                marker=markers[idx],
                markersize=6.6,
                markerfacecolor="white",
                label=SCHEME_LABELS.get(str(scheme), str(scheme)),
            )[0]
            if legend_handles is None:
                legend_handles = []
                legend_labels = []
            if SCHEME_LABELS.get(str(scheme), str(scheme)) not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(SCHEME_LABELS.get(str(scheme), str(scheme)))
            for x, value in enumerate(subset[metric]):
                ax.text(x, value + metric_offset(metric), metric_format(metric) % value, ha="center", va="bottom", fontsize=9.2)

        ax.set_xticks(range(len(scales)), scales)
        ax.set_xlabel("数据规模", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_ylim(*ylim)
        style_axes(ax)

    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
