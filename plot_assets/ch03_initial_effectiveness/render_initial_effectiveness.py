from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "initial_effectiveness_metrics.csv"
OUT_PATH = ROOT.parent.parent / "figures" / "fig3-5_initial_effectiveness_across_datasets.pdf"


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


def style_axis(ax, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_xlabel("数据集", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, labelsize=11.2)


def annotate_bars(ax, bars, offset: float) -> None:
    for rect in bars:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height + offset,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.8,
        )


def main() -> None:
    setup_cjk_font()
    df = pd.read_csv(CSV_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.6), dpi=220)
    fig.subplots_adjust(wspace=0.26, left=0.06, right=0.99, bottom=0.18, top=0.80)

    grouped_specs = [
        ("pr_auc", "各数据集上的PR-AUC", "PR-AUC"),
        ("recall_at_k", "各数据集上的Recall@K", "Recall@K"),
    ]

    legend_handles = None

    for ax, (metric, title, ylabel) in zip(axes[:2], grouped_specs):
        subset = df[df["metric"] == metric].copy()
        labels = subset["dataset"].tolist()
        x = range(len(labels))
        width = 0.37

        bars_fusion = ax.bar(
            [i - width / 2 for i in x],
            subset["fusion_system"],
            width=width,
            label="融合系统",
            color="white",
            edgecolor="black",
            linewidth=1.2,
        )
        bars_baseline = ax.bar(
            [i + width / 2 for i in x],
            subset["baseline_system"],
            width=width,
            label="基线系统",
            color="#D1D1D1",
            edgecolor="black",
            linewidth=1.2,
        )
        legend_handles = (bars_fusion[0], bars_baseline[0])

        ax.set_title(title, fontsize=15.5, pad=10)
        ax.set_xticks(list(x), labels)
        style_axis(ax, ylabel, (0, 0.33))
        annotate_bars(ax, bars_fusion, 0.004)
        annotate_bars(ax, bars_baseline, 0.004)

    subset = df[df["metric"] == "topk_overlap"].copy()
    labels = subset["dataset"].tolist()
    x = range(len(labels))
    bars = axes[2].bar(
        list(x),
        subset["single_value"],
        width=0.72,
        color="#D1D1D1",
        edgecolor="black",
        linewidth=1.2,
    )
    axes[2].set_title("各数据集上的Top-K重合度", fontsize=15.5, pad=10)
    axes[2].set_xticks(list(x), labels)
    style_axis(axes[2], "Top-K重合度", (0, 1.05))
    annotate_bars(axes[2], bars, 0.012)

    fig.legend(
        legend_handles,
        ["融合系统", "基线系统"],
        loc="upper left",
        bbox_to_anchor=(0.06, 0.975),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
