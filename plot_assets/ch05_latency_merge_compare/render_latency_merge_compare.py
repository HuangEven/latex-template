from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "latency_merge_compare.csv"
OUT_PATH = ROOT / "fig5_16_latency_merge_compare.png"

SCENARIO_LABELS = {
    "Replicated path": "索引复制 /\n数据并行",
    "Replicated primary path": "索引复制 /\n数据并行",
    "CPU aggregation": "局部评分 +\nCPU侧聚合",
    "CPU-side aggregation fallback": "局部评分 +\nCPU侧聚合",
    "GPU merge": "局部评分 +\nGPU侧归并",
    "GPU-side merge fallback": "局部评分 +\nGPU侧归并",
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


def main() -> None:
    setup_cjk_font()
    df = pd.read_csv(CSV_PATH)
    plot_values = df[["avg_latency_ms", "p99_latency_ms"]].to_numpy(dtype=float)

    x = np.arange(len(df))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=220)
    bars_avg = ax.bar(
        x - width / 2,
        plot_values[:, 0],
        width,
        label="平均延迟",
        color="white",
        edgecolor="black",
        linewidth=1.2,
    )
    bars_p99 = ax.bar(
        x + width / 2,
        plot_values[:, 1],
        width,
        label="p99延迟",
        color="#D1D1D1",
        edgecolor="black",
        linewidth=1.2,
    )

    display = df["scheme_desc"] if "scheme_desc" in df.columns else df["scheme"]
    ax.set_xticks(x, wrapped_labels(display))
    ax.set_ylim(0, 44)
    ax.set_xlabel("归并方案", fontsize=13)
    ax.set_ylabel("延迟（毫秒）", fontsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, labelsize=11.2)
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    for bars in (bars_avg, bars_p99):
        for rect in bars:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.65,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.8,
            )

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
