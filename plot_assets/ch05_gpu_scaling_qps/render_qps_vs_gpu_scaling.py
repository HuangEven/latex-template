from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "gpu_scaling_qps.csv"
OUT_PATH = ROOT / "fig5_15_qps_vs_gpu.png"

SCHEME_LABELS = {
    "Replicated path": "索引复制 / 数据并行",
    "Replicated primary path": "索引复制 / 数据并行",
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


def main() -> None:
    setup_cjk_font()
    df = pd.read_csv(CSV_PATH)
    label_column = "scheme_desc" if "scheme_desc" in df.columns else "scheme"
    schemes = list(dict.fromkeys(df[label_column].tolist()))
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "d"]

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=220)

    for idx, scheme in enumerate(schemes):
        subset = df[df[label_column] == scheme].sort_values("gpu_count")
        ax.plot(
            subset["gpu_count"],
            subset["qps"],
            color="black",
            linestyle=line_styles[min(idx, len(line_styles) - 1)],
            linewidth=1.6,
            marker=markers[min(idx, len(markers) - 1)],
            markersize=6.8,
            markerfacecolor="white",
            label=SCHEME_LABELS.get(str(scheme), str(scheme)),
        )

        for x, y in zip(subset["gpu_count"], subset["qps"]):
            ax.text(
                x,
                y + 180,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.8,
            )

    ax.set_xlim(0.8, 4.2)
    ax.set_xticks([1, 2, 4])
    ax.set_ylim(0, 15000)
    ax.set_xlabel("GPU数量", fontsize=13)
    ax.set_ylabel("吞吐率（QPS）", fontsize=13)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, labelsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
