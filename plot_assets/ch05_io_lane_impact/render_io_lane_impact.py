from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "io_lane_impact.csv"
OUT_PATH = ROOT / "fig5_18_io_lane_impact.png"

SCHEME_LABELS = {
    "Topology-aware": "拓扑感知绑定",
    "Topology-aware I/O binding": "拓扑感知绑定",
    "Shared conflict": "共享路径冲突",
    "Shared-path conflict baseline": "共享路径冲突",
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


def build_matrix(df: pd.DataFrame, value_col: str) -> tuple[list[int], list[str], np.ndarray]:
    gpu_groups = list(dict.fromkeys(df["gpu_count"].tolist()))
    label_column = "scheme_desc" if "scheme_desc" in df.columns else "scheme"
    schemes = list(dict.fromkeys(df[label_column].tolist()))
    matrix = np.zeros((len(gpu_groups), len(schemes)))

    for i, gpu_count in enumerate(gpu_groups):
        for j, scheme in enumerate(schemes):
            row = df[(df["gpu_count"] == gpu_count) & (df[label_column] == scheme)].iloc[0]
            matrix[i, j] = float(row[value_col])

    return gpu_groups, schemes, matrix


def style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, labelsize=12)


def main() -> None:
    setup_cjk_font()
    df = pd.read_csv(CSV_PATH)
    gpu_groups, schemes, qps_values = build_matrix(df, "qps")
    _, _, p99_values = build_matrix(df, "p99_ms")

    x = np.arange(len(gpu_groups))
    width = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), dpi=220)
    colors = ["white", "#D1D1D1"]
    translated_labels = [SCHEME_LABELS.get(str(label), str(label)) for label in schemes]

    for ax, values, ylabel, ylim, offset in [
        (axes[0], qps_values, "吞吐率（QPS）", (0, 15000), 220),
        (axes[1], p99_values, "p99延迟（毫秒）", (0, 60), 0.95),
    ]:
        for idx, scheme in enumerate(schemes):
            bars = ax.bar(
                x + (idx - 0.5) * width,
                values[:, idx],
                width,
                label=SCHEME_LABELS.get(str(scheme), str(scheme)),
                color=colors[idx],
                edgecolor="black",
                linewidth=1.2,
            )
            for rect in bars:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height + offset,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                )

        ax.set_xticks(x, [f"{v} GPU" for v in gpu_groups])
        ax.set_xlabel("GPU配置", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_ylim(*ylim)
        style_axes(ax)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, translated_labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
