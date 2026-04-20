from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "fusion_scheme_compare.csv"

SCENARIO_LABELS = {
    "Full scoring": "PyTOD全量\n评分基线",
    "Full-scoring baseline": "PyTOD全量\n评分基线",
    "Recall-only": "仅召回\n筛选",
    "Recall-only screening": "仅召回\n筛选",
    "Initial fusion": "初步融合\n方案",
    "Initial fusion pipeline": "初步融合\n方案",
    "Full fusion": "完整融合\n方案",
    "Full fusion pipeline": "完整融合\n方案",
}

LEGEND_LABELS = {
    "Full scoring": "PyTOD全量评分基线",
    "Full-scoring baseline": "PyTOD全量评分基线",
    "Recall-only": "仅召回筛选",
    "Recall-only screening": "仅召回筛选",
    "Initial fusion": "初步融合方案",
    "Initial fusion pipeline": "初步融合方案",
    "Full fusion": "完整融合方案",
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


def wrapped_labels(values: pd.Series, width: int = 16) -> list[str]:
    labels = []
    for item in values:
        text = SCENARIO_LABELS.get(str(item), str(item))
        if "\n" in text:
            labels.append(text)
        else:
            labels.append(fill(text, width=width, break_long_words=False))
    return labels


def apply_axis_style(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, labelsize=12)


def render_accuracy(df: pd.DataFrame) -> None:
    display = df["scheme_desc"] if "scheme_desc" in df.columns else df["scheme"]
    labels = wrapped_labels(display)
    fig, ax = plt.subplots(figsize=(7.8, 5.3), dpi=220)
    bars = ax.bar(range(len(df)), df["pr_auc"], color="white", edgecolor="black", linewidth=1.2)
    ax.set_xticks(range(len(df)), labels)
    ax.set_xlabel("验证方案", fontsize=13)
    ax.set_ylabel("PR-AUC", fontsize=13)
    ax.set_ylim(0.82, 0.94)
    apply_axis_style(ax)
    for rect, value in zip(bars, df["pr_auc"]):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.004, f"{value:.4f}", ha="center", va="bottom", fontsize=9.6)
    fig.tight_layout()
    fig.savefig(ROOT / "fig6_4_fusion_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def render_qps(df: pd.DataFrame) -> None:
    display = df["scheme_desc"] if "scheme_desc" in df.columns else df["scheme"]
    labels = wrapped_labels(display)
    fig, ax = plt.subplots(figsize=(7.8, 5.3), dpi=220)
    bars = ax.bar(range(len(df)), df["qps"], color="#D1D1D1", edgecolor="black", linewidth=1.2)
    ax.set_xticks(range(len(df)), labels)
    ax.set_xlabel("验证方案", fontsize=13)
    ax.set_ylabel("吞吐率（QPS）", fontsize=13)
    ax.set_ylim(0, 5200)
    apply_axis_style(ax)
    for rect, value in zip(bars, df["qps"]):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 140, f"{value:.2f}", ha="center", va="bottom", fontsize=9.6)
    fig.tight_layout()
    fig.savefig(ROOT / "fig6_5_fusion_qps.png", bbox_inches="tight")
    plt.close(fig)


def render_tradeoff(df: pd.DataFrame) -> None:
    markers = ["o", "s", "^", "d"]
    line_styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=220)
    for idx, row in df.iterrows():
        ax.plot(
            [row["qps"]],
            [row["pr_auc"]],
            color="black",
            linestyle=line_styles[idx],
            marker=markers[idx],
            markersize=7.5,
            markerfacecolor="white",
            linewidth=1.4,
            label=LEGEND_LABELS.get(row["scheme_desc"] if "scheme_desc" in df.columns else row["scheme"], row["scheme_desc"] if "scheme_desc" in df.columns else row["scheme"]),
        )
        xoff, yoff, halign = tradeoff_label_offset(idx)
        ax.text(
            row["qps"] + xoff,
            row["pr_auc"] + yoff,
            f"{row['pr_auc']:.4f} / {row['qps']:.2f}",
            fontsize=9.2,
            ha=halign,
        )
    ax.set_xlabel("吞吐率（QPS）", fontsize=13)
    ax.set_ylabel("PR-AUC", fontsize=13)
    ax.set_xlim(2500, 4800)
    ax.set_ylim(0.83, 0.936)
    apply_axis_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False, fontsize=10.4)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ROOT / "fig6_6_fusion_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def tradeoff_label_offset(index: int) -> tuple[float, float, str]:
    if index == 0:
        return 70, 0.0009, "left"
    if index == 1:
        return -70, 0.0006, "right"
    if index == 2:
        return 70, 0.0009, "left"
    return 70, 0.0009, "left"


def main() -> None:
    setup_cjk_font()
    df = pd.read_csv(CSV_PATH)
    render_accuracy(df)
    render_qps(df)
    render_tradeoff(df)


if __name__ == "__main__":
    main()
