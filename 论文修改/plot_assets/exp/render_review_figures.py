from __future__ import annotations

import csv
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures" / "exp"

BASELINE_SOURCE = ROOT / "plot_assets" / "ch06_fusion_compare" / "fusion_scheme_compare.csv"
SINGLE_SOURCE = ROOT / "plot_assets" / "ch04_single_gpu_overview" / "single_gpu_overview_metrics.csv"
GPU_SCALING_SOURCE = ROOT / "plot_assets" / "ch05_gpu_scaling_qps" / "gpu_scaling_qps.csv"
IO_LANE_SOURCE = ROOT / "plot_assets" / "ch05_io_lane_impact" / "io_lane_impact.csv"

BASELINE_CSV = DATA_DIR / "baseline_effect_qps_comparison.csv"
SINGLE_CSV = DATA_DIR / "single_gpu_progressive_ablation.csv"
MULTIGPU_CSV = DATA_DIR / "multigpu_scaling_bottleneck_analysis.csv"


SCHEME_LABELS = {
    "Full scoring": "PyTOD全量\n评分",
    "Recall-only": "仅召回\n方案",
    "Initial fusion": "初步融合\n方案",
    "Full fusion": "完整融合\nFlashTOD",
    "Original baseline": "原始分段\n链路",
    "Recall-accelerated": "仅召回\n加速",
    "GPU handoff": "GPU主导\n链路",
    "Full optimization": "完整优化\n方案",
}


def setup_fonts() -> None:
    candidates = [
        "Songti SC",
        "SimSun",
        "Noto Serif CJK SC",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    cjk_font = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", cjk_font],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.9,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#BEBEBE", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.9, labelsize=9.5)


def label_for(scheme: str) -> str:
    return SCHEME_LABELS.get(scheme, fill(scheme, width=12, break_long_words=False))


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", dpi=600)
    plt.close(fig)


def write_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["figure_id", "scheme", "metric", "value", "unit", "data_source", "note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_baseline_csv() -> pd.DataFrame:
    source = pd.read_csv(BASELINE_SOURCE)
    rows: list[dict[str, str | float]] = []
    for _, row in source.iterrows():
        scheme = str(row["scheme"])
        note = "来自现有图6-4/图6-5使用的 fusion_scheme_compare.csv"
        rows.append(
            {
                "figure_id": "fig_baseline_effect_qps",
                "scheme": scheme,
                "metric": "PR-AUC",
                "value": f"{float(row['pr_auc']):.4f}",
                "unit": "",
                "data_source": "existing_reported",
                "note": note,
            }
        )
        rows.append(
            {
                "figure_id": "fig_baseline_effect_qps",
                "scheme": scheme,
                "metric": "QPS",
                "value": f"{float(row['qps']):.2f}",
                "unit": "QPS",
                "data_source": "existing_reported",
                "note": note,
            }
        )
    write_rows(BASELINE_CSV, rows)
    return source


def render_baseline(df: pd.DataFrame) -> None:
    labels = [label_for(str(item)) for item in df["scheme"]]
    colors = ["#FFFFFF", "#D9D9D9", "#BFBFBF", "#7A7A7A"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=600)

    for ax, metric, ylabel, ylim, offset in [
        (axes[0], "pr_auc", "PR-AUC", (0.80, 0.95), 0.004),
        (axes[1], "qps", "吞吐率（QPS）", (0, 5000), 110),
    ]:
        values = df[metric].astype(float).to_numpy()
        bars = ax.bar(range(len(df)), values, color=colors, edgecolor="black", linewidth=0.9, width=0.66)
        ax.set_xticks(range(len(df)), labels)
        ax.set_ylabel(ylabel, fontsize=10.5)
        style_axis(ax)
        ax.tick_params(axis="x", labelsize=8.6)
        for rect, value in zip(bars, values):
            text = f"{value:.4f}" if metric == "pr_auc" else f"{value:.0f}"
            ax.text(rect.get_x() + rect.get_width() / 2, value + offset, text, ha="center", va="bottom", fontsize=8.0)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, "baseline_effect_qps_comparison")


def build_single_csv() -> pd.DataFrame:
    source = pd.read_csv(SINGLE_SOURCE)
    metric_map = {
        "qps": ("QPS", "QPS"),
        "p99_ms": ("p99 latency", "ms"),
        "pci_transfer_count": ("PCIe transfer count", "count/query"),
        "gpu_util": ("GPU utilization", "%"),
    }
    rows: list[dict[str, str | float]] = []
    for _, row in source.iterrows():
        for column, (metric, unit) in metric_map.items():
            rows.append(
                {
                    "figure_id": "fig_single_gpu_progressive_ablation",
                    "scheme": str(row["scheme"]),
                    "metric": metric,
                    "value": f"{float(row[column]):.2f}",
                    "unit": unit,
                    "data_source": "existing_reported",
                    "note": "来自现有图4-6使用的 single_gpu_overview_metrics.csv",
                }
            )
    write_rows(SINGLE_CSV, rows)
    return source


def render_single(df: pd.DataFrame) -> None:
    labels = [label_for(str(item)) for item in df["scheme"]]
    colors = ["#FFFFFF", "#D9D9D9", "#BFBFBF", "#7A7A7A"]
    metrics = [
        ("qps", "吞吐率（QPS）", (0, 5200), 90),
        ("p99_ms", "p99延迟（ms）", (0, 75), 1.3),
        ("pci_transfer_count", "PCIe传输次数", (0, 7), 0.12),
        ("gpu_util", "GPU利用率（%）", (0, 75), 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), dpi=600)
    for ax, (column, ylabel, ylim, offset) in zip(axes.ravel(), metrics):
        values = df[column].astype(float).to_numpy()
        bars = ax.bar(range(len(df)), values, color=colors, edgecolor="black", linewidth=0.9, width=0.66)
        ax.set_xticks(range(len(df)), labels)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_ylim(*ylim)
        style_axis(ax)
        ax.tick_params(axis="x", labelsize=8.6)
        for rect, value in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + offset, f"{value:.2f}", ha="center", va="bottom", fontsize=8.0)
    fig.tight_layout(h_pad=1.8, w_pad=1.8)
    save_figure(fig, "single_gpu_progressive_ablation")


def build_multigpu_csv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaling = pd.read_csv(GPU_SCALING_SOURCE)
    io_lane = pd.read_csv(IO_LANE_SOURCE)
    base_qps = float(scaling.loc[scaling["gpu_count"] == 1, "qps"].iloc[0])
    ideal = pd.DataFrame({"gpu_count": [1, 2, 4], "qps": [base_qps, base_qps * 2, base_qps * 4]})

    rows: list[dict[str, str | float]] = []
    for _, row in scaling.iterrows():
        rows.append(
            {
                "figure_id": "fig_multigpu_scaling_bottleneck",
                "scheme": "Replicated path",
                "metric": f"QPS_{int(row['gpu_count'])}GPU",
                "value": f"{float(row['qps']):.2f}",
                "unit": "QPS",
                "data_source": "existing_reported",
                "note": "来自现有图5-4使用的 gpu_scaling_qps.csv",
            }
        )
    for _, row in ideal.iterrows():
        rows.append(
            {
                "figure_id": "fig_multigpu_scaling_bottleneck",
                "scheme": "Ideal linear reference",
                "metric": f"QPS_{int(row['gpu_count'])}GPU",
                "value": f"{float(row['qps']):.2f}",
                "unit": "QPS",
                "data_source": "existing_reported",
                "note": "由现有1GPU QPS推导的理想线性参考线，仅作理论参照，不作为实测结果",
            }
        )
    for _, row in io_lane.iterrows():
        rows.append(
            {
                "figure_id": "fig_multigpu_scaling_bottleneck",
                "scheme": str(row["scheme"]),
                "metric": f"p99_{int(row['gpu_count'])}GPU",
                "value": f"{float(row['p99_ms']):.2f}",
                "unit": "ms",
                "data_source": "existing_reported",
                "note": "来自现有图5-6使用的 io_lane_impact.csv",
            }
        )
    write_rows(MULTIGPU_CSV, rows)
    return scaling, io_lane, ideal


def render_multigpu(scaling: pd.DataFrame, io_lane: pd.DataFrame, ideal: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=600)

    ax = axes[0]
    ax.plot(
        scaling["gpu_count"],
        scaling["qps"],
        marker="o",
        color="black",
        linewidth=1.4,
        markersize=5.5,
        markerfacecolor="white",
        label="Replicated实测",
    )
    ax.plot(
        ideal["gpu_count"],
        ideal["qps"],
        linestyle="--",
        color="#9E9E9E",
        linewidth=1.1,
        label="理想线性参考",
    )
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("GPU数量", fontsize=10.5)
    ax.set_ylabel("吞吐率（QPS）", fontsize=10.5)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax = axes[1]
    pivot = io_lane.pivot(index="gpu_count", columns="scheme", values="p99_ms").sort_index()
    x_positions = range(len(pivot.index))
    width = 0.34
    topo = pivot["Topology-aware"].to_numpy()
    conflict = pivot["Shared conflict"].to_numpy()
    bars1 = ax.bar([x - width / 2 for x in x_positions], topo, width=width, color="#D9D9D9", edgecolor="black", linewidth=0.9, label="拓扑感知绑定")
    bars2 = ax.bar([x + width / 2 for x in x_positions], conflict, width=width, color="#7A7A7A", edgecolor="black", linewidth=0.9, label="共享路径冲突")
    ax.set_xticks(list(x_positions), [f"{int(item)} GPU" for item in pivot.index])
    ax.set_xlabel("GPU数量", fontsize=10.5)
    ax.set_ylabel("p99延迟（ms）", fontsize=10.5)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for bars in (bars1, bars2):
        for rect in bars:
            value = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, value + 0.9, f"{value:.1f}", ha="center", va="bottom", fontsize=7.8)

    fig.tight_layout(w_pad=2.5)
    save_figure(fig, "multigpu_scaling_bottleneck_analysis")


def main() -> None:
    setup_fonts()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline_csv()
    render_baseline(baseline)
    single = build_single_csv()
    render_single(single)
    scaling, io_lane, ideal = build_multigpu_csv()
    render_multigpu(scaling, io_lane, ideal)


if __name__ == "__main__":
    main()
