import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, ScalarFormatter

from energy_monitor.common.schema import SOURCE_TIMEZONE
from energy_monitor.models.expected.residuals import MAD_SCALE

DATASETS: Final = ("d1", "d2")
HORIZONS: Final = ("1h", "24h")
HORIZON_LABELS: Final = {"1h": "1 h à frente", "24h": "24 h à frente"}
FORECAST_MODELS: Final = ("seasonal_naive", "linear_regression", "lightgbm")
MODEL_LABELS: Final = {
    "seasonal_naive": "Ingênuo sazonal",
    "linear_regression": "Regressão linear",
    "lightgbm": "LightGBM",
    "cpu_bin_mean": "Média por faixa de CPU",
    "lightgbm_quantile": "LightGBM por quantis",
}
VARIANT_LABELS: Final = {"history_only": "só histórico", "with_load": "com carga"}
DETECTORS: Final = ("deviation", "fixed_threshold", "isolation_forest")
DETECTOR_LABELS: Final = {
    "deviation": "Desvio (z robusto)",
    "fixed_threshold": "Limiar fixo",
    "isolation_forest": "Isolation Forest",
}
FAMILIES: Final = {
    "e3c_excess_without_load": "Excesso sem carga",
    "e3d_high_power_zero_load": "Potência alta com carga zero",
}
ANOMALY_KINDS: Final = {
    "offset": "Acréscimo de potência",
    "drift": "Deriva gradual",
    "stuck": "Potência travada",
    "zero_load_high_power": "Potência alta com carga zero",
}
MAGNITUDES: Final = ("0.1", "0.2", "0.4")
H2_THRESHOLD_PERCENT: Final = 10.0
INTERVAL_TARGET: Final = 0.9
LATENCY_TARGET_MS: Final = 500.0
QUERY_TARGET_MS: Final = 1000.0
NEGLIGIBLE_MAD_WATTS: Final = 0.001
STRUCTURAL_BREAK_LOCAL: Final = "2019-02-25"
ANALYSIS_START_LOCAL: Final = "2019-03-15"

BLUE: Final = "#2a78d6"
ORANGE: Final = "#eb6834"
GRAY: Final = "#9a9993"
INK: Final = "#0b0b0b"
MUTED: Final = "#52514e"
GRID_COLOR: Final = "#e4e3df"
MODEL_COLORS: Final = {
    "seasonal_naive": GRAY,
    "linear_regression": BLUE,
    "lightgbm": ORANGE,
    "cpu_bin_mean": GRAY,
    "lightgbm_quantile": ORANGE,
}
DETECTOR_STYLES: Final = {
    "deviation": (BLUE, "o", "-"),
    "fixed_threshold": (GRAY, "s", "--"),
    "isolation_forest": (ORANGE, "^", ":"),
}
PARTITION_LABELS: Final = {"train": "Treino", "validation": "Validação", "test": "Teste"}
PARTITION_COLORS: Final = {"treino": "#eaf2fc", "validação": "#fdf3dc", "teste": "#fdeee7"}

STYLE: Final = {
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@dataclass(frozen=True)
class Reports:
    e1: dict[str, Any]
    e1_d2: dict[str, Any]
    e2: dict[str, Any]
    e3: dict[str, Any]
    e4: dict[str, Any]
    e5: dict[str, dict[str, Any]]
    e6: dict[str, Any]
    e6_queries: dict[str, Any]


def read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def load_reports(directory: Path) -> Reports:
    return Reports(
        e1=read_metrics(directory / "e1_data_quality_d1.json"),
        e1_d2=read_metrics(directory / "e1_data_quality_d2.json"),
        e2=read_metrics(directory / "e2_expected_consumption_test_d1.json"),
        e3=read_metrics(directory / "e3_detection_test_d1.json"),
        e4=read_metrics(directory / "e4_indicators_d1.json"),
        e5={
            name: read_metrics(directory / f"e5_forecasting_test_{name}.json") for name in DATASETS
        },
        e6=read_metrics(directory / "e6_system_d1.json"),
        e6_queries=read_metrics(directory / "e6_system_queries_d1.json"),
    )


def decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def new_figure(width: float, height: float, columns: int = 1, **options: Any) -> tuple[Figure, Any]:
    with plt.rc_context(STYLE):
        return plt.subplots(1, columns, figsize=(width, height), **options)


def comma_number(value: float, _position: int | None = None) -> str:
    return f"{value:g}".replace(".", ",")


def log_subdivisions(limits: tuple[float, float]) -> tuple[float, ...]:
    low, high = sorted(limits)
    decades = np.log10(high) - np.log10(max(low, 1e-12))
    return (1.0, 2.0, 5.0) if decades <= 2.5 else (1.0,)


def use_decimal_comma(axis_part: Any, limits: tuple[float, float]) -> None:
    if axis_part.get_scale() == "log":
        axis_part.set_major_locator(LogLocator(base=10, subs=log_subdivisions(limits)))
        axis_part.set_major_formatter(FuncFormatter(comma_number))
        axis_part.set_minor_formatter(NullFormatter())
    elif isinstance(axis_part.get_major_formatter(), ScalarFormatter):
        axis_part.set_major_formatter(FuncFormatter(comma_number))


def apply_style(figure: Figure) -> Figure:
    for axis in figure.axes:
        use_decimal_comma(axis.xaxis, axis.get_xlim())
        use_decimal_comma(axis.yaxis, axis.get_ylim())
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, color=GRID_COLOR, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.title.set_horizontalalignment("left")
        axis.title.set_fontweight("bold")
        axis.title.set_position((0, 1))
    return figure


def bottom_legend(figure: Figure, axis: Axes, columns: int, offset: float = -0.08) -> None:
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=columns,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, offset),
    )


def label_bars(
    axis: Axes,
    positions: Sequence[float],
    values: Sequence[float],
    digits: int,
    heights: Sequence[float] | None = None,
) -> None:
    anchors = values if heights is None else heights
    for position, value, height in zip(positions, values, anchors, strict=True):
        axis.annotate(
            decimal(value, digits),
            (position, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK,
        )


def forecast_bars(horizons: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (model, variant)
        for model in FORECAST_MODELS
        for variant in ("history_only", "with_load")
        if variant in horizons["1h"]["models"][model]
    ]


def bar_label(model: str, variant: str) -> str:
    if model == "seasonal_naive":
        return MODEL_LABELS[model]
    return f"{MODEL_LABELS[model]} ({VARIANT_LABELS[variant]})"


def draw_variant_bar(
    axis: Axes, positions: list[float], values: list[float], width: float, model: str, variant: str
) -> None:
    color = MODEL_COLORS[model]
    with_load = variant == "with_load"
    axis.bar(
        positions,
        values,
        width,
        color="white" if with_load else color,
        edgecolor=color,
        hatch="////" if with_load else None,
        linewidth=1.2,
        label=bar_label(model, variant),
    )


def plot_forecast_mase(e5: dict[str, dict[str, Any]], statistic: str = "mase_median") -> Figure:
    figure, axes = new_figure(10, 4.2, 2, sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        horizons = e5[dataset]["horizons"]
        bars = forecast_bars(horizons)
        width = 0.8 / len(bars)
        for index, (model, variant) in enumerate(bars):
            values = [horizons[h]["models"][model][variant][statistic] for h in HORIZONS]
            positions = [h + (index - (len(bars) - 1) / 2) * width for h in range(len(HORIZONS))]
            draw_variant_bar(axis, positions, values, width * 0.9, model, variant)
            label_bars(axis, positions, values, 2)
        axis.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        axis.set_yscale("log")
        axis.set_xticks(range(len(HORIZONS)), [HORIZON_LABELS[h] for h in HORIZONS])
        axis.set_title(dataset.upper())
        axis.grid(axis="x", visible=False)
    name = "MASE mediano" if statistic == "mase_median" else "MASE médio"
    axes[0].set_ylabel(f"{name} no teste (menor é melhor)")
    bottom_legend(figure, axes[0], 3)
    return apply_style(figure)


def host_mase_rows(e5: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            models = e5[dataset]["horizons"][horizon]["models"]
            for model in FORECAST_MODELS:
                variant = "history_only" if model == "seasonal_naive" else "with_load"
                for host, values in models[model][variant]["per_host"].items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "model": model,
                            "host": host,
                            "mase": values["mase"],
                        }
                    )
    return pd.DataFrame(rows)


def plot_host_mase_distribution(e5: dict[str, dict[str, Any]]) -> Figure:
    rows = host_mase_rows(e5)
    figure, axes = new_figure(10, 4.4, 2, sharey=True)
    width = 0.8 / len(FORECAST_MODELS)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for index, model in enumerate(FORECAST_MODELS):
            offset = (index - (len(FORECAST_MODELS) - 1) / 2) * width
            positions = [h + offset for h in range(len(HORIZONS))]
            values = [
                rows.query("dataset == @dataset and horizon == @h and model == @model")["mase"]
                for h in HORIZONS
            ]
            boxes = axis.boxplot(
                values,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                medianprops={"color": INK},
                flierprops={"markersize": 2.5, "markeredgecolor": MUTED},
            )
            for patch in boxes["boxes"]:
                patch.set_facecolor(MODEL_COLORS[model])
                patch.set_alpha(0.75)
            means = [
                e5[dataset]["horizons"][h]["models"][model][
                    "history_only" if model == "seasonal_naive" else "with_load"
                ]["mase"]
                for h in HORIZONS
            ]
            axis.scatter(positions, means, marker="D", color=INK, s=18, zorder=3)
        axis.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        axis.set_yscale("log")
        axis.set_xticks(range(len(HORIZONS)), [HORIZON_LABELS[h] for h in HORIZONS])
        axis.set_title(dataset.upper())
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("MASE por servidor no teste (escala logarítmica)")
    handles = [
        Patch(facecolor=MODEL_COLORS[model], alpha=0.75, label=model_variant_label(model))
        for model in FORECAST_MODELS
    ]
    handles.append(
        axes[0].scatter([], [], marker="D", color=INK, s=18, label="Média entre servidores")
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    return apply_style(figure)


def model_variant_label(model: str) -> str:
    if model == "seasonal_naive":
        return MODEL_LABELS[model]
    return f"{MODEL_LABELS[model]} (com carga)"


def host_coverage_rows(e1: dict[str, Any], dataset: str) -> pd.DataFrame:
    coverage = e1["coverage"]
    excluded = set(coverage["excluded_hosts"])
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "host": host,
                "partition": partition,
                "coverage": values[partition] * 100,
                "excluded": host in excluded,
            }
            for host, values in coverage["per_host"].items()
            for partition in coverage["partitions"]
        ]
    )


def plot_host_coverage(e1: dict[str, Any], e1_d2: dict[str, Any]) -> Figure:
    figure, axes = new_figure(10, 4.2, 2, sharey=True)
    generator = np.random.default_rng(0)
    for axis, (dataset, report) in zip(axes, (("d1", e1), ("d2", e1_d2)), strict=True):
        rows = host_coverage_rows(report, dataset)
        partitions = list(report["coverage"]["partitions"])
        for position, partition in enumerate(partitions):
            subset = rows[rows["partition"] == partition]
            jitter = generator.uniform(-0.18, 0.18, len(subset))
            colors = np.where(subset["excluded"], ORANGE, BLUE)
            axis.scatter(
                position + jitter, subset["coverage"], c=colors, s=14, alpha=0.8, linewidths=0
            )
        threshold = report["coverage"]["min_coverage"] * 100
        axis.axhline(threshold, color=MUTED, linewidth=1, linestyle="--")
        axis.text(
            len(partitions) - 0.5,
            threshold + 1.5,
            f"mínimo exigido: {threshold:.0f}%",
            ha="right",
            fontsize=8,
            color=MUTED,
        )
        included = report["coverage"]["eligible_count"]
        total = len(report["coverage"]["per_host"])
        axis.set_title(f"{dataset.upper()}: {included} de {total} servidores incluídos")
        axis.set_xticks(range(len(partitions)), [PARTITION_LABELS[p] for p in partitions])
        axis.set_ylim(-5, 105)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Leituras válidas por servidor (%)")
    handles = [
        Patch(facecolor=BLUE, label="Servidor incluído"),
        Patch(facecolor=ORANGE, label="Servidor excluído"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    return apply_style(figure)


def host_mae(result: dict[str, Any]) -> pd.Series:
    return pd.Series({host: values["mae"] for host, values in result["per_host"].items()})


def plot_paired_host_errors(e5: dict[str, dict[str, Any]], horizon: str = "1h") -> Figure:
    figure, axes = new_figure(10, 4.6, 2)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        models = e5[dataset]["horizons"][horizon]["models"]
        linear = host_mae(models["linear_regression"]["with_load"])
        boosted = host_mae(models["lightgbm"]["with_load"]).reindex(linear.index)
        better = int((boosted < linear).sum())
        axis.scatter(linear, boosted, s=22, color=ORANGE, edgecolor="white", linewidth=0.5)
        limits = [min(linear.min(), boosted.min()) * 0.8, max(linear.max(), boosted.max()) * 1.2]
        axis.plot(limits, limits, color=MUTED, linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("MAE da regressão linear (W)")
        axis.set_title(f"{dataset.upper()}: LightGBM melhor em {better} de {len(linear)}")
        axis.text(
            0.97,
            0.04,
            "abaixo da diagonal:\nLightGBM erra menos",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    axes[0].set_ylabel("MAE do LightGBM (W)")
    return apply_style(figure)


def plot_total_forecast_error(e5: dict[str, dict[str, Any]]) -> Figure:
    figure, axes = new_figure(10, 4.0, 2)
    models = ("seasonal_naive", "linear_regression", "lightgbm")
    width = 0.8 / len(models)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        horizons = e5[dataset]["horizons"]
        for index, model in enumerate(models):
            variant = (
                "with_load" if "with_load" in horizons["1h"]["models"][model] else "history_only"
            )
            values = [horizons[h]["models"][model][variant]["total"]["mae"] for h in HORIZONS]
            positions = [h + (index - 1) * width for h in range(len(HORIZONS))]
            axis.bar(
                positions,
                values,
                width * 0.9,
                color=MODEL_COLORS[model],
                label=bar_label(model, variant),
            )
            label_bars(axis, positions, values, 0)
        axis.set_xticks(range(len(HORIZONS)), [HORIZON_LABELS[h] for h in HORIZONS])
        axis.set_title(f"{dataset.upper()}: soma dos servidores")
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE do total do data center (W)")
    bottom_legend(figure, axes[0], 3)
    return apply_style(figure)


def load_effect_rows(e5: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            comparisons = e5[dataset]["horizons"][horizon]["comparisons"]
            for model in ("lightgbm", "linear_regression"):
                comparison = comparisons[f"{model}_with_load_vs_history_only"]
                rows.append(
                    {
                        "label": f"{dataset.upper()}, {horizon}",
                        "model": model,
                        "reduction": comparison["mae_reduction_percent"],
                        "p_holm": comparison["p_value_holm"],
                    }
                )
    return pd.DataFrame(rows)


def p_value_text(value: float) -> str:
    return "p < 0,001" if value < 0.001 else f"p = {decimal(value, 3)}"


def plot_load_effect(e5: dict[str, dict[str, Any]]) -> Figure:
    rows = load_effect_rows(e5)
    labels = list(dict.fromkeys(rows["label"]))
    figure, axis = new_figure(10, 4.4)
    width = 0.38
    for index, model in enumerate(("lightgbm", "linear_regression")):
        subset = rows[rows["model"] == model].set_index("label").reindex(labels)
        positions = np.arange(len(labels)) + (index - 0.5) * width
        axis.bar(
            positions,
            subset["reduction"],
            width * 0.92,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        if model == "lightgbm":
            for position, (_, row) in zip(positions, subset.iterrows(), strict=True):
                offset = 0.4 if row["reduction"] >= 0 else -0.4
                axis.text(
                    position,
                    row["reduction"] + offset,
                    p_value_text(row["p_holm"]),
                    ha="center",
                    va="bottom" if offset > 0 else "top",
                    fontsize=7.5,
                    color=INK,
                )
    axis.axhline(H2_THRESHOLD_PERCENT, color=INK, linewidth=1.1, linestyle="--")
    axis.text(
        len(labels) - 0.5,
        H2_THRESHOLD_PERCENT + 0.3,
        "redução de 10%",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    axis.axhline(0, color=MUTED, linewidth=0.8)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("Redução do MAE ao incluir a carga (%)")
    axis.set_ylim(min(-6.0, rows["reduction"].min() - 2), H2_THRESHOLD_PERCENT + 3)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper left", fontsize=8.5)
    return apply_style(figure)


def coverage_rows(e5: dict[str, dict[str, Any]], e2: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            variants = e5[dataset]["horizons"][horizon]["models"]["lightgbm"]
            for variant, result in variants.items():
                rows.append(
                    {
                        "label": f"{dataset.upper()} {horizon}\n{VARIANT_LABELS[variant]}",
                        "coverage": result["interval_coverage_5_95"],
                        "group": "previsão",
                    }
                )
    rows.append(
        {
            "label": "D1\nesperado",
            "coverage": e2["models"]["lightgbm_quantile"]["coverage_5_95"],
            "group": "esperado",
        }
    )
    return pd.DataFrame(rows)


def plot_interval_coverage(e5: dict[str, dict[str, Any]], e2: dict[str, Any]) -> Figure:
    rows = coverage_rows(e5, e2)
    figure, axis = new_figure(10, 4.2)
    colors = [ORANGE if group == "previsão" else BLUE for group in rows["group"]]
    positions = np.arange(len(rows))
    values = rows["coverage"] * 100
    axis.bar(positions, values, 0.7, color=colors)
    label_bars(axis, positions, values.tolist(), 1)
    axis.axhline(INTERVAL_TARGET * 100, color=INK, linewidth=1.1, linestyle="--")
    axis.text(len(rows) - 0.5, INTERVAL_TARGET * 100 + 0.4, "alvo: 90%", ha="right", fontsize=8)
    axis.set_xticks(positions, rows["label"], fontsize=8)
    handles = [
        Patch(color=ORANGE, label="Previsão (LightGBM)"),
        Patch(color=BLUE, label="Consumo esperado (LightGBM por quantis)"),
    ]
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=8.5)
    axis.set_ylim(70, 100)
    axis.set_ylabel("Valores reais dentro do intervalo 5–95% (%)")
    axis.grid(axis="x", visible=False)
    return apply_style(figure)


def plot_expected_consumption(e2: dict[str, Any]) -> Figure:
    models = ("cpu_bin_mean", "linear_regression", "lightgbm_quantile")
    figure, axis = new_figure(10, 4.2)
    values = [
        host_mae(
            {
                "per_host": {
                    host: {"mae": mae} for host, mae in e2["models"][model]["per_host_mae"].items()
                }
            }
        )
        for model in models
    ]
    boxes = axis.boxplot(
        values,
        orientation="horizontal",
        patch_artist=True,
        widths=0.55,
        medianprops={"color": INK},
        flierprops={"markersize": 3, "markeredgecolor": MUTED},
    )
    for patch, model in zip(boxes["boxes"], models, strict=True):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.75)
    for position, model in enumerate(models, start=1):
        overall = e2["models"][model]["mae"]
        axis.scatter(overall, position, marker="D", color=INK, zorder=3, s=26)
        axis.annotate(
            f"MAE geral {decimal(overall)} W",
            (overall, position),
            xytext=(0, -27),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
        )
    axis.set_yticks(range(1, len(models) + 1), [MODEL_LABELS[model] for model in models])
    axis.set_xscale("log")
    axis.set_xlabel("MAE por servidor no teste (W, escala logarítmica)")
    return apply_style(figure)


def plot_detection_rates(e3: dict[str, Any]) -> Figure:
    rates: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        for detector, values in e3["families"][family]["detection_rate_by_kind"].items():
            rates.setdefault(detector, {}).update(values)
    figure, axes = new_figure(10, 3.4, len(ANOMALY_KINDS), sharey=True)
    percentages = [float(m) * 100 for m in MAGNITUDES]
    for axis, (kind, title) in zip(axes, ANOMALY_KINDS.items(), strict=True):
        for detector in DETECTORS:
            color, marker, line = DETECTOR_STYLES[detector]
            values = [rates[detector][f"{kind}_{m}"] * 100 for m in MAGNITUDES]
            axis.plot(
                percentages,
                values,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.8,
                linestyle=line,
                label=DETECTOR_LABELS[detector],
            )
        axis.set_title(title, fontsize=9.5)
        axis.set_xticks(percentages, [f"{p:.0f}%" for p in percentages])
        axis.set_ylim(-3, 105)
    axes[0].set_ylabel("Anomalias detectadas (%)")
    figure.supxlabel(
        "Magnitude da anomalia injetada (fração da potência média)", fontsize=9, y=-0.04
    )
    bottom_legend(figure, axes[0], 3, -0.16)
    return apply_style(figure)


def summary_value(
    e3: dict[str, Any], family: str, detector: str, metric: str
) -> tuple[float, float]:
    values = e3["families"][family]["summary"][detector][metric]
    return values["mean"], values["std"]


def detector_panel(
    axis: Axes, title: str, values: Sequence[tuple[float, float]], digits: int, percent: bool
) -> None:
    scale = 100 if percent else 1
    means = [mean * scale for mean, _ in values]
    errors = [std * scale for _, std in values]
    colors = [DETECTOR_STYLES[d][0] for d in DETECTORS]
    positions = np.arange(len(DETECTORS))
    axis.bar(positions, means, 0.65, yerr=errors, capsize=3, color=colors, ecolor=MUTED)
    heights = [mean + error for mean, error in zip(means, errors, strict=True)]
    label_bars(axis, positions.tolist(), means, digits, heights)
    axis.set_xticks(positions, ["Desvio", "Limiar\nfixo", "Isolation\nForest"])
    axis.tick_params(axis="x", labelsize=8)
    axis.set_title(title, fontsize=9.5)
    axis.grid(axis="x", visible=False)


def plot_detector_summary(e3: dict[str, Any]) -> Figure:
    figure, axes = new_figure(11, 3.8, 4)
    legit = [
        summary_value(e3, "e3b_legit_peaks", d, "false_positive_rate_on_peaks") for d in DETECTORS
    ]
    detector_panel(axes[0], "Picos legítimos com alerta (%)\nmenor é melhor", legit, 0, True)
    for axis, metric, title in (
        (axes[1], "recall", "Revocação: excesso sem carga\nmaior é melhor"),
        (axes[2], "f1", "F1: excesso sem carga\nmaior é melhor"),
    ):
        values = [summary_value(e3, "e3c_excess_without_load", d, metric) for d in DETECTORS]
        detector_panel(axis, title, values, 2, False)
    normal = [(e3["normal_operation"][d]["alert_episodes_per_host_day"], 0.0) for d in DETECTORS]
    detector_panel(
        axes[3], "Alertas por servidor por dia\nsem anomalia (menor é melhor)", normal, 2, False
    )
    figure.tight_layout()
    return apply_style(figure)


def plot_residual_scale(statistics: dict[str, dict[str, float]], open_threshold: float) -> Figure:
    mad = pd.Series({host: values["mad"] for host, values in statistics.items()})
    trigger = (open_threshold * MAD_SCALE * mad).sort_values()
    figure, axis = new_figure(10, 4.2)
    positions = np.arange(len(trigger))
    axis.bar(positions, trigger.clip(lower=0.01), 0.8, color=BLUE)
    axis.set_yscale("log")
    axis.axhline(10, color=MUTED, linewidth=1, linestyle="--")
    axis.text(0, 11, "10 W", fontsize=8, color=MUTED)
    negligible = int((mad < NEGLIGIBLE_MAD_WATTS).sum())
    axis.set_title(
        f"Desvio em watts necessário para z = {decimal(open_threshold, 0)} "
        f"(mediana {decimal(float(trigger.median()), 1)} W; "
        f"MAD < 0,001 W em {negligible} servidores)"
    )
    axis.set_xlabel(
        "Servidores do D1, em ordem crescente (valores abaixo de 0,01 W exibidos em 0,01 W)"
    )
    axis.set_ylabel("Desvio necessário (W, escala logarítmica)")
    axis.set_xticks([])
    axis.grid(axis="x", visible=False)
    return apply_style(figure)


def plot_idle_energy(e4: dict[str, Any]) -> Figure:
    thresholds = sorted(e4["thresholds"], key=float)
    figure, axis = new_figure(10, 4.0)
    width = 0.38
    for index, (metric, label, color) in enumerate(
        (
            ("idle_energy_share", "Energia em ociosidade", BLUE),
            ("idle_time_share", "Tempo em ociosidade", GRAY),
        )
    ):
        estimates = [e4["thresholds"][t]["bootstrap"][metric] for t in thresholds]
        values = np.array([item["estimate"] for item in estimates]) * 100
        lower = values - np.array([item["lower"] for item in estimates]) * 100
        upper = np.array([item["upper"] for item in estimates]) * 100 - values
        positions = np.arange(len(thresholds)) + (index - 0.5) * width
        axis.bar(
            positions,
            values,
            width * 0.92,
            yerr=[lower, upper],
            capsize=3,
            color=color,
            ecolor=MUTED,
            label=label,
        )
        label_bars(axis, positions.tolist(), values.tolist(), 1, (values + upper).tolist())
    axis.set_xticks(range(len(thresholds)), [f"CPU < {t}%" for t in thresholds])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Parcela do período de teste (%), IC 95%")
    axis.set_title(f"Energia total no teste: {decimal(e4['energy_kwh'], 1)} kWh")
    bottom_legend(figure, axis, 2)
    axis.grid(axis="x", visible=False)
    return apply_style(figure)


def plot_system_latency(e6: dict[str, Any], queries: dict[str, Any]) -> Figure:
    speeds = list(e6["speeds"])
    figure, axes = new_figure(10, 4.0, 2, gridspec_kw={"width_ratios": [3, 1]})
    width = 0.38
    for index, (metric, label, color) in enumerate(
        (("latency_ms_p50", "p50", GRAY), ("latency_ms_p95", "p95", BLUE))
    ):
        values = [e6["speeds"][speed][metric] for speed in speeds]
        positions = np.arange(len(speeds)) + (index - 0.5) * width
        axes[0].bar(positions, values, width * 0.92, color=color, label=label)
        label_bars(axes[0], positions.tolist(), values, 0)
    axes[0].axhline(LATENCY_TARGET_MS, color=INK, linewidth=1.1, linestyle="--")
    axes[0].text(-0.45, LATENCY_TARGET_MS + 40, "meta: 500 ms", ha="left", fontsize=8)
    axes[0].set_xticks(range(len(speeds)), [speed.replace("x", "×") for speed in speeds])
    axes[0].set_ylabel("Latência de processamento (ms)")
    axes[0].set_title("Ingestão com 76 servidores")
    axes[0].legend(loc="upper right", fontsize=8.5)
    p95 = queries["latency_ms_p95"]
    axes[1].bar([0], [p95], 0.5, color=BLUE)
    label_bars(axes[1], [0], [p95], 0)
    axes[1].axhline(QUERY_TARGET_MS, color=INK, linewidth=1.1, linestyle="--")
    axes[1].text(0.3, QUERY_TARGET_MS + 30, "meta: 1 s", ha="right", fontsize=8)
    axes[1].set_xticks([0], ["p95"])
    axes[1].set_xlim(-0.6, 0.6)
    axes[1].set_title("Consulta do painel")
    for axis in axes:
        axis.grid(axis="x", visible=False)
    return apply_style(figure)


def daily_means(grid: pd.DataFrame, timezone: str) -> pd.DataFrame:
    observed = grid[grid["observed"]] if "observed" in grid else grid
    local_day = observed["ts"].dt.tz_convert(timezone).dt.tz_localize(None).dt.floor("D")
    return observed.groupby(local_day)[["cpu_load", "power"]].mean()


def local_bound(value: str, timezone: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert(timezone).tz_localize(None)


def partition_spans(
    e1: dict[str, Any], timezone: str
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    split = e1["split"]
    bounds = [
        local_bound(split[key]["utc"], timezone)
        for key in ("start", "train_end", "validation_end", "end")
    ]
    return list(zip(PARTITION_COLORS, bounds[:-1], bounds[1:], strict=True))


def plot_structural_break(
    grid: pd.DataFrame, e1: dict[str, Any], timezone: str = SOURCE_TIMEZONE
) -> Figure:
    daily = daily_means(grid, timezone)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for axis, column, label, color in (
        (axes[0], "cpu_load", "CPU média (%)", BLUE),
        (axes[1], "power", "Potência média (W)", ORANGE),
    ):
        for name, start, end in partition_spans(e1, timezone):
            axis.axvspan(start, end, color=PARTITION_COLORS[name], zorder=0, label=name)
        axis.plot(daily.index, daily[column], color=color, linewidth=1.6)
        for moment in (STRUCTURAL_BREAK_LOCAL, ANALYSIS_START_LOCAL):
            axis.axvline(pd.Timestamp(moment), color=MUTED, linewidth=1, linestyle="--")
        axis.set_ylabel(label)
    axes[0].text(
        pd.Timestamp(STRUCTURAL_BREAK_LOCAL),
        1.0,
        "quebra estrutural ",
        transform=axes[0].get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    axes[0].text(
        pd.Timestamp(ANALYSIS_START_LOCAL),
        1.0,
        " início da análise",
        transform=axes[0].get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axes[1].legend(loc="lower left", fontsize=8, ncol=3)
    axes[0].set_title("Média diária no D1 (horário de Melbourne)")
    figure.tight_layout()
    return apply_style(figure)


FIGURES: Final[dict[str, Callable[[Reports], Figure]]] = {
    "h1_mase_mediano": lambda r: plot_forecast_mase(r.e5, "mase_median"),
    "h1_mase_medio": lambda r: plot_forecast_mase(r.e5, "mase"),
    "h1_mase_por_servidor": lambda r: plot_host_mase_distribution(r.e5),
    "e1_cobertura_por_servidor": lambda r: plot_host_coverage(r.e1, r.e1_d2),
    "h1_erro_por_servidor": lambda r: plot_paired_host_errors(r.e5, "1h"),
    "h1_erro_total_data_center": lambda r: plot_total_forecast_error(r.e5),
    "h2_efeito_da_carga": lambda r: plot_load_effect(r.e5),
    "e2_e5_cobertura_intervalos": lambda r: plot_interval_coverage(r.e5, r.e2),
    "e2_consumo_esperado": lambda r: plot_expected_consumption(r.e2),
    "h3_resumo_detectores": lambda r: plot_detector_summary(r.e3),
    "h3_deteccao_por_tipo": lambda r: plot_detection_rates(r.e3),
    "e4_ociosidade": lambda r: plot_idle_energy(r.e4),
    "e6_latencia": lambda r: plot_system_latency(r.e6, r.e6_queries),
}


def save_figure(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def detector_open_threshold(reports_dir: Path) -> float:
    selection = read_metrics(reports_dir / "e3_detection_selection_d1.json")
    return float(selection["tuning"]["deviation"]["chosen_parameters"]["open_threshold"])


def generate_figures(
    reports_dir: Path,
    output_dir: Path,
    grid_path: Path | None = None,
    residuals_path: Path | None = None,
) -> list[Path]:
    reports = load_reports(reports_dir)
    paths = [
        save_figure(build(reports), output_dir / f"{name}.png") for name, build in FIGURES.items()
    ]
    if residuals_path is not None and residuals_path.exists():
        statistics = json.loads(residuals_path.read_text(encoding="utf-8"))["statistics"]
        figure = plot_residual_scale(statistics, detector_open_threshold(reports_dir))
        paths.append(save_figure(figure, output_dir / "h3_escala_do_z.png"))
    if grid_path is not None and grid_path.exists():
        grid = pd.read_parquet(grid_path, columns=["ts", "cpu_load", "power", "observed"])
        paths.append(
            save_figure(
                plot_structural_break(grid, reports.e1), output_dir / "e1_quebra_estrutural.png"
            )
        )
    return paths


def parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera os gráficos dos resultados dos experimentos")
    parser.add_argument("--reports-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--grid", type=Path, default=Path("data/processed/d1_grid.parquet"))
    parser.add_argument(
        "--residuals", type=Path, default=Path("artifacts/models/d1_residual_statistics.json")
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(arguments)
    for path in generate_figures(
        parsed.reports_dir, parsed.output_dir, parsed.grid, parsed.residuals
    ):
        print(path)


if __name__ == "__main__":
    main()
