"""Analytical SQL queries, statistical fits, and publication-grade scaling-relation plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .database import GalaxyDatabase

PHYSICAL_CORR_COLUMNS: tuple[str, ...] = (
    "redshift",
    "abs_mag_r",
    "color_gr",
    "color_ug",
    "log_stellar_mass",
    "log_sfr",
    "re_kpc",
    "concentration",
    "frac_dev_r",
)

PHYSICAL_CORR_LABELS: dict[str, str] = {
    "redshift": "z",
    "abs_mag_r": "M_r",
    "color_gr": "g - r",
    "color_ug": "u - g",
    "log_stellar_mass": "log M*",
    "log_sfr": "log SFR",
    "re_kpc": "R_50 [kpc]",
    "concentration": "C (R90/R50)",
    "frac_dev_r": "fracDeV_r",
}


def load_morphology_summary(db: GalaxyDatabase) -> pd.DataFrame:
    """Return the GROUP BY morph_proxy summary DataFrame."""
    return db.get_morphology_summary()


def load_scaling_relations_sample(
    db: GalaxyDatabase,
    z_min: float | None = None,
    z_max: float | None = None,
    morph_proxy: str | None = None,
) -> pd.DataFrame:
    """Load galaxies filtered by optional redshift bounds and morphology proxy."""
    clauses: list[str] = ["1 = 1"]
    params: list[Any] = []

    if z_min is not None:
        clauses.append("redshift >= ?")
        params.append(float(z_min))
    if z_max is not None:
        clauses.append("redshift <= ?")
        params.append(float(z_max))
    if morph_proxy is not None:
        clauses.append("morph_proxy = ?")
        params.append(morph_proxy)

    sql = f"""
    SELECT
        objID,
        redshift,
        abs_mag_r,
        color_gr,
        color_ug,
        log_stellar_mass,
        log_sfr,
        re_kpc,
        concentration,
        frac_dev_r,
        p_el_debiased,
        p_cs_debiased,
        morph_proxy
    FROM galaxies
    WHERE {' AND '.join(clauses)}
    ORDER BY objID
    """
    return db.query_df(sql, params=params)


def load_redshift_binned_stats(
    db: GalaxyDatabase,
    bin_width: float = 0.02,
) -> pd.DataFrame:
    """Aggregate galaxy counts and mean physical properties in redshift bins."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    sql = """
    SELECT
        ROUND(CAST(redshift / ? AS INTEGER) * ?, 4) AS z_bin_start,
        morph_proxy,
        COUNT(*) AS galaxy_count,
        AVG(log_stellar_mass) AS avg_log_stellar_mass,
        AVG(re_kpc) AS avg_re_kpc,
        AVG(color_gr) AS avg_color_gr,
        AVG(log_sfr) AS avg_log_sfr,
        AVG(concentration) AS avg_concentration
    FROM galaxies
    GROUP BY z_bin_start, morph_proxy
    ORDER BY z_bin_start, morph_proxy
    """
    return db.query_df(sql, params=(bin_width, bin_width))


def load_galaxy_zoo_agreement(db: GalaxyDatabase) -> pd.DataFrame:
    """Compare fracDeV_r morph_proxy against Galaxy Zoo debiased vote fractions."""
    sql = """
    SELECT
        morph_proxy,
        COUNT(*) AS zoo_matched_count,
        AVG(p_el_debiased) AS avg_p_el_debiased,
        AVG(p_cs_debiased) AS avg_p_cs_debiased
    FROM galaxies
    WHERE p_el_debiased IS NOT NULL AND p_cs_debiased IS NOT NULL
    GROUP BY morph_proxy
    ORDER BY morph_proxy
    """
    return db.query_df(sql)


def red_sequence_divider(abs_mag_r: np.ndarray | float) -> np.ndarray | float:
    """Evaluate the optical Red Sequence vs. Blue Cloud color separator in (g - r) vs M_r."""
    return -0.025 * (np.asarray(abs_mag_r) + 20.0) + 0.72


def classify_color_population(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate galaxies as ``red_sequence`` or ``blue_cloud`` using the CMD divider."""
    out = df.copy()
    threshold = red_sequence_divider(out["abs_mag_r"].to_numpy(dtype=float))
    out["color_population"] = np.where(
        out["color_gr"].to_numpy(dtype=float) >= threshold,
        "red_sequence",
        "blue_cloud",
    )
    return out


def fit_mass_size_power_law(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit power-law mass-size relations log10(R_50) = alpha * (logM - 10.5) + beta by morphology."""
    fits_out: dict[str, dict[str, float]] = {}
    for morph in ("early", "late"):
        sub = df.loc[(df["morph_proxy"] == morph) & (df["re_kpc"] > 0)].copy()
        if len(sub) < 3:
            continue
        x = sub["log_stellar_mass"].to_numpy(dtype=float) - 10.5
        y = np.log10(sub["re_kpc"].to_numpy(dtype=float))
        reg = stats.linregress(x, y)
        fits_out[morph] = {
            "n": int(len(sub)),
            "alpha_slope": float(reg.slope),
            "beta_intercept_at_10p5": float(reg.intercept),
            "re_kpc_at_10p5": float(10.0**reg.intercept),
            "r_squared": float(reg.rvalue**2),
            "p_value": float(reg.pvalue),
            "stderr": float(reg.stderr),
        }
    return fits_out


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix across core physical parameters."""
    cols = [c for c in PHYSICAL_CORR_COLUMNS if c in df.columns]
    return df[cols].corr(method="pearson")


def plot_color_magnitude(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the optical color-magnitude diagram (g - r vs M_r) with Red/Blue separator."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=df,
        x="abs_mag_r",
        y="color_gr",
        hue="morph_proxy",
        palette={"early": "#c0392b", "late": "#2980b9"},
        alpha=0.65,
        s=24,
        ax=ax,
    )
    x_line = np.linspace(df["abs_mag_r"].min(), df["abs_mag_r"].max(), 100)
    ax.plot(
        x_line,
        red_sequence_divider(x_line),
        color="#111111",
        linestyle="--",
        linewidth=1.5,
        label="Red Sequence / Blue Cloud divider",
    )
    ax.invert_xaxis()
    ax.set_ylim(-0.1, 1.5)
    ax.set_xlabel(r"$M_r$ (AB mag)")
    ax.set_ylabel(r"$g - r$ (mag)")
    ax.set_title("SDSS DR17 Color–Magnitude Diagram")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_mass_size_relation(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the galaxy stellar mass-size relation (R_50 [kpc] vs log M_*) with power-law fits."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    palette = {"early": "#c0392b", "late": "#2980b9"}
    sns.scatterplot(
        data=df,
        x="log_stellar_mass",
        y="re_kpc",
        hue="morph_proxy",
        palette=palette,
        alpha=0.55,
        s=22,
        ax=ax,
    )

    fits_out = fit_mass_size_power_law(df)
    x_grid = np.linspace(df["log_stellar_mass"].quantile(0.02), df["log_stellar_mass"].quantile(0.98), 100)
    for morph, fit in fits_out.items():
        y_fit = 10.0 ** (fit["alpha_slope"] * (x_grid - 10.5) + fit["beta_intercept_at_10p5"])
        ax.plot(
            x_grid,
            y_fit,
            color=palette[morph],
            linewidth=2.2,
            linestyle="-",
            label=f"{morph} fit: slope={fit['alpha_slope']:.2f} (R²={fit['r_squared']:.2f})",
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"$R_{50}$ (kpc)")
    ax.set_title("SDSS DR17 Stellar Mass–Size Relation (with Power-Law Fits)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_concentration_morphology(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the distribution of concentration index (R90 / R50) by morph_proxy."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sns.boxplot(
        data=df,
        x="morph_proxy",
        y="concentration",
        hue="morph_proxy",
        palette={"early": "#c0392b", "late": "#2980b9"},
        legend=False,
        ax=ax,
    )
    ax.axhline(2.6, color="black", linestyle="--", linewidth=1.0, label="Strateva et al. C = 2.6")
    ax.set_xlabel("Morphology Proxy (fracDeV_r >= 0.5)")
    ax.set_ylabel(r"Concentration $C = R_{90} / R_{50}$")
    ax.set_title("Concentration Index by Morphological Proxy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_statistical_dashboard(
    df: pd.DataFrame,
    output_png: str | Path | None = None,
    output_pdf: str | Path | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build a 6-panel publication-grade statistical dashboard for SDSS DR17 galaxies.

    Panels:
    1. Morphology distribution (pie chart with counts & fractions)
    2. Stellar mass-size relation (log-log scatter + early/late power-law fits)
    3. Color-magnitude diagram (Red Sequence vs. Blue Cloud separation)
    4. Star formation rate distribution by morphology (histogram + box summary)
    5. Red Sequence vs. Blue Cloud breakdown by morphology (grouped bar chart)
    6. Pearson correlation heatmap of physical galaxy parameters
    """
    annotated = classify_color_population(df)
    fits_out = fit_mass_size_power_law(annotated)
    corr_df = compute_correlation_matrix(annotated)
    palette = {"early": "#c0392b", "late": "#2980b9"}

    fig, axes = plt.subplots(3, 2, figsize=(15, 17))
    fig.suptitle(
        f"SDSS DR17 Galaxy Scaling Relations — Statistical Dashboard (N = {len(annotated):,})",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    # Panel 1: Morphology distribution (Pie)
    ax = axes[0, 0]
    counts = annotated["morph_proxy"].value_counts().reindex(["early", "late"], fill_value=0)
    ax.pie(
        counts.values,
        labels=[f"Early-type\n(n={counts['early']:,})", f"Late-type\n(n={counts['late']:,})"],
        colors=[palette["early"], palette["late"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10},
    )
    ax.set_title("1. Morphological Proxy Distribution (fracDeV_r >= 0.5)", fontweight="bold")

    # Panel 2: Mass-Size Relation + Power-Law Fits
    ax = axes[0, 1]
    sns.scatterplot(
        data=annotated,
        x="log_stellar_mass",
        y="re_kpc",
        hue="morph_proxy",
        palette=palette,
        alpha=0.45,
        s=18,
        ax=ax,
    )
    x_grid = np.linspace(
        annotated["log_stellar_mass"].quantile(0.02),
        annotated["log_stellar_mass"].quantile(0.98),
        100,
    )
    for morph, fit in fits_out.items():
        y_fit = 10.0 ** (fit["alpha_slope"] * (x_grid - 10.5) + fit["beta_intercept_at_10p5"])
        ax.plot(
            x_grid,
            y_fit,
            color=palette[morph],
            linewidth=2.4,
            label=f"{morph} fit: α={fit['alpha_slope']:.2f}, R²={fit['r_squared']:.2f}",
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"Effective Radius $R_{50}$ (kpc)")
    ax.set_title("2. Stellar Mass–Size Relation & Power-Law Fits", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    # Panel 3: Color-Magnitude Diagram + Red Sequence / Blue Cloud Divider
    ax = axes[1, 0]
    sns.scatterplot(
        data=annotated,
        x="abs_mag_r",
        y="color_gr",
        hue="morph_proxy",
        palette=palette,
        alpha=0.45,
        s=18,
        ax=ax,
    )
    x_cmd = np.linspace(annotated["abs_mag_r"].min(), annotated["abs_mag_r"].max(), 100)
    ax.plot(
        x_cmd,
        red_sequence_divider(x_cmd),
        color="#111111",
        linestyle="--",
        linewidth=1.8,
        label="Red Sequence / Blue Cloud cut",
    )
    ax.invert_xaxis()
    ax.set_ylim(-0.1, 1.5)
    ax.set_xlabel(r"Absolute Magnitude $M_r$ (AB mag)")
    ax.set_ylabel(r"Optical Color $g - r$ (mag)")
    ax.set_title("3. Color–Magnitude Diagram (Red Sequence vs. Blue Cloud)", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)

    # Panel 4: SFR Distribution by Morphology (Histogram + Box Plot inset)
    ax = axes[1, 1]
    sns.histplot(
        data=annotated,
        x="log_sfr",
        hue="morph_proxy",
        palette=palette,
        bins=32,
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.45,
        ax=ax,
    )
    ax.set_xlabel(r"$\log_{10}(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$")
    ax.set_ylabel("Density")
    ax.set_title("4. Star Formation Rate Distribution by Morphology", fontweight="bold")
    ax.grid(alpha=0.25)

    # Panel 5: Red Sequence vs. Blue Cloud Bar + SFR Box Comparison
    ax = axes[2, 0]
    crosstab = pd.crosstab(annotated["morph_proxy"], annotated["color_population"]).reindex(
        index=["early", "late"],
        columns=["red_sequence", "blue_cloud"],
        fill_value=0,
    )
    crosstab.plot(
        kind="bar",
        color=["#922b21", "#2471a3"],
        edgecolor="white",
        ax=ax,
    )
    ax.set_xticklabels(["Early-type", "Late-type"], rotation=0)
    ax.set_xlabel("Morphology Proxy")
    ax.set_ylabel("Galaxy Count")
    ax.set_title("5. Red Sequence vs. Blue Cloud Population Counts", fontweight="bold")
    ax.legend(["Red Sequence", "Blue Cloud"], frameon=False)
    ax.grid(alpha=0.25, axis="y")
    for container in ax.containers:
        ax.bar_label(container, padding=2, fontsize=9)

    # Panel 6: Pearson Correlation Heatmap
    ax = axes[2, 1]
    renamed_corr = corr_df.rename(index=PHYSICAL_CORR_LABELS, columns=PHYSICAL_CORR_LABELS)
    sns.heatmap(
        renamed_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("6. Pearson Correlation Matrix of Physical Parameters", fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    if output_png is not None:
        out_png = Path(output_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if output_pdf is not None:
        out_pdf = Path(output_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")

    metrics = {
        "n_total": int(len(annotated)),
        "morph_counts": {k: int(v) for k, v in counts.items()},
        "color_population_crosstab": {
            m: {c: int(crosstab.loc[m, c]) for c in crosstab.columns} for m in crosstab.index
        },
        "mass_size_fits": fits_out,
        "correlation_matrix": {
            "columns": list(corr_df.columns),
            "labels": [PHYSICAL_CORR_LABELS.get(c, c) for c in corr_df.columns],
            "values": [[round(float(val), 4) for val in row] for row in corr_df.to_numpy()],
        },
    }
    return fig, metrics
