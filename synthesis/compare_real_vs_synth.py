import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


REAL_XLSX = "hodp2026_pseudonymised.csv"
SYN_XLSX = "synthetic_data.csv"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names comparable across CSV exports (strip spaces, remove BOM/NBSP)."""
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("\ufeff", "", regex=False)   # BOM (often from UTF-8-SIG)
        .str.replace("\xa0", " ", regex=False)    # non-breaking space
        .str.strip()
    )
    return df


def _find_col(df: pd.DataFrame, wanted: str) -> str:
    """Find a column name case-insensitively, ignoring leading/trailing whitespace."""
    key = str(wanted).strip().lower()
    lookup = {str(c).strip().lower(): c for c in df.columns}
    if key not in lookup:
        raise KeyError(
            f"Column '{wanted}' not found. Available columns: {list(df.columns)}"
        )
    return lookup[key]


def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    Read CSV robustly across locales:
    - sep=None + engine='python' lets pandas sniff ',' vs ';'
    - encoding='utf-8-sig' strips BOM if present
    """
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    return _normalize_columns(df)


def load_data(real_path: str, syn_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_df = _read_csv_robust(real_path)
    syn_df = _read_csv_robust(syn_path)
    return real_df, syn_df


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert to numeric safely:
    - treat comma decimals (e.g. '12,5') as '.' (rare for Age, but harmless)
    - coerce non-numeric to NaN
    """
    x = s.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def plot_age_hist(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    col="Age",
    out_path="age_hist.png",
):
    # find the matching column names robustly
    real_col = _find_col(real_df, col)
    syn_col = _find_col(syn_df, col)

    r = _to_numeric_series(real_df[real_col]).dropna()
    s = _to_numeric_series(syn_df[syn_col]).dropna()

    if r.empty or s.empty:
        raise ValueError("Age column has no numeric values in real and/or synthetic data.")

    # integer-friendly bins centered on integers
    mn = int(np.floor(min(r.min(), s.min())))
    mx = int(np.ceil(max(r.max(), s.max())))
    bins = np.arange(mn, mx + 2) - 0.5  # e.g. 18.5, 19.5, ...

    plt.figure(figsize=(9, 4.5))
    plt.hist(r, bins=bins, density=True, alpha=0.5, label="Real")
    plt.hist(s, bins=bins, density=True, alpha=0.5, label="Synthetic")
    plt.xlabel(col)
    plt.ylabel("Proportion (density)")
    plt.title(f"{col} histogram: Real vs Synthetic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _normalized_counts(series: pd.Series) -> pd.Series:
    """Value counts normalized to proportions, with NaN treated as a visible category."""
    x = series.astype("object").where(series.notna(), "<NA>")
    return x.value_counts(normalize=True)


def plot_categorical_compare(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    cols=("Sex", "Education"),
    out_path="sex_education_compare.png",
    top_n_for_long=None,  # e.g. 15 for Education; None shows all categories
):
    # resolve columns robustly (case/whitespace/BOM)
    real_cols = {c: _find_col(real_df, c) for c in cols}
    syn_cols = {c: _find_col(syn_df, c) for c in cols}

    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 4.5 * len(cols)), constrained_layout=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        rc = _normalized_counts(real_df[real_cols[col]])
        sc = _normalized_counts(syn_df[syn_cols[col]])

        # align categories (union of both)
        cats = rc.index.union(sc.index)
        rc = rc.reindex(cats, fill_value=0.0)
        sc = sc.reindex(cats, fill_value=0.0)

        # optionally keep plot readable for long category lists (e.g., Education)
        if top_n_for_long is not None and len(cats) > top_n_for_long:
            top = rc.sort_values(ascending=False).head(top_n_for_long).index
            rc2 = rc.loc[top].copy()
            sc2 = sc.loc[top].copy()
            rc2.loc["Other"] = rc.drop(top).sum()
            sc2.loc["Other"] = sc.drop(top).sum()
            rc, sc = rc2, sc2

        x = np.arange(len(rc))
        width = 0.4

        ax.bar(x - width / 2, rc.values, width, label="Real")
        ax.bar(x + width / 2, sc.values, width, label="Synthetic")

        ax.set_xticks(x)
        ax.set_xticklabels(rc.index, rotation=45, ha="right")
        ax.set_ylabel("Proportion")
        ax.set_title(f"{col} distribution: Real vs Synthetic")
        ax.legend()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    real_df, syn_df = load_data(REAL_XLSX, SYN_XLSX)

    # Plot 1: AGE histogram overlay
    plot_age_hist(real_df, syn_df, col="Age", out_path="age_hist.png")

    # Plot 2: Sex + Education distributions
    plot_categorical_compare(
        real_df, syn_df,
        cols=("Sex", "Education"),
        out_path="sex_education_compare.png",
        top_n_for_long=15
    )

    print("Saved plots: age_hist.png, sex_education_compare.png")


if __name__ == "__main__":
    main()

