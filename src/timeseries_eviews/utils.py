# Copyright (c) 2025 Eduardo Belisario Scheffer
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for details.

"""
utils.py
========

General utility functions for:
- DataFrame preprocessing and checks (time series validation, merging, etc.)
- Weighted statistics (mean, std, quantiles)
- Misc. transformations and custom logic from Jeremy Bejarano's archives
- Plotting time series data using Plotly Express (and some Matplotlib references)

Notes
-----
- The original code is adapted from Jeremy Bejarano's repository:
  https://github.com/jmbejara/blank_project

"""

import os
import math
import logging
import datetime
from dateutil.relativedelta import relativedelta
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

#------------------------------------------------------------------------------
# Basic Time Series Checks and Plotting (Plotly)
#------------------------------------------------------------------------------

def check_time_series(df: pd.DataFrame, date_col: str = None, freq: str = None) -> pd.DataFrame:
    """
    Validate and (optionally) transform a DataFrame into a proper time series.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing time series data.
    date_col : str, optional
        The name of the column to be converted to a DateTimeIndex. If None, assumes
        the existing index is already a DateTimeIndex.
    freq : str, optional
        Pandas frequency string (e.g., 'D' for daily, 'M' for monthly) to set the index frequency.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DateTimeIndex and optional frequency set.

    Raises
    ------
    ValueError
        If the specified date_col does not exist in the DataFrame or if DataFrame index
        is not DatetimeIndex (when date_col is None).
    """
    if date_col is not None:
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' does not exist in DataFrame.")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.set_index(date_col, inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex. "
                         "Provide `date_col` or manually set index.")

    if freq is not None:
        df = df.asfreq(freq)

    return df


def standardize_column_names(df: pd.DataFrame, prefix: str = "var") -> pd.DataFrame:
    """
    Standardize column names to ensure no spaces/special characters,
    optionally appending a prefix to each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    prefix : str, optional
        Prefix for each column name (default='var').

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """
    new_cols = {}
    for i, col in enumerate(df.columns):
        # Keep it simple: rename to prefix_i if prefix is specified
        clean_name = f"{prefix}_{i}" if prefix else col
        new_cols[col] = clean_name
    df.rename(columns=new_cols, inplace=True)
    return df


def plot_time_series(
    df: pd.DataFrame,
    title: str = "Time Series Plot",
    ytitle: str = "",
    output_dir: str = None,
    filename: str = "time_series_plot.png",
    figsize: Tuple[int, int] = (8, 4),
):
    """
    Plot multiple columns of a time series DataFrame using Matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a DateTimeIndex or numeric index and one or more columns to plot.
    title : str, optional
        Plot title.
    ytitle : str, optional
        Y-axis label.
    output_dir : str, optional
        If not None, saves the plot to this directory as a PNG file.
    filename : str, optional
        The filename for saving the plot. Default='time_series_plot.png'.
    figsize : Tuple[int, int], optional
        Figure size in inches (width, height). Default=(8, 4).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col))

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Index")
    ax.set_ylabel(ytitle)
    ax.legend(loc='best')
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path_out = os.path.join(output_dir, filename)
        plt.savefig(path_out, dpi=150, bbox_inches="tight")

    return ax

#------------------------------------------------------------------------------
# Correlogram Utilities
#------------------------------------------------------------------------------


def compute_correlogram(series: pd.Series, max_lag: int = 40) -> pd.DataFrame:
    """
    Compute ACF, PACF, and Ljung-Box Q-test up to a specified lag, returning
    a DataFrame with columns ['Lag', 'AC', 'PAC', 'Q-Stat', 'Prob'].

    Parameters
    ----------
    series : pd.Series
        Univariate time series.
    max_lag : int, optional
        Maximum lag to compute. Default=40.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'Lag', 'AC', 'PAC', 'Q-Stat', 'Prob'.
    """
    # 1) ACF & PACF (skip lag=0)
    ac_values = acf(series, nlags=max_lag, fft=False)
    pac_values = pacf(series, nlags=max_lag, method='ols')

    # 2) Ljung-Box Q test for lags=1..max_lag
    lb_results = acorr_ljungbox(series, lags=list(range(1, max_lag+1)), return_df=True)

    # 3) Build DataFrame
    corr_df = pd.DataFrame({
        "Lag": range(1, max_lag+1),
        "AC": ac_values[1:max_lag+1],
        "PAC": pac_values[1:max_lag+1],
        "Q-Stat": lb_results["lb_stat"].values,
        "Prob": lb_results["lb_pvalue"].values
    })

    return corr_df


def plot_correlogram(
    series: pd.Series,
    max_lag: int = 40,
    title: str = "Correlogram",
    alpha: float = 0.05,
    output_dir: str = None,
    filename: str = "correlogram.png",
    show_plot: bool = True
) -> pd.DataFrame:
    """
    Create a Matplotlib figure closely replicating EViews-style correlogram:
      1) Autocorrelation bars (horizontal) on the left
      2) Partial Correlation bars (horizontal) in the middle
      3) A numeric table on the right: [Lag, AC, PAC, Q-Stat, Prob]
         displayed from Lag=1 at the top row to Lag=max_lag at the bottom.

    Lags are displayed top-to-bottom, matching EViews.

    Parameters
    ----------
    series : pd.Series
        Time series to analyze.
    max_lag : int, optional
        Max lag for ACF/PACF. Default=40.
    title : str, optional
        Figure title, displayed at the top.
    alpha : float, optional
        Significance level for confidence intervals. If alpha=0.05 => ±1.96/sqrt(n).
    output_dir : str, optional
        If provided, directory to save the figure as PNG.
    filename : str, optional
        Name of the file saved, default 'correlogram.png'.
    show_plot : bool, optional
        If True, show the plot via plt.show().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Lag, AC, PAC, Q-Stat, Prob].
    """
    corr_df = compute_correlogram(series, max_lag=max_lag)
    n = len(series.dropna())
    if n < 2:
        raise ValueError("Not enough non-NaN data to compute correlations.")

    # Confidence bounds
    z_val = norm.ppf(1 - alpha / 2)
    bound = z_val * (1 / math.sqrt(n))

    # Figure size
    fig_height = max(5, min(16, max_lag * 0.35))
    fig = plt.figure(figsize=(10, fig_height))

    # Use GridSpec: 3 columns -> AC, PAC, Table
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=1, ncols=3, width_ratios=[1.0, 1.0, 1.7], wspace=0.3)

    # ----------------------------------------------------------------------------
    # Subplot 1: ACF
    # ----------------------------------------------------------------------------
    ax_ac = fig.add_subplot(gs[0, 0])
    ax_ac.barh(
        y=corr_df.index, 
        width=corr_df["AC"], 
        color="#c6d9f0", 
        edgecolor="black"
    )
    # Grey dashed lines for conf intervals
    ax_ac.axvline(bound, color="grey", linestyle="--", lw=1.2)
    ax_ac.axvline(-bound, color="grey", linestyle="--", lw=1.2)
    ax_ac.axvline(0, color="lightgrey", linestyle="-", lw=0.7)

    ax_ac.invert_yaxis()  # so 0 is top row => lag=1
    ax_ac.set_xlim([-1.0, 1.0])
    ax_ac.set_xlabel("Autocorr.")
    ax_ac.set_title("Autocorrelation", fontsize=10, pad=17)
    xlim = ax_ac.get_xlim()
    line_length = int((xlim[1] - xlim[0]) * 10 + 5)
    ax_ac.text(0, -1.5, "=" * line_length, ha='center', va='center', fontsize=9, fontfamily='monospace')
    ax_ac.set_yticks(corr_df.index)
    ax_ac.set_yticklabels(corr_df["Lag"].astype(str))
    ax_ac.tick_params(axis='both', labelsize=9)
    ax_ac.spines["top"].set_visible(False)
    ax_ac.spines["right"].set_visible(False)

    # ----------------------------------------------------------------------------
    # Subplot 2: PAC
    # ----------------------------------------------------------------------------
    # We'll mirror the y-axis on the right side
    ax_pac = fig.add_subplot(gs[0, 1], sharey=ax_ac)
    ax_pac.barh(
        y=corr_df.index, 
        width=corr_df["PAC"], 
        color="#c6d9f0", 
        edgecolor="black"
    )
    ax_pac.axvline(bound, color="grey", linestyle="--", lw=1.2)
    ax_pac.axvline(-bound, color="grey", linestyle="--", lw=1.2)
    ax_pac.axvline(0, color="lightgrey", linestyle="-", lw=0.5)

    ax_pac.invert_yaxis() 
    ax_pac.set_xlim([-1.0, 1.0])
    ax_pac.yaxis.set_label_position("right")
    ax_pac.yaxis.tick_right()
    ax_pac.tick_params(axis='both', labelsize=9)
    ax_pac.set_xlabel("Partial Corr.")
    ax_pac.set_title("Partial Correlation", fontsize=10, pad=17)
    xlim = ax_ac.get_xlim()
    line_length = int((xlim[1] - xlim[0]) * 10 + 5)
    ax_pac.text(0, -1.5, "=" * line_length, ha='center', va='center', fontsize=9, fontfamily='monospace')
    ax_pac.spines["top"].set_visible(False)
    ax_pac.spines["left"].set_visible(False)

    # ----------------------------------------------------------------------------
    # Subplot 3: Table (align y with ax_ac so rows match the bars)
    # ----------------------------------------------------------------------------
    ax_tbl = fig.add_subplot(gs[0, 2], sharey=ax_ac)
    ax_tbl.set_xlim([0, 1])
    # Hide spines & ticks
    ax_tbl.xaxis.set_visible(False)
    ax_tbl.yaxis.set_visible(False)
    for spine in ax_tbl.spines.values():
        spine.set_visible(False)

    # We'll place text at each y=i for reversed DF, 
    # but we want the table in ascending order of Lag so that row at top is lag=1
    # Actually, rev_df[0] is lag= max_lag, so we need to map back to corr_df row
    # Let's store a map from i -> the row in corr_df
    # index i in rev_df => corr_df index = len(corr_df)-i-1
    # But we want lines from top=lag=1 => i=0 in rev => actually lag= max_lag
    # We'll just reuse rev_df to keep the top row as lag=1 in the bar chart, 
    # so the table's top row is also lag=1. That means we should iterate over rev_df from i=0.. 
    # Yes, we do that, but each row's "Lag" is rev_df.Lag[i]. 
    # This ensures row i lines up with bar i.
    # We'll put a header at y=-1 so it's above the top bar.

    # For spacing, let's see how tall each bar is. We'll do center alignment at i.
    # We'll define a "monospace" string. EViews has columns: Lag, AC, PAC, Q-Stat, Prob
    # We'll place the header just above the top bar at y=-1
    ax_tbl.set_ylim([-1, len(corr_df) - 0.5])  # so there's room for the header
    ax_tbl.invert_yaxis()  # keep the same orientation as bars

    header_str = "  Lag    AC     PAC    Q-Stat    Prob"
    ax_tbl.text(0, -1.8, header_str, fontname="monospace", fontsize=10)
    ax_tbl.text(0, -1.3, "=" * (len(header_str)+5), fontname="monospace", fontsize=9)

    for i in corr_df.index:
        # rev_df[i] => row with reversed. 
        # i=0 => top bar => lag=1
        row = corr_df.loc[i]
        lag = int(row["Lag"])
        ac_ = f"{row['AC']:.3f}"
        pac_ = f"{row['PAC']:.3f}"
        qst_ = f"{row['Q-Stat']:.2f}"
        prb_ = f"{row['Prob']:.3f}"
        table_str = f"{lag:4d}  {ac_:>6}  {pac_:>6}  {qst_:>7}  {prb_:>6}"
        ax_tbl.text(0, i, table_str, fontname="monospace", fontsize=10, va="center")

    # ----------------------------------------------------------------------------
    # Title with "====" line
    # ----------------------------------------------------------------------------
    fig.suptitle(f"{title} ({series.name})", fontsize=11)

    # Save
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(output_dir, filename)
        plt.savefig(outfile, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()

    return corr_df


def plot_correlation(data: pd.DataFrame, output_dir=None, filename="corr_heatmap.png", show_plot=True):
    """
    Plot a correlation matrix heatmap (Seaborn) for the columns of 'self.data'.
    """
    corr_mat = data.corr()  # or self.fitted_model.resid.corr() if you want residuals
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", ax=ax, fmt=".3f")

    col_names = ", ".join(data.columns)
    ax.set_title(f"Correlation Matrix - {col_names}", fontsize=11)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path_out = os.path.join(output_dir, filename)
        plt.savefig(path_out, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()

    return corr_mat

#------------------------------------------------------------------------------
# Additional helper functions from Jeremy Bejarano:
#   - df_to_literal, merge_stats, dataframe_set_difference
#   - freq_counts, move_column_inplace, ...
#   - Weighted stats, quantiles, lagged columns, ...
#   - date manipulations (quarter ends, etc.)
#------------------------------------------------------------------------------

def df_to_literal(df, missing_value="None"):
    """
    Convert a pandas DataFrame into a literal string representing code to recreate it.
    Missing values (NaN) are replaced with 'None' by default.
    """
    cols = df.to_dict("list")
    lines = []
    lines.append("df = pd.DataFrame(")
    lines.append("{")
    for idx, (col, values) in enumerate(cols.items()):
        line = f"    '{col}': {values}"
        if idx < len(cols) - 1:
            line += ","
        lines.append(line)
    lines.append("}")

    # Add index if not a simple RangeIndex
    if (not isinstance(df.index, pd.RangeIndex) or
        not (df.index == pd.RangeIndex(len(df))).all()):
        index_values = list(df.index)
        lines[-1] += f", index={index_values}"

    lines.append(")")
    output = "\n".join(lines)
    return output.replace("nan", missing_value)


def merge_stats(df_left, df_right, on=[]):
    """
    Provide stats on merges:
    Count how many unique keys on left/right, intersection, union, etc.
    Returns a Series with stats: union, intersection, difference, etc.
    """
    left_index = df_left.set_index(on).index.unique()
    right_index = df_right.set_index(on).index.unique()
    union = left_index.union(right_index)
    intersection = left_index.intersection(right_index)
    stats = [
        "union",
        "intersection",
        "union-intersection",
        "intersection/union",
        "left",
        "right",
        "left-intersection",
        "right-intersection",
        "intersection/left",
        "intersection/right",
    ]
    df_stats = pd.Series(index=stats, dtype=float)
    df_stats["union"] = len(union)
    df_stats["intersection"] = len(intersection)
    df_stats["union-intersection"] = len(union) - len(intersection)
    df_stats["intersection/union"] = len(intersection) / len(union) if len(union) != 0 else float('nan')
    df_stats["left"] = len(left_index)
    df_stats["right"] = len(right_index)
    df_stats["left-intersection"] = len(left_index) - len(intersection)
    df_stats["right-intersection"] = len(right_index) - len(intersection)
    df_stats["intersection/left"] = len(intersection) / len(left_index) if len(left_index) != 0 else float('nan')
    df_stats["intersection/right"] = len(intersection) / len(right_index) if len(right_index) != 0 else float('nan')
    return df_stats


def dataframe_set_difference(dff, df, library="pandas", show="rows_and_numbers"):
    """
    Returns the rows that appear in dff but not in df. Optionally returns row indices.

    Parameters
    ----------
    dff, df : DataFrames (pandas or polars).
    library : str
        'pandas' or 'polars'.
    show : str
        'rows_and_numbers' to return (row_numbers, rows),
        otherwise just row_numbers.

    Returns
    -------
    tuple or list
        row_numbers, and optionally the subset of rows from `dff` not in `df`.
    """
    if library == "pandas":
        dff_reset = dff.reset_index().rename(columns={"index": "original_row_number"})
        df_reset = df.reset_index(drop=True)

        # Merge on all columns
        merged = dff_reset.merge(
            df_reset, how="left", indicator=True, on=dff.columns.tolist()
        )
        only_in_dff = merged[merged["_merge"] == "left_only"]
        row_numbers = only_in_dff["original_row_number"].tolist()
        ret = row_numbers

    elif library == "polars":
        assert dff.columns == df.columns, "Columns must match for polars anti-join."
        dff_with_index = dff.with_columns(pl.arange(0, dff.height).alias("row_number"))
        df_with_index = df.with_columns(pl.arange(0, df.height).alias("dummy_row_number"))

        diff = dff_with_index.join(
            df_with_index, on=list(dff.columns), how="anti", join_nulls=True
        )
        row_numbers = diff.select("row_number").to_series().to_list()
        ret = row_numbers
    else:
        raise ValueError("Unknown library for set difference.")

    if show == "rows_and_numbers":
        rows = dff.iloc[row_numbers] if library == "pandas" else None
        # Polars version would be dff.filter(pl.col("row_number").is_in(row_numbers))
        return row_numbers, rows

    return ret


def freq_counts(df, col=None, with_count=True, with_cum_freq=True):
    """
    Like value_counts() but normalizes to give frequencies (%) and optional cumulative frequency.
    For polars DataFrame usage.
    """
    s = df[col]
    ret = (
        s.value_counts(sort=True)
        .with_columns(
            freq=pl.col("count") / s.shape[0] * 100,
        )
        .with_columns(cum_freq=pl.col("freq").cumsum())
    )
    if not with_count:
        ret = ret.drop("count")
    if not with_cum_freq:
        ret = ret.drop("cum_freq")
    return ret


def move_column_inplace(df, col, pos=0):
    """
    Move a single column in a pandas DataFrame to a specified position in-place.
    """
    col_series = df.pop(col)
    df.insert(pos, col_series.name, col_series)


def move_columns_to_front(df, cols=[]):
    """
    Move a list of columns to the front of the DataFrame, preserving order.
    """
    for col in reversed(cols):
        move_column_inplace(df, col, pos=0)


def weighted_average(data_col=None, weight_col=None, data=None):
    """
    Calculate a weighted average for a specified column of a DataFrame.
    """
    def weights_function(row):
        return data.loc[row.index, weight_col]

    def wm(row):
        return np.average(row, weights=weights_function(row))

    return wm(data[data_col])


def groupby_weighted_average(
    data_col=None,
    weight_col=None,
    by_col=None,
    data=None,
    transform=False,
    new_column_name="",
):
    """
    Grouped weighted average, with an option to transform to a column in the original DataFrame.
    """
    data["_data_times_weight"] = data[data_col] * data[weight_col]
    data["_weight_where_notnull"] = data[weight_col] * pd.notnull(data[data_col])
    g = data.groupby(by_col)
    result = g["_data_times_weight"].sum() / g["_weight_where_notnull"].sum()
    del data["_data_times_weight"], data["_weight_where_notnull"]

    if transform:
        result.name = f"__{data_col}"
        merged = data.merge(result.reset_index(), how="left", on=by_col)
        new_series = merged[f"__{data_col}"]
        new_series.name = new_column_name
        return new_series
    return result


def groupby_weighted_std(data_col=None, weight_col=None, by_col=None, data=None, ddof=1):
    """
    Grouped weighted standard deviation. Adapts the formula from known references.
    """
    def weighted_sd(group):
        weights = group[weight_col]
        vals = group[data_col]
        weighted_avg = np.average(vals, weights=weights)
        numer = np.sum(weights * (vals - weighted_avg)**2)
        denom = ((vals.count() - ddof) / vals.count()) * np.sum(weights)
        return np.sqrt(numer / denom) if denom != 0 else np.nan

    return data.groupby(by_col).apply(weighted_sd)


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """
    Weighted quantiles for 1D numeric data. Very close to np.percentile but supports weights.
    `quantiles` should be in [0, 1].
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ*@#"


@np.vectorize
def calc_check_digit(number):
    """
    Calculate a check digit for an 8-digit CUSIP.
    Taken from python-stdnum library for demonstration.
    """
    number = "".join(str((1, 2)[i % 2] * _alphabet.index(n)) for i, n in enumerate(number))
    return str((10 - sum(int(n) for n in number)) % 10)


def convert_cusips_from_8_to_9_digit(cusip_8dig_series):
    """
    Convert 8-digit CUSIPs to 9-digit by appending the check digit.
    """
    dig9 = calc_check_digit(cusip_8dig_series)
    return cusip_8dig_series + dig9


def _with_lagged_column_no_resample(
    df=None,
    columns_to_lag=None,
    id_columns=None,
    lags=1,
    prefix="L",
):
    """
    Helper method using groupby().shift() to add lagged columns without resampling.
    """
    subsub_df = df.groupby(id_columns)[columns_to_lag].shift(lags)
    lag_sub_df = df.copy()
    for col in columns_to_lag:
        lag_sub_df[f"{prefix}{lags}_{col}"] = subsub_df[col]
    return lag_sub_df


def with_lagged_columns(
    df=None,
    column_to_lag=None,
    id_column=None,
    lags=1,
    date_col="date",
    prefix="L",
    freq=None,
    resample=True,
):
    """
    Add lagged columns to a DataFrame, optionally resampling missing dates
    at a specified frequency to ensure consistent time intervals.
    """
    if resample:
        df_wide = df.pivot(index=date_col, columns=id_column, values=column_to_lag)
        new_col = f"{prefix}{lags}_{column_to_lag}"
        df_resampled = df_wide.resample(freq).last()
        df_lagged = df_resampled.shift(lags)
        df_lagged = df_lagged.stack(dropna=False).reset_index(name=new_col)
        df_lagged = df.merge(df_lagged, on=[date_col, id_column], how="right")
        df_lagged = df_lagged.dropna(subset=[column_to_lag, new_col], how="all")
        df_lagged = df_lagged.sort_values(by=[id_column, date_col])
    else:
        df_lagged = _with_lagged_column_no_resample(
            df=df,
            columns_to_lag=[column_to_lag],
            id_columns=[id_column],
            lags=lags,
            prefix=prefix,
        )
    return df_lagged


def leave_one_out_sums(df, groupby=[], summed_col=""):
    """
    Compute leave-one-out sums: sum of the group minus the current row's value.
    Handy for shift-share style instruments (Borusyak, Hull, Jaravel).
    """
    return df.groupby(groupby)[summed_col].transform(lambda x: x.sum() - x)


def get_most_recent_quarter_end(d):
    """
    Take a datetime and find the most recent quarter end date.
    """
    quarter_month = (d.month - 1) // 3 * 3 + 1
    quarter_end = datetime.datetime(d.year, quarter_month, 1) - relativedelta(days=1)
    return quarter_end


def get_next_quarter_start(d):
    """
    Take a datetime and find the start date of the next quarter.
    """
    quarter_month = (d.month - 1) // 3 * 3 + 4
    years_to_add = quarter_month // 12
    quarter_month_mod = quarter_month % 12
    return datetime.datetime(d.year + years_to_add, quarter_month_mod, 1)


def get_end_of_current_month(d):
    """
    Find the last date of the current month, resetting time to zero.
    """
    d = pd.DatetimeIndex([d]).normalize()[0]
    next_month = d.replace(day=28) + datetime.timedelta(days=4)
    end_of_current_month = next_month - datetime.timedelta(days=next_month.day)
    return end_of_current_month


def get_end_of_current_quarter(d):
    """
    Find the last date of the current quarter, resetting time to zero.
    """
    quarter_start = get_next_quarter_start(d)
    quarter_end = quarter_start - datetime.timedelta(days=1)
    return quarter_end


def add_vertical_lines_to_plot(
    start_date,
    end_date,
    ax=None,
    freq="Q",
    adjust_ticks=True,
    alpha=0.1,
    extend_to_nearest_quarter=True,
):
    """
    Add vertical lines to a Matplotlib plot for each quarter boundary (or other freq).
    """
    if extend_to_nearest_quarter:
        start_date = get_most_recent_quarter_end(start_date)
        end_date = get_next_quarter_start(end_date)
    if freq == "Q":
        dates = pd.date_range(
            pd.to_datetime(start_date),
            pd.to_datetime(end_date) + pd.offsets.QuarterBegin(1),
            freq="Q",
        )
        mask = (dates >= start_date) & (dates <= end_date)
        dates = dates[mask]
        months = mdates.MonthLocator((1, 4, 7, 10))
        if adjust_ticks and ax is not None:
            for d in dates:
                ax.axvline(d, color="k", alpha=alpha)
            ax.xaxis.set_major_locator(months)
        if ax is not None:
            ax.xaxis.set_tick_params(rotation=90)
    else:
        raise ValueError(f"Unsupported freq={freq}")


def plot_weighted_median_with_distribution_bars(
    data=None,
    variable_name=None,
    date_col="date",
    weight_col=None,
    percentile_bars=True,
    percentiles=[0.25, 0.75],
    rolling_window=1,
    rolling=False,
    rolling_min_periods=None,
    rescale_factor=1,
    ax=None,
    add_quarter_lines=True,
    ylabel=None,
    xlabel=None,
    label=None,
):
    """
    Plot the weighted median of a variable over time using Matplotlib. Optionally show
    percentile bands and vertical lines for quarter boundaries. Rescale factor can
    shift the axis (e.g., for basis points).
    """
    if ax is None:
        _, ax = plt.subplots()

    median_series = data.groupby(date_col).apply(
        lambda x: weighted_quantile(x[variable_name], 0.5, sample_weight=x[weight_col])
    )

    if rolling:
        wavrs = median_series.rolling(rolling_window, min_periods=rolling_min_periods).mean()
    else:
        wavrs = median_series

    (wavrs * rescale_factor).plot(ax=ax, label=label)

    if percentile_bars:
        lower = data.groupby(date_col).apply(
            lambda x: weighted_quantile(x[variable_name], percentiles[0], sample_weight=x[weight_col])
        )
        upper = data.groupby(date_col).apply(
            lambda x: weighted_quantile(x[variable_name], percentiles[1], sample_weight=x[weight_col])
        )
        if rolling:
            lower = lower.rolling(rolling_window, min_periods=rolling_min_periods).mean()
            upper = upper.rolling(rolling_window, min_periods=rolling_min_periods).mean()

        ax.plot(wavrs.index, lower * rescale_factor, color="tab:blue", alpha=0.1)
        ax.plot(wavrs.index, upper * rescale_factor, color="tab:blue", alpha=0.1)
        ax.fill_between(
            wavrs.index, lower * rescale_factor, upper * rescale_factor, alpha=0.2
        )

    if add_quarter_lines:
        start_date = data[date_col].min()
        end_date = data[date_col].max()
        add_vertical_lines_to_plot(start_date, end_date, ax=ax, freq="Q", alpha=0.05)

        ax.xaxis.set_tick_params(rotation=90)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if ylabel is None:
        if rolling_window > 1:
            ylabel = f"{variable_name} ({rolling_window}-period rolling avg)"
        else:
            ylabel = f"{variable_name}"
    ax.set_ylabel(ylabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    plt.tight_layout()
    return ax