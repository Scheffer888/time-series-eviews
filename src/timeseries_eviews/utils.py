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

import pandas as pd
import numpy as np
import plotly.express as px
import logging

# Additional libraries from Jeremy's snippet
import polars as pl
import datetime
from dateutil.relativedelta import relativedelta
# Matplotlib is used in certain advanced or legacy plots:
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    template: str = "plotly_white"
):
    """
    Plot multiple columns of a time series DataFrame using Plotly Express.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a DateTimeIndex and one or more columns to plot.
    title : str, optional
        Plot title.
    ytitle : str, optional
        Y-axis label.
    template : str, optional
        Plotly template for styling.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object.
    """
    plot_df = df.copy()
    plot_df['time_index'] = plot_df.index
    melted = plot_df.melt(id_vars='time_index', var_name='Series', value_name='Value')

    fig = px.line(
        melted,
        x="time_index",
        y="Value",
        color="Series",
        title=title,
        template=template
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=ytitle,
        legend_title_text="Series"
    )
    fig.show()
    return fig


#------------------------------------------------------------------------------
# Jeremy Bejarano's Extended Utilities (Weighted Stats, etc.)
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