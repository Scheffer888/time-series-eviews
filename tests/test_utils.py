import pytest
import pandas as pd
import numpy as np
from timeseries_eviews.utils import (
    check_time_series,
    standardize_column_names,
    compute_correlogram,
    plot_correlogram,
)


@pytest.fixture
def simple_data():
    """
    Returns a DataFrame with 'date' and 'value' as columns,
    not setting 'date' as the index.
    """
    data = {
        "date": pd.date_range(start="2020-01-01", periods=5, freq="D"),
        "value": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)
    return df


def test_check_time_series(simple_data):
    """
    Tests check_time_series function with date_col specified,
    then with no date_col assuming index is already DateTimeIndex.
    """
    df = simple_data
    # normal usage: pass 'date_col'
    df_ts = check_time_series(df.copy(), date_col="date", freq="D")
    assert isinstance(df_ts.index, pd.DatetimeIndex)
    assert df_ts.index.freqstr == "D"
    assert "date" not in df_ts.columns, "After setting index, 'date' should not be a column."

    # Now let's assume the user has already set the index
    # We'll mimic that scenario by setting it ourselves
    df2 = df.copy().set_index("date")
    df_ts2 = check_time_series(df2, date_col=None, freq="D")
    assert isinstance(df_ts2.index, pd.DatetimeIndex)
    assert df_ts2.index.freqstr == "D"


def test_standardize_column_names():
    """Test that column names are renamed to 'var_0', 'var_1', etc."""
    df = pd.DataFrame({
        " col 1 ": [1, 2],
        " col 2 ": [3, 4]
    })
    df_std = standardize_column_names(df.copy(), prefix="var")
    # check if columns are renamed
    assert df_std.columns.tolist() == ["var_0", "var_1"]


def test_compute_correlogram():
    """Test the shape and columns of the correlogram DataFrame."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 3, 4, 5, 6, 4, 8, 4, 2, 1, 5, 2, 8, 3, 4, 6, 7, 2, 2, 4, 
                        12, 12, 12, 14, 14, 14, 15, 12, 12, 12, 14, 14, 14, 15, 12, 12, 12, 14, 14, 14, 15, 12, 12, 12, 14, 14, 14,
                        15, 12, 12, 12, 14, 14, 14, 15, 12, 12, 12, 14, 14, 14, 15])
    corr_df = compute_correlogram(series, max_lag=24)
    # expect 3 rows
    assert corr_df.shape[0] == 24
    # check columns
    expected_cols = ["Lag", "AC", "PAC", "Q-Stat", "Prob"]
    assert corr_df.columns.tolist() == expected_cols
    # confirm lag values 1, 2, 3
    #assert corr_df["Lag"].tolist() == [1, 2, 3]


def test_plot_correlogram():
    """Smoke test to ensure no errors occur when plotting."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 3, 4, 5, 6, 4, 8, 4, 2, 1, 5, 2, 8, 3, 4, 6, 7, 2, 2, 4, 
                        12, 12, 12, 14, 14, 14, 15, 12, 12, 12, 14, 14, 14, 15, 12, 12, 12, 14, 14, 14, 15, 12, 12, 12, 14, 14, 14,
                        15, 12, 12, 12, 14, 14, 14, 15, 12, 12, 12, 14, 14, 14, 15], name="Some series")
    fig = plot_correlogram(series, max_lag=24, title="Test Correlogram")
    assert fig is not None, "plot_correlogram should return a figure object."