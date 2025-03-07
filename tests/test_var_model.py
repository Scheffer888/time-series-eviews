import pytest
import pandas as pd
import numpy as np
from timeseries_eviews.var_model import VarModel


@pytest.fixture
def var_data():
    """
    Create a simple 2-dimensional dataset for VAR testing.
    Example: x_t = 0.7*x_{t-1} + e_t, y_t = 0.4*y_{t-1} + e'_t
    """
    np.random.seed(42)
    n = 50
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t-1] + np.random.normal(0, 1)
        y[t] = 0.4 * y[t-1] + np.random.normal(0, 1)

    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    df = pd.DataFrame({"X": x, "Y": y}, index=dates)
    return df

def test_var_model_fit(var_data):
    """
    Test fitting a simple VAR(1).
    """
    model = VarModel(data=var_data, lags=1, model_name="VAR")
    model.fit()
    assert model.fitted_model is not None, "VAR model should be fitted."

def test_var_model_forecast_static(var_data):
    """
    Test static forecast on VarModel.
    """
    model = VarModel(data=var_data, lags=1)
    model.fit()
    fc = model.forecast(steps=5, method='static', plot='none')
    assert fc.shape == (5, 2), "Should produce a 5x2 forecast DataFrame for 'X' and 'Y'"

def test_var_model_forecast_dynamic(var_data):
    """
    Test dynamic forecast on VarModel.
    """
    model = VarModel(data=var_data, lags=1)
    model.fit()
    fc = model.forecast(steps=5, method='dynamic', plot='none')
    assert fc.shape == (5, 2), "Should produce a 5x2 forecast DataFrame for 'X' and 'Y'"

def test_residual_correlograms(var_data):
    """
    Test residual correlograms for VAR model.
    """
    model = VarModel(data=var_data, lags=12)
    model.fit()
    model.residual_correlograms()