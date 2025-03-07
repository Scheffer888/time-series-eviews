import pytest
import pandas as pd
import numpy as np
from timeseries_eviews.ts_model import TsModel


@pytest.fixture
def arma_data():
    """
    Create a simple AR(1) process for test: y_t = 0.5*y_{t-1} + e_t
    """
    np.random.seed(123)
    n = 50
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t-1] + np.random.normal(0, 1)
    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    return pd.Series(y, index=dates)


def test_ts_model_fit(arma_data):
    """
    Test fitting a TsModel (ARMA only, no GARCH).
    """
    model = TsModel(data=arma_data, arma_order=(1, 0, 0), garch_order=None, model_name="TestAR1")
    model.fit()
    assert model.arma_result is not None, "ARMA result should not be None after fit."
    assert model.residuals is not None, "Residuals must be stored after fit."

def test_ts_model_forecast_static(arma_data):
    """
    Test static forecast on TsModel.
    """
    model = TsModel(data=arma_data, arma_order=(1, 0, 1), model_name="TestARMA")
    model.fit()
    out = model.forecast(steps=5, method='static', plot='none')
    assert "mean" in out, "Forecast should include 'mean' key"
    assert len(out["mean"]) == 5, "Should return 5-step forecast"

def test_ts_model_forecast_dynamic(arma_data):
    """
    Test dynamic forecast on TsModel.
    """
    model = TsModel(data=arma_data, arma_order=(1, 0, 1), model_name="TestARMA")
    model.fit()
    out = model.forecast(steps=5, method='dynamic', plot='none')
    assert "mean" in out, "Forecast should include 'mean' key"
    assert len(out["mean"]) == 5, "Should return 5-step forecast"

