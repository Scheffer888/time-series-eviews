# **Time Series Analysis Package (EViews-Style Replication)**
*A Comprehensive Python Package for ARMA, GARCH, and VAR Modeling with Full EViews-Like Diagnostics*

## **Overview**
This package provides a **professional-grade** framework for **time series analysis** that replicates **EViews-style outputs and functionality**. It enables users to estimate, analyze, and forecast **ARMA, GARCH, and VAR models**, with extensive diagnostic tools including impulse response functions, correlograms, Johansen cointegration tests, and stability analysis.

The package is designed to:
- **Mimic EViews' output formatting** for estimation results, tests, and diagnostics.
- **Provide deep statistical insights** into time series dynamics.
- **Offer flexible forecasting tools**, including dynamic and stochastic simulation.
- **Support both numerical and graphical outputs** for enhanced data visualization.

## **Features**
### **1. ARMA/GARCH Modeling (`ts_model`)**
- Estimate **ARMA(p, q)** and **GARCH(p, q)** models.
- Display **exact** EViews-style coefficient tables with:
  - Coefficient estimates, standard errors, t-statistics, and p-values.
  - Fit statistics: R², AIC, SC, Log-likelihood, Durbin-Watson, and more.
- **Residual Diagnostics**:
  - Residual correlograms (with user-defined lags).
  - Portmanteau autocorrelation tests.
  - Normality and ARCH effect tests.
- **Impulse Response Analysis**:
  - Compute responses for ARMA models with user-defined impulse size.
  - Display results in both tabular and graphical formats.
- **Forecasting**:
  - Static, dynamic, and stochastic simulation forecasts.
  - Forecasts with confidence intervals and visualization.

### **2. Vector Autoregression (VAR) Modeling (`var_model`)**
- Estimate **VAR(p)** models with multiple endogenous variables.
- Display **EViews-style estimation tables** with:
  - Coefficient estimates, standard errors, t-statistics, p-values.
  - Model statistics: R², Log-likelihood, Akaike Information Criterion (AIC), Schwarz Criterion (SC), etc.
- **Residual Tests**:
  - Residual correlograms (graph and table).
  - Portmanteau tests for autocorrelation.
  - Cross-correlation matrices.
- **Cointegration Testing**:
  - Full **Johansen Cointegration Test** implementation.
  - User-defined trend assumptions, critical values, and lags.
- **Impulse Response Functions (IRF)**:
  - Response analysis for shocks to VAR models.
  - Confidence interval estimation via **Analytic, Monte Carlo, or Bootstrap** methods.
- **Stability and Lag Structure Analysis**:
  - Compute **roots of the characteristic polynomial** to assess stability.
  - Display EViews-style stability test results.
  - Suggest **optimal lag structure** using AIC, SC, and other criteria.
- **Forecasting**:
  - Static and dynamic forecasts with visualization.
  - Stochastic simulation with scenario analysis.

## **Installation**
```bash
pip install timeseries-eviews
```

## **Usage**
ARMA/GARCH Example

```python
from timeseries_eviews import ts_model

# Load dataset
import pandas as pd
data = pd.read_csv("timeseries_data.csv")

# Fit ARMA(2,1) with GARCH(1,1)
model = ts_model(data["returns"], order=(2,0,1), garch_order=(1,1))
model.fit()

# Display estimation output (EViews-style table)
model.display_estimation()

# Residual correlogram
model.plot_correlogram(lags=12)

# Impulse Response Function
model.impulse_response(periods=10, output='plot')

# Forecasting
forecast_results = model.forecast(steps=10, method='dynamic', plot='forecast')
```

VAR Example

```python
from timeseries_eviews import var_model

# Load multivariate dataset
df = pd.read_csv("multivariate_timeseries.csv", index_col=0, parse_dates=True)

# Fit VAR(2)
var = var_model(df, lags=2)
var.fit()

# Display estimation results
var.display_estimation()

# Check for autocorrelation in residuals
var.portmanteau_test(max_lag=12)

# Johansen Cointegration Test
var.johansen_cointegration_test(det_order=3, k_ar_diff=1, significance='mhm')

# Impulse Response
var.impulse_response(steps=10, ident='cholesky', ci_method='analytic')

# Forecasting
var.forecast(steps=10, method='dynamic', plot=True)
```