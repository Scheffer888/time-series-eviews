# Copyright (c) 2025 Eduardo Belisario Scheffer
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for details.

"""
cli.py
======

Command Line Interface (CLI) module. Provides `main()` entry point for the package
if the user wants to run commands directly from the terminal. This example uses the 
`click` library to demonstrate how you might structure CLI commands for creating,
fitting, and diagnosing time series models.
"""

import click
import pandas as pd
from .ts_model import TsModel
from .var_model import VarModel

@click.group()
def main():
    """
    timeseries_eviews CLI.
    Use subcommands to operate on the ARMA/GARCH (TsModel) or VAR (VarModel).
    """
    pass

@main.command()
@click.option("--csv-file", type=click.Path(exists=True), required=True, help="Path to CSV file of univariate time series.")
@click.option("--p", default=1, help="AR order.")
@click.option("--d", default=0, help="Differencing order.")
@click.option("--q", default=1, help="MA order.")
@click.option("--garch", is_flag=True, help="Flag to also fit a GARCH(1,1) model on residuals.")
def arma(csv_file, p, d, q, garch):
    """
    Fit an ARMA/ARIMA model (and optional GARCH) on a univariate time series.
    """
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    series = df.iloc[:, 0]  # assume first column is the series
    if garch:
        model = TsModel(data=series, arma_order=(p, d, q), garch_order=(1,1))
    else:
        model = TsModel(data=series, arma_order=(p, d, q))
    model.fit()
    model.display_estimation()

@main.command()
@click.option("--csv-file", type=click.Path(exists=True), required=True, help="Path to CSV file of multivariate time series.")
@click.option("--lags", default=1, help="Number of lags for VAR.")
@click.option("--model-type", default="VAR", help="Model type: VAR, SVAR, VECM.")
def var(csv_file, lags, model_type):
    """
    Fit a Vector Autoregression (or SVAR, VECM) on multivariate time series.
    """
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    model = VarModel(data=df, lags=lags, model_type=model_type)
    model.fit()
    model.display_estimation()