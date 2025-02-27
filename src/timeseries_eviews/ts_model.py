# Copyright (c) 2025 Eduardo Belisario Scheffer
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for details.

"""
ts_model.py
===========

A module for ARMA and GARCH models with EViews-like output. This file contains
the TsModel class for:
- ARMA(,d,) fitting using statsmodels
- GARCH fitting using arch
- EViews-style residual diagnostics, impulse response, forecasting, etc.

"""

import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

class TsModel:
    """
    A comprehensive class for ARMA and (optionally) GARCH modeling,
    aiming to replicate EViews-style outputs.

    Parameters
    ----------
    data : pd.Series
        The univariate time series data.
    arma_order : tuple of int, default=(1, 0, 1)
        (p, d, q) specifying ARIMA order.
    garch_order : tuple of int or None, optional
        (p, q) specifying GARCH order if GARCH is required.

    Attributes
    ----------
    data : pd.Series
        The original time series.
    arma_order : tuple
        The ARIMA order.
    garch_order : tuple or None
        The GARCH order or None if no GARCH model is to be fit.
    arma_result : ARIMAResultsWrapper or None
        statsmodels result object for ARMA(ARIMA).
    garch_result : arch.univariate.base.ARCHModelResult or None
        arch model result if GARCH was requested.
    residuals : pd.Series or None
        Residuals from the fitted ARIMA model (used for GARCH as well).
    """

    def __init__(self, data: pd.Series, arma_order=(1, 0, 1), garch_order=None):
        self.data = data
        self.arma_order = arma_order
        self.garch_order = garch_order

        # Internal placeholders
        self.arma_result = None
        self.garch_result = None
        self.residuals = None

    def fit(self):
        """
        Fit the ARIMA (p,d,q) model, then optionally fit a GARCH model to the residuals.
        """
        logging.info("Fitting ARIMA model with order=%s", self.arma_order)
        model = ARIMA(self.data, order=self.arma_order)
        self.arma_result = model.fit()
        self.residuals = self.arma_result.resid

        if self.garch_order is not None:
            logging.info("Fitting GARCH model with order=%s", self.garch_order)
            p, q = self.garch_order
            garch_mod = arch_model(self.residuals, p=p, q=q, rescale=False)
            self.garch_result = garch_mod.fit(disp="off")

    def display_estimation(self):
        """
        Print an EViews-style results table, including ARMA and (optionally) GARCH coefficients,
        standard errors, t-stats, and basic fit metrics. This is a simplified version,
        but can be expanded to match EViews formatting exactly.
        """
        if self.arma_result is None:
            print("Model not fitted yet.")
            return

        # ARMA portion
        print("==================================================")
        print("               ARIMA Estimation Results          ")
        print("==================================================")
        print(f"Order: ARIMA{self.arma_order}")
        print(f"Number of observations: {self.arma_result.nobs}")
        print(f"Log-likelihood: {self.arma_result.llf:.3f}")
        print(f"AIC: {self.arma_result.aic:.3f}")
        print(f"BIC: {self.arma_result.bic:.3f}")
        print("--------------------------------------------------")

        params = self.arma_result.params
        bse = self.arma_result.bse
        tvals = params / bse
        pvals = self.arma_result.pvalues
        coef_table_header = f"{'Param':<15} {'Coef':>12} {'Std.Err':>12} {'t-Stat':>12} {'P-value':>12}"
        print(coef_table_header)
        print("-"*len(coef_table_header))
        for idx in params.index:
            print(f"{idx:<15} {params[idx]:>12.4f} {bse[idx]:>12.4f} {tvals[idx]:>12.4f} {pvals[idx]:>12.4f}")
        print("==================================================")

        # GARCH portion (if applicable)
        if self.garch_result is not None:
            print("\n==================================================")
            print("              GARCH Estimation Results           ")
            print("==================================================")
            print(f"Order: GARCH{self.garch_order}")
            print(f"Number of observations: {self.garch_result.nobs}")
            print(f"Log-likelihood: {self.garch_result.loglikelihood:.3f}")
            print(f"AIC: {self.garch_result.aic:.3f}")
            print(f"BIC: {self.garch_result.bic:.3f}")
            print("--------------------------------------------------")

            gparams = self.garch_result.params
            gse = self.garch_result.std_err
            gtvals = gparams / gse
            gpvals = self.garch_result.pvalues
            print(coef_table_header)
            print("-"*len(coef_table_header))
            for idx in gparams.index:
                print(f"{idx:<15} {gparams[idx]:>12.4f} {gse[idx]:>12.4f} {gtvals[idx]:>12.4f} {gpvals[idx]:>12.4f}")
            print("==================================================")

    def forecast(self, steps=5, alpha=0.05):
        """
        Forecast the ARIMA mean equation forward 'steps' periods. If GARCH is present,
        optionally we can retrieve volatility forecasts as well.

        Parameters
        ----------
        steps : int
            Number of periods ahead to forecast.
        alpha : float
            Significance level for forecast confidence intervals.

        Returns
        -------
        dict
            Dictionary with 'mean' and 'mean_ci' keys. If GARCH is fitted,
            also returns 'vol' (volatility) forecast.
        """
        if self.arma_result is None:
            logging.warning("ARIMA model not fitted; cannot forecast.")
            return {}

        forecast_obj = self.arma_result.get_forecast(steps=steps)
        mean_forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=alpha)

        forecast_out = {
            "mean": mean_forecast,
            "mean_ci": conf_int
        }

        if self.garch_result is not None:
            # arch_model forecast horizon
            arch_fore = self.garch_result.forecast(horizon=steps)
            # The variance forecasts are in arch_fore.variance
            vol_fore = np.sqrt(arch_fore.variance.iloc[-1])
            forecast_out["vol"] = vol_fore

        return forecast_out