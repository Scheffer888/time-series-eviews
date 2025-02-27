# Copyright (c) 2025 Eduardo Belisario Scheffer
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for details.

"""
ts_model.py
===========

A module for ARMA and (optionally) GARCH models with expanded EViews-like output.
Includes the following:
- Attributes for representing ARMA/ARIMA and GARCH model parameters/stats
- Methods for actual/fitted/residuals table, residual plots, standardized residuals
- ARMA diagnostics (roots, IRF, correlogram)
- Covariance matrix of parameters
- Forecasting: dynamic, static, stochastic simulation
"""

import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt



class TsModel:
    """
    A comprehensive class for ARMA and (optionally) GARCH modeling,
    aiming to replicate EViews-like outputs and diagnostics.

    Parameters
    ----------
    data : pd.Series
        The univariate time series data.
    arma_order : tuple of int, default=(1, 0, 1)
        (p, d, q) specifying ARIMA order.
    garch_order : tuple of int or None, optional
        (p, q) specifying GARCH order if GARCH is required.
    model_name : str, optional
        Arbitrary name for the model, e.g. 'My ARMA-GARCH Model'.

    Attributes
    ----------
    data : pd.Series
        The original time series.
    arma_order : tuple
        The ARIMA order (p, d, q).
    garch_order : tuple or None
        The GARCH order or None if no GARCH model is to be fit.
    model_name : str
        Name/label for the model.
    arma_result : statsmodels.tsa.arima.model.ARIMAResults
        Fitted ARIMA results object.
    garch_result : arch.univariate.base.ARCHModelResult
        Fitted GARCH model result object (if any).
    residuals : pd.Series
        Residuals from the ARIMA model.
    st_resid : pd.Series
        Standardized residuals (if GARCH fitted).
    """

    def __init__(self, data: pd.Series, arma_order=(1, 0, 1), garch_order=None, model_name="TS_Model"):
        self.data = data
        self.arma_order = arma_order
        self.garch_order = garch_order
        self.model_name = model_name

        self.arma_result = None
        self.garch_result = None
        self.residuals = None
        self.st_resid = None  # standardized residuals if GARCH is fitted

        # Representation table / fit stats placeholders
        self.loglik_arma = None
        self.aic_arma = None
        self.bic_arma = None
        self.loglik_garch = None
        self.aic_garch = None
        self.bic_garch = None

    def fit(self, disp="off"):
        """
        Fit the ARMA/ARIMA model, then optionally fit GARCH on its residuals.

        Parameters
        ----------
        disp : str
            Display option passed to arch_model.fit. 'off' means silent mode.
        """
        # Fit ARIMA
        model = ARIMA(self.data, order=self.arma_order)
        self.arma_result = model.fit()
        self.residuals = self.arma_result.resid

        # Store ARMA fit stats
        self.loglik_arma = self.arma_result.llf
        self.aic_arma = self.arma_result.aic
        self.bic_arma = self.arma_result.bic

        # Fit GARCH (if requested)
        if self.garch_order is not None:
            p, q = self.garch_order
            garch_mod = arch_model(self.residuals, p=p, q=q, rescale=False)
            self.garch_result = garch_mod.fit(disp=disp)
            self.loglik_garch = self.garch_result.loglikelihood
            self.aic_garch = self.garch_result.aic
            self.bic_garch = self.garch_result.bic

            # Standardized residuals = residual / conditional st. dev
            self.st_resid = self.garch_result.std_resid
        else:
            self.st_resid = None

    def display_estimation(self):
        """
        Print an EViews-style results table, including ARMA and (optionally) GARCH parameters,
        standard errors, t-stats, p-values, and fit statistics.
        """
        print("==================================================")
        print(f"          {self.model_name} Representation       ")
        print("==================================================")
        print(f"ARMA Order: {self.arma_order}")
        if self.garch_order is not None:
            print(f"GARCH Order: {self.garch_order}")
        else:
            print("GARCH: None")

        # ARMA portion
        print("--------------------------------------------------")
        print("ARMA(ARIMA) Estimation Results:")
        print(f"Log-likelihood: {self.loglik_arma:.3f}")
        print(f"AIC: {self.aic_arma:.3f}, BIC: {self.bic_arma:.3f}")
        params = self.arma_result.params
        bse = self.arma_result.bse
        tvals = params / bse
        pvals = self.arma_result.pvalues

        header = f"{'Param':<15} {'Coef':>12} {'Std.Err':>12} {'t-Stat':>12} {'P-value':>12}"
        print(header)
        print("-" * len(header))
        for idx in params.index:
            print(f"{idx:<15} {params[idx]:>12.4f} {bse[idx]:>12.4f} {tvals[idx]:>12.4f} {pvals[idx]:>12.4f}")

        # GARCH portion
        if self.garch_result is not None:
            print("--------------------------------------------------")
            print("GARCH Estimation Results:")
            print(f"Log-likelihood: {self.loglik_garch:.3f}")
            print(f"AIC: {self.aic_garch:.3f}, BIC: {self.bic_garch:.3f}")

            gparams = self.garch_result.params
            gse = self.garch_result.std_err
            gtvals = gparams / gse
            gpvals = self.garch_result.pvalues

            print(header)
            print("-" * len(header))
            for idx in gparams.index:
                print(f"{idx:<15} {gparams[idx]:>12.4f} {gse[idx]:>12.4f} {gtvals[idx]:>12.4f} {gpvals[idx]:>12.4f}")

        print("==================================================")

    def get_actual_fitted_residuals(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns ['Actual', 'Fitted', 'Residual'].

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by the same index as the original data, containing
            actual, fitted, and residual series.
        """
        fitted_vals = self.arma_result.fittedvalues
        df_out = pd.DataFrame({
            'Actual': self.data,
            'Fitted': fitted_vals,
            'Residual': self.residuals
        })
        return df_out

    def plot_residuals(self):
        """
        Plot the raw residuals over time (Matplotlib).
        """
        if self.residuals is None:
            logging.error("No residuals found. Fit the model first.")
            return
        self.residuals.plot(title=f"{self.model_name} - Residuals", figsize=(10, 4))
        plt.axhline(0, color='red', linestyle='--')
        plt.tight_layout()
        plt.show()

    def plot_std_residuals(self):
        """
        Plot the standardized residuals (residual / cond. st. dev.) if GARCH is fitted;
        otherwise uses overall std of residuals.
        """
        if self.st_resid is not None:
            sr = self.st_resid
            title = f"{self.model_name} - Standardized Residuals (GARCH)"
        else:
            sr = self.residuals / self.residuals.std()
            title = f"{self.model_name} - Standardized Residuals (No GARCH)"

        sr.plot(title=title, figsize=(10, 4))
        plt.axhline(0, color='red', linestyle='--')
        plt.tight_layout()
        plt.show()

    def plot_roots(self):
        """
        Plot the AR and MA roots in the complex plane for stationarity checks.
        """
        arroots = self.arma_result.arroots
        maroots = self.arma_result.maroots

        fig, ax = plt.subplots(figsize=(6, 6))
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

        # Plot AR roots
        if arroots.size > 0:
            ax.scatter(arroots.real, arroots.imag, color='blue', label='AR Roots')
        # Plot MA roots
        if maroots.size > 0:
            ax.scatter(maroots.real, maroots.imag, color='red', marker='x', label='MA Roots')

        ax.set_title(f"{self.model_name} - AR/MA Roots")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.legend()
        plt.grid(True)
        plt.show()

    def impulse_response(self, periods=10, impulse=1.0, output='plot'):
        """
        Compute a simple impulse response function for the ARMA model, applying
        an impulse shock of size 'impulse' to the white noise at t=0. This is
        a simplified approach using statsmodels' arma_impulse_response or custom logic.

        Parameters
        ----------
        periods : int
            Number of periods to compute IRF.
        impulse : float
            Size of the shock to epsilon_t.
        output : str
            'plot' or 'table'.

        Returns
        -------
        np.ndarray or None
            The IRF array if output='table'. Otherwise, displays a plot and returns None.
        """
        from statsmodels.tsa.api import arma_impulse_response
        arparams = self.arma_result.arparams if len(self.arma_result.arparams) > 0 else []
        maparams = self.arma_result.maparams if len(self.arma_result.maparams) > 0 else []

        irf_values = arma_impulse_response(ar=arparams, ma=maparams, nobs=periods) * impulse
        if output.lower() == 'table':
            return irf_values
        else:
            plt.stem(range(1, periods+1), irf_values, use_line_collection=True)
            plt.title(f"{self.model_name} - Impulse Response (shock={impulse})")
            plt.xlabel("Lag")
            plt.ylabel("Response")
            plt.tight_layout()
            plt.show()

    def correlogram(self, lags=24, squared=False, plot=True):
        """
        Compute and optionally plot the correlogram (ACF, PACF, Q-stat) for the residuals or
        squared residuals to check for autocorrelation or ARCH effects.

        Parameters
        ----------
        lags : int, optional
            Maximum lag to compute. Default=24.
        squared : bool, optional
            If True, use squared residuals.
        plot : bool, optional
            If True, display ACF/PACF plots using Matplotlib.

        Returns
        -------
        pd.DataFrame
            Table with columns ['Lag', 'AC', 'PAC', 'Q-Stat', 'Prob'].
        """
        from .utils import compute_correlogram, plot_correlogram

        if self.residuals is None:
            logging.error("No residuals found. Fit the model first.")
            return pd.DataFrame()

        data_series = (self.residuals**2) if squared else self.residuals
        corr_table = compute_correlogram(data_series, max_lag=lags)

        if plot:
            plot_title = f"{self.model_name} - {'Squared ' if squared else ''}Residuals Correlogram"
            plot_correlogram(data_series, max_lag=lags, title=plot_title)

        return corr_table

    def get_covariance_matrix(self):
        """
        Return the covariance matrix of the ARMA (and optionally GARCH) coefficients.

        Returns
        -------
        dict
            Dictionary with keys 'ARMA_cov' and possibly 'GARCH_cov' if GARCH is fitted.
        """
        cov_matrices = {}
        if self.arma_result is not None:
            cov_matrices["ARMA_cov"] = self.arma_result.cov_params()
        if self.garch_result is not None:
            # arch_model doesn't always store a direct covariance matrix, but we have .variance_params
            # Instead, we can reconstruct from standard errors if needed.
            # For demonstration, we fetch a robust or default covariance matrix if provided:
            garch_cov = self.garch_result.variance_params
            cov_matrices["GARCH_cov"] = garch_cov

        return cov_matrices

    def forecast(self, steps=5, method='dynamic', sample=None, alpha=0.05, plot='none'):
        """
        Forecast ARIMA model for a specified horizon. If GARCH is present, optionally retrieve volatility forecasts.

        Parameters
        ----------
        steps : int, optional
            Number of steps ahead to forecast.
        method : str, optional
            'dynamic' or 'static'. 'dynamic' uses forecasted values for future lags.
            'static' does one-step-ahead forecasts using actual data. (Simplified example.)
        sample : str or None
            String specifying forecast sample range, e.g. '1:30' or '-10:0'. For demonstration only.
        alpha : float, optional
            Significance level for confidence intervals.
        plot : str, optional
            'none', 'forecast', or 'forecast_and_actuals'.

        Returns
        -------
        dict
            A dictionary with forecast results and optionally volatility if GARCH is fitted.
        """
        # For demonstration, we won't parse 'sample' to change the sample. A real implementation might slice data.
        if self.arma_result is None:
            logging.error("No fitted ARMA/ARIMA model to forecast. Fit first.")
            return {}

        # ARIMA forecast
        # statsmodels ARIMA: 'dynamic' argument means only in-sample dynamic forecast from a certain start
        # We'll do a simplified approach
        forecast_obj = self.arma_result.get_forecast(steps=steps)
        mean_forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=alpha)

        forecast_out = {
            "mean": mean_forecast,
            "mean_ci": conf_int
        }

        # GARCH forecast
        if self.garch_result is not None:
            garch_fore = self.garch_result.forecast(horizon=steps)
            # Variance forecasts in garch_fore.variance
            vol_fore = np.sqrt(garch_fore.variance.iloc[-1])
            forecast_out["vol"] = vol_fore

        # Plot if requested
        if plot in ('forecast', 'forecast_and_actuals'):
            plt.figure(figsize=(10, 4))
            if plot == 'forecast_and_actuals':
                plt.plot(self.data, label='Actuals')
            plt.plot(mean_forecast, label='Forecast', color='orange')
            plt.fill_between(mean_forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                             color='orange', alpha=0.2, label='Confidence Interval')
            plt.legend()
            plt.title(f"{self.model_name} Forecast")
            plt.tight_layout()
            plt.show()

        return forecast_out