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

import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_impulse_response
from arch import arch_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

class TsModel:
    """
    A comprehensive class for ARMA/ARIMA and (optionally) GARCH modeling,
    with EViews-like outputs, diagnostics, and forecasts.

    Parameters
    ----------
    data : pd.Series
        Original time series.
    arma_order : tuple
        The ARIMA order (p, d, q).
    garch_order : tuple or None
        The GARCH order, if any.
    model_name : str
        Model label.
    arma_result : statsmodels ARIMAResultsWrapper
        Fitted ARIMA results.
    garch_result : arch.univariate.ARCHModelResult or None
        Fitted GARCH results (if any).
    residuals : pd.Series
        ARIMA residuals.
    st_resid : pd.Series
        Standardized residuals (if GARCH is fitted), otherwise None.
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

        # For storing ARMA fit stats
        self.loglik_arma = None
        self.aic_arma = None
        self.bic_arma = None

        # For storing GARCH fit stats
        self.loglik_garch = None
        self.aic_garch = None
        self.bic_garch = None


    def fit(self, disp="off"):
        """
        Fit the ARMA/ARIMA model, then optionally fit a GARCH model on the ARIMA residuals.

        Parameters
        ----------
        disp : str, optional
            Display option for arch_model fitting. 'off' = no console output.
        """
        logging.info(f"Fitting ARIMA with order={self.arma_order} on {self.model_name}")
        model = ARIMA(self.data, order=self.arma_order)
        self.arma_result = model.fit()
        self.residuals = self.arma_result.resid
        self.loglik_arma = self.arma_result.llf
        self.aic_arma = self.arma_result.aic
        self.bic_arma = self.arma_result.bic

        if self.garch_order is not None:
            logging.info(f"Fitting GARCH with order={self.garch_order} on {self.model_name} residuals")
            p, q = self.garch_order
            garch_mod = arch_model(self.residuals, p=p, q=q, rescale=False)
            self.garch_result = garch_mod.fit(disp=disp)
            self.loglik_garch = self.garch_result.loglikelihood
            self.aic_garch = self.garch_result.aic
            self.bic_garch = self.garch_result.bic

            # Standardized residuals
            self.st_resid = self.garch_result.std_resid
        else:
            self.st_resid = None


    def display_estimation(self):
        """
        Print an EViews-style results table, including ARIMA (and GARCH) parameters,
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
            DataFrame with actual, fitted, and residual series.
        """
        fitted_vals = self.arma_result.fittedvalues
        df_out = pd.DataFrame({
            'Actual': self.data,
            'Fitted': fitted_vals,
            'Residual': self.residuals
        })
        return df_out


    def plot_residuals(self, fig_size=(8,4), output_dir=None, filename="residuals.png"):
        """
        Plot raw residuals over time using Matplotlib.
        """
        if self.residuals is None:
            logging.error("No residuals. Fit the model first.")
            return

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(self.residuals.index, self.residuals, label="Residuals", color='blue')
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f"{self.model_name} - Residuals")
        ax.set_xlabel("Index")
        ax.set_ylabel("Residual")
        ax.legend()
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path_out = os.path.join(output_dir, filename)
            plt.savefig(path_out, dpi=150, bbox_inches="tight")

        fig.show()


    def plot_std_residuals(self, fig_size=(8,4), output_dir=None, filename="std_residuals.png"):
        """
        Plot standardized residuals if GARCH is fitted; else fallback to residual/std.
        """
        if self.st_resid is not None:
            sr = self.st_resid
            title = f"{self.model_name} - Standardized Residuals (GARCH)"
        else:
            sr = self.residuals / self.residuals.std()
            title = f"{self.model_name} - Standardized Residuals (No GARCH)"

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(sr.index, sr, label="Std. Residuals", color='blue')
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.set_ylabel("Std. Resid.")
        ax.legend()
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path_out = os.path.join(output_dir, filename)
            plt.savefig(path_out, dpi=150, bbox_inches="tight")

        plt.show()


    def plot_roots(self, fig_size=(6,6), output_dir=None, filename="arma_roots.png", show_plot=True):
        """
        Plot AR and MA roots for stationarity/invertibility checks in Matplotlib.
        """
        arroots = self.arma_result.arroots
        maroots = self.arma_result.maroots

        fig, ax = plt.subplots(figsize=fig_size)
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

        if len(arroots) > 0:
            ax.scatter(arroots.real, arroots.imag, color='blue', label='AR Roots')
        if len(maroots) > 0:
            ax.scatter(maroots.real, maroots.imag, color='red', marker='x', label='MA Roots')

        ax.set_title(f"{self.model_name} - AR/MA Roots")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.legend()
        plt.grid(True)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path_out = os.path.join(output_dir, filename)
            plt.savefig(path_out, dpi=150, bbox_inches="tight")

        plt.show()


    def impulse_response(self, fig_size=(6,4), periods=10, impulse=1.0, output_dir=None, filename="impulse_response.png"):
        """
        Compute a simple impulse response function for the ARMA model, with user-defined
        shock size. This uses statsmodels' arma_impulse_response internally.

        Parameters
        ----------
        periods : int
            Number of periods to compute IRF.
        impulse : float
            Size of shock to epsilon_t.
        output : str, optional
            'plot' or 'table'. If 'plot', display a stem plot.

        Returns
        -------
        np.ndarray or None
            IRF array if output='table', otherwise None.
        """
        arparams = self.arma_result.arparams if len(self.arma_result.arparams) > 0 else []
        maparams = self.arma_result.maparams if len(self.arma_result.maparams) > 0 else []

        irf_values = arma_impulse_response(ar=arparams, ma=maparams, nobs=periods) * impulse

        fig, ax = plt.subplots(figsize=fig_size)
        ax.stem(range(1, periods + 1), irf_values, use_line_collection=True)
        ax.set_title(f"{self.model_name} - Impulse Response (shock={impulse})")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Response")
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path_out = os.path.join(output_dir, filename)
            plt.savefig(path_out, dpi=150, bbox_inches="tight")

        plt.show()


    def correlogram(self, lags=24, squared=False, output_dir=None, filename="correlogram.png", show_plot=True):
        """
        Compute and optionally show a two-panel Plotly figure (ACF, PACF, Ljung-Box Q) for residuals
        or squared residuals to detect autocorrelation or ARCH effects.

        Parameters
        ----------
        lags : int, optional
            Number of lags to compute. Default=24.
        squared : bool, optional
            If True, use squared residuals.
        plot : bool, optional
            If True, display ACF/PACF plots.

        Returns
        -------
        pd.DataFrame
            A table with columns ['Lag', 'AC', 'PAC', 'Q-Stat', 'Prob'].
        """

        if self.residuals is None:
            logging.error("No residuals found. Fit the model first.")
            return pd.DataFrame()

        data_series = self.residuals**2 if squared else self.residuals
        corr_df = plot_correlogram(
            data_series,
            max_lag=lags,
            title=f"{self.model_name} - {'Squared ' if squared else ''}Residuals Correlogram",
            alpha=0.05,
            output_dir=output_dir,
            filename=filename,
            show_plot=show_plot
        )
        return corr_df
    

    def get_covariance_matrix(self):
        """
        Return covariance matrices for ARMA and (if fitted) GARCH parameters.

        Returns
        -------
        dict
            Keys 'ARMA_cov' and 'GARCH_cov' with covariance matrices (if available).
        """
        cov_matrices = {}
        if self.arma_result is not None:
            cov_matrices["ARMA_cov"] = self.arma_result.cov_params()
        if self.garch_result is not None:
            # arch_model doesn't always store a direct covariance matrix but we can approximate:
            garch_cov = self.garch_result.variance_params
            cov_matrices["GARCH_cov"] = garch_cov
        return cov_matrices


    def forecast(self, steps=5, method='static', alpha=0.05, plot='none',
                 output_dir=None, filename="forecast.png", show_plot=True):
        """
        Forecast the ARIMA model for a specified horizon. Optionally retrieve GARCH volatility.

        **Static Forecast (One-Step Ahead)**:
            - Uses actual historical values for lagged dependent variables in each step of the forecast.
            - Minimizes error accumulation because each new forecast uses real observed data.
            - Suitable for short-term forecasts and in-sample evaluation.

        **Dynamic Forecast (Multi-Step Ahead)**:
            - Uses previously forecasted values for lagged dependent variables once the forecast period starts.
            - Errors accumulate over the forecast horizon.
            - Suitable for long-term forecasting and out-of-sample projections.

        Parameters
        ----------
        steps : int, optional
            Forecast horizon.
        method : {'static', 'dynamic'}, optional
            Forecast approach.
        alpha : float, optional
            Significance level for confidence intervals.
        plot : {'none', 'forecast', 'forecast_and_actuals'}, optional
            Controls plotting of the forecast results.

        Returns
        -------
        dict
            {
                'mean': pd.Series,
                'mean_ci': pd.DataFrame,
                'vol': (if GARCH fitted) np.ndarray,
            }
        """
        if self.arma_result is None:
            logging.error("Cannot forecast. ARMA model not fitted.")
            return {}

        # In statsmodels ARIMA, 'dynamic' parameter can implement multi-step forecasting:
        # - dynamic=False => 1-step ahead (uses actual for all lags)
        # - dynamic=True => multi-step ahead (uses forecasted values once you step into the forecast horizon)
        dynamic_param = True if method.lower() == 'dynamic' else False

        forecast_obj = self.arma_result.get_forecast(steps=steps, dynamic=dynamic_param)
        mean_forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=alpha)

        forecast_out = {
            "mean": mean_forecast,
            "mean_ci": conf_int
        }

        if self.garch_result is not None:
            garch_fc = self.garch_result.forecast(horizon=steps)
            vol_fc = np.sqrt(garch_fc.variance.iloc[-1])
            forecast_out["vol"] = vol_fc

        if plot in ["forecast", "forecast_and_actuals"]:
            fig, ax = plt.subplots(figsize=(8,4))
            if plot == "forecast_and_actuals":
                ax.plot(self.data.index, self.data, label="Actual", color="blue")

            ax.plot(mean_forecast.index, mean_forecast, label="Forecast", color="orange")
            ax.fill_between(mean_forecast.index,
                            conf_int.iloc[:, 0],
                            conf_int.iloc[:, 1],
                            color="orange", alpha=0.2,
                            label="Confidence Interval")
            ax.set_title(f"{self.model_name} - {method.capitalize()} Forecast")
            ax.legend()
            plt.tight_layout()

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                path_out = os.path.join(output_dir, filename)
                plt.savefig(path_out, dpi=150, bbox_inches="tight")

            if show_plot:
                plt.show()

        return forecast_out
    

if __name__ == "__main__":
    """
    Example script demonstrating usage with a hypothetical /data_manual/qqq.xlsm file
    containing daily prices. We'll:
      1) Load data, compute log-returns.
      2) Plot & compute correlogram, check stationarity via ADF.
      3) Fit ARMA(1,1) GARCH(1,1).
      4) Check residuals, squared residuals correlogram.
      5) Display model output.
      6) Make static & dynamic forecasts, plot with confidence intervals.
    """
    from pathlib import Path
    from utils import compute_correlogram, plot_correlogram
    
    from statsmodels.tsa.stattools import adfuller

    # 1) Load data (example code, adjust path as needed)
    
    current_dir = Path(__file__).parent.parent
    file_path = current_dir.parent / 'tests' / 'data_manual' / 'qqq.csv'
    df_prices = pd.read_csv(file_path, index_col="date", parse_dates=True)
    
    # Compute log returns
    df_prices["Returns"] = np.log(df_prices["PRC"]).diff().dropna()
    returns = df_prices["Returns"].dropna()

    # 2) Plot & compute correlogram, ADF test
    print("---- Correlogram for Returns ----")
    plot_correlogram(returns, max_lag=20, title="SPY500 Log Returns Correlogram")
    corr_table = compute_correlogram(returns, max_lag=20)
    print(corr_table.head(10))

    print("\n---- Augmented Dickey-Fuller Test ----")
    adf_result = adfuller(returns.dropna(), autolag='AIC')
    print(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    print("Critical Values:")
    for k, v in adf_result[4].items():
        print(f"  {k}: {v:.4f}")

    # 3) Fit ARMA(1,1) GARCH(1,1)
    print("\n---- Fitting ARMA(1,1) + GARCH(1,1) ----")
    model = TsModel(data=returns, arma_order=(1,0,1), garch_order=(1,1), model_name="SPY500_ARMA_GARCH")
    model.fit()

    # 4) Check residuals correlogram, squared residuals correlogram
    print("\n---- Residuals Correlogram ----")
    resid_corr = model.correlogram(lags=20, squared=False, show_plot=True)
    print(resid_corr.head(10))

    print("\n---- Squared Residuals Correlogram ----")
    sq_resid_corr = model.correlogram(lags=20, squared=True, show_plot=True)
    print(sq_resid_corr.head(10))

    # 5) Display model output
    print("\n---- Model Estimation Results ----")
    model.display_estimation()

    # 6) Static & Dynamic Forecasts
    print("\n---- Static Forecast (One-Step Ahead) ----")
    forecast_static = model.forecast(steps=10, method='static', alpha=0.05, plot='forecast_and_actuals')
    print("Forecast Values:\n", forecast_static["mean"])
    if "vol" in forecast_static:
        print("Volatility Forecast:\n", forecast_static["vol"])

    print("\n---- Dynamic Forecast (Multi-Step Ahead) ----")
    forecast_dynamic = model.forecast(steps=10, method='dynamic', alpha=0.05, plot='forecast_and_actuals')
    print("Forecast Values:\n", forecast_dynamic["mean"])
    if "vol" in forecast_dynamic:
        print("Volatility Forecast:\n", forecast_dynamic["vol"])

    print("\nDone.")


