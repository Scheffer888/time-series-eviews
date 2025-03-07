# Copyright (c) 2025 Eduardo Belisario Scheffer
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for details.

"""
var_model.py
============

Module for Vector Autoregression (VAR) and its variants (SVAR, VECM). This
implementation focuses on a `VarModel` class that replicates EViews-like
functionality, including:

- Fit: standard VAR model using statsmodels
- Display EViews-like parameter tables and stats
- Residual diagnostics
- Impulse Response functions
- Forecasting with both "Static" (one-step ahead) and "Dynamic" (multi-step) approaches
"""

import os	
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objects as go

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.svar_model import SVAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import VARMAX

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


class VarModel:
    """
    A class to fit and analyze VAR (and extended) models, replicating EViews diagnostics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing multiple endogenous variables (time series).
    lags : int, optional
        Number of lags for the VAR model. Default=1.
    exog : pd.DataFrame or None, optional
        Optional exogenous regressors.
    model_type : str, optional
        'VAR', 'SVAR', 'VECM'. Determines which type of model to fit. Default='VAR'.
    svar_ident : str, optional
        Identification scheme for SVAR (e.g. 'A' or 'B') if `model_type='SVAR'`.
    """

    def __init__(self, data, lags=1, exog=None, model_name="VAR", svar_ident="A"):
        self.data = data
        self.lags = lags
        self.exog = exog
        self.model_name = model_name.upper()
        self.svar_ident = svar_ident

        # Fitted model result placeholder
        self.fitted_model = None

        # For VECM specifically
        self.vecm_rank = None  # Will be determined if model_type='VECM'
        self.vecm_trend = None
        self.vecm_results = None


    def fit(self, deterministic="nc", coint_rank=None):
        """
        Fit the model (VAR, SVAR, or VECM) with specified parameters.

        Parameters
        ----------
        deterministic : str
            Deterministic term specification for VECM (e.g., 'nc', 'co', 'ci', etc.).
            Only used if model_type='VECM'.
        coint_rank : int, optional
            Number of cointegrating relationships (rank). Only used if model_type='VECM'.

        Notes
        -----
        - If model_type='VAR', uses `VAR(self.data, exog=self.exog).fit(self.lags)`.
        - If model_type='SVAR', first fits a standard VAR, then applies structural identification
          using `SVAR(fitted_model, svar_type=self.svar_ident)`.
        - If model_type='VECM', uses `VECM(self.data, k_ar_diff=self.lags, deterministic=deterministic,
          rank=coint_rank, ...)`.
        """
        logging.info(f"Fitting {self.model_name} model with {self.lags} lags.")

        if self.model_name == "VAR":
            # Fit a standard VAR
            var_mod = VAR(self.data, exog=self.exog)
            self.fitted_model = var_mod.fit(self.lags)
            logging.info("VAR model fitted successfully.")

        elif self.model_name == "SVAR":
            # First fit standard VAR
            var_mod = VAR(self.data, exog=self.exog)
            base_fit = var_mod.fit(self.lags)
            # Then create SVAR with identification
            svar_mod = SVAR(base_fit, svar_type=self.svar_ident)
            self.fitted_model = svar_mod.fit()
            logging.info("SVAR model fitted successfully with identification=%s", self.svar_ident)

        elif self.model_name == "VECM":
            # VECM requires cointegration rank. If user didn't specify, attempt coint test
            if coint_rank is None:
                # Attempt to guess from Johansen if not provided
                logging.info("Cointegration rank not specified; attempting Johansen test.")
                joh_results = coint_johansen(self.data, det_order=0, k_ar_diff=self.lags)
                # Heuristic to pick rank: number of significant eigenvalues
                trace_stat = joh_results.lr1
                crit_vals = joh_results.cvt[:, 1]  # 5% critical values
                rank = sum(trace_stat > crit_vals)
                coint_rank = rank
                logging.info("Inferred cointegration rank from Johansen test: %d", coint_rank)

            self.vecm_rank = coint_rank
            self.vecm_trend = deterministic
            vecm_mod = VECM(self.data, k_ar_diff=self.lags, deterministic=deterministic, rank=coint_rank, exog=self.exog)
            self.vecm_results = vecm_mod.fit()
            self.fitted_model = self.vecm_results  # store for consistency
            logging.info("VECM fitted successfully with rank=%d, deterministic=%s", coint_rank, deterministic)
        else:
            raise ValueError(f"Unknown model_type={self.model_name}")


    def display_estimation(self):
        """
        Display EViews-like output table with coefficients, standard errors, t-stats, etc.
        Also prints key fit stats (AIC, SC, log-likelihood, etc.) depending on model_type.
        """
        if self.fitted_model is None:
            print("No fitted model available. Please call .fit() first.")
            return

        print("==================================================")
        print(f"          {self.model_name} Estimation Results")
        print("==================================================")

        # Standard VAR results
        if self.model_name in ["VAR", "SVAR"]:
            print(f"Lags: {self.lags}")
            print(f"Number of observations: {self.fitted_model.nobs}")
            print(f"AIC: {self.fitted_model.aic:.3f}")
            print(f"BIC: {self.fitted_model.bic:.3f}")
            print(f"FPE: {self.fitted_model.fpe:.3e}")
            print(f"Log-likelihood: {self.fitted_model.llf:.3f}")
            print("--------------------------------------------------")

            # Coefficients per equation
            for eqn_name in self.fitted_model.names:
                print(f"\nEquation: {eqn_name}")
                coefs = self.fitted_model.params[eqn_name]
                stderrs = self.fitted_model.stderr[eqn_name]
                tvals = coefs / stderrs
                # p-values not always directly accessible, we can approximate
                # or rely on self.fitted_model.pvalues if available
                coef_table_header = f"{'Variable':<20} {'Coef':>12} {'Std.Err':>12} {'t-Stat':>12}"
                print(coef_table_header)
                print("-" * len(coef_table_header))
                for param_name, coef_val in coefs.items():
                    std_err = stderrs.get(param_name, np.nan)
                    tval = tvals.get(param_name, np.nan)
                    print(f"{param_name:<20} {coef_val:>12.4f} {std_err:>12.4f} {tval:>12.4f}")

        elif self.model_name == "VECM":
            print(f"Lags (AR diff): {self.lags}")
            print(f"Cointegration rank: {self.vecm_rank}")
            print("--------------------------------------------------")
            params = self.vecm_results.params  # The short-run params
            alpha = self.vecm_results.alpha    # The speed of adjustment (long-run)
            beta = self.vecm_results.beta      # Cointegration vectors

            print("Short-run coefficients (params):")
            print(params)
            print("\nAlpha (speed of adjustment):")
            print(alpha)
            print("\nBeta (cointegration vectors):")
            print(beta)

        print("==================================================")


    def residual_correlograms(self, max_lag=24, output_dir=None, filename="var_resid_correl.png", show_plot=True):
        """
        Plot the autocorrelation (AC) for each equation's residual in a single figure
        with horizontal bars, EViews style. All variables shown in the same figure,
        either stacked vertically or side-by-side.

        Returns
        -------
        dict of pd.DataFrame
            AC DataFrames for each variable: { var_name: DataFrame of [Lag, AC] }.
        """
        if self.fitted_model is None:
            raise ValueError("VAR model not fitted.")

        residuals = self.fitted_model.resid
        var_names = list(residuals.columns)
        n_vars = len(var_names)

        # We'll create a figure with n_vars rows, 1 column, each row is a horizontal bar chart
        fig_height = max(4, n_vars * 2.5)
        fig, axes = plt.subplots(n_vars, 1, figsize=(8, fig_height), sharex=False)
        if n_vars == 1:
            axes = [axes]  # ensure it's a list

        df_dict = {}

        for i, var_name in enumerate(var_names):
            ax = axes[i]
            series = residuals[var_name].dropna()
            ac_vals = acf(series, nlags=max_lag, fft=False)

            n = len(series)
            se = 1.0 / np.sqrt(n)
            # For alpha=0.05 => ~1.96
            bound = 1.96 * se

            # Build a DataFrame
            data_rows = []
            for lag in range(1, max_lag+1):
                data_rows.append({"Lag": lag, "AC": ac_vals[lag]})
            ac_df = pd.DataFrame(data_rows)
            df_dict[var_name] = ac_df

            # Reverse so lag=1 is top
            rev_df = ac_df.iloc[::-1].reset_index(drop=True)

            # Horizontal bar chart
            ax.barh(rev_df.index, rev_df["AC"], color="#c6d9f0", edgecolor="black")
            ax.axvline(bound, color="grey", linestyle="--")
            ax.axvline(-bound, color="grey", linestyle="--")

            ax.invert_yaxis()
            ax.set_xlim([-1.0, 1.0])
            ax.set_ylabel(f"{var_name}  (Lag)")  # or just var_name
            ax.set_yticks(rev_df.index)
            ax.set_yticklabels(rev_df["Lag"].astype(str))
            # If you want an x-axis label for just the last subplot:
            if i == n_vars - 1:
                ax.set_xlabel("Autocorr.")
            else:
                ax.set_xlabel("")

            ax.set_title(f"{var_name}", fontsize=10, pad=10)

        fig.suptitle(f"{self.model_name} - Residual Correlograms", fontsize=11, y=0.96)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            outfile = os.path.join(output_dir, filename)
            plt.savefig(outfile, dpi=150, bbox_inches="tight")

        if show_plot:
            plt.show()

        return df_dict
    

    def portmanteau_test(self, max_lag=12):
        """
        Perform Portmanteau (Q) test for autocorrelation in VAR residuals up to `max_lag`.
        Return or print a table with Q-Stat, p-values, etc.
        """
        if self.model_name == "VECM":
            logging.warning("Portmanteau tests are typically for VAR residuals. Will run on short-run VECM residuals.")
        if self.fitted_model is None:
            logging.error("No fitted model to run portmanteau test. Call fit() first.")
            return

        if self.model_name in ["VAR", "SVAR"]:
            results = sm.stats.acorr_ljungbox(self.fitted_model.resid, lags=[max_lag], return_df=True)
            print("Portmanteau (Ljung-Box) Test:")
            print(results)
        elif self.model_name == "VECM":
            results = sm.stats.acorr_ljungbox(self.vecm_results.resid, lags=[max_lag], return_df=True)
            print("Portmanteau (Ljung-Box) Test on VECM residuals:")
            print(results)


    def johansen_cointegration_test(self, det_order=0, k_ar_diff=1):
        """
        Conduct Johansen cointegration test with user-specified deterministic trend assumptions
        and lag differences. This is helpful even if not strictly doing VECM.
        """
        logging.info("Running Johansen cointegration test with det_order=%d, k_ar_diff=%d.", det_order, k_ar_diff)
        results = coint_johansen(self.data, det_order=det_order, k_ar_diff=k_ar_diff)
        # Print or return results
        print("--------------------------------------------------")
        print("Johansen Cointegration Test Results")
        print("--------------------------------------------------")
        print(f"Eigenvalues:\n{results.eig}")
        print(f"\nTrace Statistics:\n{results.lr1}")
        print(f"Critical Values (trace):\n{results.cvt}")
        print(f"\nMax-Eigenvalue Statistics:\n{results.lr2}")
        print(f"Critical Values (max-eig):\n{results.cvm}")
        print("--------------------------------------------------")


    def impulse_response(self, steps=10, orth=True, impulselist=None, method='default'):
        """
        Compute IRF for the fitted model. Can do advanced bootstrapping or Monte Carlo if desired.

        Parameters
        ----------
        steps : int
            Number of periods for impulse response.
        orth : bool
            If True, compute orthogonalized IRF via Cholesky decomposition.
        impulselist : list of str, optional
            Variables for which impulses are computed. If None, all endogenous variables.
        method : str
            'default' for standard IRF, 'bootstrap' or 'mc' for advanced uncertainty analysis.

        Returns
        -------
        object
            IRF results or a custom structure. This is simplified; actual usage
            might rely on statsmodels or a custom IRF approach.
        """
        if self.fitted_model is None:
            logging.error("No fitted model for IRF. Call fit() first.")
            return None

        # For standard VAR or SVAR
        if self.model_name in ["VAR", "SVAR"]:
            irf = self.fitted_model.irf(steps=steps)
            if method.lower() == 'bootstrap':
                logging.info("Running bootstrap-based IRF confidence intervals.")
                # statsmodels IRF object supports bootstrap
                irf_plot = irf.plot(stderr=True, orth=orth, impulse=impulselist, plot_stderr=True, repl=1000)
                return irf_plot
            else:
                # default or 'mc'
                irf_plot = irf.plot(orth=orth, impulse=impulselist)
                return irf_plot

        elif self.model_name == "VECM":
            logging.warning("IRF for VECM not fully supported in statsmodels. "
                            "Implement custom approach if needed.")
            # We might do a custom approach or rely on the FEVD from the companion VAR representation.


    def white_heteroskedasticity_test(self):
        """
        Placeholder for White's test for heteroskedasticity in each VAR equation residual.
        Typically used for univariate regressions, but can be adapted for each equation in the system.

        Returns
        -------
        dict
            Dictionary of test statistics per equation.
        """
        if self.fitted_model is None:
            logging.error("No fitted model. Call fit() first.")
            return {}

        if self.model_name not in ["VAR", "SVAR"]:
            logging.warning("White test primarily relevant for (S)VAR. For VECM, adapt as needed.")

        # For each equation in residuals, run a White test as if it were a standard OLS regression:
        # This is a simplified approach, can be improved for true multivariate white test logic.
        residuals = self.fitted_model.resid
        results = {}
        for eqn_name in residuals.columns:
            eq_resid = residuals[eqn_name]
            # Build the design matrix (the original regressors) - here, we assume a standard approach
            # In a real scenario, we'd reconstruct the design from self.fitted_model.
            # We'll do a placeholder: y = alpha + Beta1 * y_{t-1} + ...
            # Then run White test with statsmodels. For illustration, we skip actual regressor retrieval.
            # A real implementation would require a full OLS object with exog expansions.
            results[eqn_name] = "White test placeholder"

        return results
    

    def forecast(self, steps=5, method='static', plot='none'):
        """
        Produce either a static or dynamic forecast of the fitted VAR model.

        **Static Forecast (One-Step Ahead)**:
            - For each forecasted period, we use the actual data if it's within the sample
              or newly available. Minimizes accumulation of forecast errors.

        **Dynamic Forecast (Multi-Step Ahead)**:
            - Starting from the forecast origin, we use only forecasted values for future lags.
            - Errors can accumulate over the forecast horizon.

        Implementation
        -------------
        - statsmodels' VARResults provides `.forecast()` which requires the last `p` observations
          to generate forecasts. For dynamic, we simply feed the model's own predictions once
          we step beyond the available data. For truly 'in-sample static' forecasts,
          we can iteratively incorporate actual data if it's in-sample.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.
        method : {'static', 'dynamic'}
            Forecasting approach.
        plot : {'none', 'forecast', 'forecast_and_actuals'}
            If 'forecast' or 'forecast_and_actuals', plots the forecast vs. time.

        Returns
        -------
        pd.DataFrame
            DataFrame of forecasted values for each endogenous variable.
        """
        if self.fitted_model is None:
            logging.error("No fitted VAR model. Call `fit()` first.")
            return pd.DataFrame()

        # Retrieve last `lags` observations
        endog_vals = self.data.values
        p = self.fitted_model.k_ar
        last_obs = endog_vals[-p:, :]  # shape: (p, k_endog)

        # Implementation detail:
        # - For 'dynamic' forecasting, we just rely on .forecast() from the model
        #   over the entire horizon, substituting predicted values after the origin.
        # - For 'static' in-sample, we can do iterative one-step ahead:
        #   if the forecast horizon is within the sample, we can progressively
        #   use actual data. If out-of-sample, we revert to typical forecast approach.

        forecasts = []
        if method.lower() == 'dynamic':
            forecast_vals = self.fitted_model.forecast(last_obs, steps=steps)
            idx_forecast = pd.date_range(start=self.data.index[-1], periods=steps+1, freq=self._infer_freq())[1:]
            df_forecast = pd.DataFrame(forecast_vals, columns=self.fitted_model.names, index=idx_forecast)
            forecasts = df_forecast
        else:  # static
            # We'll simulate a "rolling" one-step ahead approach.
            # If we remain in-sample, we can substitute actual data. For out-of-sample, we use forecast.
            # For simplicity, assume all forecast steps are out-of-sample for demonstration.
            # A more advanced approach could check if part of the forecast horizon overlaps the sample.
            obs_seq = last_obs.copy()
            forecast_out = []
            for step in range(steps):
                step_forecast = self.fitted_model.forecast(obs_seq, steps=1)[0]  # 1D array
                forecast_out.append(step_forecast)
                # shift obs_seq up by one and place the newly forecasted row at the bottom
                obs_seq = np.vstack([obs_seq[1:], step_forecast])

            idx_forecast = pd.date_range(start=self.data.index[-1], periods=steps+1, freq=self._infer_freq())[1:]
            df_forecast = pd.DataFrame(forecast_out, columns=self.fitted_model.names, index=idx_forecast)
            forecasts = df_forecast

        # Plot if requested
        if plot in ("forecast", "forecast_and_actuals"):
            plt.figure(figsize=(10, 5))
            for col in self.fitted_model.names:
                if plot == "forecast_and_actuals":
                    plt.plot(self.data.index, self.data[col], label=f"{col} Actual", alpha=0.7)
                plt.plot(forecasts.index, forecasts[col], label=f"{col} Forecast")
            plt.title(f"{self.model_name} - {method.capitalize()} Forecast")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return forecasts


    def _infer_freq(self):
        """Utility to try inferring frequency from the DataFrame index, default to 'D' (daily)."""
        freq = pd.infer_freq(self.data.index)
        if freq is None:
            freq = 'D'
        return freq


if __name__ == "__main__":
    