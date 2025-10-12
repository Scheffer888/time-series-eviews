# Copyright (c) 2025 Eduardo Belisario Scheffer
# Licensed under the Apache License, Version 2.0
# See the LICENSE file in the project root for details.

"""
unit_root.py
===========
A module for conducting unit root and stationarity tests, including:
- Simulation of the Dickey-Fuller sampling distribution using discretized Brownian motion.

"""
import numpy as np
import matplotlib.pyplot as plt
import os

def sim_dickey_fulley(
        num_sims=10000,
        n=1000,
        seed=42,
        title: str = "Simulated Dickey-Fuller Distribution",
        percentile: float = 0.05,
        output_dir: str = None,
        filename: str = "dickey_fuller_sim.png",
        show_plot: bool = True):
    """
    Simulate the Dickey-Fuller sampling distribution using a discretized Brownian motion.

    The exercise states:
        T(β_hat - 1) approx ( [chi^2(1) - 1 ] / integral_0^1 W(r)^2 dr ) * 1/2,
        or a simplified correlation between numerator = \chi^2(1)-1
        and denominator = \int_0^1 W(r)^2 dr.

    We'll approximate by:
      1) Generating Brownian motion via partial sums of eps ~ N(0, 1/n).
      2) The integral of W^2 is sum(W_t^2) * (1/n).
      3) The \chi^2(1) realization is sum(eps_t)^2, then minus 1.

    Parameters
    ----------
    num_sims : int
        Number of simulations (10k in the problem).
    n : int
        The discretization steps for Brownian motion (1000 in the problem).
    seed : int
        Random seed for reproducibility.
    title : str
        Title for the histogram plot.
    percentile : float
        The percentile to calculate (default is 0.05 for the 5th percentile).
    output_dir : str or None
        If provided, a directory path where we can save a histogram figure.
    filename : str
        The filename to save the histogram plot.
    show_plot : bool
        Whether to display the histogram plot.

    Returns
    -------
    df : dict
        A dictionary with keys:
            'T_values' : 1D array of simulated T-values
            'percentile_5' : the 5th percentile
    """
    np.random.seed(seed)

    T_values = np.zeros(num_sims)

    for i in range(num_sims):
        # 1) Generate eps ~ N(0, 1/n)
        eps = np.random.normal(loc=0, scale=np.sqrt(1/n), size=n)

        # 2) Construct Brownian motion W(r). 
        #    W[k] = sum_{t=1..k} eps[t], for k=1..n.
        #    So W is length n, from step 1..n
        W = np.cumsum(eps)

        # 3) Denominator ~ integral_0^1 [W(r)]^2 dr ~ sum( W[k]^2 ) * (1/n)
        denom = np.sum(W**2) * (1/n)

        # 4) Numerator: \chi^2(1) - 1 => sum(eps)^2 - 1
        #    According to the problem, we use the same eps for that.
        #    sum(eps)^2 is basically (W[n])^2 if W started at 0.
        #    Or you might do sum(eps_t^2). The exercise says we want the square 
        #    of sum(eps), i.e. ( eps_1 + ... + eps_n )^2, minus 1
        numer = (np.sum(eps)**2) - 1

        # 5) Take the ratio, and multiply by 1/2
        #    T(β_hat - 1) ~ [1/2] * [numer / denom]
        ratio = 0.5 * (numer / denom)

        T_values[i] = ratio

    # 6) percentile
    perc_T = np.percentile(T_values, percentile*100)

    # 7) Plot a histogram if output_dir is set
    
    plt.figure(figsize=(7,4))
    plt.hist(T_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(perc_T, color='red', linestyle='--', label=f"{percentile} pct = {perc_T:.2f}")
    plt.title(title)
    plt.xlabel("T values")
    plt.ylabel("Frequency")
    plt.legend()

    # Save
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(output_dir, filename)
        plt.savefig(outfile, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
        
    return {
        "T_values": T_values,
        "percentile_5": perc_T
    }


def adf_test(series, max_lag=None):
    """
    Wrapper for the augmented Dickey-Fuller test using statsmodels.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    max_lag : int, optional
        Maximum number of lags. If None, uses AIC-based selection.

    Returns
    -------
    dict
        ADF statistic, p-value, number of lags, and critical values.
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series, autolag='AIC' if max_lag is None else max_lag)

    return {
        'ADF Stat': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Critical Values': result[4]
    }