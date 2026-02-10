#
"""
dcc_garch.py

Lightweight, pure-Python DCC-GARCH implementation suitable for Colab / Python 3.8+.
Dependencies: numpy, scipy, pandas (optional for convenience).

Provides:
- Univariate GARCH(1,1) estimator (MLE)
- DCC(1,1) estimator (Engle 2002)
- DCCGARCH wrapper to fit univariate GARCHs + DCC, and forecast H_{t+1}

This module is intended for prototyping and pipeline integration. It's
vectorized where sensible but prioritizes clarity and numerical stability.

Author: ChatGPT (adapted for user's pipeline)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, slogdet
from scipy.optimize import minimize
from typing import Tuple, Dict, Any, Optional


# -----------------------------
# Univariate GARCH(1,1) helper
# -----------------------------
class GARCH11:
    """Simple GARCH(1,1) MLE estimator.

    Methods
    -------
    fit(x)
        Fit GARCH(1,1) to 1D array x (assumed zero-mean residuals).
    h()
        Return estimated conditional variances (array length T).
    params
        (omega, alpha, beta)
    """

    def __init__(self) -> None:
        self.params: Optional[Tuple[float, float, float]] = None
        self.h_: Optional[np.ndarray] = None

    def _negloglik(self, params: np.ndarray, eps: np.ndarray) -> float:
        omega, alpha, beta = params
        T = len(eps)
        # initialize h with sample variance
        h = np.empty(T, dtype=float)
        var0 = np.var(eps)
        h[0] = var0 if var0 > 0 else 1e-6
        for t in range(1, T):
            h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
            if h[t] <= 0:
                return 1e12
        # Gaussian quasi-likelihood
        ll = 0.5 * (np.log(h) + (eps ** 2) / h).sum()
        return ll

    def fit(self, eps: np.ndarray, x0: Optional[np.ndarray] = None, disp: bool = False) -> Dict[str, Any]:
        eps = np.asarray(eps).astype(float)
        T = len(eps)
        if x0 is None:
            var0 = np.var(eps)
            x0 = np.array([0.1 * var0 if var0>0 else 1e-6, 0.05, 0.9])
        bounds = [(1e-12, None), (1e-8, 0.9999), (1e-8, 0.9999)]

        res = minimize(self._negloglik, x0, args=(eps,), bounds=bounds, method="L-BFGS-B",
                       options={"disp": disp})
        self.params = tuple(res.x)
        # compute h
        omega, alpha, beta = self.params
        h = np.empty(T, dtype=float)
        var0 = np.var(eps)
        h[0] = var0 if var0>0 else 1e-6
        for t in range(1, T):
            h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        self.h_ = h
        return {"params": self.params, "h": h, "opt_res": res}


# -----------------------------
# DCC(1,1) Estimation
# -----------------------------
class DCC:
    """DCC(1,1) estimator.

    Fit is done on standardized residuals z (T x N). Returns a, b and sequences Q_t and R_t.
    """

    def __init__(self) -> None:
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.Q_: Optional[np.ndarray] = None
        self.R_: Optional[np.ndarray] = None
        self.opt_res = None

    @staticmethod
    def _dcc_negloglik(params: np.ndarray, z: np.ndarray) -> float:
        a, b = params
        T, N = z.shape
        Qbar = np.cov(z.T, bias=True)
        Qt = Qbar.copy()
        negll = 0.0
        eps = 1e-8
        # loop through time
        for t in range(T):
            if t > 0:
                outer = np.outer(z[t - 1], z[t - 1])
                Qt = (1 - a - b) * Qbar + a * outer + b * Qt
            D12 = np.diag(1.0 / np.sqrt(np.diag(Qt) + eps))
            Rt = D12 @ Qt @ D12
            sign, logdet = slogdet(Rt)
            if sign <= 0:
                return 1e12
            invRt = inv(Rt)
            negll += 0.5 * (logdet + z[t] @ invRt @ z[t])
        return negll

    def fit(self, z: np.ndarray, a0: float = 0.02, b0: float = 0.97, disp: bool = False) -> Dict[str, Any]:
        z = np.asarray(z).astype(float)
        T, N = z.shape
        bounds = [(1e-8, 0.9999), (1e-8, 0.9999)]
        cons = ({"type": "ineq", "fun": lambda x: 0.9999 - (x[0] + x[1])})
        x0 = np.array([a0, b0])
        res = minimize(self._dcc_negloglik, x0, args=(z,), bounds=bounds, constraints=cons,
                       method="SLSQP", options={"disp": disp, "maxiter": 200})
        self.a, self.b = float(res.x[0]), float(res.x[1])
        self.opt_res = res
        # compute Q_t and R_t series
        Qbar = np.cov(z.T, bias=True)
        Qt = Qbar.copy()
        Q_list = np.empty((T, N, N), dtype=float)
        R_list = np.empty((T, N, N), dtype=float)
        eps_small = 1e-8 # Using a different name to avoid conflict with `eps` parameter
        for t in range(T):
            if t > 0:
                Qt = (1 - self.a - self.b) * Qbar + self.a * np.outer(z[t - 1], z[t - 1]) + self.b * Qt
            D12 = np.diag(1.0 / np.sqrt(np.diag(Qt) + eps_small))
            Rt = D12 @ Qt @ D12
            Q_list[t] = Qt
            R_list[t] = Rt
        self.Q_ = Q_list
        self.R_ = R_list
        return {"a": self.a, "b": self.b, "Q": Q_list, "R": R_list, "opt_res": res}


# -----------------------------
# High-level wrapper: DCC-GARCH
# -----------------------------
class DCCGARCH:
    """Wrapper that fits independent GARCH(1,1) on columns and then fits DCC on standardized residuals.

    Usage:
        model = DCCGARCH()
        model.fit(eps_matrix)  # eps_matrix: T x N, residuals from VAR/VARX or returns (demeaned)
        H_t = model.get_cov()  # T x N x N
        R_t = model.get_corr()
        H_next = model.forecast(horizon=1)
    """

    def __init__(self):
        self.N: Optional[int] = None
        self.T: Optional[int] = None
        self.garch_models: Optional[list] = None
        self.garch_params: Optional[list] = None
        self.h_: Optional[np.ndarray] = None  # T x N
        self.z_: Optional[np.ndarray] = None  # T x N
        self.dcc: Optional[DCC] = None
        self.H_: Optional[np.ndarray] = None  # T x N x N
        self.column_names: Optional[list] = None # NEW: Store column names

    def fit(self, eps: np.ndarray, column_names: Optional[list] = None, disp: bool = False) -> Dict[str, Any]: # NEW: Add column_names
        eps = np.asarray(eps).astype(float)
        if eps.ndim != 2:
            raise ValueError("eps must be 2D array with shape (T, N)")
        T, N = eps.shape
        self.T, self.N = T, N
        self.column_names = column_names if column_names is not None else [f"Series_{i}" for i in range(N)]

        # fit individual GARCH
        self.garch_models = []
        self.garch_params = []
        H = np.zeros_like(eps)
        Z = np.zeros_like(eps)
        for i in range(N):
            g = GARCH11()
            res = g.fit(eps[:, i], disp=disp)
            self.garch_models.append(g)
            self.garch_params.append(res["params"])
            H[:, i] = res["h"]
            Z[:, i] = eps[:, i] / np.sqrt(res["h"])  # standardized residuals
        self.h_ = H
        self.z_ = Z
        # fit DCC
        dcc = DCC()
        dcc_res = dcc.fit(Z, disp=disp)
        self.dcc = dcc
        # build H_t = D_t R_t D_t
        H_t = np.empty((T, N, N), dtype=float)
        for t in range(T):
            Dt = np.diag(np.sqrt(H[t, :]))
            Rt = dcc_res["R"][t]
            H_t[t] = Dt @ Rt @ Dt
        self.H_ = H_t
        return {"garch_params": self.garch_params, "dcc_params": (dcc_res["a"], dcc_res["b"]), "H": H_t}

    def get_corr(self) -> np.ndarray:
        if self.dcc is None or self.dcc.R_ is None:
            raise RuntimeError("Model not fitted yet")
        return self.dcc.R_

    def get_cov(self) -> np.ndarray:
        if self.H_ is None:
            raise RuntimeError("Model not fitted yet")
        return self.H_

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast conditional covariance for horizon steps ahead (only up to t+1 implemented robustly).

        For horizon=1, we use last estimated GARCH parameters to forecast h_{T+1} per series and
        the DCC recursion to forecast Q_{T+1} then form H_{T+1} = D_{T+1} R_{T+1} D_{T+1}.
        """
        if self.H_ is None or self.h_ is None or self.dcc is None:
            raise RuntimeError("Model not fitted yet")
        T = self.T
        N = self.N
        # forecast h_{T+1} using last residual and last h
        h_next = np.empty(N, dtype=float)
        for i in range(N):
            omega, alpha, beta = self.garch_params[i]
            eps_last = (np.sqrt(self.h_[-1, i]) * self.z_[-1, i])
            h_last = self.h_[-1, i]
            h_next[i] = omega + alpha * (eps_last ** 2) + beta * h_last
        # forecast Q_{T+1}
        a, b = self.dcc.a, self.dcc.b
        Qbar = np.cov(self.z_.T, bias=True)
        Qt_last = self.dcc.Q_[-1]
        outer = np.outer(self.z_[-1], self.z_[-1])
        Q_next = (1 - a - b) * Qbar + a * outer + b * Qt_last
        eps_small = 1e-8 # Using a different name to avoid conflict with `eps` parameter
        D12 = np.diag(1.0 / np.sqrt(np.diag(Q_next) + eps_small))
        R_next = D12 @ Q_next @ D12
        D_next = np.diag(np.sqrt(h_next))
        H_next = D_next @ R_next @ D_next
        return H_next


# -----------------------------
# If run as script: quick demo
# -----------------------------
if __name__ == "__main__":
    import time
    print("Running built-in demo: simulate 3 series, fit DCC-GARCH")
    np.random.seed(42)
    T = 500
    N = 3
    # simulate simple heteroskedastic eps
    true_params = [(1e-6, 0.05, 0.9), (2e-6, 0.04, 0.92), (1.5e-6, 0.06, 0.88)]
    eps = np.zeros((T, N))
    for i in range(N):
        omega, alpha, beta = true_params[i]
        h = np.zeros(T)
        h[0] = 1e-6 / (1 - alpha - beta) if (1 - alpha - beta) > 0 else 1e-6
        for t in range(1, T):
            h[t] = omega + alpha * eps[t - 1, i] ** 2 + beta * h[t - 1]
            eps[t, i] = np.random.randn() * np.sqrt(h[t])
    start = time.time()
    model = DCCGARCH()
    out = model.fit(eps, disp=False)
    elapsed = time.time() - start
    print(f"Done fit in {elapsed:.2f}s")
    print("GARCH params:", out["garch_params"])
    print("DCC params:", out["dcc_params"])
    print("Last corr matrix:\n", model.get_corr()[-1])
