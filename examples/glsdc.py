"""Generic gauss least-squares differential correction (GLSDC).

It solves

    minimize (y - F(x))^T R^{-1} (y - F(x))

using differential corrections and a finite-difference Jacobian by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


VectorFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class GLSDCResults:
    x_hat: np.ndarray
    y_hat: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    normal_matrix: np.ndarray
    covariance: np.ndarray
    dx: np.ndarray
    cost: float
    iterations: int
    converged: bool
    x_history: list[np.ndarray] = field(default_factory=list)
    cost_history: list[float] = field(default_factory=list)


class FiniteDifferenceJacobian:
    """Central finite-difference Jacobian for vector-valued functions."""

    def __init__(self, function: VectorFunction, eps: float = 1.0e-6):
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        self.function = function
        self.eps = float(eps)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _as_column(x, "x")
        y0 = _as_column(self.function(x), "F(x)")

        ny = y0.shape[0]
        nx = x.shape[0]
        jac = np.empty((ny, nx), dtype=float)

        for j in range(nx):
            h = self.eps * max(1.0, abs(float(x[j, 0])))
            xp = x.copy()
            xm = x.copy()
            xp[j, 0] += h
            xm[j, 0] -= h

            yp = _as_column(self.function(xp), "F(x + h)")
            ym = _as_column(self.function(xm), "F(x - h)")

            if yp.shape != y0.shape or ym.shape != y0.shape:
                raise ValueError("measurement model output dimension changed during finite difference")

            jac[:, j] = ((yp - ym) / (2.0 * h)).ravel()

        return jac


class GLSDC:
    def __init__(self, max_iters: int = 50, tol: float = 1.0e-10, fd_eps: float = 1.0e-6):
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be positive and finite")
        if not np.isfinite(fd_eps) or fd_eps <= 0.0:
            raise ValueError("fd_eps must be positive and finite")

        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.fd_eps = float(fd_eps)

    def solve(
        self,
        F: VectorFunction,
        x0: np.ndarray,
        y: np.ndarray,
        R: Optional[np.ndarray] = None,
        jacobian: Optional[VectorFunction] = None,
        verbose: bool = True,
    ) -> GLSDCResults:
        x = _as_column(x0, "x0")
        y = _as_column(y, "y")

        nx = x.shape[0]
        ny = y.shape[0]

        Rinv = _inverse_covariance(R, ny)
        df = jacobian if jacobian is not None else FiniteDifferenceJacobian(F, eps=self.fd_eps)

        x_history = [x.copy()]
        cost_history: list[float] = []
        dx = np.zeros_like(x)
        converged = False

        for k in range(self.max_iters):
            y_model = _as_column(F(x), "F(x)")
            H = np.asarray(df(x), dtype=float)
            _validate_dimensions(y_model, H, nx, ny)

            residual = y - y_model
            cost = float((residual.T @ Rinv @ residual)[0, 0])
            normal_matrix = H.T @ Rinv @ H
            rhs = H.T @ Rinv @ residual

            dx = _solve_normal_equations(normal_matrix, rhs)
            x = x + dx

            x_history.append(x.copy())
            cost_history.append(cost)

            dx_norm = float(np.linalg.norm(dx))
            if verbose:
                print("=" * 72)
                print(f"GLSDC iteration {k + 1}")
                print(f"x         = {x.ravel()}")
                print(f"dx        = {dx.ravel()}")
                print(f"|dx|      = {dx_norm:.16e}")
                print(f"cost      = {cost:.16e}")
                print(f"rms resid = {np.sqrt(np.mean(residual ** 2)):.16e}")

            if dx_norm < self.tol:
                converged = True
                break

        y_hat = _as_column(F(x), "F(x_hat)")
        H = np.asarray(df(x), dtype=float)
        _validate_dimensions(y_hat, H, nx, ny)

        residual = y - y_hat
        normal_matrix = H.T @ Rinv @ H
        covariance = np.linalg.pinv(normal_matrix)
        final_cost = float((residual.T @ Rinv @ residual)[0, 0])

        return GLSDCResults(
            x_hat=x,
            y_hat=y_hat,
            residual=residual,
            jacobian=H,
            normal_matrix=normal_matrix,
            covariance=covariance,
            dx=dx,
            cost=final_cost,
            iterations=len(cost_history),
            converged=converged,
            x_history=x_history,
            cost_history=cost_history,
        )


def _as_column(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        pass
    else:
        raise ValueError(f"{name} must be convertible to a column vector")
    return arr


def _inverse_covariance(R: Optional[np.ndarray], ny: int) -> np.ndarray:
    if R is None:
        return np.eye(ny)

    R = np.asarray(R, dtype=float)
    if R.shape != (ny, ny):
        raise ValueError(f"R must have shape ({ny}, {ny})")

    try:
        chol = np.linalg.cholesky(R)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(ny)))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(R)


def _validate_dimensions(y_model: np.ndarray, H: np.ndarray, nx: int, ny: int) -> None:
    if y_model.shape != (ny, 1):
        raise ValueError(f"F(x) must have shape ({ny}, 1), got {y_model.shape}")
    if H.shape != (ny, nx):
        raise ValueError(f"Jacobian must have shape ({ny}, {nx}), got {H.shape}")


def _solve_normal_equations(normal_matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(normal_matrix) @ rhs
