"""
.. module:: orbitpy.glsdc
    :synopsis: Generalized least-squares differential correction.

The GLSDC module provides classes and functions for solving nonlinear least-squares
problems using differential correction.

It solves problems of the form

    minimize (y - F(x))^T R^{-1} (y - F(x))

where ``y`` is the measurement vector, ``F`` is the measurement model, ``x`` is
the solve-for parameter vector, and ``R`` is the measurement covariance matrix.

Classes:

- GLSDC: Solves nonlinear least-squares problems using iterative differential
         correction.

- FiniteDifferenceJacobian: Computes a central finite-difference Jacobian for a
                            vector-valued measurement model.

**Example:**

    solver = GLSDC(max_iters=20, tol=1.0e-10, fd_eps=1.0e-6)
    results = solver.solve(F=measurement_model, x0=x0, y=measurements)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

VectorFunction = Callable[[np.ndarray], np.ndarray]

@dataclass
class GLSDCResults:
    """Container for GLSDC solution results.

    Args:
        x_hat (np.ndarray): Estimated parameter vector in physical units.
        y_hat (np.ndarray): Estimated measurement vector, ``F(x_hat)``.
        residual (np.ndarray): Final measurement residual, ``y - F(x_hat)``.
        jacobian (np.ndarray): Final Jacobian with respect to physical parameters.
        normal_matrix (np.ndarray): Final normal matrix in physical parameters.
        covariance (np.ndarray): Pseudo-inverse of the final normal matrix.
        dx (np.ndarray): Final parameter correction in physical units.
        cost (float): Final weighted least-squares cost.
        iterations (int): Number of completed iterations.
        converged (bool): Flag indicating whether the convergence tolerance was met.
        x_history (list[np.ndarray]): Parameter estimate history in physical units.
        cost_history (list[float]): Cost history.
    """

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
    """Computes a central finite-difference Jacobian.

    Args:
        function (VectorFunction): Vector-valued function to differentiate.
        eps (float): Relative finite-difference step size.

    Raises:
        ValueError: If ``eps`` is not positive and finite.
    """

    def __init__(self, function: VectorFunction, eps: float = 1.0e-6) -> None:
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite.")

        self.function = function
        self.eps = float(eps)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the finite-difference Jacobian.

        Args:
            x (np.ndarray): Parameter vector.

        Returns:
            np.ndarray: Jacobian matrix with shape ``(ny, nx)``.

        """
        x = _as_column(x, "x")
        y0 = _as_column(self.function(x), "F(x)")

        ny = y0.shape[0]
        nx = x.shape[0]
        jacobian = np.empty((ny, nx), dtype=float)

        for j in range(nx):
            step = self.eps * max(1.0, abs(float(x[j, 0])))

            xp = x.copy()
            xm = x.copy()

            xp[j, 0] += step
            xm[j, 0] -= step

            yp = _as_column(self.function(xp), "F(x + h)")
            ym = _as_column(self.function(xm), "F(x - h)")

            jacobian[:, j] = ((yp - ym) / (2.0 * step)).ravel()

        return jacobian


class GLSDC:
    """Generalized least-squares differential-correction solver.

    Args:
        max_iters (int): Maximum number of differential-correction iterations.
        tol (float): Convergence tolerance on the parameter correction norm.
        fd_eps (float): Relative finite-difference step size.
        x_scale (Optional[np.ndarray]): Optional parameter scale vector. If provided,
            it must have the same dimension as ``x0`` passed to :meth:`solve`.
            Scaling is used internally only; the measurement model receives physical
            parameters.

    Raises:
        ValueError: If solver options are invalid.
    """

    def __init__(
        self,
        max_iters: int = 50,
        tol: float = 1.0e-10,
        fd_eps: float = 1.0e-6,
        x_scale: Optional[np.ndarray] = None,
    ) -> None:
        if max_iters <= 0:
            raise ValueError("max_iters must be positive.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be positive and finite.")
        if not np.isfinite(fd_eps) or fd_eps <= 0.0:
            raise ValueError("fd_eps must be positive and finite.")

        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.fd_eps = float(fd_eps)
        self.x_scale = (
            None if x_scale is None else _as_column(x_scale, "x_scale")
        )

        if self.x_scale is not None:
            if (
                np.any(~np.isfinite(self.x_scale))
                or np.any(self.x_scale <= 0.0)
            ):
                raise ValueError("x_scale entries must be positive and finite.")

    def solve(
        self,
        F: VectorFunction,
        x0: np.ndarray,
        y: np.ndarray,
        R: Optional[np.ndarray] = None,
        jacobian: Optional[VectorFunction] = None,
        verbose: bool = True,
    ) -> GLSDCResults:
        """Solve a nonlinear least-squares problem.

        Args:
            F (VectorFunction): Measurement model. The function must accept a
                physical parameter vector ``x`` and return a measurement vector.
            x0 (np.ndarray): Initial parameter estimate.
            y (np.ndarray): Measurement vector.
            R (Optional[np.ndarray]): Measurement covariance matrix. If ``None``,
                identity covariance is used.
            jacobian (Optional[VectorFunction]): Optional analytic or user-supplied
                Jacobian with respect to physical parameters. If ``None``, a central
                finite-difference Jacobian is used.
            verbose (bool): If ``True``, print iteration information.

        Returns:
            GLSDCResults: Solver results.

        Raises:
            ValueError: If dimensions are inconsistent.
        """
        x0 = _as_column(x0, "x0")
        y = _as_column(y, "y")

        nx = x0.shape[0]
        ny = y.shape[0]

        x_scale = self._get_x_scale(nx)
        z = x0 / x_scale

        def F_scaled(z_in: np.ndarray) -> np.ndarray:
            x_physical = _as_column(z_in, "z") * x_scale
            return F(x_physical)

        if jacobian is None:
            df_scaled = FiniteDifferenceJacobian(
                F_scaled, eps=self.fd_eps
            )
        else:

            def df_scaled(z_in: np.ndarray) -> np.ndarray:
                x_physical = _as_column(z_in, "z") * x_scale
                H_x = np.asarray(jacobian(x_physical), dtype=float)
                return H_x @ np.diagflat(x_scale)

        Rinv = _inverse_covariance(R, ny)

        x_history = [(z * x_scale).copy()]
        cost_history: list[float] = []

        dz = np.zeros_like(z)
        converged = False

        for k in range(self.max_iters):
            y_model = _as_column(F_scaled(z), "F(x)")
            H_z = np.asarray(df_scaled(z), dtype=float)
            _validate_dimensions(y_model, H_z, nx, ny)

            residual = y - y_model
            cost = float((residual.T @ Rinv @ residual)[0, 0])

            normal_matrix_z = H_z.T @ Rinv @ H_z
            rhs_z = H_z.T @ Rinv @ residual

            dz = _solve_normal_equations(normal_matrix_z, rhs_z)
            z = z + dz

            x = z * x_scale
            dx = dz * x_scale

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

        x_hat = z * x_scale
        y_hat = _as_column(F(x_hat), "F(x_hat)")

        H_z = np.asarray(df_scaled(z), dtype=float)
        H_x = H_z @ np.diagflat(1.0 / x_scale)
        _validate_dimensions(y_hat, H_x, nx, ny)

        residual = y - y_hat
        normal_matrix_x = H_x.T @ Rinv @ H_x
        covariance_x = np.linalg.pinv(normal_matrix_x)

        final_cost = float((residual.T @ Rinv @ residual)[0, 0])
        dx_final = dz * x_scale

        return GLSDCResults(
            x_hat=x_hat,
            y_hat=y_hat,
            residual=residual,
            jacobian=H_x,
            normal_matrix=normal_matrix_x,
            covariance=covariance_x,
            dx=dx_final,
            cost=final_cost,
            iterations=len(cost_history),
            converged=converged,
            x_history=x_history,
            cost_history=cost_history,
        )

    def _get_x_scale(self, nx: int) -> np.ndarray:
        """Get the internal parameter scale vector.

        Args:
            nx (int): Number of solve-for parameters.

        Returns:
            np.ndarray: Parameter scale vector with shape ``(nx, 1)``.

        Raises:
            ValueError: If ``x_scale`` has the wrong shape.
        """
        if self.x_scale is None:
            return np.ones((nx, 1), dtype=float)

        if self.x_scale.shape != (nx, 1):
            raise ValueError(
                f"x_scale must have shape ({nx}, 1), "
                f"got {self.x_scale.shape}."
            )

        return self.x_scale.copy()


def _as_column(value: np.ndarray, name: str) -> np.ndarray:
    """Convert an input value to a column vector.

    Args:
        value (np.ndarray): Input value.
        name (str): Name used in error messages.

    Returns:
        np.ndarray: Column vector.

    Raises:
        ValueError: If the input cannot be converted to a column vector.
    """
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        pass
    else:
        raise ValueError(f"{name} must be convertible to a column vector.")

    return arr


def _inverse_covariance(R: Optional[np.ndarray], ny: int) -> np.ndarray:
    """Compute the inverse measurement covariance matrix.

    Args:
        R (Optional[np.ndarray]): Measurement covariance matrix. If ``None``,
            identity covariance is used.
        ny (int): Number of measurements.

    Returns:
        np.ndarray: Inverse covariance matrix.

    Raises:
        ValueError: If ``R`` has an invalid shape.
    """
    if R is None:
        return np.eye(ny)

    R = np.asarray(R, dtype=float)

    if R.shape != (ny, ny):
        raise ValueError(f"R must have shape ({ny}, {ny}).")

    try:
        chol = np.linalg.cholesky(R)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(ny)))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(R)


def _validate_dimensions(
    y_model: np.ndarray,
    H: np.ndarray,
    nx: int,
    ny: int,
) -> None:
    """Validate measurement-model and Jacobian dimensions.

    Args:
        y_model (np.ndarray): Measurement-model output.
        H (np.ndarray): Jacobian matrix.
        nx (int): Number of solve-for parameters.
        ny (int): Number of measurements.

    Raises:
        ValueError: If dimensions are inconsistent.
    """
    if y_model.shape != (ny, 1):
        raise ValueError(f"F(x) must have shape ({ny}, 1), got {y_model.shape}.")

    if H.shape != (ny, nx):
        raise ValueError(f"Jacobian must have shape ({ny}, {nx}), got {H.shape}.")


def _solve_normal_equations(
    normal_matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve the normal equations.

    Args:
        normal_matrix (np.ndarray): Normal matrix.
        rhs (np.ndarray): Right-hand side vector.

    Returns:
        np.ndarray: Parameter correction vector.
    """
    try:
        return np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(normal_matrix) @ rhs