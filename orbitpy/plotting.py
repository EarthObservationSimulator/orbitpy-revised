"""Plotting functions for visualizing coverage data."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from orbitpy.coverage import DiscreteCoverageGP
from eosimutils.state import Cartesian3DPositionArray

def plot_covered_steps(coverage: DiscreteCoverageGP, positions: Cartesian3DPositionArray, max_steps: int = None):
    """
    Plot grid points in 3D, colored by the number of time steps covered.

    Args:
        coverage (DiscreteCoverageGP): Accessed time indices for each grid point.
        positions (Cartesian3DPositionArray): Locations of grid points.
        max_steps (int): Maximum number of covered steps for normalization in the colormap.
        
    Returns:
        Axes3D: The 3D axes object with the plotted grid points.
    """

    # Extract positions and covered steps
    coords = positions.to_numpy()
    steps_covered = coverage.coverage_steps()

    if max_steps is None:
        max_steps = np.max(steps_covered)

    # Normalize covered steps for colormap
    norm = plt.Normalize(vmin=0, vmax=max_steps)

    # Create colormap
    cmap = cm.get_cmap('viridis')

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=steps_covered,
        cmap=cmap,
        norm=norm,
        s=100,
        alpha=1
    )

    # Add color bar to represent the number of covered steps
    cbar = plt.colorbar(sc, pad=0.1, shrink=0.75, aspect=15)
    cbar.set_label('Number of Covered Steps', fontsize=12)

    # Label axes
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title('Covered Time Steps per Grid Point', fontsize=14)

    plt.tight_layout()

    return ax