"""
.. module:: orbitpy.coverage
   :synopsis: Coverage classes for storing output of coverage simulations
"""

from typing import List, Dict, Any
from math import ceil, floor
import numpy as np

from eosimutils.time import AbsoluteDateArray, AbsoluteDate, AbsoluteDateIntervalArray


class DiscreteCoverageTP:
    """
    Stores the results of a discrete-time coverage simulation.

    Stored in "TP-first" format, a list of covered GP indices is given for each time point.

    Attributes:
        time (AbsoluteDateArray): An array of time points.
        coverage (List[List[int]]): For each time point, the list of accessed indices.
        n_grid_points (int): Total number of grid points.
    """

    def __init__(
        self,
        time: AbsoluteDateArray,
        coverage: List[List[int]],
        n_grid_points: int,
    ) -> None:
        """
        Initializes a DiscreteCoverageTP instance.

        Args:
            time (AbsoluteDateArray): Array of time points.
            coverage (List[List[int]]): For each time point, a list of accessed indices.
            n_grid_points (int): Total number of grid points.
        """
        self.time = time
        self.coverage = coverage
        self.n_grid_points = n_grid_points

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing
                'time' : dict of an AbsoluteDateArray
                'coverage': List[List[int]] of accessed indices per time point,
                'n_grid_points': Total number of grid points.
        """
        return {
            "time": self.time.to_dict(),
            "coverage": self.coverage,
            "n_grid_points": self.n_grid_points,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscreteCoverageTP":
        """
        Deserializes a DiscreteCoverageTP from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary with keys:
                'time': dict of an AbsoluteDateArray,
                'coverage': List[List[int]] of accessed indices per time point,
                'n_grid_points': Total number of grid points.

        Returns:
            DiscreteCoverageTP: The deserialized object.
        """
        time = AbsoluteDateArray.from_dict(data["time"])
        coverage = data["coverage"]
        n_grid_points = data["n_grid_points"]
        return cls(time=time, coverage=coverage, n_grid_points=n_grid_points)

    @classmethod
    def from_gp(cls, gp: "DiscreteCoverageGP") -> "DiscreteCoverageTP":
        """
        Construct a DiscreteCoverageTP from a DiscreteCoverageGP instance.

        Args:
            gp (DiscreteCoverageGP): Space-first coverage instance.

        Returns:
            DiscreteCoverageTP: Time-first coverage instance.
        """
        # prepare empty list per time index
        n_times = len(gp.time)
        tp_cov: List[List[int]] = [[] for _ in range(n_times)]
        for grid_idx, times in enumerate(gp.coverage):
            for t_idx in times:
                tp_cov[t_idx].append(grid_idx)
        return cls(
            time=gp.time, coverage=tp_cov, n_grid_points=len(gp.coverage)
        )

    def to_gp(self) -> "DiscreteCoverageGP":
        """
        Convert this DiscreteCoverageTP to a DiscreteCoverageGP instance.

        Returns:
            DiscreteCoverageGP: Space-first coverage instance.
        """
        return DiscreteCoverageGP.from_tp(self)


class DiscreteCoverageGP:
    """
    Stores the results of a discrete-time coverage simulation.

    Stored in "GP-first" format: a list of covered time indices is given for each grid point.

    Attributes:
        time (AbsoluteDateArray): Array of time points.
        coverage (List[List[int]]): For each grid point, list of time indices
            at which that grid point is accessed.
    """

    def __init__(
        self, time: AbsoluteDateArray, coverage: List[List[int]]
    ) -> None:
        """
        Initializes a DiscreteCoverageGP instance.

        Args:
            time (AbsoluteDateArray): Array of time points.
            coverage (List[List[int]]): For each grid point, indices of accessed time points.
        """
        self.time = time
        self.coverage = coverage

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary with keys:
                'time': dict from AbsoluteDateArray.to_dict(),
                'coverage': List[List[int]] of time indices per grid point.
        """
        return {"time": self.time.to_dict(), "coverage": self.coverage}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscreteCoverageGP":
        """
        Deserializes a DiscreteCoverageGP from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary with keys:
                'time': dict for AbsoluteDateArray,
                'coverage': List[List[int]] of time indices per grid point.

        Returns:
            DiscreteCoverageGP: The reconstructed instance.
        """
        time = AbsoluteDateArray.from_dict(data["time"])
        coverage = data["coverage"]
        return cls(time=time, coverage=coverage)

    @classmethod
    def from_tp(cls, tp: DiscreteCoverageTP) -> "DiscreteCoverageGP":
        """
        Construct a DiscreteCoverageGP from a DiscreteCoverageTP instance.

        Args:
            tp (DiscreteCoverageTP): Time-first coverage instance.

        Returns:
            DiscreteCoverageGP: Space-first coverage instance.
        """
        # use total grid points from TP
        n_grids = tp.n_grid_points
        gp_cov: List[List[int]] = [[] for _ in range(n_grids)]
        for t_idx, grids in enumerate(tp.coverage):
            for grid_idx in grids:
                gp_cov[grid_idx].append(t_idx)
        return cls(time=tp.time, coverage=gp_cov)

    def to_tp(self) -> DiscreteCoverageTP:
        """
        Convert this DiscreteCoverageGP to a DiscreteCoverageTP instance.

        Returns:
            DiscreteCoverageTP: Time-first coverage instance.
        """
        return DiscreteCoverageTP.from_gp(self)

    def coverage_steps(self) -> np.ndarray:
        """
        Returns a NumPy array where each entry is the total number of steps
            covered for the corresponding grid point.

        Returns:
            np.ndarray: An integer numpy array of length equal to the number of grid points,
                where each entry is the total number of time steps for which the corresponding grid
                point is covered.
        """
        return np.array([len(times) for times in self.coverage])

    @staticmethod
    def symmetric_difference(
        a: "DiscreteCoverageGP", b: "DiscreteCoverageGP"
    ) -> "DiscreteCoverageGP":
        """
        Compute the symmetric difference between two DiscreteCoverageGP instances.

        The symmetric difference of two lists of time steps will return those time steps which are
        present in one list but not the other.

        Args:
            a (DiscreteCoverageGP): First coverage instance.
            b (DiscreteCoverageGP): Second coverage instance.

        Returns:
            DiscreteCoverageGP: A new instance with the same time array and, for each grid
                point, the symmetric difference of accessed time indices.

        Raises:
            ValueError: If the time arrays or number of grid points differ.
        """
        if len(a.time) != len(b.time):
            raise ValueError(
                "Time arrays must be the same size for symmetric difference"
            )
        if len(a.coverage) != len(b.coverage):
            raise ValueError(
                "Both instances must have the same number of grid points"
            )

        new_cov: List[List[int]] = []
        for cov_a, cov_b in zip(a.coverage, b.coverage):
            diff = sorted(set(cov_a).symmetric_difference(cov_b))
            new_cov.append(diff)

        return DiscreteCoverageGP(time=a.time, coverage=new_cov)


class ContinuousCoverageGP:
    """
    Stores the results of a continuous-time coverage simulation.

    Stored in "GP-first" format, coverage intervals are stored per grid point.

    Attributes:
        coverage (List[AbsoluteDateIntervalArray]):
            For each grid point, an AbsoluteDateIntervalArray of (start, end) coverage intervals.
    """

    def __init__(self, coverage: List[AbsoluteDateIntervalArray]) -> None:
        """
        Initializes a ContinuousCoverageGP instance.

        Args:
            coverage (List[AbsoluteDateIntervalArray]):
                For each grid point, an AbsoluteDateIntervalArray of coverage intervals.
        """
        self.coverage = coverage

    def to_discrete(
        self, start: AbsoluteDate, step: float, num_points: int
    ) -> DiscreteCoverageGP:
        """
        Convert continuous coverage representation to a discrete coverage representation.

        Uses a fixed step size for the discrete representation.

        Args:
            start (AbsoluteDate): The time for the first point in the time grid.
            step (float): Step size in seconds between consecutive points in the time grid.
            num_points (int): Total number of time points.

        Returns:
            DiscreteCoverageGP: Space-first coverage with each grid point
                mapped to the list of time indices covered.
        """
        # build time array
        ephemer = np.array([start.ephemeris_time + i * step for i in range(num_points)])
        time_array = AbsoluteDateArray(ephemer)

        # convert intervals to indices
        discrete_cov: List[List[int]] = []
        t0 = start.ephemeris_time
        for gp_intervals in self.coverage:
            idxs: List[int] = []
            starts_et, stops_et = gp_intervals.to_spice_ephemeris_time()
            for s_et, e_et in zip(starts_et, stops_et):
                i0 = int(ceil((s_et - t0) / step))
                i1 = int(floor((e_et - t0) / step))
                lo = max(0, i0)
                hi = min(num_points - 1, i1)
                if lo <= hi:
                    idxs.extend(range(lo, hi + 1))
            discrete_cov.append(idxs)

        return DiscreteCoverageGP(time=time_array, coverage=discrete_cov)

    @classmethod
    def from_stk(cls, file_path: str) -> "ContinuousCoverageGP":
        """
        Construct a ContinuousCoverageGP from an STK coverage report (.cvaa).

        Args:
            file_path (str): Path to the STK .cvaa file.

        Returns:
            ContinuousCoverageGP: Stores coverage intervals for each grid point.
        """
        coverage_map: Dict[int, List[tuple[AbsoluteDate, AbsoluteDate]]] = {}
        epoch = None
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "EpochTime":
                    epoch_str = line.split(None, 1)[1].strip()
                    epoch = AbsoluteDate.from_dict(
                        {
                            "time_format": "GREGORIAN_DATE",
                            "calendar_date": epoch_str,
                            "time_scale": "UTC",
                        }
                    )
                elif parts[0] == "PointNumber":
                    point_idx = int(parts[1])
                elif parts[0] == "NumberOfAccesses":
                    num_acc = int(parts[1])
                    intervals: List[tuple[AbsoluteDate, AbsoluteDate]] = []
                    for _ in range(num_acc):
                        rec = next(lines).split()
                        start_off = float(rec[1])
                        end_off = float(rec[2])
                        intervals.append((epoch + start_off, epoch + end_off))
                    coverage_map[point_idx] = intervals

        # build ordered list of AbsoluteDateIntervalArray per grid point
        max_idx = max(coverage_map.keys(), default=-1)
        coverage: List[AbsoluteDateIntervalArray] = []
        for i in range(max_idx + 1):
            ivs = coverage_map.get(i, [])
            if ivs:
                starts = np.array([s.ephemeris_time for (s, _) in ivs], dtype=float)
                stops = np.array([e.ephemeris_time for (_, e) in ivs], dtype=float)
            else:
                starts = np.array([], dtype=float)
                stops = np.array([], dtype=float)
            coverage.append(
                AbsoluteDateIntervalArray(
                    AbsoluteDateArray(starts), AbsoluteDateArray(stops)
                )
            )
        return cls(coverage=coverage)


def get_integer_intervals(sorted_array: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a sorted numpy array of integers into a list of integer intervals.

    For example, if the input array is [1,2,3,7,9,11,12,13], the output list will
    be [(1,3), (7,7), (9,9), (11,13)]

    Parameters:
        sorted_array (np.ndarray): A 1D numpy array of sorted integers.

    Returns:
        List[Tuple[int, int]]: A list of (start, end) tuples representing intervals.
    """
    if sorted_array.size == 0:
        return []

    # Find the indices where the difference is not 1, i.e., gap in the sequence
    diff = np.diff(sorted_array)
    gap_indices = np.where(diff != 1)[0]

    # Starts: first element + elements after gaps
    starts = np.r_[sorted_array[0], sorted_array[gap_indices + 1]]
    # Ends: elements at gaps + last element
    ends = np.r_[sorted_array[gap_indices], sorted_array[-1]]

    return list(zip(starts, ends))
