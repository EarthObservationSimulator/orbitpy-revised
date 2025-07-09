"""
.. module:: orbitpy.coverage
   :synopsis: Coverage representations for storing output of coverage simulations
"""


from typing import List, Dict, Any
from eosimutils.time import AbsoluteDateArray


class DiscreteCoverageTP:
    """
    Stores the results of a discrete-time coverage simulation.

    Attributes:
        time (AbsoluteDateArray): An array of absolute time points.
        coverage (List[List[int]]): For each time point, the list of accessed indices.
    """

    def __init__(self, time: AbsoluteDateArray, coverage: List[List[int]]) -> None:
        """
        Initializes a DiscreteCoverageTP instance.

        Args:
            time (AbsoluteDateArray): Array of time points.
            coverage (List[List[int]]): For each time point, one list of accessed indices.
        """
        self.time = time
        self.coverage = coverage

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing
                'time': dict from AbsoluteDateArray.to_dict()
                'coverage': List[List[int]] of accessed indices per time point
        """
        return {
            'time': self.time.to_dict(),
            'coverage': self.coverage
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscreteCoverageTP':
        """
        Deserializes a DiscreteCoverageTP from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary with keys:
                'time': dict representing an AbsoluteDateArray
                'coverage': List[List[int]] of accessed indices per time point

        Returns:
            DiscreteCoverageTP: The reconstructed instance.
        """
        time = AbsoluteDateArray.from_dict(data['time'])
        coverage = data['coverage']
        return cls(time=time, coverage=coverage)
