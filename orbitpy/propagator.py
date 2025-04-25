"""
.. module:: eosimutils.propagator
   :synopsis: Spacecraft propagator module for orbitpy.
"""

from typing import Type, Dict, Any, Union
import math

from skyfield.api import (
    EarthSatellite as Skyfield_EarthSatellite,
    load as Skyfield_load,
)

from eosimutils.base import EnumBase, ReferenceFrame
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.trajectory import StateSeries

from .orbits import TwoLineElementSet, OrbitalMeanElementsMessage

NUMBER_OF_SECONDS_IN_A_DAY = 86400.0


class PropagatorType(EnumBase):
    """Enumeration of supported propagator types."""

    SGP4_PROPAGATOR = "SGP4_PROPAGATOR"


class PropagatorFactory:
    """Factory class to register and invoke the appropriate propagator class.

    This class allows registering propagator classes and retrieving instances
    of the appropriate propagator based on specifications.

    example:
        factory = PropagatorFactory()
        factory.register_propagator("Custom_Propagator", CustomPropagator)
        specs = {"propagator_type": "Custom_Propagator", "step_size": 60}
        propagator = factory.get_propagator(specs)

    Attributes:
        _creators (Dict[str, Type]): A dictionary mapping propagator type 
                                     labels to their respective classes.
    """

    def __init__(self):
        """Initializes the PropagatorFactory and registers default propagators."""
        self._creators: Dict[str, Type] = {}
        self.register_propagator(
            PropagatorType.SGP4_PROPAGATOR.value, SGP4Propagator
        )

    def register_propagator(self, propagator_type: str, creator: Type) -> None:
        """Registers a propagator class with a specific type label.

        Args:
            propagator_type (str): The label for the propagator type.
            creator (Type): The propagator class to register.
        """
        self._creators[propagator_type] = creator

    def get_propagator(self, specs: Dict[str, Any]) -> Any:
        """Retrieves an instance of the appropriate propagator based on specifications.

        Args:
            specs (Dict[str, Any]): A dictionary containing propagator specifications.
                Must include a valid propagator type in the "propagator_type" key.

        Returns:
            Any: An instance of the appropriate propagator class initialized 
                 with the given specifications.

        Raises:
            KeyError: If the "propagator_type" key is missing in the specifications dictionary.
            ValueError: If the specified propagator type is not registered.
        """
        propagator_type_str = specs.get("propagator_type")
        if propagator_type_str is None:
            raise KeyError(
                'Propagator type key "propagator_type" not found in specifications dictionary.'
            )

        if propagator_type_str not in self._creators:
            raise ValueError(
                f'Propagator type "{propagator_type_str}" is not registered.'
            )

        creator = self._creators[propagator_type_str]
        return creator.from_dict(specs)


class SGP4Propagator:
    """A Simplified General Perturbations 4 (SGP4) orbit propagator class.

    This class is a wrapper around the Skyfield SGP4 orbit propagator.

    Attributes:
        step_size (float): Orbit propagation time-step in seconds. Default is 60 seconds.
        propagator_type (str): Type of propagator. Value is always 'SGP4_PROPAGATOR'.
    """

    def __init__(self, step_size=None):
        """Initializes the SGP4Propagator.

        Args:
            step_size (float, optional): Orbit propagation time-step in seconds. 
                                         Default is 60 seconds.
        """
        self.step_size = float(step_size) if step_size is not None else 60.0
        self.propagator_type = "SGP4_PROPAGATOR"

    @classmethod
    def from_dict(cls, specs):
        """Parses an SGP4Propagator object from a dictionary.

        Args:
            specs (dict): Dictionary with the SGP4 propagator specifications.
                The following keys are expected:
                - "step_size" (float): Step size in seconds. Default value is 60 seconds.

        Returns:
            SGP4Propagator: An instance of the SGP4Propagator class.
        """
        step_size = specs.get("step_size", 60.0)
        return cls(step_size=step_size)

    def execute(
        self,
        t0: AbsoluteDate,
        duration_days: float,
        orbit: Union[TwoLineElementSet, OrbitalMeanElementsMessage],
    ) -> StateSeries:
        """Propagates the orbit from t0 to t1 using the SGP4 propagator.

        Args:
            t0 (AbsoluteDate): Start time for propagation.
            duration_days (float): Duration of propagation in days.
            orbit (Union[TwoLineElementSet, OrbitalMeanElementsMessage]): 
                    Orbital specifications to be propagated.
                    This can be either a TLE set or an Orbital Mean Elements Message.

        Returns:
            StateSeries: A StateSeries object containing the propagated trajectory.

        Raises:
            ValueError: If the orbit type is invalid for SGP4 propagation.
        """
        if isinstance(orbit, TwoLineElementSet) or isinstance(
            orbit, OrbitalMeanElementsMessage
        ):
            skyfield_ts = Skyfield_load.timescale()

            skyfield_t0 = t0.to_skyfield_time()
            skyfield_t1 = (
                t0 + duration_days * NUMBER_OF_SECONDS_IN_A_DAY
            ).to_skyfield_time()
            propagate_time = skyfield_ts.linspace(
                skyfield_t0,
                skyfield_t1,
                num=math.floor(
                    duration_days * NUMBER_OF_SECONDS_IN_A_DAY / self.step_size
                )
                + 1,
            )

            tle = orbit.get_tle_as_tuple()
            skyfield_sat = Skyfield_EarthSatellite(
                tle[0], tle[1], None, skyfield_ts
            )

            geocentric = skyfield_sat.at(
                propagate_time
            )  # in GCRS (ECI) coordinates
            pos = geocentric.position.km
            vel = geocentric.velocity.km_per_s

            # Convert Skyfield Time object to AbsoluteDateArray
            absolute_date_array = AbsoluteDateArray.from_dict(
                {
                    "time_format": "Gregorian_Date",
                    "calendar_date": propagate_time.utc_strftime(
                        format="%Y-%m-%dT%H:%M:%S"
                    ),
                    "time_scale": "utc",
                }
            )

            # Define the reference frame. GCRS of Skyfield is considered
            # equivalent to ReferenceFrame.ICRF_EC
            # See: https://rhodesmill.org/skyfield/earth-satellites.html#generating-a-satellite-position # pylint: disable=line-too-long
            reference_frame = ReferenceFrame.ICRF_EC

            # Instantiate the StateSeries object
            state_series = StateSeries(
                time=absolute_date_array,
                data=[
                    pos.T,
                    vel.T,
                ],  # Transpose pos and vel to match StateSeries format
                frame=reference_frame,
            )
            return state_series

        else:
            raise ValueError(
                "Invalid orbit type for SGP4 propagation. Must be TLE or mean elements message."
            )
