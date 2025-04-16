"""
.. module:: eosimutils.propagator
   :synopsis: Spacecraft propagator module for orbitpy.
"""

from typing import Type, Dict, Any, Union

from eosimutils.base import EnumBase
from eosimutils.time import AbsoluteDate

from .orbits import TwoLineElementSet, OrbitalMeanElementsMessage

class PropagatorType(EnumBase):
    """Enumeration of supported propagator types."""
    SGP4_PROPAGATOR = "SGP4_PROPAGATOR"

class PropagatorFactory:
    """Factory class to register and invoke the appropriate propagator class.

    This class allows registering propagator classes and retrieving instances
    of the appropriate propagator based on specifications.

    Attributes:
        _creators (Dict[str, Type]): A dictionary mapping propagator type labels to their respective classes.
    """

    def __init__(self):
        """Initializes the PropagatorFactory and registers default propagators."""
        self._creators: Dict[str, Type] = {}
        self.register_propagator(PropagatorType.SGP4_PROPAGATOR.value, SGP4Propagator)

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
            Any: An instance of the appropriate propagator class initialized with the given specifications.

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
            raise ValueError(f'Propagator type "{propagator_type_str}" is not registered.')

        creator = self._creators[propagator_type_str]
        return creator.from_dict(specs)

class SGP4Propagator():
   """A Simplified General Perturbations 4 (SGP4) orbit propagator class.
    THe implementation of this class is a wrapper around the Skyfield SGP4 orbit propagator.

    The instance variable(s) correspond to the propagator setting(s). 

    :ivar stepSize: Orbit propagation time-step in seconds.
    :vartype stepSize: float or None

    :ivar propagator_type: Type of propagator. Value is always 'SGP4 PROPAGATOR'.
    :vartype propagator_type: str

    """
   def __init__(self, stepSize=None):
      self.stepSize = float(stepSize) if stepSize is not None else None
      self.propagator_type = "SGP4_PROPAGATOR"
   
   @classmethod
   def from_dict(cls, specs):
      """ Parses an SGP4Propagator object from a dictionary.
        
        :param d: Dictionary with the SGP4 PROPAGATOR specifications.

                Following keys are to be specified.
                
                * "stepSize": (float) Step size in seconds. Default value is 60s.

        :paramtype d: dict

        :return: ``SGP4Propagator`` object.
        :rtype: :class:`orbitpy.propagator.SGP4Propagator`

        """ 
      step_size = specs.get("step_size", 60.0)
      return cls(stepSize=step_size)