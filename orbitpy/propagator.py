"""
.. module:: eosimutils.propagator
   :synopsis: Spacecraft propagator module for orbitpy.
"""

from typing import Type, Dict, Any, Union

from eosimutils.time import AbsoluteDate

from .orbits import TwoLineElementSet, OrbitalMeanElementsMessage


class PropagatorFactory:
    """Factory class to register and invoke the appropriate propagator class.

    This class allows registering propagator classes and retrieving instances
    of the appropriate propagator based on specifications.

    Attributes:
        _creators (Dict[str, Type]): A dictionary mapping propagator type labels 
                                     to their respective classes.
    """

    def __init__(self):
        """Initializes the PropagatorFactory and registers default propagators."""
        self._creators: Dict[str, Type] = {}
        self.register_propagator("SGP4 PROPAGATOR", SGP4Propagator)

    def register_propagator(self, propagator_type: str, creator: Type) -> None:
        """Registers a propagator class with a specific type label.

        Args:
            _type (str): The label for the propagator type.
            creator (Type): The propagator class to register.
        """
        self._creators[propagator_type] = creator

    def get_propagator(self, specs: Dict[str, Any]) -> Any:
        """Retrieves an instance of the appropriate propagator based on specifications.

        Args:
            specs (Dict[str, Any]): A dictionary containing propagator specifications.
                Must include a valid propagator type in the "propagator_type" key.
                The other keys depend on the specific propagator class being requested.
                For example, for SGP4 PROPAGATOR, the "step_size" key is expected.

        Returns:
            Any: An instance of the appropriate propagator class 
                 initialized with the given specifications.

        Raises:
            KeyError: If the "propagator_type" key is missing in the specifications dictionary.
            ValueError: If the specified propagator type is not registered.
        """
        propagator_type = specs.get("propagator_type")
        if propagator_type is None:
            raise KeyError(
                'Propagator type key "propagator_type" not found in specifications dictionary.'
            )

        creator = self._creators.get(propagator_type)
        if not creator:
            raise ValueError(f'Propagator type "{propagator_type}" is not registered.')
        return creator.from_dict(specs)
