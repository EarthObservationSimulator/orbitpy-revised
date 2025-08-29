"""
.. module:: orbitpy.contactfinder
    :synopsis: Module to handle contact opportunity finder between two entities (satellite to ground-station,
                satellite to satellite, etc).

    A class to calculate line-of-sight contact opportunities is implemented.
"""

from typing import Type, Dict, Any, Union, Callable

from eosimutils.base import EnumBase, WGS84_EARTH_POLAR_RADIUS
from eosimutils.time import AbsoluteDateArray, AbsoluteDateIntervalArray
from eosimutils.framegraph import FrameGraph
from eosimutils.state import (
    CartesianState,
    GeographicPosition,
    Cartesian3DPosition,
)
from eosimutils.trajectory import StateSeries, PositionSeries
import eosimutils.utils

from . import utils

class ContactFinderType(EnumBase):
    """Enumeration of supported contact finder types.

    Attributes:
        LOS_CONTACT_FINDER (str): Line-of-sight contact finder type.
    """

    LOS_CONTACT_FINDER = "LOS_CONTACT_FINDER"


class ContactFinderFactory:
    """Factory class to register and invoke the appropriate contact-finder calculator class.

    This class allows registering contact-finder calculator classes and retrieving instances
    of the appropriate contact-finder calculator based on specifications.

    Example:
        class CustomContactFinder:
            @classmethod
            def from_dict(cls, specs):
                return cls()
        ContactFinderFactory.register_type("CustomContactFinder")(CustomContactFinder)
        specs = {"contact_finder_type": "CustomContactFinder", ...}
        contact_finder = ContactFinderFactory.from_dict(specs)

    Attributes:
        _registry (Dict[str, Type]): A dictionary mapping contact-finder type
                                     labels to their respective classes.
    """

    # Registry for factory pattern
    _registry: Dict[str, Type] = {}

    @classmethod
    def register_type(cls, type_name: str) -> Callable[[Type], Type]:
        """
        Decorator to register a contact-finder class under a type name.
        """

        def decorator(contact_finder_class: Type) -> Type:
            cls._registry[type_name] = contact_finder_class
            return contact_finder_class

        return decorator

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> object:
        """Retrieves an instance of the appropriate contact-finder calculator based on specifications.

        Args:
            specs (Dict[str, Any]): A dictionary containing contact-finder calculator specifications.
                Must include a valid contact-finder calculator type in the "contact_finder_type" key.

        Returns:
            object: An instance of the appropriate contact-finder calculator class initialized
                 with the given specifications.

        Raises:
            KeyError: If the "contact_finder_type" key is missing in the specifications dictionary.
            ValueError: If the specified contact-finder calculator type is not registered.
        """
        contact_finder_type_str = specs.get("contact_finder_type")
        if contact_finder_type_str is None:
            raise KeyError(
                'Contact finder type key "contact_finder_type" not found in specifications dictionary.'
            )
        contact_finder_class = cls._registry.get(contact_finder_type_str)
        if not contact_finder_class:
            raise ValueError(
                f'Contact finder type "{contact_finder_type_str}" is not registered.'
            )
        return contact_finder_class.from_dict(specs)


@ContactFinderFactory.register_type(
    ContactFinderType.LOS_CONTACT_FINDER.to_string()
)
class LineOfSightContactFinder:
    """Handles line-of-sight contact opportunities between two entities.
    Presently only supports the scenarios when:
        - Both entities are fixed, at the same reference frame.
        - At least one of the entities is fixed (in a reference frame) while
            the other entity is in motion.
    """

    def execute(
        self,
        frame_graph: FrameGraph,
        entity1_state: Union[
            StateSeries,
            PositionSeries,
            CartesianState,
            GeographicPosition,
            Cartesian3DPosition,
        ],
        entity2_state: Union[
            StateSeries,
            PositionSeries,
            CartesianState,
            GeographicPosition,
            Cartesian3DPosition,
        ],
    ) -> Union[bool, list[bool]]:
        """
        Calculate the line-of-sight contact opportunities between two entities.
        Presently only supports the scenarios when:
            - One of the entities is fixed (in a reference frame) over the entire period of time when the other entity could be in motion.
            - Both entities are fixed in the same reference frames.

        TODO: Support the scenario when both entities are in motion or fixed in different reference frame.
        This would require additional calculations such as cases when the entities have time-series at different time-steps,
        or for different durations.


        Args:
            frame_graph (FrameGraph): The frame graph containing transformations between 
                                        reference frames.
            entity1_state (Union[StateSeries, PositionSeries, CartesianState, GeographicPosition, Cartesian3DPosition]):
                State or position of the first entity.
            entity2_state (Union[StateSeries, PositionSeries, CartesianState, GeographicPosition, Cartesian3DPosition]):
                State or position of the second entity.

        Returns:
            ContactIntervals: An object containing time-intervals of contact opportunities.
        """

        # Convert all the entities to Cartesian3DPosition (fixed) or PositionSeries (moving)
        entity1_fixed = False
        entity2_fixed = False

        if isinstance(entity1_state, (GeographicPosition, Cartesian3DPosition, CartesianState)):
            entity1_fixed = True
            entity1_state = eosimutils.utils.convert_object(source_obj=entity1_state, target_type=Cartesian3DPosition)
        elif isinstance(entity1_state, (StateSeries, PositionSeries)):
            entity1_fixed = False
            entity1_state = eosimutils.utils.convert_object(source_obj=entity1_state, target_type=PositionSeries)

        if isinstance(entity2_state, (GeographicPosition, Cartesian3DPosition, CartesianState)):
            entity2_fixed = True
            entity2_state = eosimutils.utils.convert_object(source_obj=entity2_state, target_type=Cartesian3DPosition)
        elif isinstance(entity2_state, (StateSeries, PositionSeries)):
            entity2_fixed = False
            entity2_state = eosimutils.utils.convert_object(source_obj=entity2_state, target_type=PositionSeries)

        if not (entity1_fixed or entity2_fixed):
            raise NotImplementedError(
                "At least one of the entities must be of type GeographicPosition or Cartesian3DPosition, i.e. it has to be fixed."
            )

        #### Handle the scenario when both the entities are fixed. ####
        if entity1_fixed and entity2_fixed:
            if entity1_state.frame != entity2_state.frame:
                raise NotImplementedError(
                    "Both the fixed entities must be fixed in the same reference frame."
                )
            # Both entities are fixed in the same reference frame, so we can proceed with the calculations.
            los = utils.check_line_of_sight(entity1_state.to_numpy(), entity2_state.to_numpy(), WGS84_EARTH_POLAR_RADIUS)

            return los

        #### Handle the scenario when one of the entities is moving. ####
        # The fixed entity is Cartesian3DPosition type and the moving entity is PositionSeries type.
        if entity1_fixed is False:
            moving_entity_position_series = entity1_state
            fixed_entity_cartesian_3d_position = entity2_state
        else:
            moving_entity_position_series = entity2_state
            fixed_entity_cartesian_3d_position = entity1_state

        # Helper function to transform position vectors to a target frame
        def transform_to_target_frame(frame_graph, from_frame, to_frame, input_pos_vector, times):
            """Transform the input position vector to the target frame."""
            rot_array, _ = frame_graph.get_orientation_transform(
                from_frame, to_frame, times
            )
            return rot_array.apply(input_pos_vector)

        if fixed_entity_cartesian_3d_position.frame != moving_entity_position_series.frame:
            # Transform the moving entity's state to the fixed entity's reference frame
            moving_entity_position_series = utils.transform_to_target_frame(
                frame_graph,
                moving_entity_position_series.frame,
                fixed_entity_cartesian_3d_position.frame,
                moving_entity_position_series.position.to_numpy(),
                moving_entity_position_series.time
            )
        
        # Check for line-of-sight contact opportunities
        los = []
        fixed_entity_np_position = fixed_entity_cartesian_3d_position.to_numpy()
        for i, cart_position in enumerate(moving_entity_position_series.position):
            los.append(utils.check_line_of_sight(fixed_entity_np_position, cart_position.to_numpy(), WGS84_EARTH_POLAR_RADIUS))

        return los