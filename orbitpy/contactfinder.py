"""
.. module:: orbitpy.contactfinder
    :synopsis: Module to handle contact opportunity finder between two entities
    (satellite to ground-station, satellite to satellite, etc).

    A class to calculate line-of-sight contact opportunities is implemented.
"""

from typing import Type, Dict, Any, Union, Callable
import numpy as np

from eosimutils.base import EnumBase, WGS84_EARTH_POLAR_RADIUS
from eosimutils.time import AbsoluteDateArray
from eosimutils.framegraph import FrameGraph
from eosimutils.state import (
    CartesianState,
    GeographicPosition,
    Cartesian3DPosition,
)
from eosimutils.timeseries import Timeseries, _group_contiguous
from eosimutils.trajectory import StateSeries, PositionSeries
import eosimutils.utils

from . import utils


class ContactFinderType(EnumBase):
    """Enumeration of supported contact finder types.

    Attributes:
        LOS_CONTACT_FINDER (str): Line-of-sight contact finder type.
    """

    LOS_CONTACT_FINDER = "LOS_CONTACT_FINDER"


class ContactInfo(Timeseries):
    """
    A class to store the results of the contact-finder execute function.

    This class inherits from the Timeseries class and ensures that the data stored
    is an array of booleans representing contact opportunities.

    Attributes:
        time (AbsoluteDateArray or None): Time values.
        data (np.ndarray): A numpy array of booleans. 'T' indicates contact.
                        'F' indicates no contact.
        headers (list): A list containing headers for the data array.
    """

    def __init__(self, time: Union[AbsoluteDateArray, None], data: np.ndarray):
        """
        Initialize a ContactInfo instance.

        Args:
            time (AbsoluteDateArray or None): Time values.
            data (np.ndarray): Array of booleans representing contact opportunities.

        Raises:
            TypeError: If `data` is not a numpy array of booleans.
        """
        headers = [["contact"]]
        if not isinstance(data, np.ndarray) or not np.issubdtype(
            data.dtype, np.bool_
        ):
            raise TypeError("data must be a numpy array of booleans.")

        if time is not None:
            # Call the parent class initializer if time is provided
            super().__init__(time, [data], headers)
        else:
            # Handle the case where time is None
            self.time = None
            self.data = np.array([data], dtype=bool)
            self.headers = headers

    def has_contact(self, index: int = None) -> Union[bool, None]:
        """
        Check if there is any contact opportunity in the data, or at a specific index.

        Args:
            index (int, optional): The index corresponding to a specific time.
                                    If None, checks for any contact.

        Returns:
            bool: True if there is contact at the specified index or at least one contact exists.
            None: If the index is out of bounds.
        """
        if index is None:
            return np.any(self.data[0])
        if 0 <= index < len(self.data[0]):
            return self.data[0][index]
        return None

    def contact_intervals(self) -> list:
        """
        Get the time intervals where contact opportunities exist.

        Returns:
            list: A list of tuples representing the start and end times of contact intervals.
        """
        contact_indices = np.where(self.data[0])[0]
        groups = _group_contiguous(contact_indices)
        intervals = [
            (self.time[i[0]], self.time[i[-1]]) for i in groups if len(i) > 0
        ]
        return intervals


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
        """Retrieves an instance of the appropriate contact-finder based on specifications.

        Args:
            specs (Dict[str, Any]): A dictionary containing contact-finder specifications.
                Must include a valid contact-finder type in the "contact_finder_type" key.

        Returns:
            object: An instance of the appropriate contact-finder class initialized
                 with the given specifications.

        Raises:
            KeyError: If the "contact_finder_type" key is missing in the specifications dictionary.
            ValueError: If the specified contact-finder type is not registered.
        """
        contact_finder_type_str = specs.get("contact_finder_type")
        if contact_finder_type_str is None:
            raise KeyError(
                "Contact finder type key 'contact_finder_type' not found "
                "in specifications dictionary."
            )
        contact_finder_class = cls._registry.get(contact_finder_type_str)
        if not contact_finder_class:
            raise ValueError(
                f'Contact finder type "{contact_finder_type_str}" is not registered.'
            )
        return contact_finder_class.from_dict(specs)


def get_entities_position_as_numpy(
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
) -> tuple[bool, np.ndarray, bool, np.ndarray]:
    """Helper function to extract position as numpy arrays for entities.
    The positions are returned in a common reference frame.

    Presently only supports the scenarios when:
        - Both entities are fixed, at the same reference frame.
        - At least one of the entities is fixed (in a reference frame) while
            the other entity is in motion.

    Args:
        frame_graph (FrameGraph): The frame graph containing transformations between
                                        reference frames.
        entity1_state (Union[StateSeries, PositionSeries, CartesianState,
                                GeographicPosition, Cartesian3DPosition]): State or position
                                                                        of the first entity.
        entity2_state (Union[StateSeries, PositionSeries, CartesianState,
                                GeographicPosition, Cartesian3DPosition]): State or position
                                                                        of the second entity.

    Returns:
        tuple: A tuple containing:
            - bool: True if the first entity is fixed, False otherwise.
            - np.ndarray: Numpy array of position(s) for the first entity.
                        Multiple positions are returned as a numpy array of shape (N, 3).
            - bool: True if the second entity is fixed, False otherwise.
            - np.ndarray: Numpy array of position(s) for the second entity.
                        Multiple positions are returned as a numpy array of shape (N, 3).

    Raises:
        NotImplementedError: If a particular scenario is not supported
                        (e.g., fixed entities in different reference frames).
    """
    entity1_fixed_flag = False
    entity2_fixed_flag = False

    # Convert all the entities to Cartesian3DPosition (fixed) or PositionSeries (moving)
    if isinstance(
        entity1_state,
        (GeographicPosition, Cartesian3DPosition, CartesianState),
    ):
        entity1_fixed_flag = True
        entity1_state = eosimutils.utils.convert_object(
            source_obj=entity1_state, target_type=Cartesian3DPosition
        )
    elif isinstance(entity1_state, (StateSeries, PositionSeries)):
        entity1_fixed_flag = False
        entity1_state = eosimutils.utils.convert_object(
            source_obj=entity1_state, target_type=PositionSeries
        )

    if isinstance(
        entity2_state,
        (GeographicPosition, Cartesian3DPosition, CartesianState),
    ):
        entity2_fixed_flag = True
        entity2_state = eosimutils.utils.convert_object(
            source_obj=entity2_state, target_type=Cartesian3DPosition
        )
    elif isinstance(entity2_state, (StateSeries, PositionSeries)):
        entity2_fixed_flag = False
        entity2_state = eosimutils.utils.convert_object(
            source_obj=entity2_state, target_type=PositionSeries
        )

    #### Handle the scenario when both the entities are fixed. ####
    if entity1_fixed_flag and entity2_fixed_flag:
        if entity1_state.frame != entity2_state.frame:
            raise NotImplementedError(
                "Both the fixed entities must be fixed in the same reference frame."
            )
        else:
            entity1_position_np = entity1_state.to_numpy()
            entity2_position_np = entity2_state.to_numpy()
            return (
                entity1_fixed_flag,
                entity1_position_np,
                entity2_fixed_flag,
                entity2_position_np,
            )

    #### Handle the scenario when both the entities are moving. ####
    if not (entity1_fixed_flag or entity2_fixed_flag):
        raise NotImplementedError(
            "At least one of the entities must be of type GeographicPosition"
            " or Cartesian3DPosition, i.e. it has to be fixed."
        )

    #### Handle the scenario when one of the entities is moving (logical XOR operation). ####
    if (entity1_fixed_flag and not entity2_fixed_flag) or (
        not entity1_fixed_flag and entity2_fixed_flag
    ):
        # The fixed entity is Cartesian3DPosition type and the moving entity is PositionSeries type.
        if entity1_fixed_flag is False:
            moving_entity_position_series = entity1_state
            fixed_entity_cartesian_3d_position = entity2_state
        else:
            moving_entity_position_series = entity2_state
            fixed_entity_cartesian_3d_position = entity1_state

        # Helper function to transform position vectors to a target frame
        def transform_to_target_frame(
            frame_graph, from_frame, to_frame, input_pos_vector, times
        ):
            """Transform the input position vector to the target frame."""
            # TODO: Should the position transform be also applied? I.e. should
            # the relative position difference in the center of the from and to
            # frames be considered?
            rot_array, _ = frame_graph.get_orientation_transform(
                from_frame, to_frame, times
            )
            return rot_array.apply(input_pos_vector)

        if (
            fixed_entity_cartesian_3d_position.frame
            != moving_entity_position_series.frame
        ):
            # Transform the moving entity's state to the fixed entity's reference frame.
            # The reason the moving entity is transformed is because the time information
            # is readily available with the moving entity's position series object.
            moving_entity_position_np = transform_to_target_frame(
                frame_graph,
                moving_entity_position_series.frame,
                fixed_entity_cartesian_3d_position.frame,
                moving_entity_position_series.position.to_numpy(),
                moving_entity_position_series.time,
            )
        else:
            moving_entity_position_np = (
                moving_entity_position_series.position.to_numpy()
            )

        fixed_entity_position_np = fixed_entity_cartesian_3d_position.to_numpy()

        if entity1_fixed_flag is True:
            return (
                True,
                fixed_entity_position_np,
                False,
                moving_entity_position_np,
            )
        else:
            return (
                False,
                moving_entity_position_np,
                True,
                fixed_entity_position_np,
            )


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
    ) -> ContactInfo:
        """
        Calculate the line-of-sight contact opportunities between two entities.
        Presently only supports the scenarios when:
            - One of the entities is fixed (in a reference frame) over the entire
                period of time when the other entity could be in motion.
            - Both entities are fixed in the same reference frames.

        TODO: Support the scenario when both entities are in motion or fixed in
        different reference frame. This would require additional calculations such as
        cases when the entities have time-series at different time-steps,
        or for different durations.


        Args:
            frame_graph (FrameGraph): The frame graph containing transformations between
                                        reference frames.
            entity1_state (Union[StateSeries, PositionSeries, CartesianState,
                                 GeographicPosition, Cartesian3DPosition]): State or position
                                                                            of the first entity.
            entity2_state (Union[StateSeries, PositionSeries, CartesianState,
                                 GeographicPosition, Cartesian3DPosition]): State or position
                                                                            of the second entity.

        Returns:
            ContactIntervals: An object containing time-intervals of contact opportunities.
        """

        (
            entity1_fixed_flag,
            entity1_state_np,
            entity2_fixed_flag,
            entity2_state_np,
        ) = get_entities_position_as_numpy(
            frame_graph, entity1_state, entity2_state
        )

        #### Handle the scenario when both the entities are fixed. ####
        if entity1_fixed_flag and entity2_fixed_flag:
            los = utils.check_line_of_sight(
                entity1_state_np,
                entity2_state_np,
                WGS84_EARTH_POLAR_RADIUS,
            )
            return ContactInfo(None, np.array([los], dtype=bool))

        #### Handle the scenario when both the entities are moving. ####
        if not (entity1_fixed_flag or entity2_fixed_flag):
            raise NotImplementedError(
                "The scenario where both the observer and target are moving is not supported."
            )

        #### Handle the scenario when one of the entities is moving (logical XOR operation). ####
        if (entity1_fixed_flag and not entity2_fixed_flag) or (
            not entity1_fixed_flag and entity2_fixed_flag
        ):
            if entity1_fixed_flag is False:
                moving_entity_position_series = entity1_state
                moving_entity_position_np = entity1_state_np
                fixed_entity_position_np = entity2_state_np
            else:
                moving_entity_position_series = entity2_state
                moving_entity_position_np = entity2_state_np
                fixed_entity_position_np = entity1_state_np

            # Check for line-of-sight contact opportunities
            los = []

            for _, cart_position_np in enumerate(moving_entity_position_np):
                los.append(
                    utils.check_line_of_sight(
                        fixed_entity_position_np,
                        cart_position_np,
                        WGS84_EARTH_POLAR_RADIUS,
                    )
                )

            # Convert results to ContactInfo object
            time = moving_entity_position_series.time
            los_array = np.array(los, dtype=bool)
            return ContactInfo(time, los_array)


@ContactFinderFactory.register_type("ELEVATION_AWARE_CONTACT_FINDER")
class ElevationAwareContactFinder(LineOfSightContactFinder):
    """Handles line-of-sight contact opportunities with elevation angle considerations.
    This class extends the LineOfSightContactFinder to include a minimum elevation angle
    constraint for contact opportunities.
    It first determines the line-of-sight (LOS) using the parent class and then applies 
    an additional filter to ensure that the elevation angle between the observer and 
    the target meets or exceeds the specified minimum threshold. 
    The limitations of the parent class also apply here.
    """

    def execute(
        self,
        frame_graph: FrameGraph,
        observer_state: Union[
            StateSeries,
            PositionSeries,
            CartesianState,
            GeographicPosition,
            Cartesian3DPosition,
        ],
        target_state: Union[
            StateSeries,
            PositionSeries,
            CartesianState,
            GeographicPosition,
            Cartesian3DPosition,
        ],
        min_elevation_angle: float,
    ) -> ContactInfo:
        """
        Calculate the line-of-sight contact opportunities between the observer and target,
        considering a minimum elevation angle.

        Args:
            frame_graph (FrameGraph): The frame graph containing transformations between
                                      reference frames.
            observer_state: State or position of the observer entity (e.g., ground station).
            target_state: State or position of the target entity (e.g., satellite).
            min_elevation_angle (float): Minimum elevation angle (in degrees) for contact.

        Returns:
            ContactInfo: An object containing time-intervals of contact opportunities.
        """
        # Use the parent class logic to get initial line-of-sight results
        contact_info = super().execute(
            frame_graph, observer_state, target_state
        )

        obs_fixed_flag, obs_state_np, target_fixed_flag, target_state_np = (
            get_entities_position_as_numpy(
                frame_graph, observer_state, target_state
            )
        )

        # Filter results based on elevation angle
        los_with_elevation = []

        # Handle the scenario when both the entities are fixed.
        if obs_fixed_flag and target_fixed_flag:
            has_los = contact_info.data[0]
            if has_los:
                elevation_angle = utils.calculate_elevation_angle(
                    obs_state_np, target_state_np
                )
                los_with_elevation = elevation_angle >= min_elevation_angle
            else:
                los_with_elevation = False
            return ContactInfo(None, np.array([los_with_elevation], dtype=bool))

        # Handle the scenario when both the entities are moving.
        if not (obs_fixed_flag or target_fixed_flag):
            raise NotImplementedError(
                "The scenario where both the observer and target are moving is not supported."
            )

        # Handle the scenario when one of the entities is moving (logical XOR operation).
        if (obs_fixed_flag and not target_fixed_flag) or (
            not obs_fixed_flag and target_fixed_flag
        ):

            if obs_fixed_flag is False:
                moving_entity_position_series = observer_state
                moving_entity_position_np = obs_state_np
                fixed_entity_position_np = target_state_np
            else:
                moving_entity_position_series = target_state
                moving_entity_position_np = target_state_np
                fixed_entity_position_np = obs_state_np

            for i, has_los in enumerate(contact_info.data[0]):
                if has_los:
                    elevation_angle = utils.calculate_elevation_angle(
                        fixed_entity_position_np, moving_entity_position_np[i]
                    )
                    los_with_elevation.append(
                        elevation_angle >= min_elevation_angle
                    )
                else:
                    los_with_elevation.append(False)

            # Convert results to ContactInfo object
            time = moving_entity_position_series.time
            los_with_elevation_array = np.array(los_with_elevation, dtype=bool)
            return ContactInfo(time, los_with_elevation_array)
