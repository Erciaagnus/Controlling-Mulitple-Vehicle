from control.matlab import StateSpace
from abc import ABC, abstractmethod


# Abstract base class to define the interface for vehicle models
class InterfaceModel(ABC):
    """
    Abstract class representing an interface for vehicle dynamic models.

    Subclasses should implement the 'model' property, which should return a
    StateSpace system that describes the vehicle's dynamic behavior.
    """

    @property
    @abstractmethod
    def model(self) -> StateSpace:
        """
        Abstract property to retrieve the vehicle model as a StateSpace system.

        This property should be implemented by subclasses to return a
        StateSpace object that represents the dynamics of the vehicle.

        Returns:
            StateSpace: A StateSpace model that describes the vehicle's behavior.
        """
        pass

    def __str__(self) -> str:
        """
        String representation of the vehicle model.

        This method returns a string representation of the vehicle's dynamic model.

        Returns:
            str: A string representation of the vehicle's dynamic system model.
        """
        return str(self.model)
