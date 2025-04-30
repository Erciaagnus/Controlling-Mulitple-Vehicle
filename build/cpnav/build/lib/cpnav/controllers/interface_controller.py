from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class ControllerInterface(ABC):
    """
    Abstract base class that defines the interface for a control system controller.
    Any specific controller inherits from this class and implements its abstract methods.
    """

    @abstractmethod
    def setup(self, x_init: np.ndarray) -> None:
        """
        Initialize the controller with an initial state.

        This method should be implemented by subclasses to set up any necessary internal
        parameters or state of the controller based on the provided initial state.

        Args:
            x_init (np.ndarray): The initial state vector of the system, typically used to set up
                                  the controller’s initial conditions or internal variables.
        """
        pass

    @abstractmethod
    def step(self, y_measured: np.ndarray, y_reference: np.ndarray, y_min: np.ndarray = None,
             y_max: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one control step based on the measured and reference outputs.

        This method should be implemented by subclasses to calculate the control input at each time step.
        It will use the measured output `y_measured`, the reference output `y_reference`, and optionally,
        constraints on the output (`y_min`, `y_max`) to compute the control input and any associated state updates.

        Args:
            y_measured (np.ndarray): The measured output from the system at the current time step.
            y_reference (np.ndarray): The desired reference or target output for the system.
            y_min (np.ndarray, optional): The lower bounds for the output. Defaults to None.
            y_max (np.ndarray, optional): The upper bounds for the output. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple where the first element is the control input
                                           and the second element is the updated state (if applicable).
        """
        pass
