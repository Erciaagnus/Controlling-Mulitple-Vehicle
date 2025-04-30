import pickle
from control.matlab import c2d, ss, StateSpace
import numpy as np

from cpnav.models import InterfaceModel

import dill as pickle

# class LongitudinalModel(InterfaceModel):
#     """
#     Longitudinal dynamic model for a vehicle in a state-space form.

#     This class implements the InterfaceModel to represent the longitudinal dynamics of a vehicle, i.e., the state-space matrices for the system.
#     """

#     def __init__(self, t_s: float):
#         """
#         Initializes the LongitudinalModel with state-space matrices.

#         Args:
#             t_s (float): The sample time (time step). Only useful when the identified model is continuous.
#         """
#         super().__init__()

#         # Load the model for the longitudinal dynamics of the vehicle
#         with open('state-space-serialized.pkl', 'rb') as f:
#             self._model: StateSpace = pickle.load(f)

#     @property
#     def model(self) -> StateSpace:
#         """
#         Returns the discrete-time state-space model for the longitudinal dynamics of the vehicle.

#         Returns:
#             StateSpace: The discrete-time vehicle longitudinal model.
#         """
#         return self._model


import pickle
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import c2d, ss, StateSpace, lsim
from cpnav.models import InterfaceModel


class LongitudinalModel(InterfaceModel):
    """
    Longitudinal dynamic model for a vehicle in a state-space form.
    """

    def __init__(self, t_s: float):
        """
        Initializes the LongitudinalModel with state-space matrices.

        Args:
            t_s (float): The sample time (time step). Only useful when the identified model is continuous.
        """
        super().__init__()

        # Load the model for the longitudinal dynamics of the vehicle
        with open('state-space-serialized.pkl', 'rb') as f:
            self._model: StateSpace = pickle.load(f)


    @property
    def model(self) -> StateSpace:
        """Returns the discrete-time state-space model."""
        return self._model