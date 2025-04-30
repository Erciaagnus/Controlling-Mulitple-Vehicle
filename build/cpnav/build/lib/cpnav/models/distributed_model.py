from control.matlab import c2d, ss, StateSpace
import numpy as np
import dill

from cpnav.models import InterfaceModel
from sympy import gamma


# import dill as pickle

class DistributedModel(InterfaceModel):
    """
    Longitudinal dynamic model for a vehicle in a state-space form.

    This class implements the InterfaceModel to represent the longitudinal dynamics of a vehicle, i.e., the state-space matrices for the system.
    """

    def __init__(self, control_type:str, t_s: float):
        """
        Initializes the LongitudinalModel with state-space matrices.

        Args:
            t_s (float): The sample time (time step). Only useful when the identified model is continuous.
        """
        super().__init__()
        
        # Create a discrete-time state-space model using the defined matrices and the sample time Ts
        with open('state-space-serialized.pkl', 'rb') as f:
            self._model: StateSpace = dill.load(f)

        A = self._model.A
        B = self._model.B
        C = self._model.C
        D = self._model.D

        self.control_type = control_type
        self._model: StateSpace = ss(A, B, C, D)

    @property
    def model(self) -> StateSpace:
        """
        Returns the discrete-time state-space model for the longitudinal dynamics of the vehicle.

        Returns:
            StateSpace: The discrete-time vehicle longitudinal model.
        """
        return self._model
