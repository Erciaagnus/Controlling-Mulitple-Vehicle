from control.matlab import c2d, ss, StateSpace

from cpnav.models import InterfaceModel

import dill
import numpy as np

class CentralModel(InterfaceModel):
    """
    Centralized model for a platoon of vehicles, representing their longitudinal dynamics.

    This class models the behavior of a vehicle platoon by using the state-space models of individual
    vehicles and combines them into a single large state-space model for the entire platoon.
    The model includes vehicle inter-vehicle distances and velocities as outputs.
    """

    def __init__(self, t_s: float, n_vehicles: int):
        """
        Initializes the CentralModel for a platoon of vehicles.

        Args:
            t_s (float): The sample time (time step) for discretizing the model.
            n_vehicles (int): The number of vehicles in the platoon.
        """
        with open('state-space-serialized.pkl', 'rb') as f:
            self._model: StateSpace = dill.load(f)

        A = self._model.A
        B = self._model.B
        C = self._model.C
        D = self._model.D

        self.n_states = A.shape[0]

        A_n = np.kron(np.eye(n_vehicles), A)
        B_n = np.kron(np.eye(n_vehicles), B)

        C_s, C_v = C[0], C[1]

        I = np.eye(n_vehicles)
        I[-1][-1] = 0

        Z = np.zeros((n_vehicles, n_vehicles))
        Z[-1][-1] = 1

        R = np.roll(np.eye(n_vehicles), 1, axis=1)
        R[-1][0] = 0

        C_n = np.add(np.add(np.kron(I, -1 * C_s), np.kron(Z, C_v)), np.kron(R, C_s))

        # D_n = np.kron(np.eye(n_vehicles), self._model.D)
        D_n = np.zeros((n_vehicles, 1))

        self._model.A = A_n
        self._model.B = B_n
        self._model.C = C_n
        self._model.D = D_n

    @property
    def model(self) -> StateSpace:
        """
        Returns the state-space model for the entire platoon.

        The model combines the longitudinal dynamics of each vehicle in the platoon,
        inter-vehicle distances, and velocities.

        Returns:
            StateSpace: The state-space model of the platoon.
        """
        

        return self._model
