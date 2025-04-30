#! /usr/bin/env python3

import dill as pickle  # Use dill to handle lambda serialization

from typing import Dict, BinaryIO

import numpy as np
from control import StateSpace

from cpnav.vendor.sippy import system_identification
from cpnav.vendor.sippy.OLSims_methods import SS_model

import matplotlib.pyplot as plt

from cpnav.vendor.sippy import functionsetSIM as fsetSIM

def main(data_file_path: str, model_save_path: str, t_s: float):
    # Load data
    data: Dict[str, np.ndarray] = pickle.load(open(data_file_path, 'rb'))

    # Extract timestamps, input and output
    u_in = np.array(data['u_in'])
    u_out = np.array(data['u_out'])
    
    # Assertions for shapes
    assert len(u_in) == len(u_out), f"u_in and u_out must have the same number of data points, but got {len(u_in)} and {len(u_out)}."
    n_data_points = len(u_in)
    assert u_in.ndim == 1, f"u_in must be a 1D array with shape ({n_data_points},), but got shape {u_in.shape}."
    assert u_out.ndim == 2 and u_out.shape[1] == 2, f"u_out must have shape ({n_data_points}, 2), but got shape {u_out.shape}."

    print(f"u_in[0]: {u_in[0]}")
    print(f"u_out[0]: {u_out[0]}")
    print(f"Number of data points: {n_data_points}")

    # Do system identification
    # Hints:
    #   - Do not change sample time. The identified model is already in the discrete domain. Functions such as c2d is no longer needed.
    #   - Try different orders, such as two and three, and compare their accurancies. Observe the relation between the size of the system matrix A and the order.
    method: str = 'N4SID'
    ss_model: SS_model = system_identification(u_out, u_in, method, SS_fixed_order=5,
                                               tsample=t_s)

    # Extract data to dict
    state_space: StateSpace = ss_model.G
    print(state_space)

    
    d = dict()
    d["A"] = state_space.A
    d["B"] = state_space.B
    d["C"] = state_space.C
    d["D"] = state_space.D
    d["dt"] = state_space.dt
    print("saved")
    # Serialize state space
    f: BinaryIO = open('state-space-serialized.pkl', 'wb')
    pickle.dump(d, f)
    f.close()
    
    print(f"The identified model is saved at: {model_save_path}.")

def validate_model(data_file_path: str, model_save_path: str, t_s: float, fig_save_path: str):
    # Load the saved model from the pickle file
    with open(model_save_path, 'rb') as f:
        state_space: StateSpace = pickle.load(f)

    # Extract matrices from the loaded model
    A: np.ndarray = state_space.A
    B: np.ndarray = state_space.B
    C: np.ndarray = state_space.C
    D: np.ndarray = state_space.D

    # Load the input and output data collected during the identification phase
    with open(data_file_path, 'rb') as f:
        data = pickle.load(f)

    u_in = np.array(data['u_in'])
    u_out_original = np.array(data['u_out'])

    time_points = np.arange(len(u_in)) * t_s
    
    # Determine the initial state x_0 based on the output and C matrix
    y_0 = u_out_original[0, :]  # Initial output
    # Check properties of C
    if C.shape[0] == C.shape[1] and np.linalg.matrix_rank(C) == C.shape[0]:
        # Square and invertible
        x_0 = np.linalg.inv(C) @ y_0
    else:
        # Non-square and not invertible: use pseudo-inverse
        x_0 = np.linalg.pinv(C) @ y_0  # pseudo-inverse(C) := (C^T * C)^(−1) * C^T
    x_0 = x_0.reshape(-1, 1)  # Ensure x_0 is a column vector
    print(f"Initial state x_0: {x_0}")
    
    # Simulate the identified model
    x, u_out_simulated = fsetSIM.SS_lsim_process_form(A, B, C, D, u_in.reshape(1, -1), x_0)

    u_out_distance_target = np.cumsum(u_in * t_s) + y_0[0]  # Accumulated target distance
    
    plt.rcParams.update({'font.size': 16})

    # Create subplots for distance and speed
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # Plot distances
    axs[0].plot(time_points, u_out_distance_target, label='Target distance', linestyle='-')
    axs[0].plot(time_points, u_out_original[:, 0], label='Original distance', linestyle='--')
    axs[0].plot(time_points, u_out_simulated[0, :], label='Simulated distance', linestyle='-')

    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Distance')
    axs[0].set_title('Distance Output')
    axs[0].legend()
    axs[0].grid(True)

    # Plot speeds
    axs[1].plot(time_points, u_in, label='Target speed', linestyle='-')
    axs[1].plot(time_points, u_out_original[:, 1], label='Original speed', linestyle='--')
    axs[1].plot(time_points, u_out_simulated[1, :], label='Simulated speed', linestyle='-')

    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Speed')
    axs[1].set_title('Speed Output')
    axs[1].legend()
    axs[1].grid(True)

    # Save figure
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    print(f"The figure is saved at: {fig_save_path}.")

    plt.show()
    
if __name__ == '__main__':
    data_file_path: str = 'time-series-data.pkl'
    model_save_path: str = 'state-space-serialized.pkl'
    fig_save_path: str = 'plot-system-identification.png'
    t_s: float = 0.4  # Sample time in seconds

    main(data_file_path, model_save_path, t_s)
    validate_model(data_file_path, model_save_path, t_s, fig_save_path)
