#! /usr/bin/env python3

import pickle
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import lsim, StateSpace, ss
from cpnav.vendor.sippy import functionsetSIM as fsetSIM, system_identification

from cpnav.vendor.sippy.OLSims_methods import SS_model


def load_data(pickle_file: str):
    """
    Load data from the pickle file and align input-output.

    Args:
        pickle_file (str): Path to the pickle file containing input-output data.

    Returns:
        tuple: Tuple containing aligned input data (u_in) and output data (u_out).
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    u_in = np.array(data["u_in"][:-1])  # Exclude the last input
    u_out = np.array(data["u_out"][1:])  # Exclude the first output
    return u_in, u_out


def load_model(serialized_file: str):
    """
    Load the state-space model from the serialized file.

    Args:
        serialized_file (str): Path to the serialized state-space model file.

    Returns:
        StateSpace: The state-space model.
    """
    with open(serialized_file, "rb") as f:
        data = pickle.load(f)
    return ss(data["A"], data["B"], data["C"], data["D"], data["dt"])


def plot_input_vs_output(u_in: np.ndarray, u_out: np.ndarray, output_file: str):
    """
    Plot and save a graph showing the input vs. actual output.

    Args:
        u_in (np.ndarray): Input data.
        u_out (np.ndarray): Output data (speed).
        output_file (str): Path to save the plot image.
    """
    t = np.arange(len(u_in))  # Time steps
    velocity = u_out[:, 0]


    plt.figure(figsize=(10, 6))
    plt.plot(t, u_in, label="Input (u_in)", linestyle="--")
    plt.plot(t, velocity, label="Actual Output (Speed)")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("Input vs. Actual Output Used for Identification")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_file}_input_actual_output.png")
    plt.show()


def plot_input_vs_model_output(
    u_in: np.ndarray, u_out: np.ndarray, model: StateSpace, output_file: str
):
    """
    Plot and save a graph showing the input vs. model output.

    Args:
        u_in (np.ndarray): Input data.
        u_out (np.ndarray): Output data (speed).
        model (StateSpace): State-space model.
        output_file (str): Path to save the plot image.

        
    """

    with open("state-space-serialized.pkl", "rb") as f:
        data = pickle.load(f)

    state_x = np.zeros(5)

    ss_v_out = []
    for v_ in u_in:
        ss_v_out.append(model.output(0, state_x, v_))
        state_x = model.dynamics(0, state_x, v_)

    

    u_in = u_in[:-1]  # Exclude the last input
    u_out = [x[0] for x in ss_v_out[1:]] # Exclude the first output
    
    t = np.arange(len(u_in))
    
    

    plt.plot(t, u_in, label="Input (u_in)", linestyle="--")
    plt.plot(t, u_out, label="Model Output (Speed)")
    plt.show()


def main():
    """
    Main function to load data and plot identification results.
    """
    # Paths to the input-output data and serialized state-space model
    time_data_file = "time-series-data.pkl"
    serialized_file = "state-space-serialized.pkl"
    output_file = "identification_results"

    # Load data and model
    u_in, u_out = load_data(time_data_file)
    model = load_model(serialized_file)

    # Plot input vs. actual output
    plot_input_vs_output(u_in, u_out, output_file)

    # Plot input vs. model output
    plot_input_vs_model_output(u_in, u_out, model, output_file)


if __name__ == "__main__":
    main()
