#! /usr/bin/env python3
from threading import Lock, Thread
from typing import List, Dict, Tuple, Callable

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rclpy import Parameter
from rclpy.executors import MultiThreadedExecutor

from cpnav.utils import outer_lane_path, MeasurementTransformer, compute_relative_distance_on_path
import rclpy
from cpm_lab_lab_msgs.msg import VehicleStateList, VehicleCommand, VehicleState
from rclpy.subscription import Subscription
from rclpy.time import Time

from rclpy.node import Node
from cpnav.internal_types import VehicleStateList as VehicleStateListInternal, \
    VehicleState as VehicleStateInternal, PathPoint as PathPointInternal

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PlotPlatooningNode(Node):
    def __init__(self):
        super().__init__('plot_platooning_node')

        # Parameters
        self.declare_parameter('vehicle_ids', rclpy.Parameter.Type.INTEGER_ARRAY)

        # Retrieve and validate the parameter
        param = self.get_parameter('vehicle_ids')

        if param.type_ != Parameter.Type.INTEGER_ARRAY:
            raise ValueError(
                "The 'vehicle_ids' parameter is not an integer array. Using an empty list.")
        else:
            try:
                self._vehicle_ids = param.get_parameter_value().integer_array_value
                if not all(isinstance(id, int) for id in self._vehicle_ids):
                    raise ValueError("All elements in 'vehicle_ids' must be integers.")
            except Exception as e:
                raise ValueError(f"Error with 'vehicle_ids': {e}")
        self._path_internal: List[PathPointInternal] = outer_lane_path()

        # Subscriptions
        self._vehicle_command_subscription: Subscription = self.create_subscription(
            VehicleCommand, 'command', self._on_vehicle_command, 10)
        self._vehicle_state_list_subscription: Subscription = self.create_subscription(
            VehicleStateList, 'vehicle_states', self._on_vehicle_state_list, 10)

        # Input data
        self._vehicle_commands: List[VehicleCommand] = []
        self._vehicle_state_lists: List[VehicleStateList] = []

        # Start time
        self._start_time: Time | None = None

        # Leader reference velocity
        self._v0_reference: Callable[[float], float] = lambda t: 0.5 if 0 <= t < 15 else \
            1.4 if 15 <= t < 25 else \
                0.8 if 25 <= t < 35 else 0

        # Data access mutex
        self._mutex: Lock = Lock()

        # General plotting setup

        res = plt.subplots(4, 1, figsize=(8, 12))
        self._fig: Figure = res[0]
        self._axs: List[Axes] = res[1].tolist()
        self._plotting_data: Dict[str, Dict[str, Tuple[Line2D, List[float], List[float]]]] = {}

        # Input velocity plot setup
        self._plotting_data['v_in'] = {
            'v_min': (self._axs[0].plot([], [], lw=2, color='red', label='v_min')[0], [], []),
            'v_max': (self._axs[0].plot([], [], lw=2, color='red', label='v_max')[0], [], []),
        }
        if len(self._vehicle_ids) > 1:
            self._plotting_data['v_in']['v0_ref'] = (
                self._axs[0].plot([], [], lw=2, color=(0.6, 0.6, 0.6), label='v0_ref')[0], [], [])

        for vehicle_id in self._vehicle_ids:
            self._plotting_data['v_in'][str(vehicle_id)] = (
                self._axs[0].plot([], [], lw=2, label=f'vehicle_{vehicle_id}')[0], [], [])

        self._axs[0].legend()
        self._axs[0].set_ylim(-0.1, 1.75)
        self._axs[0].set_xlim(0, 60)
        self._axs[0].set_title('Input Velocity over Time')
        self._axs[0].set_xlabel('Time (s)')
        self._axs[0].set_ylabel('Speed (m/s)')

        # Output velocity
        self._plotting_data['v_out'] = {
            'v_min': (self._axs[1].plot([], [], lw=2, color='red', label='v_min')[0], [], []),
            'v_max': (self._axs[1].plot([], [], lw=2, color='red', label='v_max')[0], [], []),
        }
        if len(self._vehicle_ids) > 1:
            self._plotting_data['v_out']['v0_ref'] = (
                self._axs[1].plot([], [], lw=2, color=(0.6, 0.6, 0.6), label='v0_ref')[0], [], [])

        for vehicle_id in self._vehicle_ids:
            self._plotting_data['v_out'][str(vehicle_id)] = (
                self._axs[1].plot([], [], lw=2, label=f'vehicle_{vehicle_id}')[0], [], [])

        self._axs[1].legend()
        self._axs[1].set_ylim(-0.1, 1.6)
        self._axs[1].set_xlim(0, 60)
        self._axs[1].set_title('Output Velocity over Time')
        self._axs[1].set_xlabel('Time [s]')
        self._axs[1].set_ylabel('Speed [m/s]')

        # Input acceleration
        self._plotting_data['a_in'] = {
            'a_min': (self._axs[2].plot([], [], lw=2, color='red', label='a_min')[0], [], []),
            'a_max': (self._axs[2].plot([], [], lw=2, color='red', label='a_max')[0], [], []),
        }
        for vehicle_id in self._vehicle_ids:
            self._plotting_data['a_in'][str(vehicle_id)] = (
                self._axs[2].plot([], [], lw=2, label=f'vehicle_{vehicle_id}')[0], [], [])

        self._axs[2].legend()
        self._axs[2].set_ylim(-1.1, 0.6)
        self._axs[2].set_xlim(0, 60)
        self._axs[2].set_title('Input Acceleration over Time')
        self._axs[2].set_xlabel('Time [s]')
        self._axs[2].set_ylabel('Acceleration [m^2/s]')

        self._measurement_transformer = MeasurementTransformer(outer_lane_path(),
                                                               self._vehicle_ids)
        if len(self._vehicle_ids) > 1:
            self._plotting_data['d'] = {}
            self._plotting_data['d']['d_min'] = \
                (self._axs[3].plot([], [], lw=2, label=f'd_min')[0], [], [])
            self._plotting_data['d']['d_ref'] = \
                (self._axs[3].plot([], [], lw=2, label=f'd_ref')[0], [], [])
            for index, vehicle_id in enumerate(self._vehicle_ids[:-1]):
                vehicle_id_next: int = self._vehicle_ids[index + 1]
                pair_str: str = f'd_{vehicle_id}{vehicle_id_next}'
                self._plotting_data['d'][pair_str] = \
                    (self._axs[3].plot([], [], lw=2, label=pair_str)[0], [], [])

                self._axs[3].legend()
                self._axs[3].set_ylim(0, 1)
                self._axs[3].set_xlim(0, 60)
                self._axs[3].set_title('Distance between pairs of vehicles over Time')
                self._axs[3].set_xlabel('Time [s]')
                self._axs[3].set_ylabel('Distance d [m]')
            
            self._position_cost_cur_text = None
            self._position_cost_acc_text = None
            self._position_cost_mea_text = None

        else:
            vehicle_id: int = self._vehicle_ids[0]
            # Vehicle Position
            self._plotting_data['s'] = {}
            self._plotting_data['s_ref'] = {}
            self._plotting_data['s'][str(vehicle_id)] = \
                (self._axs[3].plot([], [], lw=2, label=f's')[0], [], [])
            self._plotting_data['s_ref'][str(vehicle_id)] = \
                (self._axs[3].plot([], [], lw=2, label=f's_ref')[0], [], [])

            # Current position cost
            self._position_cost_cur_text = self._axs[3].text(
                0.02, 0.95, '',
                transform=self._axs[3].transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
            )
            # Accumulated position cost
            self._position_cost_acc_text = self._axs[3].text(
                0.02, 0.8, '',
                transform=self._axs[3].transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
            )
            # Mean position cost
            self._position_cost_mea_text = self._axs[3].text(
                0.02, 0.65, '',
                transform=self._axs[3].transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
            )
            
            self._axs[3].legend()
            self._axs[3].set_ylim(0, 200)
            self._axs[3].set_xlim(0, 60)
            self._axs[3].set_title('Position of vehicle over Time')
            self._axs[3].set_xlabel('Time [s]')
            self._axs[3].set_ylabel('Position s on path [m]')

            # Reference generation (for PMPC only)
            self._y_ref: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda y, t: np.ravel(
                np.stack((1.1 * t + 0.5 * np.sin(t) + y[0], 1.1 + 0.5 * np.cos(t)), axis=1)).flatten()
            self._s0: float | None = None
            self._position_cost_acc: float = 0.0

    def plot(self):
        # Start plotting
        plt.tight_layout()  # Adjust layout to prevent overlap of subplots
        self._animation: FuncAnimation = FuncAnimation(self._fig, self._update_plots, blit=True,
                                                       interval=50)
        plt.show()

    def _on_vehicle_command(self, msg: VehicleCommand) -> None:
        with self._mutex:
            if self._start_time is None:
                self._start_time = self.get_clock().now()
            self._vehicle_commands.append(msg)

    def _on_vehicle_state_list(self, msg: VehicleStateList):
        with self._mutex:
            if self._start_time is not None:
                self._vehicle_state_lists.append(msg)

    def _plot_input_velocity(self):
        relative_time_s: float | None = None
        for vehicle_command in self._vehicle_commands:
            vehicle_id: str = vehicle_command.header.frame_id
            vehicle_id_int: int
            try:
                vehicle_id_int: int = int(vehicle_id)
            except ValueError:
                print(f"Non-integer frame id {vehicle_id} are not supported!")
                continue

            if vehicle_id_int not in self._vehicle_ids:
                print(f"Vehicle id {vehicle_id} not in specified vehicle ids!")
                continue

            relative_time_s: float = (Time.from_msg(
                vehicle_command.header.stamp) - self._start_time).nanoseconds * 1e-9
            input_velocity: float = vehicle_command.speed
            self._plotting_data['v_in'][str(vehicle_id)][1].append(relative_time_s)
            self._plotting_data['v_in'][str(vehicle_id)][2].append(input_velocity)
            self._plotting_data['v_in'][str(vehicle_id)][0].set_data(
                self._plotting_data['v_in'][str(vehicle_id)][1],
                self._plotting_data['v_in'][str(vehicle_id)][2])

        # Update min/max/ref
        if relative_time_s is not None:
            # v_min
            self._plotting_data['v_in']['v_min'][1].append(relative_time_s)
            self._plotting_data['v_in']['v_min'][2].append(0.0)
            self._plotting_data['v_in']['v_min'][0].set_data(
                self._plotting_data['v_in']['v_min'][1],
                self._plotting_data['v_in']['v_min'][2])

            # v_max
            self._plotting_data['v_in']['v_max'][1].append(relative_time_s)
            self._plotting_data['v_in']['v_max'][2].append(1.5)
            self._plotting_data['v_in']['v_max'][0].set_data(
                self._plotting_data['v_in']['v_max'][1],
                self._plotting_data['v_in']['v_max'][2])

            # v0_ref
            if len(self._vehicle_ids) > 1:
                self._plotting_data['v_in']['v0_ref'][1].append(relative_time_s)
                self._plotting_data['v_in']['v0_ref'][2].append(self._v0_reference(relative_time_s))
                self._plotting_data['v_in']['v0_ref'][0].set_data(
                    self._plotting_data['v_in']['v0_ref'][1],
                    self._plotting_data['v_in']['v0_ref'][2])

    def _plot_output_velocity(self):
        for vehicle_state_list in self._vehicle_state_lists:
            relative_time_s: float = (
                    (vehicle_state_list.t_now - self._start_time.nanoseconds) * 1e-9)
            for vehicle_state in vehicle_state_list.state_list:
                vehicle_state: VehicleState
                vehicle_id: int = vehicle_state.vehicle_id
                if vehicle_id not in self._vehicle_ids:
                    print(f"Vehicle id {vehicle_id} not in specified vehicle ids!")
                    continue

                output_velocity: float = vehicle_state.speed
                self._plotting_data['v_out'][str(vehicle_id)][1].append(relative_time_s)
                self._plotting_data['v_out'][str(vehicle_id)][2].append(output_velocity)
                self._plotting_data['v_out'][str(vehicle_id)][0].set_data(
                    self._plotting_data['v_out'][str(vehicle_id)][1],
                    self._plotting_data['v_out'][str(vehicle_id)][2])

            # Update min/max/ref
            # v_min
            self._plotting_data['v_out']['v_min'][1].append(relative_time_s)
            self._plotting_data['v_out']['v_min'][2].append(0.0)
            self._plotting_data['v_out']['v_min'][0].set_data(
                self._plotting_data['v_out']['v_min'][1],
                self._plotting_data['v_out']['v_min'][2])

            # v_max
            self._plotting_data['v_out']['v_max'][1].append(relative_time_s)
            self._plotting_data['v_out']['v_max'][2].append(1.5)
            self._plotting_data['v_out']['v_max'][0].set_data(
                self._plotting_data['v_out']['v_max'][1],
                self._plotting_data['v_out']['v_max'][2])

            # v0_ref
            if len(self._vehicle_ids) > 1:
                self._plotting_data['v_out']['v0_ref'][1].append(relative_time_s)
                self._plotting_data['v_out']['v0_ref'][2].append(
                    self._v0_reference(relative_time_s))
                self._plotting_data['v_out']['v0_ref'][0].set_data(
                    self._plotting_data['v_out']['v0_ref'][1],
                    self._plotting_data['v_out']['v0_ref'][2])

    def _plot_input_acceleration(self):
        relative_time_s: float | None = None
        for vehicle_id in self._vehicle_ids:
            if len(self._plotting_data['v_in'][str(vehicle_id)][1]) >= 2:
                dt: float = self._plotting_data['v_in'][str(vehicle_id)][1][-1] - \
                            self._plotting_data['v_in'][str(vehicle_id)][1][-2]
                dv: float = self._plotting_data['v_in'][str(vehicle_id)][2][-1] - \
                            self._plotting_data['v_in'][str(vehicle_id)][2][-2]
                input_acceleration: float = dv / dt
                relative_time_s: float = self._plotting_data['v_in'][str(vehicle_id)][1][-1]
                self._plotting_data['a_in'][str(vehicle_id)][1].append(relative_time_s)
                self._plotting_data['a_in'][str(vehicle_id)][2].append(input_acceleration)
                self._plotting_data['a_in'][str(vehicle_id)][0].set_data(
                    self._plotting_data['a_in'][str(vehicle_id)][1],
                    self._plotting_data['a_in'][str(vehicle_id)][2])

        if relative_time_s is not None:
            # Update min/max/ref
            # a_min
            self._plotting_data['a_in']['a_min'][1].append(relative_time_s)
            self._plotting_data['a_in']['a_min'][2].append(-1.0)
            self._plotting_data['a_in']['a_min'][0].set_data(
                self._plotting_data['a_in']['a_min'][1],
                self._plotting_data['a_in']['a_min'][2])

            # a_max
            self._plotting_data['a_in']['a_max'][1].append(relative_time_s)
            self._plotting_data['a_in']['a_max'][2].append(0.5)
            self._plotting_data['a_in']['a_max'][0].set_data(
                self._plotting_data['a_in']['a_max'][1],
                self._plotting_data['a_in']['a_max'][2])

    def _plot_distance_between_vehicles(self):
        # Limits
        d_min: float = 0.3
        d_ref: float = 0.5

        path: List[PathPointInternal] = outer_lane_path()

        for vehicle_state_list in self._vehicle_state_lists:
            vehicle_state_list: VehicleStateList
            relative_time_s: float = (
                    (vehicle_state_list.t_now - self._start_time.nanoseconds) * 1e-9)

            s_vehicles: List[Tuple[int, float]] = []

            vehicle_state_list_internal: VehicleStateListInternal = VehicleStateListInternal(
                {}, vehicle_state_list.t_now,
                vehicle_state_list.period_ms)

            for vehicle_state in vehicle_state_list.state_list:
                vehicle_state: VehicleState
                vehicle_id: int = vehicle_state.vehicle_id
                if vehicle_id not in self._vehicle_ids:
                    print(f"Vehicle id {vehicle_id} not in specified vehicle ids!")
                    continue
                x: float = vehicle_state.pose.x
                y: float = vehicle_state.pose.y
                theta: float = vehicle_state.pose.theta
                speed: float = vehicle_state.speed

                vehicle_state_internal: VehicleStateInternal = VehicleStateInternal(vehicle_id, x,
                                                                                    y,
                                                                                    theta, speed)

                vehicle_state_list_internal.vehicle_states[vehicle_id] = vehicle_state_internal

            result: np.ndarray = self._measurement_transformer.measure_longitudinal(
                vehicle_state_list_internal)

            for index, vehicle_id in enumerate(vehicle_state_list_internal.vehicle_states.keys()):
                s_vehicles.append((vehicle_id, float(result[index * 2])))

            for index, (vehicle_id, s) in enumerate(s_vehicles[:-1]):
                vehicle_id_next: int = s_vehicles[index + 1][0]
                s_next: float = s_vehicles[index + 1][1]

                rel_distance: float = compute_relative_distance_on_path(path, s, s_next)

                pair_str: str = f'd_{vehicle_id}{vehicle_id_next}'

                self._plotting_data['d'][pair_str][1].append(relative_time_s)
                self._plotting_data['d'][pair_str][2].append(rel_distance)
                self._plotting_data['d'][pair_str][0].set_data(
                    self._plotting_data['d'][pair_str][1],
                    self._plotting_data['d'][pair_str][2])

            self._plotting_data['d']['d_min'][1].append(relative_time_s)
            self._plotting_data['d']['d_min'][2].append(d_min)
            self._plotting_data['d']['d_min'][0].set_data(
                self._plotting_data['d']['d_min'][1],
                self._plotting_data['d']['d_min'][2])
            self._plotting_data['d']['d_ref'][1].append(relative_time_s)
            self._plotting_data['d']['d_ref'][2].append(d_ref)
            self._plotting_data['d']['d_ref'][0].set_data(
                self._plotting_data['d']['d_ref'][1],
                self._plotting_data['d']['d_ref'][2])

    def _plot_position(self):
        for vehicle_state_list in self._vehicle_state_lists:
            vehicle_state: VehicleState = vehicle_state_list.state_list[0]
            vehicle_id: int = vehicle_state.vehicle_id
            if vehicle_id not in self._vehicle_ids:
                print(f"Vehicle id {vehicle_id} not in specified vehicle ids!")
                continue

            relative_time_s: float = (
                    (vehicle_state_list.t_now - self._start_time.nanoseconds) * 1e-9)
            x: float = vehicle_state.pose.x
            y: float = vehicle_state.pose.y
            theta: float = vehicle_state.pose.theta
            speed: float = vehicle_state.speed

            vehicle_state_internal: VehicleStateInternal = VehicleStateInternal(vehicle_id, x, y,
                                                                                theta, speed)
            vehicle_state_internal.x = x
            vehicle_state_internal.y = y
            vehicle_state_list_internal: VehicleStateListInternal = VehicleStateListInternal(
                {vehicle_id: vehicle_state_internal}, vehicle_state_list.t_now,
                vehicle_state_list.period_ms)

            s: float = float(self._measurement_transformer.measure_longitudinal(
                vehicle_state_list_internal)[0])

            if self._s0 is None:
                self._s0 = s

            s_ref: float = float(self._y_ref(np.array([self._s0]), np.array([relative_time_s]))[0])

            self._plotting_data['s'][str(vehicle_id)][1].append(relative_time_s)
            self._plotting_data['s'][str(vehicle_id)][2].append(s)
            self._plotting_data['s'][str(vehicle_id)][0].set_data(
                self._plotting_data['s'][str(vehicle_id)][1],
                self._plotting_data['s'][str(vehicle_id)][2])

            self._plotting_data['s_ref'][str(vehicle_id)][1].append(relative_time_s)
            self._plotting_data['s_ref'][str(vehicle_id)][2].append(s_ref)
            self._plotting_data['s_ref'][str(vehicle_id)][0].set_data(
                self._plotting_data['s_ref'][str(vehicle_id)][1],
                self._plotting_data['s_ref'][str(vehicle_id)][2])
            
            n_data_points = len(self._plotting_data['s'][str(vehicle_id)][1])
            position_cost_cur = compute_relative_distance_on_path(outer_lane_path(), s,
                                                                     s_ref) ** 2
            self._position_cost_acc += position_cost_cur
            self._position_cost_cur_text.set_text(f'Current position cost: {position_cost_cur:.2f}')
            self._position_cost_acc_text.set_text(f'Accumulated position cost: {self._position_cost_acc:.2f}')
            self._position_cost_mea_text.set_text(f'Mean position cost: {(self._position_cost_acc / n_data_points):.2f}')

    def _update_plots(self, _: int) -> List[Line2D]:
        with self._mutex:
            if self._start_time is not None:

                self._plot_input_velocity()
                self._plot_output_velocity()
                self._plot_input_acceleration()

                if len(self._vehicle_ids) > 1:
                    self._plot_distance_between_vehicles()
                else:
                    self._plot_position()

                self._vehicle_state_lists = []
                self._vehicle_commands = []
        
        updated_elements = [
            self._plotting_data[plot][line][0]
            for plot in self._plotting_data.keys()
            for line in self._plotting_data[plot].keys()
        ]
        
        if len(self._vehicle_ids) == 1:
            updated_elements.extend(
                [
                    self._position_cost_cur_text,
                    self._position_cost_acc_text,
                    self._position_cost_mea_text,
                ]
            )

        return updated_elements if updated_elements else []

def main(args=None):
    rclpy.init(args=args)
    plot_platooning_node: PlotPlatooningNode = PlotPlatooningNode()

    try:
        plot_platooning_node.get_logger().info('Started plotting')
        executor: MultiThreadedExecutor = MultiThreadedExecutor()
        executor.add_node(plot_platooning_node)
        thread: Thread = Thread(target=executor.spin, daemon=True)
        thread.start()
        plot_platooning_node.plot()

    except KeyboardInterrupt:
        plot_platooning_node.get_logger().info('KeyboardInterrupt caught! Shutting down...')

    finally:
        plot_platooning_node.get_logger().info('Stopping plotting')

        try:
            file_name = "plot-platooning.png"
            plot_platooning_node._fig.savefig(file_name)
            plot_platooning_node.get_logger().info(f'The figure was saved at: {file_name}.')
        except Exception as e:
            plot_platooning_node.get_logger().error(f'Failed to save the figure: {e}.')
            
        plot_platooning_node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
