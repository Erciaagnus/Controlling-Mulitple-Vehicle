from typing import List, Dict, Type

from rclpy import Parameter

from cpnav.utils import outer_lane_path
import rclpy
from cpm_lab_lab_msgs.msg import SystemTrigger, VehicleStateList, VehicleCommand, PathPoint, \
    VehicleState
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.time import Time, Duration

from cpnav.hlcs import InterfaceHlc, IdentificationHlc, PmpcHlc, CmpcHlc, DmpcHlc
from rclpy.node import Node
from cpnav.internal_types import VehicleStateList as VehicleStateListInternal, \
    VehicleState as VehicleStateInternal, PathPoint as PathPointInternal


def convert_vehicle_state_list(msg: VehicleStateList) -> VehicleStateListInternal:
    vehicle_state_dict: Dict[int, VehicleStateInternal] = {}
    vehicle_state: VehicleState
    for vehicle_state in msg.state_list:
        vehicle_state_internal: VehicleStateInternal = VehicleStateInternal(
            vehicle_state.vehicle_id, vehicle_state.pose.x, vehicle_state.pose.y,
            vehicle_state.pose.theta, vehicle_state.speed)
        vehicle_state_dict[vehicle_state.vehicle_id] = vehicle_state_internal

    return VehicleStateListInternal(vehicle_state_dict, msg.t_now, msg.period_ms)


def convert_path(path_internal: List[PathPointInternal]) -> List[PathPoint]:
    path: List[PathPoint] = []
    for path_point_internal in path_internal:
        path_point: PathPoint = PathPoint()
        path_point.pose.x = path_point_internal.x
        path_point.pose.y = path_point_internal.y
        path_point.pose.theta = path_point_internal.yaw
        path_point.s = path_point_internal.s
        path.append(path_point)

    return path


class HlcNode(Node):
    def __init__(self, hlc: Type[InterfaceHlc], t_s: float):
        super().__init__('hlc_node', namespace='/planner')

        # Parameters
        self.declare_parameter('vehicle_ids', rclpy.Parameter.Type.INTEGER_ARRAY)

        # Retrieve and validate the parameter
        param = self.get_parameter('vehicle_ids')

        if param.type_ != Parameter.Type.INTEGER_ARRAY:
            raise ValueError(
                "The 'vehicle_ids' parameter is not an integer array. Using an empty list.")
        else:
            try:
                vehicle_ids = param.get_parameter_value().integer_array_value
                if not all(isinstance(id, int) for id in vehicle_ids):
                    raise ValueError("All elements in 'vehicle_ids' must be integers.")
            except Exception as e:
                raise ValueError(f"Error with 'vehicle_ids': {e}")

        self._is_first_time_step: bool = True
        self._hlc: InterfaceHlc = hlc(vehicle_ids, t_s)
        self._vehicle_ids: List[int] = vehicle_ids
        self._path_internal: List[PathPointInternal] = outer_lane_path()
        self._path_external: List[PathPoint] = convert_path(self._path_internal)
        self._system_trigger_publisher: Publisher = self.create_publisher(
            SystemTrigger, 'system_trigger', 10)
        self._system_trigger_subscription: Subscription = self.create_subscription(
            SystemTrigger, 'system_trigger', self._on_system_trigger, 10)
        self._vehicle_command_publisher: Publisher = self.create_publisher(
            VehicleCommand, 'command', 10)
        self._vehicle_state_list_subscription: Subscription = self.create_subscription(
            VehicleStateList, 'vehicle_states', self._on_vehicle_state_list, 10)

    def _on_system_trigger(self, msg: SystemTrigger):
        # Check if message is ready request command
        if msg.type == SystemTrigger.READY_REQUEST:
            # Resend ready signal as it might have been missed
            self.get_logger().info('Resending ready signal.')
            self._send_ready_signal()

        # Check if message is stop command
        if msg.type == SystemTrigger.STOP:
            # Stop planner is stopped because lab requested it
            self.get_logger().info('Stopping planner.')
            rclpy.shutdown()

    def _send_ready_signal(self):
        msg: SystemTrigger = SystemTrigger()
        msg.type = SystemTrigger.READY
        msg.stamp = self.get_clock().now().to_msg()
        for vehicle_id in self._vehicle_ids:
            msg.source_id = "planner_" + str(vehicle_id)
            self._system_trigger_publisher.publish(msg)

    def _on_vehicle_state_list(self, vehicle_state_list: VehicleStateList):
        # Convert message to internal format
        vehicle_state_list_internal: VehicleStateListInternal = convert_vehicle_state_list(
            vehicle_state_list)

        if self._is_first_time_step:
            self._t_start_ns: int = vehicle_state_list.t_now
            dt_period_ns: int = int(vehicle_state_list.period_ms * 1e6)
            dt_max_computation_communication_ns: int = int(300e6 + 100e6)  # multiples of dt_period
            self._dt_valid_after_ns: int = max(dt_max_computation_communication_ns, dt_period_ns)

            self._hlc.on_first_timestep(vehicle_state_list_internal)

            self._is_first_time_step = False

        self._t_expired_s: float = (vehicle_state_list.t_now - self._t_start_ns) * 1e-9

        # Measure time of HLC execution step
        # Start
        start_time: Time = self.get_clock().now()

        # Execute HLC
        speeds: Dict[int, float] = self._hlc.on_each_timestep(vehicle_state_list_internal,
                                                              self._t_expired_s)
        # End time
        end_time: Time = self.get_clock().now()

        # Calculate elapsed time
        elapsed_time: Duration = end_time - start_time
        self.get_logger().info(
            f"Elapsed time for update: {elapsed_time.nanoseconds * 1e-6:.4f} milliseconds")
        for vehicle_id in self._vehicle_ids:
            vehicle_command: VehicleCommand = VehicleCommand()
            vehicle_command.header.stamp = self.get_clock().now().to_msg()
            vehicle_command.header.frame_id = str(vehicle_id)
            vehicle_command.type = VehicleCommand.PATH_TRACKING
            vehicle_command.path = self._path_external
            vehicle_command.speed = float(speeds[vehicle_id])
            vehicle_command.valid_after_stamp = (self.get_clock().now() + Duration(
                nanoseconds=self._dt_valid_after_ns)).to_msg()
            self._vehicle_command_publisher.publish(vehicle_command)


def main(args=None):
    ### INSERT CORRECT HLC AND Ts ###

    t_s: float = 0.4  # s

    # hlc: Type[IdentificationHlc] = IdentificationHlc
    #hlc: Type[PmpcHlc] = PmpcHlc
    #hlc: Type[CmpcHlc] = CmpcHlc
    hlc: Type[DmpcHlc] = DmpcHlc

    ### DO NOT CHANGE ANYTHING FROM HERE ###

    rclpy.init(args=args)
    hlc_node: HlcNode = HlcNode(hlc, t_s)

    try:
        hlc_node.get_logger().info(f'Starting planner {hlc.__name__}')
        
        rclpy.spin(hlc_node)

    except KeyboardInterrupt:
        hlc_node.get_logger().info('KeyboardInterrupt caught! Shutting down...')

    finally:
        hlc_node.get_logger().info('Stopping planner')

        hlc_node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
