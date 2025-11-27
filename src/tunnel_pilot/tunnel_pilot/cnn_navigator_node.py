#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import onnxruntime as ort
import os
from ament_index_python.packages import get_package_share_directory

TOP_LIDAR_TOPIC = '/lidar_top/scan'
BOT_LIDAR_TOPIC = '/lidar_bottom/scan'
IMU_TOPIC = '/mavros/imu/data'
CMD_TOPIC = '/mavros/setpoint_velocity/cmd_vel'

LIDAR_FOV_V = 45.0 * (np.pi / 180.0)
IMG_H, IMG_W = 60, 80
MAX_Y = 4.0
MAX_Z = 3.0
MAX_X = 10.0


class CNNNavigator(Node):
    def __init__(self):
        super().__init__('cnn_navigator')

        # Load ONNX Model
        pkg_share = get_package_share_directory('tunnel_pilot')
        model_path = os.path.join(pkg_share, 'models', 'tunnel_cnn.onnx')

        self.get_logger().info(f'Loading Brain: {model_path}')
        try:
            self.session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.get_logger().info('Brain Loaded!')
        except Exception as e:
            self.get_logger().error(f'Model Load Failed: {e}')
            return

        self.scan_top = None
        self.scan_bot = None
        self.current_roll = 0.0
        self.current_pitch = 0.0

        # Subscribers
        self.create_subscription(
            LaserScan, TOP_LIDAR_TOPIC, self.top_cb, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, BOT_LIDAR_TOPIC, self.bot_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, IMU_TOPIC, self.imu_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)

        # Control Timer( freq 20 Hz lidar freq is 10 Hz )
        self.create_timer(0.05, self.control_loop)

    def top_cb(self, msg):
        self.scan_top = msg.ranges

    def bot_cb(self, msg):
        self.scan_bot = msg.ranges

    def imu_cb(self, msg):
        q = msg.orientation
        # Quaternion -> Euler (Roll/Pitch)
        # Roll - Phi | Pitch - Theta | Yaw - Psi
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        self.current_roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            self.current_pitch = np.sign(sinp) * (np.pi / 2)
        else:
            self.current_pitch = np.arcsin(sinp)

    def resize_scan(self, ranges, target_size=360):
        """Resize a single scan to exactly 360 points"""
        ranges = np.array(ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0

        x_old = np.linspace(0, 1, len(ranges))
        x_new = np.linspace(0, 1, target_size)
        return np.interp(x_new, x_old, ranges).astype(np.float32)

    def preprocess(self, ranges_top, ranges_bot, roll, pitch):
        """
        Combines Top(360) + Bot(360) -> 720 -> Image
        """
        angles = np.linspace(-np.pi, np.pi, 360)

        # Top Sensor (+45 deg)
        x_t = ranges_top * np.cos(angles) * np.cos(LIDAR_FOV_V)
        y_t = ranges_top * np.sin(angles)
        z_t = ranges_top * np.cos(angles) * np.sin(LIDAR_FOV_V)

        # Bottom Sensor (-45 deg)
        x_b = ranges_bot * np.cos(angles) * np.cos(-LIDAR_FOV_V)
        y_b = ranges_bot * np.sin(angles)
        z_b = ranges_bot * np.cos(angles) * np.sin(-LIDAR_FOV_V)

        # Concatenate to form the full cloud
        x_final = np.concatenate([x_t, x_b])  # Depth
        y = np.concatenate([y_t, y_b])
        z = np.concatenate([z_t, z_b])

        # Stabilize (Rotation Matrix)
        # Pitch (Y-axis rotation)
        cp, sp = np.cos(-pitch), np.sin(-pitch)
        # Roll (X-axis rotation)
        cr, sr = np.cos(-roll), np.sin(-roll)

        # Apply rotations
        # 1. Apply Pitch
        x_p = x_final * cp + z * sp
        z_p = -x_final * sp + z * cp
        # 2. Apply Roll
        y_final = y * cr - z_p * sr
        z_final = y * sr + z_p * cr

        # Project to Image
        u = ((y_final / MAX_Y) * (IMG_W / 2) + (IMG_W / 2)).astype(int)
        v = (-(z_final / MAX_Z) * (IMG_H / 2) + (IMG_H / 2)).astype(int)

        img = np.zeros((3, IMG_H, IMG_W), dtype=np.float32)
        valid = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        u, v = u[valid], v[valid]

        img[0, v, u] = np.clip(x_p[valid] / MAX_X, 0, 1)
        img[1, v, u] = (pitch + 0.35) / 0.7
        img[2, v, u] = (roll + 0.35) / 0.7

        return np.expand_dims(img, axis=0)

    def control_loop(self):
        # Check if we have data from both eyes
        if self.scan_top is None or self.scan_bot is None:
            self.get_logger().warn('Waiting for LiDARs...', throttle_duration_sec=2.0)
            return

        start_t = self.get_clock().now()

        # 1. Resize both to 360 points (Standardize)
        r_top = self.resize_scan(self.scan_top, 360)
        r_bot = self.resize_scan(self.scan_bot, 360)

        # 2. Preprocess (Merge & Paint)
        input_tensor = self.preprocess(
            r_top, r_bot, self.current_roll, self.current_pitch)

        # 3. Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predicted_yaw = outputs[0][0][0]

        # 4. Act
        twist = Twist()
        twist.angular.z = float(predicted_yaw) * 1.5
        twist.linear.x = 0.5
        self.cmd_pub.publish(twist)

        # Logging speed
        dt = (self.get_clock().now() - start_t).nanoseconds / 1e6
        # self.get_logger().info(f'Yaw: {predicted_yaw:.3f} | Inference: {dt:.1f}ms')


def main(args=None):
    rclpy.init(args=args)
    node = CNNNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
