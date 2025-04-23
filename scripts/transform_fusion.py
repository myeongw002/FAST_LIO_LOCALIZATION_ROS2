#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import copy
import struct

import open3d as o3d

from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# ROS2 tf2_ros 사용
import tf_transformations  # pip install tf-transformations
import tf2_ros

# PointCloud2 메시지 생성을 위한 라이브러리 (sensor_msgs_py 필요)
from sensor_msgs_py import point_cloud2

class TransformFusionNode(Node):
    def __init__(self):
        super().__init__('transform_fusion')
        self.FREQ_PUB_LOCALIZATION = 50.0

        # 파라미터 선언: map_file_path 파라미터가 있으면 해당 PCD 파일을 로드하여 맵으로 사용
        self.declare_parameter('map_file_path', '')
        map_file_path = self.get_parameter('map_file_path').value

        self.global_map = None
        if map_file_path != "":
            try:
                self.global_map = o3d.io.read_point_cloud(map_file_path)
                self.get_logger().info(f"Loaded map from: {map_file_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load map from {map_file_path}: {e}")
        else:
            self.get_logger().info("No map_file_path provided; global_map is not loaded.")

        # Publisher for map publishing (5초 주기)
        self.pub_map = self.create_publisher(PointCloud2, '/global_map', 10)

        # Subscribers for odometry and map_to_odom
        self.create_subscription(
            Odometry,
            '/Odometry',
            self.cb_save_cur_odom,
            10
        )
        self.create_subscription(
            Odometry,
            '/map_to_odom',
            self.cb_save_map_to_odom,
            10
        )

        # Publisher for localization message
        self.pub_localization = self.create_publisher(Odometry, '/localization', 10)

        # TF 브로드캐스터 (ROS2 tf2_ros 사용)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 타이머: 50Hz 주기로 변환 융합 및 발행 수행
        self.create_timer(1.0 / self.FREQ_PUB_LOCALIZATION, self.transform_fusion_callback)
        # 타이머: 5초 주기로 글로벌 맵 퍼블리시
        self.create_timer(5.0, self.publish_map_callback)

        self.get_logger().info('Transform Fusion Node Inited...')

        # 최신 odometry, 전역 변환 정보를 저장할 변수들
        self.cur_odom_to_baselink = None
        self.cur_map_to_odom = None

    def pose_to_mat(self, pose_msg: Odometry):
        """
        Odometry 메시지의 pose를 4x4 변환 행렬로 변환.
        """
        pos = pose_msg.pose.pose.position
        ori = pose_msg.pose.pose.orientation
        trans_mat = tf_transformations.translation_matrix([pos.x, pos.y, pos.z])
        rot_mat = tf_transformations.quaternion_matrix([ori.x, ori.y, ori.z, ori.w])
        return np.matmul(trans_mat, rot_mat)

    def transform_fusion_callback(self):
        # cur_map_to_odom가 존재하면 변환 행렬을 계산, 없으면 단위행렬 사용
        if self.cur_map_to_odom is not None:
            T_map_to_odom = self.pose_to_mat(self.cur_map_to_odom)
        else:
            T_map_to_odom = np.eye(4)

        # odometry 정보 복사 (스레드 안전을 위해 deepcopy 사용)
        if self.cur_odom_to_baselink is not None:
            cur_odom = copy.deepcopy(self.cur_odom_to_baselink)
        else:
            cur_odom = None

        # TF 브로드캐스트: map -> camera_init 변환 발행
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera_init'
        translation = tf_transformations.translation_from_matrix(T_map_to_odom)
        rotation = tf_transformations.quaternion_from_matrix(T_map_to_odom)
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]
        self.tf_broadcaster.sendTransform(t)

        # cur_odom이 존재할 경우 localization 메시지 생성 및 발행
        if cur_odom is not None:
            T_odom_to_base_link = self.pose_to_mat(cur_odom)
            T_map_to_base_link = np.matmul(T_map_to_odom, T_odom_to_base_link)
            xyz = tf_transformations.translation_from_matrix(T_map_to_base_link)
            quat = tf_transformations.quaternion_from_matrix(T_map_to_base_link)

            localization = Odometry()
            
            # ROS2에서는 Point 생성 후 속성 할당
            p = Point()
            p.x = xyz[0]
            p.y = xyz[1]
            p.z = xyz[2]
            localization.pose.pose.position = p
            
            # Quaternion의 경우에도 속성 할당 방식 사용 가능
            q = Quaternion()
            q.x = quat[0]
            q.y = quat[1]
            q.z = quat[2]
            q.w = quat[3]
            localization.pose.pose.orientation = q
            
            localization.twist = cur_odom.twist
            localization.header.stamp = cur_odom.header.stamp
            localization.header.frame_id = 'map'
            localization.child_frame_id = 'body'
            self.pub_localization.publish(localization)


    def publish_map_callback(self):
        """
        5초마다 global_map (PCD 파일에서 읽은 맵)이 존재할 경우, 이를 PointCloud2 메시지로 변환하여 퍼블리시.
        """
        if self.global_map is None:
            self.get_logger().warn("No global map loaded; skipping map publish.")
            return

        # Open3D 포인트 클라우드를 NumPy 배열로 변환
        map_points = np.asarray(self.global_map.points)
        if map_points.size == 0:
            self.get_logger().warn("Global map is empty; skipping publish.")
            return

        # std_msgs.msg.Header를 사용하여 헤더 생성
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        
        # 필드 정의 (x,y,z: float32)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        # NumPy 배열을 리스트로 변환하여 cloud_msg 생성
        cloud_msg = point_cloud2.create_cloud(header, fields, map_points.tolist())
        self.pub_map.publish(cloud_msg)
        self.get_logger().info("Published global map.")


    def cb_save_cur_odom(self, msg: Odometry):
        self.cur_odom_to_baselink = msg

    def cb_save_map_to_odom(self, msg: Odometry):
        self.cur_map_to_baselink = msg
        self.cur_map_to_baselink = msg  # 기존에 저장된 최신 맵 변환 정보

def main(args=None):
    rclpy.init(args=args)
    node = TransformFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
