#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import copy
import struct
import time

import open3d as o3d

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
from builtin_interfaces.msg import Time

# TF 관련 모듈 (ROS2 tf2_ros 사용)
import tf_transformations  # pip install tf-transformations
import tf2_ros

# PointCloud2 생성용 (ROS2의 경우 sensor_msgs_py 패키지 활용 가능)
from sensor_msgs_py import point_cloud2

class GlobalLocalizationNode(Node):
    def __init__(self):
        super().__init__('global_localization')
        
        # 상수 파라미터 (필요 시 ROS2 파라미터로 설정 가능)
        self.declare_parameter('map_voxel_size', 0.5)
        self.declare_parameter('scan_voxel_size', 0.5)
        self.declare_parameter('freq_localization', 1)  # Hz
        self.declare_parameter('localization_th', 0.9)
        self.declare_parameter('fov', 2 * np.pi)           # radian; 전체 시야
        self.declare_parameter('fov_far', 100)             # 최대 거리

        # 파라미터 값 할당
        self.MAP_VOXEL_SIZE = self.get_parameter('map_voxel_size').value
        self.SCAN_VOXEL_SIZE = self.get_parameter('scan_voxel_size').value
        self.FREQ_LOCALIZATION = self.get_parameter('freq_localization').value
        self.LOCALIZATION_TH = self.get_parameter('localization_th').value
        self.FOV = self.get_parameter('fov').value
        self.FOV_FAR = self.get_parameter('fov_far').value

        self.get_logger().info(f"Parameters: MAP_VOXEL_SIZE={self.MAP_VOXEL_SIZE}, "
                               f"SCAN_VOXEL_SIZE={self.SCAN_VOXEL_SIZE}, "
                               f"FREQ_LOCALIZATION={self.FREQ_LOCALIZATION}, "
                               f"LOCALIZATION_TH={self.LOCALIZATION_TH}, "
                               f"FOV={self.FOV}, FOV_FAR={self.FOV_FAR}")
        
        # 변수 초기화
        self.global_map = None
        self.initialized = False
        self.T_map_to_odom = np.eye(4)
        self.cur_odom = None
        self.cur_scan = None

        # 퍼블리셔 생성
        self.pub_pc_in_map = self.create_publisher(PointCloud2, '/cur_scan_in_map', 10)
        self.pub_submap = self.create_publisher(PointCloud2, '/submap', 10)
        self.pub_map_to_odom = self.create_publisher(Odometry, '/map_to_odom', 10)

        # 서브스크라이버 생성
        self.create_subscription(PointCloud2, '/cloud_registered', self.cb_save_cur_scan, 10)
        self.create_subscription(Odometry, '/Odometry', self.cb_save_cur_odom, 10)
        self.create_subscription(PointCloud2, '/global_map', self.cb_initialize_global_map, 10)

        # 타이머 설정 : FREQ_LOCALIZATION 주기로 전역 로컬라이제이션 수행
        self.create_timer(1.0 / self.FREQ_LOCALIZATION, self.localization_callback)

        # TF 브로드캐스터 (만약 로컬라이제이션 결과를 TF로도 브로드캐스트 하고 싶다면 별도 노드에서 수행)
        self.get_logger().info("Global Localization Node Initialized.")

    def pose_to_mat(self, pose_msg):
        """
        Pose 메시지(Pose 혹은 Odometry 내부의 pose)를 4x4 변환 행렬로 변환.
        """
        pos = pose_msg.pose.pose.position
        ori = pose_msg.pose.pose.orientation
        trans = tf_transformations.translation_matrix([pos.x, pos.y, pos.z])
        rot = tf_transformations.quaternion_matrix([ori.x, ori.y, ori.z, ori.w])
        return np.matmul(trans, rot)

    def msg_to_array(self, pc_msg: PointCloud2):
        """
        PointCloud2 메시지를 NumPy 배열로 변환.
        단, 필드 순서가 ['x','y','z']로 되어 있다고 가정.
        """
        # 각 포인트의 바이트 수
        point_step = pc_msg.point_step
        num_points = int(len(pc_msg.data) / point_step)
        fmt = 'fff'  # x,y,z : float32
        # 포맷 길이 계산 (4바이트 * 3 = 12바이트)
        fmt_size = struct.calcsize(fmt)
        
        points = []
        for i in range(num_points):
            offset = i * point_step
            # 추출: x, y, z 는 앞의 12바이트라고 가정
            x, y, z = struct.unpack_from(fmt, pc_msg.data, offset)
            points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def voxel_down_sample(self, pcd: o3d.geometry.PointCloud, voxel_size):
        try:
            return pcd.voxel_down_sample(voxel_size)
        except Exception as e:
            self.get_logger().warn(f'Voxel downsample failed: {e}')
            return pcd

    def registration_at_scale(self, pc_scan: o3d.geometry.PointCloud,
                              pc_map: o3d.geometry.PointCloud,
                              initial: np.ndarray,
                              scale: float):
        """
        ICP 기반 정합을 수행한다. 스케일에 따라 down sample voxel 크기를 조절.
        """
        source = self.voxel_down_sample(pc_scan, self.SCAN_VOXEL_SIZE * scale)
        target = self.voxel_down_sample(pc_map, self.MAP_VOXEL_SIZE * scale)
        
        reg = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=1.0 * scale,
            init=initial,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        )
        return reg.transformation, reg.fitness

    def inverse_se3(self, trans: np.ndarray):
        """
        SE(3) 변환의 역행렬 계산.
        """
        trans_inv = np.eye(4)
        trans_inv[:3, :3] = trans[:3, :3].T
        trans_inv[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inv

    def publish_point_cloud(self, publisher, header, pc: np.ndarray):
        """
        NumPy 배열 (Nx3 혹은 Nx4) 형태의 포인트 클라우드를 PointCloud2 메시지로 발행.
        """
        # 여기서는 sensor_msgs_py의 point_cloud2.create_cloud를 사용
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        # intensity 필드가 있다면 추가 (pc.shape[1]==4인 경우)
        if pc.shape[1] == 4:
            fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
        cloud_msg = point_cloud2.create_cloud(header, fields, pc.tolist())
        publisher.publish(cloud_msg)

    def crop_global_map_in_FOV(self, global_map: o3d.geometry.PointCloud,
                               pose_estimation: np.ndarray,
                               cur_odom: Odometry):
        """
        현재 스캔 원점(odometry 기준)에서 보이는 시야(FOV) 내의 전역 맵 포인트만 추출.
        """
        # 현재 odom의 transform (기본적으로 base_link → odom)
        T_odom_to_base_link = self.pose_to_mat(cur_odom)
        T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
        T_base_link_to_map = self.inverse_se3(T_map_to_base_link)

        # 전역 맵의 점들을 동차좌표로 변환
        global_map_array = np.asarray(global_map.points)
        ones = np.ones((global_map_array.shape[0], 1))
        global_map_homo = np.hstack((global_map_array, ones))
        global_map_in_base_link = (T_base_link_to_map @ global_map_homo.T).T

        # 시야 내 추출: FOV가 전체(2pi)인 경우 단순 거리 필터링, 그 외에는 각도도 고려
        if self.FOV > np.pi:
            indices = np.where(
                (global_map_in_base_link[:, 0] < self.FOV_FAR) &
                (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < self.FOV / 2.0)
            )[0]
        else:
            indices = np.where(
                (global_map_in_base_link[:, 0] > 0) &
                (global_map_in_base_link[:, 0] < self.FOV_FAR) &
                (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < self.FOV / 2.0)
            )[0]
        # 인덱스에 해당하는 포인트만 추출하여 새로운 포인트 클라우드 생성
        sub_map = o3d.geometry.PointCloud()
        sub_map.points = o3d.utility.Vector3dVector(global_map_array[indices])
        
        # fov 내 포인트 클라우드 발행 (샘플링: 1/10 간격)
        header = cur_odom.header
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        self.publish_point_cloud(self.pub_submap, header, np.asarray(sub_map.points)[::10])
        
        return sub_map

    def global_localization(self, pose_estimation: np.ndarray):
        if self.global_map is None or self.cur_scan is None or self.cur_odom is None:
            self.get_logger().warn("필요한 데이터가 아직 모두 수신되지 않았습니다.")
            return False

        # 스캔 데이터 복사 (스레드 안전)
        scan_tobe_mapped = copy.deepcopy(self.cur_scan)
        
        tic = time.time()
        # 현재 pose_estimation을 기반으로 global map의 관심 영역(Crop) 구하기
        global_map_in_FOV = self.crop_global_map_in_FOV(self.global_map, pose_estimation, self.cur_odom)
        
        # coarse registration (스케일 5)
        transformation, _ = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)
        # fine registration (스케일 1)
        transformation, fitness = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation, scale=1)
        toc = time.time()
        self.get_logger().info(f'Global localization ICP elapsed time: {toc - tic:.3f} sec')
        
        if fitness > self.LOCALIZATION_TH:
            self.T_map_to_odom = transformation
            # map_to_odom 메시지 발행
            odom_msg = Odometry()

            # 변환 행렬에서 위치와 쿼터니언 추출
            xyz = tf_transformations.translation_from_matrix(self.T_map_to_odom)
            quat = tf_transformations.quaternion_from_matrix(self.T_map_to_odom)
            
            # Point 객체 생성 후 값 할당
            p = Point()
            p.x = xyz[0]
            p.y = xyz[1]
            p.z = xyz[2]

            # Quaternion 객체 생성 후 값 할당
            q = Quaternion()
            q.x = quat[0]
            q.y = quat[1]
            q.z = quat[2]
            q.w = quat[3]

            # Pose 객체 구성
            odom_msg.pose.pose = Pose()
            odom_msg.pose.pose.position = p
            odom_msg.pose.pose.orientation = q

            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = 'map'
            self.pub_map_to_odom.publish(odom_msg)
            return True
        else:
            self.get_logger().warn("매칭 실패. fitness: {:.3f}".format(fitness))
        return False


    # 콜백 함수들
    def cb_save_cur_odom(self, msg: Odometry):
        self.cur_odom = msg

    def cb_save_cur_scan(self, msg: PointCloud2):
        # frame id 및 타임스탬프 업데이트
        msg.header.frame_id = 'camera_init'
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_pc_in_map.publish(msg)
        # 포인트 클라우드 변환 (여기서는 msg_to_array를 이용)
        pc_array = self.msg_to_array(msg)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_array)
        self.cur_scan = pcd

    def cb_initialize_global_map(self, msg: PointCloud2):
        if self.global_map is None:
            pc_array = self.msg_to_array(msg)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_array)
            self.global_map = self.voxel_down_sample(pcd, self.MAP_VOXEL_SIZE)
            self.get_logger().info("Global map 수신 완료.")

    def localization_callback(self):
        """
        주기적으로 현재 T_map_to_odom를 기반으로 전역 로컬라이제이션 수행.
        초기 pose (예: /initialpose를 통해 받는 pose)의 경우,
        단 한번 정합에 성공하면 initialized = True로 변경.
        """
        if self.cur_scan is None or self.cur_odom is None or self.global_map is None:
            self.get_logger().warn("데이터 미수신: global_map, cur_scan, 또는 cur_odom")
            return

        # 현재 T_map_to_odom를 초기 pose로 사용하여 정합 수행
        success = self.global_localization(self.T_map_to_odom)
        if success and not self.initialized:
            self.initialized = True
            self.get_logger().info("초기화 성공!")

def main(args=None):
    rclpy.init(args=args)
    node = GlobalLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("키보드 인터럽트로 종료합니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
