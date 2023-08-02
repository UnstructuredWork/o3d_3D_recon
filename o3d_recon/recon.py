#!/usr/bin/env python
import os
import sys
import time
import psutil
from collections import namedtuple
from ctypes import *
from queue import Queue

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import coord_transform as ct
from o3d_recon.vio.vio import VIO
from panoptic.panoptic import Panoptic

# import rospy
# from std_msgs.msg import Header
# from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2



def kill_ros_node():
    pass
    # nodes = os.popen("rosnode list").readlines()
    # for i in range(len(nodes)):
    #     nodes[i] = nodes[i].replace("\n", "")
    #
    # for node in nodes:
    #     os.system("rosnode kill " + node)
# import rospy
# from std_msgs.msg import Header
# from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2


class RealtimeRecon:
    def __init__(self,
                 intrinsic: np.ndarray,
                 voxel_size: float = 0.006,
                 panoptic: bool = True,
                 device: str = 'CUDA:0',
                 send_ros=False):

        self.intrinsic  = o3d.core.Tensor(intrinsic, o3d.core.Dtype.Float64)
        self.device_str = device
        self.device     = o3d.core.Device(device)
        self.voxel_size = voxel_size
        self.panoptic   = panoptic

        self.send_ros = send_ros
        if self.send_ros:
            kill_ros_node()

            self.ros_frame_id = 'odom'
            self.ros_publisher = None
            self.set_ros()

            rospy.on_shutdown(self.shutdown_ros)

        # Reconstruction config
        self.depth_scale = 1000.0
        self.depth_min   = 0.2
        self.depth_max   = 3.0
        self.trunc_multiplier = 8.0
        self.est_block_count  = 100000
        self.est_point_count  = 8000000

        self.recon_model_rgb = None
        self.recon_model_pan = None
        self.panoptic_model  = None
        self.index = 0
        # self.poses = []

        self.T_frame_to_model = o3c.Tensor(np.identity(4))
        self.T_frame_to_model_vio = o3c.Tensor(np.identity(4))

        self.vio_img_queue = Queue()
        self.vio_imu_queue = Queue()

        self.vio = VIO(self.vio_img_queue, self.vio_imu_queue)

        self.timestamp = time.time()
        self.imu = None
        self.color_raw = None
        self.depth_raw = None
        self.color_ref = None
        self.depth_ref = None
        self.panop_ref = None

        self.input_frame_rgb = None
        self.input_frame_pan = None
        self.raycast_frame_rgb = None
        self.raycast_frame_pan = None

        # PCD
        self.pcd = None
        self.ros_pcd = None

        self.prev_points = None
        self.prev_colors = None
        self.prev_num_pcd = 0

        self.curr_points = None
        self.curr_colors = None
        self.curr_num_pcd = 0

        self.is_started = False

        self.set_model()

        if self.panoptic:
            # PANOPTIC
            self.panoptic_model = Panoptic('./panoptic/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py',
                                'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth',
                                            classes=[i for i in range(80, 133)],
                                            device='0')

    def shutdown_ros(self):
        kill_ros_node()

        time.sleep(1)
        os.system(f"kill -9 {os.getpid()}")

    def __call__(self, color, depth, imu=None, ext_img=None):
        self.timestamp = time.time()

        self.color_raw = color
        self.depth_raw = depth
        self.imu = imu

        self.color_ref = self.numpy2Image(color).to(self.device)
        self.depth_ref = self.numpy2Image(depth).to(self.device)
        self.set_input_frame_rgb()

        if self.raycast_frame_rgb is None:
            self.set_raycast_rgb()

        if self.panoptic:
            panoptic_img = self.panoptic_model.get_panoptic(color)
            self.panop_ref = self.numpy2Image(panoptic_img).to(self.device)
            self.set_input_frame_pan()

            if self.raycast_frame_pan is None:
                self.set_raycast_pan()

        self.get_pose()
        # if imu is not None:
        #     self.get_pose_vio()

        self.recon(self.recon_model_rgb, self.input_frame_rgb, self.raycast_frame_rgb)
        pcd_rgb, curr_points_rgb, curr_colors_rgb = self.get_pcd(self.recon_model_rgb)

        result_rgb = namedtuple('PointCloudRGB', ('pcd', 'curr_points', 'curr_colors'))(
            pcd_rgb, curr_points_rgb, curr_colors_rgb)

        if self.panoptic:
            self.recon(self.recon_model_pan, self.input_frame_pan, self.raycast_frame_pan)
            pcd_pan, curr_points_pan, curr_colors_pan = self.get_pcd(self.recon_model_pan)

            result_pan = namedtuple('PointCloudPanoptic', ('pcd', 'curr_points', 'curr_colors'))(
                pcd_pan, curr_points_pan, curr_colors_pan)
        else:
            result_pan = None

        result = namedtuple('Realtime3DMap', ('rgb', 'panoptic'))(
            result_rgb, result_pan)

        self.index += 1

        if self.send_ros and self.pcd is not None and not rospy.is_shutdown():
            try:
                ros_pcd = self.cvt_o3d_pcd_to_ros_pcd2()
                self.ros_publisher.publish(ros_pcd)

            except rospy.ROSInterruptException as e:
                pass

        return result

    def recon(self, model, input_frame, raycast_frame):
        model.update_frame_pose(self.index, self.T_frame_to_model)
        model.integrate(input_frame,
                       self.depth_scale,
                       self.depth_max,
                       self.trunc_multiplier)

        model.synthesize_model_frame(
            raycast_frame,
            self.depth_scale,
            self.depth_min,
            self.depth_max,
            self.trunc_multiplier,
            True,                       # True -> RGB + PointCloud raycast == True
        )

    def set_input_frame_rgb(self):
        self.input_frame_rgb = o3d.t.pipelines.slam.Frame(self.depth_ref.rows,
                                                          self.depth_ref.columns, self.intrinsic,
                                                          self.device)
        self.input_frame_rgb.set_data_from_image('color', self.color_ref)
        self.input_frame_rgb.set_data_from_image('depth', self.depth_ref)

    def set_input_frame_pan(self):
        self.input_frame_pan = o3d.t.pipelines.slam.Frame(self.depth_ref.rows,
                                                          self.depth_ref.columns, self.intrinsic,
                                                          self.device)
        self.input_frame_pan.set_data_from_image('color', self.panop_ref)
        self.input_frame_pan.set_data_from_image('depth', self.depth_ref)

    def set_raycast_rgb(self):
        self.raycast_frame_rgb = o3d.t.pipelines.slam.Frame(self.depth_ref.rows,
                                                        self.depth_ref.columns, self.intrinsic,
                                                        self.device)
        self.raycast_frame_rgb.set_data_from_image('color', self.color_ref)
        self.raycast_frame_rgb.set_data_from_image('depth', self.depth_ref)

    def set_raycast_pan(self):
        self.raycast_frame_pan = o3d.t.pipelines.slam.Frame(self.depth_ref.rows,
                                                        self.depth_ref.columns, self.intrinsic,
                                                        self.device)
        self.raycast_frame_pan.set_data_from_image('color', self.panop_ref)
        self.raycast_frame_pan.set_data_from_image('depth', self.depth_ref)

    def set_model(self):
        self.recon_model_rgb = o3d.t.pipelines.slam.Model(
            self.voxel_size,
            16,
            self.est_block_count,
            o3c.Tensor(np.eye(4)),
            o3c.Device('CUDA:0'))

        if self.panoptic:
            self.recon_model_pan = o3d.t.pipelines.slam.Model(
                self.voxel_size,
                16,
                self.est_block_count,
                o3c.Tensor(np.eye(4)),
                o3c.Device('CUDA:0'))

        self.is_started = True

    def get_pose(self):
        if self.index > 0:
            result = self.recon_model_rgb.track_frame_to_model(
                self.input_frame_rgb,
                self.raycast_frame_rgb,
                self.depth_scale,
                self.depth_max
            )

            self.T_frame_to_model = self.T_frame_to_model @ result.transformation

    #
    # def get_pose_vio(self):
    #     gray = cv2.cvtColor(self.color_raw, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.resize(gray, [752, 480])
    #
    #     gray_msg = namedtuple('img_msg', ['timestamp', 'image'])(self.timestamp, gray)
    #
    #     imu_msg = namedtuple('imu_msg', ['timestamp', 'angular_velocity', 'linear_acceleration'])(
    #         self.timestamp, self.imu.angular_velocity, self.imu.linear_acceleration)
    #
    #     img_msg = namedtuple('stereo_msg', ['timestamp', 'cam0_image', 'cam1_image' ,'cam0_msg', 'cam1_msg'])(
    #         self.timestamp, gray, gray, gray_msg, gray_msg)
    #
    #     self.vio_img_queue.put(img_msg)
    #     self.vio_imu_queue.put(imu_msg)
    #
    #     if self.vio.pose is not None:
    #         # print(self.vio.pose)
    #         pose = o3c.Tensor(self.vio.pose)
    #         # print(pose)
    #
    #         self.T_frame_to_model_vio = self.T_frame_to_model_vio @ pose

    def get_pcd(self, model):
        pcd = model.voxel_grid.extract_point_cloud(
            3.0,
            self.est_point_count,
        ).to(o3d.core.Device('CPU:0'))

        points = pcd.point.positions
        # colors = (self.pcd.point.colors * 255).to(o3d.core.uint8)
        colors = pcd.point.colors

        # prev_points = points[:self.prev_num_pcd, :]
        # prev_colors = colors[:self.prev_num_pcd, :]

        self.curr_points = points[self.prev_num_pcd:, :].numpy()
        self.curr_colors = colors[self.prev_num_pcd:, :].numpy()

        self.prev_num_pcd = points.shape[0]
        self.curr_num_pcd = self.curr_colors.shape[0]

        return pcd, self.curr_points, self.curr_colors

    def set_ros(self):
        # The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
        self.FIELDS_XYZ = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        self.FIELDS_XYZRGB = self.FIELDS_XYZ + \
                        [PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]

        # Bit operations
        self.BIT_MOVE_16 = 2 ** 16
        self.BIT_MOVE_8 = 2 ** 8
        convert_rgbUint32_to_tuple = lambda rgb_uint32: (
            (rgb_uint32 & 0x00ff0000) >> 16, (rgb_uint32 & 0x0000ff00) >> 8, (rgb_uint32 & 0x000000ff)
        )
        convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
            int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
        )

        rospy.init_node('keti_3dmap', anonymous=True)

        topic_name = "keti/3d_map/pointcloud"
        self.ros_publisher = rospy.Publisher(topic_name, PointCloud2, queue_size=10)

    def cvt_o3d_pcd_to_ros_pcd2(self):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.ros_frame_id

        # fields = self.FIELDS_XYZ
        fields = self.FIELDS_XYZRGB

        # points = self.pcd.point.positions.numpy()
        # colors = self.pcd.point.colors.numpy()

        points = self.curr_points.numpy()
        colors = self.curr_colors.numpy()

        colors = colors[:, ::-1]
        colors = np.floor(colors * 255)  # nx3 matrix

        colors = colors.astype(np.uint32)
        colors = colors[:, 2] * self.BIT_MOVE_16 + colors[:, 1] * self.BIT_MOVE_8 + colors[:, 1]
        colors = colors.view(np.float32)
        cloud_data = [tuple((*p, c)) for p, c in zip(points, colors)]

        # create ros_cloud
        return pc2.create_cloud(header, fields, cloud_data)

    @staticmethod
    def numpy2Image(data):
        return o3d.t.geometry.Image(np.array(data))
