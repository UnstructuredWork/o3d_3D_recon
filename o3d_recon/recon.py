import time
from collections import namedtuple
import cv2
import numpy as np

import open3d as o3d
import open3d.core as o3c
import coord_transform as ct
from o3d_recon.vio.vio import VIO


class RealtimeRecon:
    def __init__(self,
                 intrinsic: np.ndarray,
                 voxel_size: float = 0.006,
                 device: str = 'CUDA:0'):

        self.intrinsic  = o3d.core.Tensor(intrinsic, o3d.core.Dtype.Float64)
        self.device_str = device
        self.device     = o3d.core.Device(device)
        self.voxel_size = voxel_size

        # Reconstruction config
        self.depth_scale = 1000.0
        self.depth_min   = 0.2
        self.depth_max   = 3.0
        self.trunc_multiplier = 8.0
        self.est_block_count  = 100000
        self.est_point_count  = 8000000

        self.model = None
        self.index = 0
        # self.poses = []

        self.T_frame_to_model = o3c.Tensor(np.identity(4))
        self.T_frame_to_model_vio = o3c.Tensor(np.identity(4))

        self.timestamp = time.time()
        self.imu = None
        self.color_raw = None
        self.depth_raw = None
        self.color_ref = None
        self.depth_ref = None

        self.input_frame   = None
        self.raycast_frame = None

        # PCD
        self.pcd = None

        self.prev_points = None
        self.prev_colors = None
        self.prev_num_pcd = 0

        self.curr_points = None
        self.curr_colors = None
        self.curr_num_pcd = 0

        self.is_started = False

        self.set_model()

    def __call__(self, color, depth, imu=None, ext_img=None):
        self.timestamp = time.time()

        self.color_raw = color
        self.depth_raw = depth
        self.imu = imu

        self.color_ref = self.numpy2Image(color).to(self.device)
        self.depth_ref = self.numpy2Image(depth).to(self.device)

        self.set_frame()

        if self.raycast_frame is None:
            self.set_raycast()

        self.get_pose()
        # self.get_pose_vio()
        # if imu is None:
        #     self.get_pose()
        # if imu is not None:
        #     self.get_pose_vio()

        self.recon()
        self.get_pcd()

        return self.curr_points, self.curr_colors, self.pcd

    def recon(self):
        # self.poses.append(self.T_frame_to_model.cpu().numpy())

        self.model.update_frame_pose(self.index, self.T_frame_to_model)
        self.model.integrate(self.input_frame,
                             self.depth_scale,
                             self.depth_max,
                             self.trunc_multiplier,
                             )

        self.model.synthesize_model_frame(
            self.raycast_frame,
            self.depth_scale,
            self.depth_min,
            self.depth_max,
            self.trunc_multiplier,
            True,                       # True -> RGB + PointCloud raycast == True
        )

        self.index += 1

    def set_frame(self):
        self.input_frame   = o3d.t.pipelines.slam.Frame(self.depth_ref.rows,
                                                        self.depth_ref.columns, self.intrinsic,
                                                        self.device)
        self.input_frame.set_data_from_image('color', self.color_ref)
        self.input_frame.set_data_from_image('depth', self.depth_ref)

    def set_raycast(self):
        self.raycast_frame = o3d.t.pipelines.slam.Frame(self.depth_ref.rows,
                                                        self.depth_ref.columns, self.intrinsic,
                                                        self.device)
        self.raycast_frame.set_data_from_image('color', self.color_ref)
        self.raycast_frame.set_data_from_image('depth', self.depth_ref)

    def set_model(self):
        self.model = o3d.t.pipelines.slam.Model(
            self.voxel_size,
            16,
            self.est_block_count,
            o3c.Tensor(np.eye(4)),
            o3c.Device('CUDA:0'))

        self.is_started = True

    def get_pose(self):
        if self.index > 0:
            result = self.model.track_frame_to_model(
                self.input_frame,
                self.raycast_frame,
                self.depth_scale,
                self.depth_max
            )

            self.T_frame_to_model = self.T_frame_to_model @ result.transformation
            print(self.T_frame_to_model)

    def get_pose_vio(self):
        gray = cv2.cvtColor(self.color_raw, cv2.COLOR_BGR2GRAY)
        gray_msg = namedtuple('img_msg', ['timestamp', 'image'])(self.timestamp, gray)

        imu_msg = namedtuple('imu_msg', ['timestamp', 'angular_velocity', 'linear_acceleration'])(
            self.timestamp, self.imu.angular_velocity, self.imu.linear_acceleration)

        img_msg = namedtuple('stereo_msg', ['timestamp', 'cam0_image', 'cam1_image' ,'cam0_msg' 'cam1_msg'])(
            self.timestamp, gray, gray, gray_msg, gray_msg)


        pose = ct.Matrix()
        # pose.set_rmat()
        # pose.set_tvec()



        # self.T_frame_to_model_vio = self.T_frame_to_model_vio @ pose.get_T()
        self.T_frame_to_model_vio = pose.get_T()
        print(self.T_frame_to_model)

    def get_pcd(self):
        self.pcd = self.model.voxel_grid.extract_point_cloud(
            3.0,
            self.est_point_count,
        ).to(o3d.core.Device('CPU:0'))

        points = self.pcd.point.positions
        colors = (self.pcd.point.colors * 255).to(o3d.core.uint8)

        self.prev_points = points[:self.prev_num_pcd, :]
        self.prev_colors = colors[:self.prev_num_pcd, :]

        self.curr_points = points[self.prev_num_pcd:, :]
        self.curr_colors = colors[self.prev_num_pcd:, :]

        self.prev_num_pcd = points.shape[0]
        self.curr_num_pcd = self.curr_colors.shape[0]

    @staticmethod
    def numpy2Image(data):
        return o3d.t.geometry.Image(np.array(data))
