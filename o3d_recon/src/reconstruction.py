import numpy as np
import open3d as o3d
import open3d.core as o3c


class RealtimeReconstruction:
    def __init__(self,
                 intrinsic: np.ndarray,
                 voxel_size: float = 0.01,
                 device: str = '0'):

        self.intrinsic = o3d.core.Tensor(intrinsic, o3d.core.Dtype.Float64)
        self.device_id = device
        self.device = o3d.core.Device("CUDA:" + device)
        self.voxel_size = voxel_size

        # Reconstruction config
        self.depth_scale = 1000.0
        self.depth_min = 0.2
        self.depth_max = 3.0
        self.trunc_multiplier = 8.0
        self.est_block_count = 100000
        self.est_point_count = 8000000

        self.model = None
        self.index = 0

        self.T_frame_to_model = o3c.Tensor(np.identity(4))

        self.pcd = None

        self.color_raw = None
        self.depth_raw = None

        self.color_ref = None
        self.depth_ref = None

        self.input_frame   = None
        self.raycast_frame = None

        self.is_started = False

        self.set_model()

    def update(self, color, depth, T_frame_to_model=None):
        self.color_ref = self.numpy2Image(color).to(self.device)
        self.depth_ref = self.numpy2Image(depth).to(self.device)

        self.set_input_frame()

        if self.index == 0:
            self.set_raycast_frame()

        if T_frame_to_model is None:
            self.get_pose()

        self.recon()
        return self.get_pcd()

    def recon(self, T_frame_to_model=None):
        if T_frame_to_model is not None:
            self.T_frame_to_model = T_frame_to_model

        self.model.update_frame_pose(self.index, self.T_frame_to_model)
        self.model.integrate(self.input_frame,
                             self.depth_scale,
                             self.depth_max,
                             self.trunc_multiplier)
        self.model.synthesize_model_frame(
            self.raycast_frame,
            self.depth_scale,
            self.depth_min,
            self.depth_max,
            self.trunc_multiplier,
            True)

        self.index += 1

    def set_model(self):
        self.model = o3d.t.pipelines.slam.Model(
            self.voxel_size,
            16,
            self.est_block_count,
            o3c.Tensor(np.eye(4)),
            o3c.Device("CUDA:" + self.device_id)
        )

        self.is_started = True

    def set_input_frame(self):
        self.input_frame = o3d.t.pipelines.slam.Frame(self.depth_ref.rows,
                                                      self.depth_ref.columns,
                                                      self.intrinsic,
                                                      self.device)
        self.input_frame.set_data_from_image('color', self.color_ref)
        self.input_frame.set_data_from_image('depth', self.depth_ref)

    def set_raycast_frame(self):
        self.raycast_frame = o3d.t.pipelines.slam.Frame(self.depth_ref.rows,
                                                        self.depth_ref.columns,
                                                        self.intrinsic,
                                                        self.device)
        self.raycast_frame.set_data_from_image('color', self.color_ref)
        self.raycast_frame.set_data_from_image('depth', self.depth_ref)

    def get_pose(self):
        if self.index > 0:
            result = self.model.track_frame_to_model(
                self.input_frame,
                self.raycast_frame,
                self.depth_scale,
                self.depth_max
            )

            self.T_frame_to_model = self.T_frame_to_model @ result.transformation

    def get_pcd(self):
        self.pcd = self.model.voxel_grid.extract_point_cloud(
            3.0,
            self.est_point_count,
        ).to(o3d.core.Device('CPU:0'))
        return self.pcd

    @staticmethod
    def numpy2Image(data):
        return o3d.t.geometry.Image(np.array(data))
