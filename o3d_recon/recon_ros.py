import time
from types import SimpleNamespace as SN

from pynput import keyboard

from o3d_recon.src import RealtimeReconstruction, thread_method
from o3d_recon.src import RosPublisher
from panoptic.panoptic import Panoptic


def recon_data(dname):
    data = SN()
    data.name  = dname
    data.ready = False
    data.run   = False
    data.model = None
    data.color = None
    data.depth = None
    data.T     = None
    data.pcd   = None
    data.pcd_points = None
    data.pcd_colors = None
    data.error = ""

    return data

class ReconROS:
    def __init__(self,
                 intrinsic,
                 voxel_size,
                 panoptic=False,
                 device: str = '0',
                 node_name: str = 'keti_realtime_recon',
                 topic_name: str = 'keti/rt_recon/3d_map/pointcloud',
                 frame_id: str = 'odom'):

        self.intrinsic  = intrinsic
        self.voxel_size = voxel_size
        self.device = device

        self.color = None
        self.depth = None
        self.color_pan = None

        self.pan = panoptic

        self.ros = RosPublisher(node_name, topic_name, frame_id)

        self.recon_raw = recon_data('raw')
        self.recon_pan = recon_data('panoptic')

        self.run_raw_recon()

        if self.pan:
            self.panoptic = Panoptic('./panoptic/mask2former_custom/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py',
                                     './panoptic/mask2former_custom/iter_5000.pth',
                                     classes=[i for i in range(0, 12)],
                                     device=device)
            # self.run_pan_recon() # self.run_raw_recon()에서 panoptic 실행

        self._keyboard()

        print("Kill this process : 'ESC'")

    def update(self, color, depth):
        self.color = color
        self.depth = depth

    def publish_ros(self, data='raw'):
        if data == 'raw':
            self.ros.send_ros(self.recon_raw.pcd)
        elif data == 'panoptic':
            self.ros.send_ros(self.recon_pan.pcd)
        else:
            print('publish_ros - data type error')

    @thread_method
    def run_raw_recon(self):
        self.recon_raw.model = RealtimeReconstruction(self.intrinsic, self.voxel_size, self.device)
        self.recon_raw.ready = True

        while True:

            while self.recon_raw.ready:

                if self.color is not None and self.depth is not None:

                    try:
                        if self.pan:
                            self.recon_raw.color = self.panoptic.get_panoptic(self.color)
                        else:
                            self.recon_raw.color = self.color

                        self.recon_raw.depth = self.depth

                        self.recon_raw.pcd = self.recon_raw.model.update(self.recon_raw.color, self.recon_raw.depth)
                        self.recon_raw.T   = self.recon_raw.model.T_frame_to_model
                        self.recon_raw.run = True

                    except Exception as e:
                        self.recon_raw.error = e
                        self.recon_raw.run = False

                else:
                    self.recon_raw.run = False
                    time.sleep(0.2)

            time.sleep(0.5)

    # @thread_method         # self.run_raw_recon()에서 panoptic 실행
    # def run_pan_recon(self):
    #     self.recon_pan.model = RealtimeReconstruction(self.intrinsic, self.voxel_size, self.device)
    #     self.recon_pan.ready = True
    #
    #     while True:
    #         while self.recon_pan.ready and self.recon_raw.run:
    #             if self.color is not None and self.depth is not None:
    #
    #                 try:
    #                     self.recon_pan.color = self.panoptic.get_panoptic(self.color)
    #                     self.recon_pan.depth = self.depth
    #
    #                     self.recon_pan.pcd = self.recon_pan.model.update(self.recon_pan.color, self.recon_pan.depth)
    #                     self.recon_pan.run = True
    #
    #                 except Exception as e:
    #                     self.recon_pan.error = e
    #                     self.recon_pan.run = False
    #
    #             else:
    #                 self.recon_pan.run = False
    #                 time.sleep(0.2)
    #
    #         time.sleep(0.5)

    @thread_method
    def _keyboard(self):

        def on_press(key):
            pass

        def on_release(key):

            if key == keyboard.Key.esc:
                self.ros.shutdown_ros()

                return False

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
