from queue import Queue
from threading import Thread

import coord_transform as ct
from o3d_recon.vio.image import ImageProcessor
from o3d_recon.vio.config import ConfigEuRoC
from o3d_recon.vio.msckf import MSCKF



class VIO(object):
    def __init__(self, img_queue, imu_queue):
        config = ConfigEuRoC()

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

        self.pose = None

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()

            if img_msg is None:
                self.feature_queue.put(None)
                return

            feature_msg = self.image_processor.stareo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()

            if imu_msg is None:
                return

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()

            if feature_msg is None:
                return

            result = self.msckf.feature_callback(feature_msg)

            if result is not None:
                pose = ct.Matrix()
                pose.set_rmat(result.pose.R)
                pose.set_tvec(result.pose.t)

                self.pose = pose.get_T()
                # self.R = result.pose.R
                # self.t = result.pose.t
                # print("R:", result.pose.R)
                # print("T:", result.pose.t)
                # print('   orientation:', result.orientation)
                # print('   position:', result.position)

