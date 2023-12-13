import cv2
import time
import socket
import numpy as np
import open3d as o3d

from keti_server import RgbdServer
# from ketisdk.sensor.realsense_sensor import RSSensor
from kinect import Kinect
from o3d_recon import ReconROS


# sensor = RSSensor()
# sensor.start()
# intrinsic = np.array([[sensor.info.fx, 0, sensor.info.cx],
#                       [0, sensor.info.fy, sensor.info.cy],
#                       [0, 0, 1]])

# sensor = Kinect()
# sensor.start(1536)
# intrinsic = sensor.intrinsic_color

# color, depth = sensor.get_data()

''' UDP '''
# '''
sensor = RgbdServer(['10.252.117.101', 5003], "RGBD")
while True:
    try:
        recv_data = sensor.recv_udp()
        if recv_data is not None:
            intrinsic = recv_data["intrinsic"]
            break
    except socket.timeout:
        pass
# '''

time.sleep(1)

def main():
    slam = ReconROS(intrinsic=intrinsic, voxel_size=0.005, panoptic=False, device='0') # panoptic True False로 RGB, PANOPTIC 구분
    while True:
        if slam.recon_raw.ready:

            # color, depth = sensor.get_data()
            #
            # # color = cv2.resize(color, dsize=[1024, 768])
            # # depth = cv2.resize(depth, dsize=[1024, 768])
            #
            # slam.update(color, depth)
            #
            # try:
            #     _pcd = slam.recon_raw.pcd
            #     # _pcd = slam.recon_pan.pcd
            #
            #     if slam.recon_raw.run:
            #         slam.publish_ros()
            #
            # except Exception as e:
            #     print(e)

            ''' UDP '''
            # '''
            recv_data = sensor.recv_udp()
            if recv_data is not None:
                color, depth = recv_data["rgb"], recv_data["depth"]

                slam.update(color, depth)

                try:
                    _pcd = slam.recon_raw.pcd
                    # _pcd = slam.recon_pan.pcd

                    if slam.recon_raw.run:
                        slam.publish_ros()

                except Exception as e:
                    print(e)
            # '''

        else:
            time.sleep(0.05)

if __name__ == "__main__":
    main()
