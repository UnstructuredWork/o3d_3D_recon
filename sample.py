import time
import numpy as np

from kinect import Kinect
from ketisdk.sensor.realsense_sensor import RSSensor
from o3d_recon import RealtimeRecon


# sensor = RSSensor()
# sensor.start()
# intrinsic = np.array([[sensor.info.fx, 0, sensor.info.cx],
#                       [0, sensor.info.fy, sensor.info.cy],
#                       [0, 0, 1]])

sensor = Kinect()
sensor.start(1536)
intrinsic = sensor.intrinsic_color

color, depth = sensor.get_data()

time.sleep(1)

slam = RealtimeRecon(voxel_size=0.006, intrinsic=intrinsic, send_ros=False)

if slam.is_started:
    a = 0
    while True:
        color, depth = sensor.get_data()

        t1 = time.time()
        pcd, curr_points, curr_colors, prev_points, prev_colors = slam(color, depth)
        t2 = time.time()

        print((t2 - t1) * 1000)


