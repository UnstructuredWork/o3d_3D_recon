import time
import numpy as np
import cv2
import open3d as o3d

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

slam = RealtimeRecon(voxel_size=0.01, intrinsic=intrinsic, panoptic=False, send_ros=False)

vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

pcd = o3d.geometry.PointCloud()
points = np.random.rand(10, 3)
pcd.points = o3d.utility.Vector3dVector(points)

vis.add_geometry(pcd)

if slam.is_started:
    cnt = 0
    while True:
        color, depth = sensor.get_data()

        # color = cv2.resize(color, dsize=[1024, 768])
        # depth = cv2.resize(depth, dsize=[1024, 768])

        t1 = time.time()
        result = slam(color, depth)
        t2 = time.time()

        print((t2 - t1) * 1000)

        _pcd = result.rgb
        # _pcd = result.panoptic

        pcd.points = o3d.utility.Vector3dVector(_pcd.pcd.point.positions.numpy())
        pcd.colors = o3d.utility.Vector3dVector(_pcd.pcd.point.colors.numpy())

        # pcd.points.extend(result.rgb.curr_points)
        # pcd.colors.extend(result.rgb.curr_colors)
        # print(result.rgb.curr_colors)

        vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        cnt += 1

vis.destroy_window()
