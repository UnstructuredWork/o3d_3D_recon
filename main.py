import time
import cv2
import numpy as np
import open3d as o3d

from ketisdk.sensor.realsense_sensor import RSSensor
# from kinect import Kinect
from o3d_recon import ReconROS


sensor = RSSensor()
sensor.start()
intrinsic = np.array([[sensor.info.fx, 0, sensor.info.cx],
                      [0, sensor.info.fy, sensor.info.cy],
                      [0, 0, 1]])

# sensor = Kinect()
# sensor.start(1536)
# intrinsic = sensor.intrinsic_color

color, depth = sensor.get_data()

time.sleep(1)

# slam = RealtimeRecon(voxel_size=0.01, intrinsic=intrinsic, panoptic=False, send_ros=False)
slam = ReconROS(intrinsic=intrinsic, voxel_size=0.005, panoptic=False, device='0')

vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

pcd = o3d.geometry.PointCloud()
points = np.random.rand(10, 3)
pcd.points = o3d.utility.Vector3dVector(points)

vis.add_geometry(pcd)

# if slam.is_started:
# if True:
cnt = 0
while True:
    color, depth = sensor.get_data()

    color = cv2.resize(color, dsize=[1024, 768])
    depth = cv2.resize(depth, dsize=[1024, 768])

    t1 = time.time()
    slam.update(color, depth)

    t2 = time.time()

    # print((t2 - t1) * 1000)

    try:
        _pcd = slam.recon_raw.pcd
        # _pcd = slam.recon_pan.pcd

        if slam.recon_raw.run:
            slam.ros.send_ros(_pcd)

        pcd.points = o3d.utility.Vector3dVector(_pcd.point.positions.numpy())
        pcd.colors = o3d.utility.Vector3dVector(_pcd.point.colors.numpy())

        vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
    except Exception as e:
        # print(e)
        pass

    cnt += 1
#
vis.destroy_window()
