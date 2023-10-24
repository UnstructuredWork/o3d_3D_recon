import os
import time
from ctypes import *

import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
                [PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2 ** 16
BIT_MOVE_8 = 2 ** 8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000) >> 16, (rgb_uint32 & 0x0000ff00) >> 8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

class RosPublisher:
    def __init__(self,
                 node_name: str,
                 topic_name: str,
                 frame_id: str):

        self.node_name = node_name
        self.topic_name = topic_name
        self.frame_id = frame_id

        self.publisher = None

        self.set_ros()

    def set_ros(self):
        rospy.init_node(self.node_name, anonymous=True)

        self.publisher = rospy.Publisher(self.topic_name, PointCloud2, queue_size=10)

    def send_ros(self, pcd):
        if not rospy.is_shutdown():
            ros_pcd = self.cvt_o3d_pcd_to_ros_pcd2(pcd)
            self.publisher.publish(ros_pcd)

    def cvt_o3d_pcd_to_ros_pcd2(self, pcd):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        fields = FIELDS_XYZRGB

        points = pcd.point.positions.numpy()
        colors = pcd.point.colors.numpy()

        colors = colors[:, ::-1]
        colors = np.floor(colors * 255)  # nx3 matrix

        colors = colors.astype(np.uint32)
        colors = colors[:, 2] * BIT_MOVE_16 + colors[:, 1] * BIT_MOVE_8 + colors[:, 1]
        colors = colors.view(np.float32)
        cloud_data = [tuple((*p, c)) for p, c in zip(points, colors)]

        # create ros_cloud
        return pc2.create_cloud(header, fields, cloud_data)

    def kill_ros_node(self):
        # pass
        nodes = os.popen("rosnode list").readlines()

        for i in range(len(nodes)):
            nodes[i] = nodes[i].replace("\n", "")

        for node in nodes:
            n = "_".join(node.split('_')[:-2])[1:]
            if n == self.node_name:

                os.system("rosnode kill " + node)

    def shutdown_ros(self):
        self.kill_ros_node()

        # time.sleep(1)
        os.system(f"kill -9 {os.getpid()}")
