#/usr/bin/python
import cv2
import rospy
import numpy as np
import message_filters

from parallel import thread_method
from sensor_msgs.msg import Image, CameraInfo

class ImageConverter:
    def __init__(self):
        self.rgb   = None
        self.depth = None

        self.msg = rospy.wait_for_message('/zed/zed_node/rgb/camera_info', CameraInfo)
        self.intrinsic = np.reshape(np.array(self.msg.K), (3, 3))

        self.rgb_sub = message_filters.Subscriber('/zed/zed_node/rgb/image_rect_color', Image)
        self.depth_sub = message_filters.Subscriber('/zed/zed_node/depth/depth_registered', Image)

        self.time_sync = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 100)
        self.time_sync.registerCallback(self.callback)

        self.started = False

    def callback(self, rgb, depth):
        cv_rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)
        cv_depth = np.frombuffer(depth.data, dtype=np.float32).reshape(depth.height, depth.width)

        self.rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_BGRA2BGR)
        self.depth = cv_depth

        # self.show(cv_rgb, cv_depth)

        self.started = True

    def get_data(self):
        return self.rgb, self.depth

    @thread_method
    def show(self, rgb, depth):
        cv2.imshow("RGB", rgb)
        cv2.imshow("DEPTH", depth)
        cv2.waitKey(0)

def main():
    rospy.init_node('Open3D', anonymous=True)
    ic = ImageConverter()
    while not rospy.is_shutdown():
        if ic.started:
            rgb, depth = ic.get_data()
            cv2.imshow("RGB", rgb)
            cv2.imshow("DEPTH", depth)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    rospy.spin()

if __name__ == '__main__':
    main()