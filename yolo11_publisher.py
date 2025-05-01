import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from YOLOv11 import YOLOv11 

class YoloPosePublisher(Node):
    def __init__(self):
        super().__init__('yolo_pose_publisher')
        self.bridge = CvBridge()
        self.pose_pub = self.create_publisher(Pose, 'yolo_pose', 10)
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)

        # Download the format of the model
        # You can use your own model or a pre-trained one
        # Make sure to have a compatible YOLOv5 model
        self.model = YOLOv11("yolo11n.pt", device="cpu")  # Change the route to suit you.
        self.cap=cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            raise RuntimeError("Camera not available")
        # Camera settings
        self.fx = 625  # focal length x
        self.fy = 625  # focal length y
        self.cx = 320  # principal point x
        self.cy = 240  # principal point y
        self.depth_assumption = 0.5  # Meter, for example

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(frame)

        if results:
            for det in results:
                x1, y1, x2, y2, conf, cls = det
                u = (x1 + x2) / 2
                v = (y1 + y2) / 2
                z = self.depth_assumption

                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy

                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = float(z)
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0

                self.pose_pub.publish(pose)
                self.get_logger().info(f"Published pose x={x:.2f}, y={y:.2f}, z={z:.2f}")
                # Draw bounding box (for visualization)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{self.model.names[int(cls_id)]}: {conf:.2f}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Show live preview
        cv2.imshow("YOLOv11 Object Detection", frame)
        cv2.waitKey(1)  # Needed for OpenCV window

def main(args=None):
    rclpy.init(args=args)
    node = YoloPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    # Cleanup
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
