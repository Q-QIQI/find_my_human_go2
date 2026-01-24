#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import os

class HumanTracker(Node):
    def __init__(self):
        super().__init__('human_tracker')
        
        self.get_logger().info("Loading YOLOv8 model for tracking...")
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(package_dir, 'yolov8n.pt')
        self.model = YOLO(model_path)
        
        self.bridge = CvBridge()
        self.target_id = None
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile
        )
        
        self.publisher_ = self.create_publisher(Image, '/human_tracker/annotated', 10)
        
        self.get_logger().info("Target Locking Node Started (D435i Mode). Waiting for video stream...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            results = self.model.track(
                cv_image, 
                classes=[0], 
                persist=True, 
                conf=0.3, 
                imgsz=320,
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                detections = {tid: box for tid, box in zip(track_ids, boxes)}
                
                if self.target_id is None:
                    max_area = 0
                    best_id = None
                    
                    for tid, box in detections.items():
                        x1, y1, x2, y2 = box
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            best_id = tid
                    
                    if best_id is not None:
                        self.target_id = best_id
                        self.get_logger().info(f"TARGET ACQUIRED. Locking onto ID: {self.target_id}")
                
                if self.target_id not in detections:
                    if self.target_id is not None:
                        self.get_logger().warn(f"Target ID {self.target_id} LOST. Resetting search.")
                    self.target_id = None
                
                for tid, box in detections.items():
                    x1, y1, x2, y2 = map(int, box)
                    
                    if tid == self.target_id:
                        color = (0, 0, 255)
                        label = f"TARGET [ID:{tid}]"
                        thickness = 3
                        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                        cv2.circle(cv_image, (cx, cy), 5, color, -1)
                    else:
                        color = (0, 255, 0)
                        label = f"ID:{tid}"
                        thickness = 2
                    
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(cv_image, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            else:
                if self.target_id is not None:
                    self.get_logger().warn(f"All targets lost. Resetting.")
                    self.target_id = None

            # Publish annotated frame
            output_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.publisher_.publish(output_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = HumanTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()