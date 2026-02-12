#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys
import datetime
import time
os.environ['YOLO_CHECKS'] = 'false'
from ultralytics import YOLO
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from .gesture_utils import GestureDetector

class DepthSkeletonTracker(Node):
    def __init__(self):
        super().__init__('depth_skeleton_tracker')
         
        self.bridge = CvBridge()
        self.get_logger().info("Initializing v18.0 ADAPTIVE POSTURE MODE...")
        
        self.model = YOLO('yolov8n-pose.pt') 
        self.gesture_detector = GestureDetector()
        
        self.log_path = 'tracking_log.txt'
        with open(self.log_path, 'w') as f:
            header = f"{'TIME':<12} | {'STATE':<8} | {'TID':<4} | {'CID':<4} | {'SIM':<5} | {'R_DIFF':<6} | {'STATUS':<12} | {'REASON'}\n"
            f.write(header)
            f.write("-" * 110 + "\n")

        self.internal_target_id = None 
        self.fixed_display_id = 1
        self.locked = False
        self.lost_frame_count = 0
        
        self.last_pos = None      
        self.last_depth = None    
        self.last_ratio = None    
        self.anchor_ratio = None  
        self.anchor_hist = None   
        
        self.last_velocity = np.array([0.0, 0.0]) 
        self.last_timestamp = None
        
        self.coexisted_strangers = set()
        self.candidate_stability = {}
        self.STABILITY_THRESHOLD = 5 
        
        self.max_depth_std = 180.0
        self.global_search_thresh = 45 
        self.recover_hist_thresh = 0.60
        
        # 🟢 [修改] 稍微放宽骨骼差异阈值，适应坐姿时的抖动
        self.strict_ratio_thresh = 0.25 
        
        self.skeleton_links = [
            (0,1), (0,2), (1,3), (2,4), (5,7), (7,9), (6,8), (8,10),
            (5,6), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
        ]
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw', qos_profile=qos)
        self.sub_ir = message_filters.Subscriber(self, Image, '/camera/camera/infra1/image_rect_raw', qos_profile=qos)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_depth, self.sub_ir], queue_size=10, slop=0.15)
        self.ts.registerCallback(self.sync_callback)
        
        self.pub_annotated = self.create_publisher(Image, '/human_tracker/output', 10)
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    def log_data(self, state, t_id, c_id, sim, r_diff, status, reason):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        s_tid = str(t_id) if t_id is not None else "-"
        s_cid = str(c_id) if c_id is not None else "-"
        line = f"{now:<12} | {state:<8} | {s_tid:<4} | {s_cid:<4} | {sim:<5.2f} | {r_diff:<6.2f} | {status:<12} | {reason}\n"
        with open(self.log_path, 'a') as f: f.write(line)

    def compute_histogram(self, img, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10: return None, 0
        roi = img[y1:y2, x1:x2]
        mean_val = np.mean(roi)
        hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist, mean_val

    def compare_histograms(self, hist1, hist2):
        if hist1 is None or hist2 is None: return 0.0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def analyze_skeleton_smart(self, kps):
        if len(kps) == 0: return False, None
        ls, rs = kps[5], kps[6]
        lh, rh = kps[11], kps[12]
        has_head = any(kps[i][0] != 0 for i in range(5))
        has_shoulders = (ls[0]!=0 and rs[0]!=0)
        has_hips = (lh[0]!=0 and rh[0]!=0)
        joints_indices = [7, 8, 9, 10, 13, 14] 
        has_limbs = any(kps[i][0] != 0 for i in joints_indices)
        if has_head: return self._calc_ratio(ls, rs, lh, rh)
        if (has_shoulders or has_hips) and has_limbs: return True, self._calc_ratio_safe(ls, rs, lh, rh)
        return False, None

    def _calc_ratio(self, ls, rs, lh, rh):
        if ls[0]!=0 and rs[0]!=0 and lh[0]!=0 and rh[0]!=0:
            w = np.linalg.norm(ls - rs)
            h = np.linalg.norm((ls+rs)/2 - (lh+rh)/2)
            if h > 20: return True, w/h
        return True, None

    def _calc_ratio_safe(self, ls, rs, lh, rh):
        is_valid, val = self._calc_ratio(ls, rs, lh, rh)
        return val

    def analyze_depth_texture(self, cv_depth, kps):
        ls, rs = kps[5], kps[6]
        lh, rh = kps[11], kps[12]
        has_shoulders = (ls[0] != 0 and rs[0] != 0)
        has_hips = (lh[0] != 0 and rh[0] != 0)
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, 0, 0
        img_h, img_w = cv_depth.shape
        if has_shoulders and has_hips:
            xs = [ls[0], rs[0], lh[0], rh[0]]
            ys = [ls[1], rs[1], lh[1], rh[1]]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w, h = x_max - x_min, y_max - y_min
            roi_x1, roi_x2 = int(x_min + w * 0.3), int(x_max - w * 0.3)
            roi_y1, roi_y2 = int(y_min + h * 0.3), int(y_max - h * 0.3)
        elif has_shoulders:
            cx, cy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
            w = np.linalg.norm(ls - rs)
            roi_x1, roi_x2 = int(cx - w*0.2), int(cx + w*0.2)
            roi_y1, roi_y2 = int(cy + w*0.2), int(cy + w*0.8)
        elif has_hips:
            cx, cy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
            w = np.linalg.norm(lh - rh)
            roi_x1, roi_x2 = int(cx - w*0.2), int(cx + w*0.2)
            roi_y1, roi_y2 = int(cy - w*0.8), int(cy - w*0.2)
        else: return 0, 999 
        roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
        roi_x2, roi_y2 = min(img_w, roi_x2), min(img_h, roi_y2)
        if (roi_x2 - roi_x1) < 5 or (roi_y2 - roi_y1) < 5: return 0, 999
        roi = cv_depth[roi_y1:roi_y2, roi_x1:roi_x2]
        valid = roi[roi > 0]
        if len(valid) < 20: return 0, 999
        return np.mean(valid), np.std(valid)

    def draw_skeleton(self, img, kps, color, thickness=2):
        for i, (start, end) in enumerate(self.skeleton_links):
            if start < len(kps) and end < len(kps):
                pt1 = (int(kps[start][0]), int(kps[start][1]))
                pt2 = (int(kps[end][0]), int(kps[end][1]))
                if pt1[0] != 0 and pt1[1] != 0 and pt2[0] != 0 and pt2[1] != 0:
                    cv2.line(img, pt1, pt2, color, thickness)
                    cv2.circle(img, pt1, 3, color, -1)
                    cv2.circle(img, pt2, 3, color, -1)

    def sync_callback(self, depth_msg, ir_msg):
        try:
            is_global_search = False 
            search_mode_text = "NONE"
            current_time = time.time()

            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            cv_ir = self.bridge.imgmsg_to_cv2(ir_msg, desired_encoding='mono8')
            
            norm_ir = cv2.normalize(cv_ir, None, 0, 255, cv2.NORM_MINMAX)
            gamma = 1.8 
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_ir = cv2.LUT(norm_ir, table)
            enhanced_ir = self.clahe.apply(gamma_ir)
            cv_ir_color = cv2.cvtColor(enhanced_ir, cv2.COLOR_GRAY2BGR)

            results = self.model.track(cv_ir_color, persist=True, conf=0.5, verbose=False)
            
            candidates = {}
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.xy.cpu().numpy()
                
                for i, tid in enumerate(track_ids):
                    is_human, ratio = self.analyze_skeleton_smart(keypoints[i])
                    box = boxes[i]
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    mean_d, std_d = self.analyze_depth_texture(cv_depth, keypoints[i])
                    hist, brightness = self.compute_histogram(enhanced_ir, box)
                    
                    candidates[tid] = {
                        'box': box, 'center': np.array([cx, cy]), 
                        'depth': mean_d, 'ratio': ratio, 'kps': keypoints[i],
                        'depth_std': std_d, 'hist': hist, 'bright': brightness
                    }

            matched_tid = None
            
            # --- 场景 A: 尚未锁定 ---
            if not self.locked:
                cv2.putText(cv_ir_color, "RAISE HAND TO LOCK", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                hand_raisers = []
                img_center_x = cv_ir.shape[1] / 2
                
                for tid, data in candidates.items():
                    x1, y1 = int(data['box'][0]), int(data['box'][1])
                    debug_status = "WAITING"
                    
                    is_raising, _ = self.gesture_detector.check_raise_hand(data['kps'])
                    is_solid = data['depth_std'] < self.max_depth_std
                    
                    if is_raising:
                        if data['depth'] <= 300: debug_status = "TOO CLOSE"
                        elif not is_solid: debug_status = f"NOISY ({data['depth_std']:.0f})"
                        else:
                            debug_status = "LOCKING..."
                            dist_to_center = abs(data['center'][0] - img_center_x)
                            hand_raisers.append({'id': tid, 'dist': dist_to_center, 'data': data})
                            self.log_data("UNLOCKED", None, tid, 0.0, 0.0, "DETECTED", "Hand Raised")
                    
                    cv2.putText(cv_ir_color, debug_status, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if len(hand_raisers) > 0:
                    hand_raisers.sort(key=lambda x: x['dist'])
                    winner = hand_raisers[0]
                    self.internal_target_id = winner['id']
                    self.locked = True
                    matched_tid = winner['id']
                    d = winner['data']
                    self.last_pos = d['center']
                    self.last_depth = d['depth']
                    self.anchor_ratio = d['ratio'] 
                    self.anchor_hist = d['hist']
                    self.last_timestamp = current_time
                    self.last_velocity = np.array([0.0, 0.0])
                    self.get_logger().info(f"LOCKED! ID:{matched_tid}")
                    self.log_data("LOCKING", matched_tid, matched_tid, 1.0, 0.0, "LOCKED", "Init")
                    self.coexisted_strangers = set()
                    for tid in candidates.keys():
                        if tid != matched_tid: self.coexisted_strangers.add(tid)

            # --- 场景 B: 已锁定 ---
            else:
                is_global_search = (self.lost_frame_count > self.global_search_thresh)
                search_mode_text = "GLOBAL" if is_global_search else "LOCAL"

                if self.internal_target_id in candidates:
                    d = candidates[self.internal_target_id]
                    curr_hist = d['hist']
                    curr_bright = d['bright']
                    curr_pos = d['center']
                    
                    dt = current_time - self.last_timestamp
                    if dt > 0:
                        velocity = (curr_pos - self.last_pos) / dt 
                        self.last_velocity = self.last_velocity * 0.7 + velocity * 0.3
                    self.last_timestamp = current_time 
                    
                    sim = 0.0
                    if self.anchor_hist is not None and curr_hist is not None:
                        sim = self.compare_histograms(self.anchor_hist, curr_hist)
                        
                        is_good_lighting = (30 < curr_bright < 220)
                        if sim > 0.7 and is_good_lighting: 
                            cv2.accumulateWeighted(curr_hist, self.anchor_hist, 0.05)
                            
                    # 🟢 [核心修改] 允许自适应更新 Ratio，不再歧视坐姿
                    if d['ratio'] and self.anchor_ratio:
                        # 只要确定是本人（YOLO ID一致），就缓慢学习他的新姿态
                        self.anchor_ratio = self.anchor_ratio * 0.95 + d['ratio'] * 0.05
                    
                    matched_tid = self.internal_target_id
                    self.log_data("TRACKING", self.internal_target_id, matched_tid, sim, 0.0, "MATCH", "Consistent")
                    
                    if self.internal_target_id in self.candidate_stability:
                        del self.candidate_stability[self.internal_target_id]

                    for tid in candidates.keys():
                        if tid != matched_tid: self.coexisted_strangers.add(tid)
                
                elif self.last_pos is not None:
                    best_score = -9999
                    best_tid = None
                    self.log_data(search_mode_text, self.internal_target_id, -1, 0, 0, "SEARCHING", "Lost")
                    
                    if is_global_search: allowed_radius = 99999.0 
                    else: allowed_radius = 300 + (self.lost_frame_count * 20)

                    if self.lost_frame_count > 30:
                        predicted_pos = self.last_pos 
                        use_momentum = False
                    else:
                        time_lost = self.lost_frame_count * 0.033
                        predicted_pos = self.last_pos + self.last_velocity * time_lost
                        use_momentum = True

                    current_candidate_ids = set(candidates.keys())
                    for cid in list(self.candidate_stability.keys()):
                        if cid not in current_candidate_ids:
                            del self.candidate_stability[cid]

                    for tid, data in candidates.items():
                        if data['depth'] == 0: continue
                        if data['depth_std'] > self.max_depth_std: continue
                        
                        sim = 0.0
                        if self.anchor_hist is not None and data['hist'] is not None:
                            sim = self.compare_histograms(self.anchor_hist, data['hist'])
                            
                        dist_to_pred = np.linalg.norm(data['center'] - predicted_pos)
                        
                        ratio_diff = 1.0
                        curr_ratio = data['ratio']
                        if self.anchor_ratio and curr_ratio:
                            ratio_diff = abs(curr_ratio - self.anchor_ratio)

                        reject_reason = ""
                        is_overexposed = (data['bright'] > 220)
                        
                        if tid in self.coexisted_strangers:
                            if sim < 0.95: reject_reason = "Coexisted Stranger"
                        
                        if dist_to_pred > allowed_radius: 
                            if use_momentum: reject_reason = "Momentum Mismatch"
                            else: reject_reason = "Too Far"

                        # 🟢 [修改] 删除了 Posture Clash (Sit vs Stand) 检查
                        # 取而代之的是纯粹的 Ratio Diff 检查
                        if not reject_reason and ratio_diff > self.strict_ratio_thresh:
                            reject_reason = f"Skeleton Diff ({ratio_diff:.2f})"

                        if not is_global_search:
                            depth_jump = abs(data['depth'] - self.last_depth)
                            allowed_depth_jump = 500 + (self.lost_frame_count * 100)
                            if depth_jump > allowed_depth_jump:
                                reject_reason = f"Depth Jump ({depth_jump:.0f})"

                        if is_global_search:
                            thresh = 0.40 if is_overexposed else self.recover_hist_thresh
                            if sim < thresh: reject_reason = "Global Sim Low"
                            score = sim * 1000 
                        else:
                            thresh = 0.30 if is_overexposed else 0.60
                            if sim < thresh: reject_reason = "Local Sim Low"
                            score = sim*500 - dist_to_pred 
                        
                        if reject_reason:
                            self.candidate_stability[tid] = 0
                        else:
                            self.candidate_stability[tid] = self.candidate_stability.get(tid, 0) + 1
                            if self.candidate_stability[tid] < self.STABILITY_THRESHOLD:
                                reject_reason = f"Verifying ({self.candidate_stability[tid]}/{self.STABILITY_THRESHOLD})"
                        
                        self.log_data(search_mode_text, self.internal_target_id, tid, sim, ratio_diff, "CANDIDATE" if not reject_reason else "REJECT", reject_reason)
                            
                        if not reject_reason:
                            if score > best_score:
                                best_score = score
                                best_tid = tid
                    
                    min_score = 500 if is_global_search else 50
                    if best_tid and best_score > min_score:
                        self.internal_target_id = best_tid
                        matched_tid = best_tid
                        self.get_logger().warn(f">>> RECOVERED: {best_tid} Sim:{sim:.2f} <<<")
                        self.log_data(search_mode_text, self.internal_target_id, best_tid, sim, ratio_diff, "RECOVERED", "Success")
                        self.last_velocity = np.array([0.0, 0.0])
                        self.last_timestamp = current_time

            # --- 绘图 ---
            display_img = cv_ir_color.copy()
            if matched_tid is not None:
                self.lost_frame_count = 0
                d = candidates[matched_tid]
                self.last_pos = d['center']
                if d['depth'] > 0: self.last_depth = d['depth']
                
                # 🟢 锁定时，始终缓慢更新 Ratio，不管是站是坐
                if d['ratio'] and self.anchor_ratio:
                    self.anchor_ratio = self.anchor_ratio * 0.95 + d['ratio'] * 0.05
                
                x1, y1, x2, y2 = map(int, d['box'])
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                self.draw_skeleton(display_img, d['kps'], (0, 0, 255))
                
                cv2.arrowedLine(display_img, 
                                (int(d['center'][0]), int(d['center'][1])),
                                (int(d['center'][0] + self.last_velocity[0]*0.5), int(d['center'][1] + self.last_velocity[1]*0.5)),
                                (0, 255, 255), 2)
                
                label = f"OWNER [{self.fixed_display_id}]"
                cv2.putText(display_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                curr_sim = 1.0
                if self.anchor_hist is not None and d['hist'] is not None:
                    curr_sim = self.compare_histograms(self.anchor_hist, d['hist'])
                cv2.putText(display_img, f"Sim:{curr_sim:.2f}", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            else:
                if self.locked:
                    self.lost_frame_count += 1
                    status = f"SEARCHING...({search_mode_text})"
                    cv2.putText(display_img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
                    if self.lost_frame_count <= 30 and self.last_pos is not None:
                        time_lost = self.lost_frame_count * 0.033
                        pred_pos = self.last_pos + self.last_velocity * time_lost
                        cv2.circle(display_img, (int(pred_pos[0]), int(pred_pos[1])), 20, (0, 255, 255), -1)

            for tid, data in candidates.items():
                if tid != matched_tid:
                    x1, y1, x2, y2 = map(int, data['box'])
                    color = (0, 255, 0)
                    msg_extra = ""
                    if tid in self.candidate_stability and self.candidate_stability[tid] > 0:
                        color = (0, 255, 255)
                        msg_extra = f" Verifying {self.candidate_stability[tid]}/5"
                    
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                    self.draw_skeleton(display_img, data['kps'], color)
                    
                    if not self.locked: pass
                    else:
                        sim_str = "0.00"
                        if self.anchor_hist is not None and data['hist'] is not None:
                            sim = self.compare_histograms(self.anchor_hist, data['hist'])
                            sim_str = f"{sim:.2f}"
                        
                        tag = ""
                        if tid in self.coexisted_strangers: tag = "[CO-EXIST]"
                        cv2.putText(display_img, f"Sim:{sim_str}{tag}{msg_extra}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            self.pub_annotated.publish(self.bridge.cv2_to_imgmsg(display_img, encoding="bgr8"))
            sys.stdout.flush() 
            
        except Exception as e:
            import traceback
            self.get_logger().error(f"Error: {e}\n{traceback.format_exc()}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthSkeletonTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
