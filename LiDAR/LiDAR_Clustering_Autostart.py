#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


# =========================
# Utility Functions
# =========================

def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


# =========================
# Data Structures
# =========================

@dataclass
class Cluster:
    points_xy: np.ndarray          # shape (N, 2)
    centroid: np.ndarray           # shape (2,)
    width: float                   # meters
    n_points: int


@dataclass
class Track:
    track_id: int
    x: np.ndarray                  # state [px, py, vx, vy] shape (4,)
    P: np.ndarray                  # covariance shape (4,4)
    age: int = 0
    hits: int = 0
    misses: int = 0
    last_update_time: float = 0.0

    @property
    def pos(self) -> np.ndarray:
        return self.x[0:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:4]


# =========================
# Kalman Filter (CV model)
# =========================

class KalmanCV:
    """
    Constant-velocity (CV) Kalman filter for 2D.
    State: [px, py, vx, vy]
    Measurement: [px, py]
    """

    def __init__(self, q_pos: float, q_vel: float, r_meas: float):
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_meas = r_meas

    def F(self, dt: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def H(self) -> np.ndarray:
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

    def Q(self, dt: float) -> np.ndarray:
        return np.diag([
            self.q_pos * dt * dt,
            self.q_pos * dt * dt,
            self.q_vel * dt,
            self.q_vel * dt
        ]).astype(np.float32)

    def R(self) -> np.ndarray:
        return np.diag([self.r_meas, self.r_meas]).astype(np.float32)

    def predict(self, tr: Track, dt: float) -> None:
        F = self.F(dt)
        tr.x = F @ tr.x
        tr.P = F @ tr.P @ F.T + self.Q(dt)

    def update(self, tr: Track, z: np.ndarray) -> None:
        H = self.H()
        R = self.R()
        y = z - (H @ tr.x)
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ y
        I = np.eye(4, dtype=np.float32)
        tr.P = (I - K @ H) @ tr.P


# =========================
# Main ROS2 Node
# =========================

class LidarPersonTrackerNode(Node):
    def __init__(self):
        super().__init__("lidar_person_tracker")

        # ---------- ROS params ----------
        # ROI
        self.declare_parameter("roi_angle_deg", 90.0)
        self.declare_parameter("range_min", 0.2)
        self.declare_parameter("range_max", 6.0)

        # Clustering
        self.declare_parameter("cluster_a", 0.1)
        self.declare_parameter("cluster_b", 0.1)

        # Candidate filtering
        self.declare_parameter("width_min", 0.15)
        self.declare_parameter("width_max", 0.80)
        self.declare_parameter("min_points", 5)

        # Tracking
        self.declare_parameter("gate_radius", 0.60)
        self.declare_parameter("hit_min", 3)
        self.declare_parameter("miss_max", 5)

        # Kalman noise
        self.declare_parameter("q_pos", 0.5)
        self.declare_parameter("q_vel", 1.0)
        self.declare_parameter("r_meas", 0.03)

        # Target tracking
        self.declare_parameter("max_lost_frames", 100)
        self.declare_parameter("target_min_distance", 0.5)

        # Visualization
        self.declare_parameter("viz_enable", True)
        self.declare_parameter("viz_scale", 80.0)
        self.declare_parameter("viz_size", 800)
        self.declare_parameter("viz_show_roi", True)

        scan_topic = self.declare_parameter("scan_topic", "/scan").value
        self.sub = self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)

        # Tracker state
        self.kf = KalmanCV(
            q_pos=float(self.get_parameter("q_pos").value),
            q_vel=float(self.get_parameter("q_vel").value),
            r_meas=float(self.get_parameter("r_meas").value)
        )
        self.tracks: List[Track] = []
        self.next_track_id: int = 1
        self.last_stamp_sec: Optional[float] = None

        # Target tracking state
        self.target_track_id: Optional[int] = None
        self.target_lost_frames: int = 0
        self.space_pressed: bool = False

        # OpenCV window
        self.viz_enable = bool(self.get_parameter("viz_enable").value)
        if self.viz_enable:
            cv2.namedWindow("LiDAR Person Tracking (Top View)", cv2.WINDOW_NORMAL)

        self.get_logger().info("LidarPersonTrackerNode started (REAR ±90deg mode with Target Tracking)")

    # -------------------------
    # Step 1: Preprocess
    # -------------------------
    def preprocess_scan(self, msg: LaserScan) -> Tuple[np.ndarray, np.ndarray]:
        """
        검증된 데이터 수집 방식 사용 - 후방(180도) 기준 ±90도
        Returns:
          points_xy: shape (N, 2) - [x, y] coordinates
          ranges: shape (N,) - corresponding ranges
        """
        roi_angle_deg = float(self.get_parameter("roi_angle_deg").value)
        rmin = float(self.get_parameter("range_min").value)
        rmax = float(self.get_parameter("range_max").value)
        half_fov = math.radians(roi_angle_deg)

        points_xy = []
        ranges_filtered = []

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            if r < rmin or r > rmax:
                continue

            theta = msg.angle_min + i * msg.angle_increment

            # 후방(180도) 기준 ±90도 ROI
            theta_rel = wrap_to_pi(theta - math.pi)
            if abs(theta_rel) > half_fov:
                continue

            # Cartesian 좌표 변환
            x = r * math.cos(theta)
            y = r * math.sin(theta)

            # 후방을 정면으로 보이게 180도 회전
            x_vis = -x
            y_vis = -y

            points_xy.append([x_vis, y_vis])
            ranges_filtered.append(r)

        if not points_xy:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        return np.array(points_xy, dtype=np.float32), np.array(ranges_filtered, dtype=np.float32)

    # -------------------------
    # Step 2: Clustering
    # -------------------------
    def cluster_points_angle_adaptive(self, points_xy: np.ndarray) -> List[np.ndarray]:
        """
        Angle-based clustering with adaptive threshold.
        """
        a = float(self.get_parameter("cluster_a").value)
        b = float(self.get_parameter("cluster_b").value)
        min_points = int(self.get_parameter("min_points").value)

        if points_xy.shape[0] == 0:
            return []

        clusters: List[List[np.ndarray]] = []
        current = [points_xy[0]]

        for i in range(points_xy.shape[0] - 1):
            p = points_xy[i]
            q = points_xy[i + 1]

            rp = float(np.hypot(p[0], p[1]))
            thr = a * rp + b

            dx = float(q[0] - p[0])
            dy = float(q[1] - p[1])
            dist2 = dx * dx + dy * dy

            if dist2 <= thr * thr:
                current.append(q)
            else:
                if len(current) >= min_points:
                    clusters.append(current)
                current = [q]

        if len(current) >= min_points:
            clusters.append(current)

        return [np.asarray(c, dtype=np.float32) for c in clusters]

    # -------------------------
    # Step 3: Feature + Candidate Filter
    # -------------------------
    def compute_cluster(self, pts: np.ndarray) -> Cluster:
        centroid = np.mean(pts, axis=0)
        span_x = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        span_y = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        width = max(span_x, span_y)
        return Cluster(points_xy=pts, centroid=centroid, width=width, n_points=int(pts.shape[0]))

    def filter_person_candidates(self, clusters_pts: List[np.ndarray]) -> Tuple[List[Cluster], np.ndarray]:
        width_min = float(self.get_parameter("width_min").value)
        width_max = float(self.get_parameter("width_max").value)
        min_points = int(self.get_parameter("min_points").value)

        clusters: List[Cluster] = []
        detections: List[np.ndarray] = []

        for pts in clusters_pts:
            if pts.shape[0] < min_points:
                continue
            c = self.compute_cluster(pts)
            if (c.width >= width_min) and (c.width <= width_max):
                clusters.append(c)
                detections.append(c.centroid)

        if len(detections) == 0:
            return clusters, np.zeros((0, 2), dtype=np.float32)

        return clusters, np.asarray(detections, dtype=np.float32)

    # -------------------------
    # Step 4: Tracking
    # -------------------------
    def associate_nn_gate(self, tracks: List[Track], detections: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        gate = float(self.get_parameter("gate_radius").value)
        gate2 = gate * gate

        T = len(tracks)
        D = detections.shape[0]
        if T == 0 or D == 0:
            return [], list(range(T)), list(range(D))

        cost = np.full((T, D), np.inf, dtype=np.float32)
        for ti, tr in enumerate(tracks):
            px, py = float(tr.pos[0]), float(tr.pos[1])
            dx = detections[:, 0] - px
            dy = detections[:, 1] - py
            dist2 = dx * dx + dy * dy
            cost[ti, dist2 <= gate2] = dist2[dist2 <= gate2]

        matches: List[Tuple[int, int]] = []
        used_t = set()
        used_d = set()

        candidates = np.argwhere(np.isfinite(cost))
        if candidates.size == 0:
            return [], list(range(T)), list(range(D))

        candidates = sorted(candidates, key=lambda ij: cost[ij[0], ij[1]])

        for ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            if not np.isfinite(cost[ti, di]):
                continue
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)

        unmatched_tracks = [i for i in range(T) if i not in used_t]
        unmatched_dets = [j for j in range(D) if j not in used_d]
        return matches, unmatched_tracks, unmatched_dets

    def step_tracker(self, detections: np.ndarray, dt: float, now_sec: float) -> None:
        for tr in self.tracks:
            self.kf.predict(tr, dt)
            tr.age += 1

        matches, un_tr, un_det = self.associate_nn_gate(self.tracks, detections)

        for ti, di in matches:
            z = detections[di]
            tr = self.tracks[ti]
            self.kf.update(tr, z)
            tr.hits += 1
            tr.misses = 0
            tr.last_update_time = now_sec

        for ti in un_tr:
            self.tracks[ti].misses += 1

        for di in un_det:
            z = detections[di]
            x0 = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float32)
            P0 = np.diag([0.2, 0.2, 1.0, 1.0]).astype(np.float32)
            self.tracks.append(Track(
                track_id=self.next_track_id,
                x=x0,
                P=P0,
                age=1,
                hits=1,
                misses=0,
                last_update_time=now_sec
            ))
            self.next_track_id += 1

        miss_max = int(self.get_parameter("miss_max").value)
        self.tracks = [t for t in self.tracks if t.misses <= miss_max]

    def get_confirmed_tracks(self) -> List[Track]:
        hit_min = int(self.get_parameter("hit_min").value)
        return [t for t in self.tracks if t.hits >= hit_min]

    # -------------------------
    # Target Tracking Functions
    # -------------------------
    def select_center_track(self) -> Optional[int]:
        """
        현재 confirmed tracks 중 가장 중앙에 가까운 트랙 선택
        Returns: track_id or None
        """
        confirmed = self.get_confirmed_tracks()
        
        if len(confirmed) == 0:
            return None
        
        min_distance = 0.5  # 최소 거리
        best_track = None
        min_score = float('inf')
        
        for tr in confirmed:
            y_offset = abs(tr.pos[1])  # 중앙선(y=0)으로부터의 거리
            x_dist = tr.pos[0]         # x축 거리
            
            # 최소 거리 이상인 객체만
            if x_dist > min_distance:
                # 중앙성 점수: y 오프셋 우선, x는 보조
                score = y_offset + (x_dist * 0.05)
                
                if score < min_score:
                    min_score = score
                    best_track = tr
        
        return best_track.track_id if best_track else None

    def update_target_tracking(self) -> None:
        """
        추적 대상이 현재 프레임에 있는지 확인
        """
        if self.target_track_id is None:
            return
        
        target_exists = False
        hit_min = int(self.get_parameter("hit_min").value)
        
        for tr in self.tracks:
            if tr.track_id == self.target_track_id:
                target_exists = True
                if tr.hits >= hit_min:
                    self.target_lost_frames = 0
                else:
                    self.target_lost_frames += 1
                break
        
        if not target_exists:
            self.target_lost_frames += 1
        
        max_lost = int(self.get_parameter("max_lost_frames").value)
        if self.target_lost_frames >= max_lost:
            self.get_logger().warn(f"Target lost: ID {self.target_track_id} after {self.target_lost_frames} frames")
            self.target_track_id = None
            self.target_lost_frames = 0

    def reset_target_tracking(self) -> None:
        """
        추적 대상 리셋
        """
        if self.target_track_id is not None:
            self.get_logger().info(f"Tracking reset: ID {self.target_track_id}")
        
        self.target_track_id = None
        self.target_lost_frames = 0

    # -------------------------
    # Step 5: Visualization
    # -------------------------
    def draw_tracking_status(self, img: np.ndarray) -> None:
        """
        화면 상단에 추적 상태 표시
        """
        status_y = 55
        max_lost = int(self.get_parameter("max_lost_frames").value)
        
        if self.target_track_id is None:
            status_text = "Tracking: NONE (Press SPACE to lock center object)"
            color = (100, 100, 100)  # 회색
        else:
            if self.target_lost_frames == 0:
                status_text = f"Tracking: ID {self.target_track_id} [LOCKED]"
                color = (0, 255, 0)  # 초록색
            elif self.target_lost_frames < max_lost:
                status_text = f"Tracking: ID {self.target_track_id} [LOST {self.target_lost_frames}/{max_lost}]"
                color = (0, 165, 255)  # 주황색
            else:
                status_text = "Tracking: LOST"
                color = (0, 0, 255)  # 빨간색
        
        cv2.putText(img, status_text, (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 도움말
        help_text = "Keys: SPACE=Lock Center | R=Reset"
        cv2.putText(img, help_text, (10, status_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def draw_top_view(self, points_xy: np.ndarray, clusters: List[Cluster], detections: np.ndarray) -> None:
        if not self.viz_enable:
            return

        size = int(self.get_parameter("viz_size").value)
        scale = float(self.get_parameter("viz_scale").value)
        show_roi = bool(self.get_parameter("viz_show_roi").value)

        img = np.zeros((size, size, 3), dtype=np.uint8)
        origin = (size // 2, int(size * 0.85))

        def to_px(xy: np.ndarray) -> Tuple[int, int]:
            x, y = float(xy[0]), float(xy[1])
            u = int(origin[0] + y * scale)
            v = int(origin[1] - x * scale)
            return u, v

        if show_roi:
            roi_angle_deg = float(self.get_parameter("roi_angle_deg").value)
            rmax = float(self.get_parameter("range_max").value)
            a = math.radians(roi_angle_deg)
            p1 = to_px(np.array([rmax * math.cos(-a), rmax * math.sin(-a)], dtype=np.float32))
            p2 = to_px(np.array([rmax * math.cos(+a), rmax * math.sin(+a)], dtype=np.float32))
            cv2.line(img, origin, p1, (50, 50, 50), 1)
            cv2.line(img, origin, p2, (50, 50, 50), 1)
            cv2.circle(img, origin, int(rmax * scale), (30, 30, 30), 1)

        for p in points_xy:
            u, v = to_px(p)
            if 0 <= u < size and 0 <= v < size:
                img[v, u] = (80, 80, 80)

        for c in clusters:
            pts = c.points_xy
            min_xy = np.min(pts, axis=0)
            max_xy = np.max(pts, axis=0)

            corners = np.array([
                [min_xy[0], min_xy[1]],
                [min_xy[0], max_xy[1]],
                [max_xy[0], max_xy[1]],
                [max_xy[0], min_xy[1]],
            ], dtype=np.float32)
            corners_px = np.array([to_px(xy) for xy in corners], dtype=np.int32)
            cv2.polylines(img, [corners_px.reshape(-1, 1, 2)], True, (0, 255, 255), 1)

            cu, cv = to_px(c.centroid)
            cv2.circle(img, (cu, cv), 4, (0, 255, 255), -1)

        for z in detections:
            u, v = to_px(z)
            cv2.circle(img, (u, v), 3, (0, 180, 0), -1)

        # Tracks 시각화 (색상 구분)
        confirmed = self.get_confirmed_tracks()
        for t in confirmed:
            u, v = to_px(t.pos)
            
            # 타겟 추적 객체 vs 일반 객체 색상 구분
            if t.track_id == self.target_track_id:
                color = (0, 0, 255)      # 빨간색
                thickness = 3
                label_color = (0, 0, 255)
                prefix = "[TARGET] "
            else:
                color = (255, 255, 255)  # 흰색
                thickness = 2
                label_color = (255, 255, 255)
                prefix = ""
            
            cv2.circle(img, (u, v), 6, color, thickness)
            label = f"{prefix}ID:{t.track_id} v={np.linalg.norm(t.vel):.2f}"
            cv2.putText(img, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

            end = t.pos + 0.5 * t.vel
            ue, ve = to_px(end)
            cv2.arrowedLine(img, (u, v), (ue, ve), color, 2, tipLength=0.25)

        # 모드 정보
        wmin = float(self.get_parameter("width_min").value)
        wmax = float(self.get_parameter("width_max").value)
        mp = int(self.get_parameter("min_points").value)
        gate = float(self.get_parameter("gate_radius").value)
        txt = f"REAR Mode (180deg ±90deg) | width[{wmin:.2f},{wmax:.2f}]m | minPts={mp} | gate={gate:.2f}m"
        cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 추적 상태 표시
        self.draw_tracking_status(img)

        cv2.imshow("LiDAR Person Tracking (Top View)", img)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            self.space_pressed = True
        elif key == ord('r') or key == ord('R'):  # R key
            self.reset_target_tracking()

    # -------------------------
    # ROS callback
    # -------------------------
    def on_scan(self, msg: LaserScan):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_stamp_sec is None:
            self.last_stamp_sec = now_sec
            return

        dt = max(1e-3, now_sec - self.last_stamp_sec)
        self.last_stamp_sec = now_sec

        # Step 1: preprocess
        points_xy, ranges_filtered = self.preprocess_scan(msg)

        if points_xy.shape[0] == 0:
            self.draw_top_view(points_xy, [], np.zeros((0, 2), dtype=np.float32))
            return

        # Step 2: clustering
        clusters_pts = self.cluster_points_angle_adaptive(points_xy)

        # Step 3: candidate filter
        candidate_clusters, detections = self.filter_person_candidates(clusters_pts)

        # Step 4: tracking
        self.step_tracker(detections, dt, now_sec)

        # Step 4.5: target tracking update
        self.update_target_tracking()

        # Space 키 입력 처리
        if self.space_pressed:
            self.space_pressed = False
            selected_id = self.select_center_track()
            if selected_id is not None:
                self.target_track_id = selected_id
                self.target_lost_frames = 0
                # 선택된 트랙 정보 로깅
                for tr in self.tracks:
                    if tr.track_id == selected_id:
                        self.get_logger().info(
                            f"Target locked: ID {selected_id} at ({tr.pos[0]:.2f}, {tr.pos[1]:.2f})"
                        )
                        break
            else:
                self.get_logger().warn("No valid track to lock!")

        # Step 5: visualization
        self.draw_top_view(points_xy, candidate_clusters, detections)


def main():
    rclpy.init()
    node = LidarPersonTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.viz_enable:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
