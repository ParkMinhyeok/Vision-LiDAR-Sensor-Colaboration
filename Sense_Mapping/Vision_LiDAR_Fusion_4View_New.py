#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-LiDAR Fusion 4-Split View
================================
4분할 뷰로 Vision(YOLO)와 LiDAR 데이터를 통합 시각화

┌─────────────────┬─────────────────┐
│                 │                 │
│    Vision       │    LiDAR        │
│    (YOLO)       │    (탑뷰)       │
│    카메라 영상  │    점군/클러스터 │
│                 │                 │
├─────────────────┼─────────────────┤
│                 │                 │
│    BEV 퓨전     │    정보 패널    │
│    (Bird's Eye) │    (통계)       │
│    전체 배치도  │    수치 데이터  │
│                 │                 │
└─────────────────┴─────────────────┘
"""

import math
import time
import argparse
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque

import numpy as np
import cv2

# ROS2 imports (optional - will work without ROS2 for testing)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[WARN] ROS2 not available. LiDAR functionality will be simulated.")

# Ultralytics YOLO import (optional)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] Ultralytics not available. Vision functionality will be simulated.")


# =========================
# Utility Functions
# =========================

def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def pixel_to_angle(pixel_x: float, frame_width: int, fov_deg: float) -> float:
    """
    Convert pixel x-coordinate to angle (radians).
    Center of frame = 0 degrees, right = positive, left = negative.
    """
    center_x = frame_width / 2.0
    offset = pixel_x - center_x
    angle_deg = (offset / frame_width) * fov_deg
    return math.radians(angle_deg)


def angle_to_bev_xy(angle_rad: float, distance: float) -> Tuple[float, float]:
    """Convert angle + distance to BEV x,y coordinates."""
    x = distance * math.cos(angle_rad)
    y = distance * math.sin(angle_rad)
    return x, y


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
    # 클러스터 정보 (튜닝용)
    n_points: int = 0              # 클러스터 포인트 수
    cluster_width: float = 0.0     # 클러스터 폭 (m)

    @property
    def pos(self) -> np.ndarray:
        return self.x[0:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:4]


@dataclass
class YoloDetection:
    """YOLO detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]          # center x, y
    track_id: Optional[int] = None   # ByteTrack ID
    angle: float = 0.0               # angle from camera center (radians)


@dataclass
class FusionPair:
    """Vision-LiDAR 퓨전 매칭 쌍"""
    # Vision 기준 (변하지 않음)
    vision_id: int
    vision_angle: float = 0.0
    vision_bbox: Optional[Tuple[int, int, int, int]] = None

    # LiDAR 매칭 정보
    lidar_id: Optional[int] = None
    last_lidar_distance: float = 3.0  # 마지막 알려진 거리
    last_lidar_angle: float = 0.0

    # 매칭 상태
    state: str = "SEARCHING"  # MATCHING, SEARCHING, VISION_ONLY
    match_confidence: int = 0  # 연속 매칭 횟수
    lidar_lost_count: int = 0  # LiDAR 못 찾은 횟수

    # 설정
    CONFIDENCE_THRESHOLD: int = 3  # 확정 매칭까지 필요한 연속 매칭
    MAX_LOST_COUNT: int = 15  # LiDAR 포기까지 허용 프레임


@dataclass
class SharedData:
    """Thread-safe shared data between Vision and LiDAR threads"""
    # Vision data
    vision_frame: Optional[np.ndarray] = None
    vision_annotated: Optional[np.ndarray] = None
    vision_detections: List[YoloDetection] = field(default_factory=list)
    vision_fps: float = 0.0
    vision_active: bool = False
    frame_width: int = 1280
    frame_height: int = 720

    # Vision target tracking (ByteTrack)
    vision_target_id: Optional[int] = None
    vision_last_seen_ts: float = 0.0
    vision_lost_sec: float = 2.0  # timeout before auto-unlock
    vision_target_bbox: Optional[Tuple[int, int, int, int]] = None
    vision_lock_requested: bool = False  # V key pressed
    vision_reset_requested: bool = False  # reset vision lock

    # LiDAR data
    lidar_points_xy: Optional[np.ndarray] = None
    lidar_clusters: List[Cluster] = field(default_factory=list)
    lidar_tracks: List[Track] = field(default_factory=list)
    lidar_detections: Optional[np.ndarray] = None
    lidar_fps: float = 0.0
    lidar_active: bool = False

    # LiDAR target tracking
    lidar_target_id: Optional[int] = None
    lidar_target_lost_frames: int = 0
    lidar_lock_requested: bool = False  # L key pressed
    lidar_reset_requested: bool = False  # reset lidar lock
    lidar_jumped: bool = False  # LiDAR 타겟 튐 감지 플래그
    lidar_jumped_ids: List[int] = field(default_factory=list)  # 튐 감지된 ID 목록

    # ========== Fusion Tracking (SPACE key) ==========
    fusion_pair: Optional[FusionPair] = None
    fusion_lock_requested: bool = False  # SPACE key pressed
    fusion_reset_requested: bool = False  # R key - reset fusion

    # Thread lock
    lock: threading.Lock = field(default_factory=threading.Lock)


# =========================
# Kalman Filter (CV model)
# =========================

class KalmanCV:
    """Constant-velocity (CV) Kalman filter for 2D."""

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
# LiDAR Processing Module
# =========================

class LidarProcessor:
    """LiDAR data processing (clustering, tracking) - 단순 위치 기반"""

    def __init__(self, config: dict):
        self.config = config
        self.tracks: List[Track] = []
        self.next_track_id: int = 1
        self.last_stamp_sec: Optional[float] = None

        # 위치 기반 추적 설정
        self.max_move = config.get('max_move', 0.3)  # 프레임당 최대 이동 거리 (m)
        self.jump_threshold = config.get('jump_threshold', 0.5)  # 튐 감지 임계값 (m)

        # Target tracking
        self.target_track_id: Optional[int] = None
        self.fusion_lidar_id: Optional[int] = None  # Fusion에서 추적 중인 LiDAR ID
        self.target_lost_frames: int = 0
        self.last_target_position: Optional[np.ndarray] = None
        self.target_jumped: bool = False  # 튐 발생 플래그
        self.jumped_track_ids: List[int] = []  # 튐이 감지된 트랙 ID 목록

    def preprocess_scan(self, ranges: np.ndarray, angle_min: float,
                        angle_increment: float) -> Tuple[np.ndarray, np.ndarray]:
        """Process LaserScan data to cartesian coordinates"""
        roi_angle_deg = self.config.get('roi_angle_deg', 90.0)
        rmin = self.config.get('range_min', 0.2)
        rmax = self.config.get('range_max', 6.0)
        half_fov = math.radians(roi_angle_deg)

        points_data = []  # (theta_rel, x_vis, y_vis, r) 튜플 리스트

        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            if r < rmin or r > rmax:
                continue

            theta = angle_min + i * angle_increment
            theta_rel = wrap_to_pi(theta - math.pi)
            if abs(theta_rel) > half_fov:
                continue

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            x_vis = -x
            y_vis = -y

            points_data.append((theta_rel, x_vis, y_vis, r))

        if not points_data:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        # 각도 순서로 정렬 (클러스터링을 위해 연속성 보장)
        points_data.sort(key=lambda x: x[0])

        points_xy = np.array([[p[1], p[2]] for p in points_data], dtype=np.float32)
        ranges_filtered = np.array([p[3] for p in points_data], dtype=np.float32)

        return points_xy, ranges_filtered

    def cluster_points(self, points_xy: np.ndarray) -> List[np.ndarray]:
        """Angle-based adaptive clustering with minimum threshold"""
        a = self.config.get('cluster_a', 0.1)
        b = self.config.get('cluster_b', 0.1)
        min_thr = self.config.get('cluster_min_thr', 0.5)  # 최소 임계값 (가까운 거리용)
        min_points = self.config.get('min_points', 5)

        if points_xy.shape[0] == 0:
            return []

        clusters: List[List[np.ndarray]] = []
        current = [points_xy[0]]

        for i in range(points_xy.shape[0] - 1):
            p = points_xy[i]
            q = points_xy[i + 1]

            rp = float(np.hypot(p[0], p[1]))
            thr = max(a * rp + b, min_thr)  # 최소 임계값 적용

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

    def compute_cluster(self, pts: np.ndarray) -> Cluster:
        centroid = np.mean(pts, axis=0)
        span_x = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        span_y = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        width = max(span_x, span_y)
        return Cluster(points_xy=pts, centroid=centroid, width=width, n_points=int(pts.shape[0]))

    def filter_candidates(self, clusters_pts: List[np.ndarray]) -> Tuple[List[Cluster], np.ndarray]:
        width_min = self.config.get('width_min', 0.15)
        width_max = self.config.get('width_max', 0.80)
        min_points = self.config.get('min_points', 5)
        max_points = self.config.get('max_points', 10)

        clusters: List[Cluster] = []
        detections: List[np.ndarray] = []

        for pts in clusters_pts:
            # 최소/최대 포인트 수 필터링
            if pts.shape[0] < min_points or pts.shape[0] > max_points:
                continue
            c = self.compute_cluster(pts)
            if (c.width >= width_min) and (c.width <= width_max):
                clusters.append(c)
                detections.append(c.centroid)

        if len(detections) == 0:
            return clusters, np.zeros((0, 2), dtype=np.float32)

        return clusters, np.asarray(detections, dtype=np.float32)

    def associate_nn_gate(self, tracks: List[Track], detections: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        gate = self.config.get('gate_radius', 0.60)
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

    def step_tracker(self, detections: np.ndarray, clusters: List[Cluster], dt: float, now_sec: float) -> None:
        """단순 위치 기반 추적 (칼만 필터 없음)"""
        self.target_jumped = False  # 튐 플래그 리셋
        self.jumped_track_ids = []  # 튐 목록 리셋

        # 트랙 나이 증가
        for tr in self.tracks:
            tr.age += 1

        # 단순 최근접 매칭 (max_move 제한)
        matches, un_tr, un_det = self._associate_by_distance(self.tracks, detections)

        # 매칭된 트랙 업데이트
        for ti, di in matches:
            z = detections[di]
            tr = self.tracks[ti]

            # 이전 위치 저장 (속도 계산용)
            prev_pos = tr.pos.copy()

            # 위치 업데이트 (단순 대입)
            tr.x[0] = z[0]
            tr.x[1] = z[1]

            # 속도 계산 (이전 위치와의 차이)
            if dt > 0:
                tr.x[2] = (z[0] - prev_pos[0]) / dt
                tr.x[3] = (z[1] - prev_pos[1]) / dt

            tr.hits += 1
            tr.misses = 0
            tr.last_update_time = now_sec

            # 클러스터 정보 업데이트
            if di < len(clusters):
                tr.n_points = clusters[di].n_points
                tr.cluster_width = clusters[di].width

            # 튐 감지 (타겟 트랙 또는 Fusion LiDAR인 경우)
            is_target = (tr.track_id == self.target_track_id)
            is_fusion = (tr.track_id == self.fusion_lidar_id)

            if is_target or is_fusion:
                move_dist = float(np.linalg.norm(z - prev_pos))
                if move_dist > self.jump_threshold:
                    self.jumped_track_ids.append(tr.track_id)
                    if is_target:
                        self.target_jumped = True
                    print(f"[LIDAR] 튐 감지! ID={tr.track_id}, 이동거리={move_dist:.2f}m")

        # 매칭 안된 트랙 miss 증가
        for ti in un_tr:
            self.tracks[ti].misses += 1

        # 새 트랙 생성
        for di in un_det:
            z = detections[di]
            x0 = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float32)
            P0 = np.diag([0.1, 0.1, 0.1, 0.1]).astype(np.float32)  # 사용 안함

            n_pts = clusters[di].n_points if di < len(clusters) else 0
            c_width = clusters[di].width if di < len(clusters) else 0.0

            self.tracks.append(Track(
                track_id=self.next_track_id,
                x=x0,
                P=P0,
                age=1,
                hits=1,
                misses=0,
                last_update_time=now_sec,
                n_points=n_pts,
                cluster_width=c_width
            ))
            self.next_track_id += 1

        # 오래된 트랙 제거
        miss_max = self.config.get('miss_max', 5)
        self.tracks = [t for t in self.tracks if t.misses <= miss_max]

    def _associate_by_distance(self, tracks: List[Track], detections: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """단순 거리 기반 매칭 (max_move 제한)"""
        T = len(tracks)
        D = detections.shape[0] if len(detections) > 0 else 0

        if T == 0 or D == 0:
            return [], list(range(T)), list(range(D))

        # 거리 행렬 계산
        cost = np.full((T, D), np.inf, dtype=np.float32)
        for ti, tr in enumerate(tracks):
            for di in range(D):
                dist = np.linalg.norm(detections[di] - tr.pos)
                if dist <= self.max_move:
                    cost[ti, di] = dist

        # 탐욕적 매칭 (가장 가까운 것부터)
        matches = []
        used_t = set()
        used_d = set()

        while True:
            # 최소 비용 찾기
            min_val = np.inf
            min_ti, min_di = -1, -1

            for ti in range(T):
                if ti in used_t:
                    continue
                for di in range(D):
                    if di in used_d:
                        continue
                    if cost[ti, di] < min_val:
                        min_val = cost[ti, di]
                        min_ti, min_di = ti, di

            if min_val == np.inf:
                break

            matches.append((min_ti, min_di))
            used_t.add(min_ti)
            used_d.add(min_di)

        unmatched_tracks = [i for i in range(T) if i not in used_t]
        unmatched_dets = [j for j in range(D) if j not in used_d]

        return matches, unmatched_tracks, unmatched_dets

    def get_confirmed_tracks(self) -> List[Track]:
        hit_min = self.config.get('hit_min', 3)
        return [t for t in self.tracks if t.hits >= hit_min]

    def select_center_track(self) -> Optional[int]:
        """Select track closest to center"""
        confirmed = self.get_confirmed_tracks()
        if len(confirmed) == 0:
            return None

        min_distance = 0.5
        best_track = None
        min_score = float('inf')

        for tr in confirmed:
            y_offset = abs(tr.pos[1])
            x_dist = tr.pos[0]

            if x_dist > min_distance:
                score = y_offset + (x_dist * 0.05)
                if score < min_score:
                    min_score = score
                    best_track = tr

        return best_track.track_id if best_track else None


# =========================
# Fusion Manager
# =========================

class FusionManager:
    """Vision-LiDAR 퓨전 매칭 관리"""

    def __init__(self, config: dict):
        self.config = config
        self.angle_tolerance = config.get('fusion_angle_tolerance', 0.25)  # ~14도
        self.distance_tolerance = config.get('fusion_distance_tolerance', 1.0)  # 1m

    def find_center_vision_target(self, detections: List[YoloDetection]) -> Optional[YoloDetection]:
        """화면 중앙에 가장 가까운 Vision 타겟 찾기"""
        if not detections:
            return None

        # 각도가 0에 가장 가까운 (중앙) 감지 찾기
        valid_detections = [d for d in detections if d.track_id is not None]
        if not valid_detections:
            return None

        return min(valid_detections, key=lambda d: abs(d.angle))

    def find_center_lidar_target(self, lidar_tracks: List[Track]) -> Optional[Track]:
        """가장 가깝고 중앙에 있는 LiDAR 타겟 찾기 (독립 선택)"""
        if not lidar_tracks:
            return None

        # 점수 = 중앙에서 벗어난 정도(각도) + 거리
        # 낮을수록 좋음
        min_distance = 0.3  # 최소 거리 (너무 가까운 것 제외)

        candidates = []
        for track in lidar_tracks:
            dist = float(np.linalg.norm(track.pos))
            if dist < min_distance:
                continue

            angle = abs(math.atan2(track.pos[1], track.pos[0]))  # 중앙에서 벗어난 각도
            # 점수: 각도 편차 + 거리 (가깝고 중앙일수록 낮음)
            score = angle * 1.0 + dist * 0.1
            candidates.append((track, score, dist))

        if not candidates:
            return None

        # 점수가 가장 낮은 (가깝고 중앙인) 트랙 반환
        best = min(candidates, key=lambda x: x[1])
        return best[0]

    def find_lidar_by_angle_and_distance(
        self,
        vision_angle: float,
        last_distance: float,
        lidar_tracks: List[Track]
    ) -> Optional[Track]:
        """Vision 각도 + 마지막 LiDAR 거리 기준으로 LiDAR 트랙 찾기"""
        if not lidar_tracks:
            return None

        candidates = []

        for track in lidar_tracks:
            track_angle = math.atan2(track.pos[1], track.pos[0])
            track_dist = float(np.linalg.norm(track.pos))

            # 조건 1: 각도가 비슷한가?
            angle_diff = abs(wrap_to_pi(vision_angle - track_angle))
            if angle_diff > self.angle_tolerance:
                continue

            # 조건 2: 거리가 비슷한가? (마지막 LiDAR 거리 기준)
            dist_diff = abs(last_distance - track_dist)
            if dist_diff > self.distance_tolerance:
                continue

            # 점수 계산 (낮을수록 좋음)
            score = angle_diff * 2.0 + dist_diff * 0.5
            candidates.append((track, score, track_dist, track_angle))

        if candidates:
            best = min(candidates, key=lambda x: x[1])
            return best[0]

        return None

    def update_fusion(
        self,
        fusion_pair: FusionPair,
        vision_detections: List[YoloDetection],
        lidar_tracks: List[Track],
        lidar_jumped: bool = False
    ) -> FusionPair:
        """퓨전 상태 업데이트"""

        # 1. Vision ID가 아직 존재하는지 확인
        vision_det = None
        for det in vision_detections:
            if det.track_id == fusion_pair.vision_id:
                vision_det = det
                break

        # Vision ID가 사라지면 퓨전 종료
        if vision_det is None:
            return None  # 퓨전 해제

        # Vision 정보 업데이트
        fusion_pair.vision_angle = vision_det.angle
        fusion_pair.vision_bbox = vision_det.bbox

        # 2. 현재 LiDAR ID가 존재하는지 확인
        current_lidar = None
        if fusion_pair.lidar_id is not None:
            for track in lidar_tracks:
                if track.track_id == fusion_pair.lidar_id:
                    current_lidar = track
                    break

        # 3. LiDAR 튐 감지 → 강제 재매칭
        if lidar_jumped and current_lidar is not None:
            print(f"[FUSION] LiDAR 튐 감지! Vision 기반 재탐색 시작")
            # 기존 LiDAR ID 무효화, Vision 기반으로 재탐색
            fusion_pair.lidar_id = None
            current_lidar = None
            fusion_pair.state = "SEARCHING"
            fusion_pair.match_confidence = max(0, fusion_pair.match_confidence - 10)

        # 4. LiDAR 상태에 따른 처리
        if current_lidar is not None:
            # LiDAR 존재 → 매칭 유지
            fusion_pair.state = "MATCHING"
            fusion_pair.match_confidence = min(fusion_pair.match_confidence + 1, 100)
            fusion_pair.lidar_lost_count = 0
            fusion_pair.last_lidar_distance = float(np.linalg.norm(current_lidar.pos))
            fusion_pair.last_lidar_angle = math.atan2(current_lidar.pos[1], current_lidar.pos[0])

        else:
            # LiDAR 없음 → 재매칭 시도
            fusion_pair.lidar_lost_count += 1

            # Vision 각도 + 마지막 LiDAR 거리로 새 트랙 검색
            new_track = self.find_lidar_by_angle_and_distance(
                fusion_pair.vision_angle,
                fusion_pair.last_lidar_distance,
                lidar_tracks
            )

            if new_track is not None:
                # 새 LiDAR 찾음
                fusion_pair.lidar_id = new_track.track_id
                fusion_pair.last_lidar_distance = float(np.linalg.norm(new_track.pos))
                fusion_pair.last_lidar_angle = math.atan2(new_track.pos[1], new_track.pos[0])
                fusion_pair.state = "MATCHING"
                fusion_pair.lidar_lost_count = 0
                # confidence는 리셋하지 않고 유지 (같은 물체일 가능성)
                print(f"[FUSION] LiDAR 재매칭: 새 ID={new_track.track_id}")

            elif fusion_pair.lidar_lost_count > fusion_pair.MAX_LOST_COUNT:
                # 너무 오래 못 찾음 → Vision만으로 전환
                fusion_pair.state = "VISION_ONLY"
                fusion_pair.lidar_id = None
                print(f"[FUSION] LiDAR 포기, Vision만 추적")

            else:
                # 아직 찾는 중
                fusion_pair.state = "SEARCHING"

        return fusion_pair

    def create_fusion_pair(
        self,
        vision_det: YoloDetection,
        lidar_tracks: List[Track]
    ) -> FusionPair:
        """새 퓨전 쌍 생성 - Vision과 LiDAR 각각 독립적으로 중앙/가까운 객체 선택"""
        pair = FusionPair(
            vision_id=vision_det.track_id,
            vision_angle=vision_det.angle,
            vision_bbox=vision_det.bbox
        )

        # LiDAR: 가장 가깝고 중앙에 있는 객체 독립 선택
        initial_track = self.find_center_lidar_target(lidar_tracks)

        if initial_track is not None:
            pair.lidar_id = initial_track.track_id
            pair.last_lidar_distance = float(np.linalg.norm(initial_track.pos))
            pair.last_lidar_angle = math.atan2(initial_track.pos[1], initial_track.pos[0])
            pair.state = "MATCHING"
            pair.match_confidence = 1

            # 초기 각도 차이 출력
            angle_diff = abs(wrap_to_pi(vision_det.angle - pair.last_lidar_angle))
            print(f"[FUSION] 초기 매칭:")
            print(f"  Vision ID={pair.vision_id} (angle={math.degrees(vision_det.angle):.1f}deg)")
            print(f"  LiDAR ID={pair.lidar_id} (dist={pair.last_lidar_distance:.2f}m, angle={math.degrees(pair.last_lidar_angle):.1f}deg)")
            print(f"  각도 차이: {math.degrees(angle_diff):.1f}deg")
        else:
            pair.state = "SEARCHING"
            print(f"[FUSION] Vision ID={pair.vision_id} 락, LiDAR 트랙 없음")

        return pair


# =========================
# Vision Processing Module
# =========================

class VisionProcessor:
    """YOLO-based vision processing with ByteTrack"""

    PERSON_CLASS = 0

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tracker_config = config.get('tracker', 'models/bytetrack.yaml')
        self.camera_fov = config.get('camera_fov', 60.0)  # 카메라 수평 FOV (degrees)

        if YOLO_AVAILABLE:
            engine_path = config.get('engine', 'yolo11n.engine')
            try:
                self.model = YOLO(engine_path)
                print(f"[INFO] YOLO model loaded: {engine_path}")
                print(f"[INFO] Tracker config: {self.tracker_config}")
                print(f"[INFO] Camera FOV: {self.camera_fov} degrees")
            except Exception as e:
                print(f"[WARN] Failed to load YOLO model: {e}")

    def process_frame_with_tracking(self, frame: np.ndarray,
                                     target_id: Optional[int] = None,
                                     last_seen_ts: float = 0.0,
                                     lost_sec: float = 2.0
                                     ) -> Tuple[np.ndarray, List[YoloDetection], Optional[int], float, Optional[Tuple[int, int, int, int]]]:
        """
        Process frame with ByteTrack tracking.

        Returns:
            annotated: Annotated frame
            detections: List of YoloDetection
            target_id: Updated target ID (None if lost)
            last_seen_ts: Updated last seen timestamp
            target_bbox: Bounding box of target (if locked)
        """
        detections = []
        target_bbox = None
        now = time.time()

        if self.model is None:
            return frame.copy(), detections, target_id, last_seen_ts, target_bbox

        h, w = frame.shape[:2]

        # Use track() instead of predict() for ByteTrack
        results = self.model.track(
            source=frame,
            imgsz=self.config.get('imgsz', 640),
            conf=self.config.get('conf', 0.25),
            iou=self.config.get('iou', 0.45),
            device=self.config.get('device', 'cuda:0'),
            classes=[self.PERSON_CLASS],
            tracker=self.tracker_config,
            persist=True,
            verbose=False
        )

        r = results[0]
        annotated = frame.copy()

        # Draw center marker
        cv2.drawMarker(
            annotated,
            (w // 2, h // 2),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )

        boxes = r.boxes
        ids = None
        xyxy = None
        confs = None

        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None

            # Build detections list
            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = map(int, xyxy[i])
                conf = float(confs[i]) if confs is not None else 0.0
                cls_id = int(cls_ids[i]) if cls_ids is not None else 0
                cls_name = r.names[cls_id] if r.names else str(cls_id)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # 픽셀 좌표를 각도로 변환
                angle = pixel_to_angle(cx, w, self.camera_fov)

                detections.append(YoloDetection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    track_id=int(tid),
                    angle=angle
                ))

        # Handle target tracking
        if target_id is not None:
            found_same_id = False
            if ids is not None:
                for i, tid in enumerate(ids):
                    if tid == target_id:
                        found_same_id = True
                        last_seen_ts = now

                        x1, y1, x2, y2 = map(int, xyxy[i])
                        target_bbox = (x1, y1, x2, y2)

                        # Draw locked target (green)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        score_txt = ""
                        if confs is not None and i < len(confs):
                            score_txt = f" conf={confs[i]:.2f}"
                        cv2.putText(
                            annotated,
                            f"[TARGET] ID={target_id}{score_txt}",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                        break

            if not found_same_id:
                if (now - last_seen_ts) > lost_sec:
                    target_id = None
                    target_bbox = None
                else:
                    # Draw "SEARCHING" indicator
                    cv2.putText(
                        annotated,
                        f"SEARCHING ID={target_id} ({lost_sec - (now - last_seen_ts):.1f}s)",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )
        else:
            # Draw all tracks (white, thin)
            if ids is not None:
                for i, tid in enumerate(ids):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv2.putText(
                        annotated,
                        f"ID={int(tid)}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )

        return annotated, detections, target_id, last_seen_ts, target_bbox

    def pick_center_person_id(self, detections: List[YoloDetection], frame_w: int, frame_h: int) -> Optional[int]:
        """Select the person closest to frame center"""
        cx0, cy0 = frame_w / 2.0, frame_h / 2.0
        best_id, best_d = None, float("inf")

        for det in detections:
            if det.track_id is None:
                continue
            cx, cy = det.center
            d = (cx - cx0) ** 2 + (cy - cy0) ** 2
            if d < best_d:
                best_d = d
                best_id = det.track_id

        return best_id


# =========================
# 4-Split View Renderer
# =========================

class FourSplitViewRenderer:
    """Renders 4-split view visualization"""

    def __init__(self, config: dict):
        self.config = config
        self.view_size = config.get('view_size', 480)
        self.total_width = self.view_size * 2
        self.total_height = self.view_size * 2

        # Colors
        self.COLORS = {
            'background': (30, 30, 30),
            'grid': (60, 60, 60),
            'text': (200, 200, 200),
            'text_bright': (255, 255, 255),
            'accent': (0, 255, 255),
            'target': (0, 0, 255),
            'lidar_point': (80, 80, 80),
            'cluster': (0, 255, 255),
            'detection': (0, 180, 0),
            'track': (255, 255, 255),
            'roi': (50, 50, 50),
            'person': (0, 255, 0),
            'vehicle': (255, 165, 0),
        }

    def create_canvas(self) -> np.ndarray:
        """Create base canvas"""
        return np.full((self.total_height, self.total_width, 3),
                       self.COLORS['background'], dtype=np.uint8)

    def draw_vision_view(self, canvas: np.ndarray, shared_data: SharedData) -> None:
        """Draw top-left vision view"""
        x_offset, y_offset = 0, 0
        size = self.view_size

        # Draw border
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + size - 1, y_offset + size - 1),
                     self.COLORS['accent'], 1)

        # Title
        cv2.putText(canvas, "Vision (YOLO)", (x_offset + 10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['accent'], 2)

        with shared_data.lock:
            if shared_data.vision_annotated is not None:
                # Resize and place vision frame
                frame = shared_data.vision_annotated
                h, w = frame.shape[:2]

                # Calculate scaling to fit in view area (with margin)
                margin = 35
                available_h = size - margin - 10
                available_w = size - 20

                scale = min(available_w / w, available_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                resized = cv2.resize(frame, (new_w, new_h))

                # Center the frame
                start_x = x_offset + (size - new_w) // 2
                start_y = y_offset + margin

                canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

                # FPS display
                fps_text = f"FPS: {shared_data.vision_fps:.1f}"
                cv2.putText(canvas, fps_text, (x_offset + size - 100, y_offset + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)
            else:
                # No vision data
                cv2.putText(canvas, "No Camera",
                           (x_offset + size // 2 - 50, y_offset + size // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['text'], 1)

    def draw_lidar_view(self, canvas: np.ndarray, shared_data: SharedData) -> None:
        """Draw top-right LiDAR top view"""
        x_offset, y_offset = self.view_size, 0
        size = self.view_size

        # Draw border
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + size - 1, y_offset + size - 1),
                     self.COLORS['accent'], 1)

        # Title
        cv2.putText(canvas, "LiDAR (Top View)", (x_offset + 10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['accent'], 2)

        # LiDAR view origin and scale
        scale = self.config.get('lidar_scale', 60.0)
        origin = (x_offset + size // 2, y_offset + int(size * 0.85))

        def to_px(xy: np.ndarray) -> Tuple[int, int]:
            x, y = float(xy[0]), float(xy[1])
            u = int(origin[0] + y * scale)
            v = int(origin[1] - x * scale)
            return u, v

        with shared_data.lock:
            # Draw ROI boundary
            roi_angle = math.radians(90)
            rmax = self.config.get('range_max', 6.0)
            p1 = to_px(np.array([rmax * math.cos(-roi_angle), rmax * math.sin(-roi_angle)]))
            p2 = to_px(np.array([rmax * math.cos(roi_angle), rmax * math.sin(roi_angle)]))
            cv2.line(canvas, origin, p1, self.COLORS['roi'], 1)
            cv2.line(canvas, origin, p2, self.COLORS['roi'], 1)

            # Range rings
            for r in [2, 4, 6]:
                if r <= rmax:
                    cv2.circle(canvas, origin, int(r * scale), self.COLORS['roi'], 1)

            # Draw LiDAR points
            if shared_data.lidar_points_xy is not None:
                for p in shared_data.lidar_points_xy:
                    u, v = to_px(p)
                    if x_offset <= u < x_offset + size and y_offset + 30 <= v < y_offset + size:
                        canvas[v, u] = self.COLORS['lidar_point']

            # Draw clusters
            for c in shared_data.lidar_clusters:
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
                cv2.polylines(canvas, [corners_px.reshape(-1, 1, 2)], True, self.COLORS['cluster'], 1)

                cu, cv_pt = to_px(c.centroid)
                cv2.circle(canvas, (cu, cv_pt), 3, self.COLORS['cluster'], -1)

            # Draw tracks
            for t in shared_data.lidar_tracks:
                u, v = to_px(t.pos)

                is_target = (t.track_id == shared_data.lidar_target_id)
                color = self.COLORS['target'] if is_target else self.COLORS['track']
                thickness = 3 if is_target else 2

                cv2.circle(canvas, (u, v), 5, color, thickness)

                # Track ID and velocity
                label = f"ID:{t.track_id}"
                if is_target:
                    label = f"[L]{label}"
                cv2.putText(canvas, label, (u + 8, v - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                # Velocity arrow
                end = t.pos + 0.5 * t.vel
                ue, ve = to_px(end)
                cv2.arrowedLine(canvas, (u, v), (ue, ve), color, 1, tipLength=0.3)

            # FPS
            fps_text = f"FPS: {shared_data.lidar_fps:.1f}"
            cv2.putText(canvas, fps_text, (x_offset + size - 100, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

    def draw_bev_fusion_view(self, canvas: np.ndarray, shared_data: SharedData) -> None:
        """Draw bottom-left Vision+LiDAR Fusion view (각도 기반 매핑)"""
        x_offset, y_offset = 0, self.view_size
        size = self.view_size

        # Draw border
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + size - 1, y_offset + size - 1),
                     (255, 0, 255), 1)  # 마젠타 (퓨전 색상)

        # Title
        cv2.putText(canvas, "Vision + LiDAR Fusion", (x_offset + 10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # BEV view settings
        scale = self.config.get('bev_scale', 50.0)
        origin = (x_offset + size // 2, y_offset + int(size * 0.90))

        def to_px(xy: np.ndarray) -> Tuple[int, int]:
            x, y = float(xy[0]), float(xy[1])
            u = int(origin[0] + y * scale)
            v = int(origin[1] - x * scale)
            return u, v

        with shared_data.lock:
            # Draw grid
            for i in range(-4, 5):
                x_pos = x_offset + size // 2 + int(i * scale)
                if x_offset < x_pos < x_offset + size:
                    cv2.line(canvas, (x_pos, y_offset + 35), (x_pos, y_offset + size - 5),
                            self.COLORS['grid'], 1)

            for i in range(1, 9):
                y_pos = origin[1] - int(i * scale)
                if y_offset + 35 < y_pos < y_offset + size:
                    cv2.line(canvas, (x_offset + 5, y_pos), (x_offset + size - 5, y_pos),
                            self.COLORS['grid'], 1)
                    cv2.putText(canvas, f"{i}m", (x_offset + 10, y_pos - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLORS['text'], 1)

            # Draw robot position
            cv2.circle(canvas, origin, 8, self.COLORS['accent'], -1)
            cv2.putText(canvas, "ROBOT", (origin[0] - 25, origin[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['accent'], 1)

            # Fusion 타겟 ID 가져오기
            fusion_vision_id = None
            fusion_lidar_id = None
            if shared_data.fusion_pair is not None:
                fusion_vision_id = shared_data.fusion_pair.vision_id
                fusion_lidar_id = shared_data.fusion_pair.lidar_id

            # ========== LiDAR Tracks (원형, 시안색) ==========
            lidar_positions = {}  # track_id -> (u, v, angle, pos, track)
            for t in shared_data.lidar_tracks:
                u, v = to_px(t.pos)

                # 퓨전 타겟 여부 확인
                is_fusion_target = (t.track_id == fusion_lidar_id)
                is_lidar_target = (t.track_id == shared_data.lidar_target_id)

                if is_fusion_target:
                    color = (255, 0, 255)  # 마젠타 (퓨전)
                    thickness = 3
                elif is_lidar_target:
                    color = self.COLORS['target']
                    thickness = 2
                else:
                    color = self.COLORS['cluster']
                    thickness = 2

                # LiDAR 트랙 원형으로 표시
                cv2.circle(canvas, (u, v), 10, color, thickness)
                cv2.circle(canvas, (u, v), 3, color, -1)

                # 각도 계산
                track_angle = math.atan2(t.pos[1], t.pos[0])
                lidar_positions[t.track_id] = (u, v, track_angle, t.pos, t)  # 트랙 객체 포함

                # 라벨 (클러스터 정보 포함)
                dist = np.linalg.norm(t.pos)
                prefix = "[F-L]" if is_fusion_target else "[L]"
                label = f"{prefix}{t.track_id} {dist:.1f}m"
                cv2.putText(canvas, label, (u + 12, v - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                # 클러스터 정보 (포인트 수, 폭)
                cluster_info = f"pts:{t.n_points} w:{t.cluster_width:.2f}m"
                cv2.putText(canvas, cluster_info, (u + 12, v + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

                # 속도 벡터
                vel_mag = np.linalg.norm(t.vel)
                if vel_mag > 0.1:
                    end = t.pos + 0.8 * t.vel
                    ue, ve = to_px(end)
                    cv2.arrowedLine(canvas, (u, v), (ue, ve), color, 2, tipLength=0.2)

            # ========== Vision Detections (사각형, 초록색) ==========
            # 각도 기반으로 BEV에 투영 (기본 거리 3m 사용, LiDAR 매칭 시 실제 거리 사용)
            vision_positions = {}  # track_id -> (u, v, angle)
            default_distance = 3.0  # Vision만 있을 때 기본 표시 거리
            angle_tolerance = 0.25  # 각도 매칭 허용 오차 (radians, ~14도)

            for det in shared_data.vision_detections:
                if det.track_id is None:
                    continue

                vision_angle = det.angle

                # LiDAR 트랙과 각도 매칭 시도
                matched_lidar = None
                matched_distance = default_distance
                matched_track = None
                min_angle_diff = float('inf')

                for lid, (lu, lv, langle, lpos, ltrack) in lidar_positions.items():
                    angle_diff = abs(wrap_to_pi(vision_angle - langle))
                    if angle_diff < angle_tolerance and angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                        matched_lidar = lid
                        matched_distance = np.linalg.norm(lpos)
                        matched_track = ltrack

                # Vision 객체를 BEV에 투영
                # 퓨전 타겟이면 퓨전의 마지막 거리 사용
                display_distance = matched_distance
                if det.track_id == fusion_vision_id and shared_data.fusion_pair is not None:
                    display_distance = shared_data.fusion_pair.last_lidar_distance

                vx, vy = angle_to_bev_xy(vision_angle, display_distance)
                u, v = to_px(np.array([vx, vy]))

                # 퓨전 타겟 여부 확인
                is_fusion_target = (det.track_id == fusion_vision_id)
                is_vision_target = (det.track_id == shared_data.vision_target_id)

                if is_fusion_target:
                    color = (255, 0, 255)  # 마젠타 (퓨전)
                    thickness = 3
                elif is_vision_target:
                    color = self.COLORS['target']
                    thickness = 2
                else:
                    color = self.COLORS['person']
                    thickness = 2

                # Vision 감지는 사각형으로 표시
                cv2.rectangle(canvas, (u - 8, v - 8), (u + 8, v + 8), color, thickness)

                vision_positions[det.track_id] = (u, v, vision_angle)

                # 라벨
                angle_deg = math.degrees(vision_angle)
                prefix = "[F-V]" if is_fusion_target else "[V]"
                label = f"{prefix}{det.track_id} {angle_deg:.1f}deg"
                cv2.putText(canvas, label, (u + 12, v + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                # ========== 매칭된 경우 연결선 그리기 ==========
                if matched_lidar is not None and matched_track is not None:
                    lu, lv, _, _, _ = lidar_positions[matched_lidar]

                    # 퓨전 타겟이면 더 굵은 선
                    is_fusion_link = (det.track_id == fusion_vision_id and matched_lidar == fusion_lidar_id)
                    line_color = (255, 0, 255) if is_fusion_link else (255, 100, 255)
                    line_thickness = 3 if is_fusion_link else 2

                    cv2.line(canvas, (u, v), (lu, lv), line_color, line_thickness)

                    # 매칭 정보 표시 (각도 차이 + 클러스터 정보)
                    mid_x, mid_y = (u + lu) // 2, (v + lv) // 2
                    angle_diff_deg = math.degrees(min_angle_diff)

                    # 퓨전 타겟이면 FUSION 표시
                    if is_fusion_link:
                        cv2.putText(canvas, "FUSION",
                                   (mid_x - 25, mid_y - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)

                    # 매칭 각도 차이
                    cv2.putText(canvas, f"MATCH {angle_diff_deg:.1f}deg",
                               (mid_x - 30, mid_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color, 1)

                    # 클러스터 상세 정보 (튜닝용)
                    cluster_detail = f"pts:{matched_track.n_points} w:{matched_track.cluster_width:.2f}m"
                    cv2.putText(canvas, cluster_detail,
                               (mid_x - 40, mid_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

                    # 거리 정보
                    cv2.putText(canvas, f"d:{display_distance:.2f}m",
                               (mid_x - 25, mid_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

            # ========== 퓨전 상태 표시 ==========
            if shared_data.fusion_pair is not None:
                fp = shared_data.fusion_pair
                state_color = {
                    "MATCHING": (0, 255, 0),
                    "SEARCHING": (0, 165, 255),
                    "VISION_ONLY": (0, 255, 255)
                }.get(fp.state, (255, 255, 255))

                cv2.putText(canvas, f"FUSION: {fp.state}", (x_offset + 10, y_offset + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 2)
                cv2.putText(canvas, f"V:{fp.vision_id} L:{fp.lidar_id} conf:{fp.match_confidence}",
                           (x_offset + 10, y_offset + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, state_color, 1)

            # ========== 범례 및 통계 ==========
            legend_y = y_offset + size - 60
            cv2.putText(canvas, "[F]=Fusion [V]=Vision [L]=LiDAR", (x_offset + 10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLORS['text'], 1)
            cv2.putText(canvas, f"Vision: {len(vision_positions)}  LiDAR: {len(lidar_positions)}",
                       (x_offset + 10, legend_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)

            # 매칭 수 계산
            matched_count = sum(1 for det in shared_data.vision_detections
                               if det.track_id is not None and
                               any(abs(wrap_to_pi(det.angle - math.atan2(t.pos[1], t.pos[0]))) < angle_tolerance
                                   for t in shared_data.lidar_tracks))
            cv2.putText(canvas, f"Matched: {matched_count}",
                       (x_offset + size - 100, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)

    def draw_info_panel(self, canvas: np.ndarray, shared_data: SharedData) -> None:
        """Draw bottom-right info panel with dual tracking status"""
        x_offset, y_offset = self.view_size, self.view_size
        size = self.view_size

        # Draw border
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + size - 1, y_offset + size - 1),
                     self.COLORS['vehicle'], 1)

        # Title
        cv2.putText(canvas, "Info Panel (Dual Tracking)", (x_offset + 10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['vehicle'], 2)

        with shared_data.lock:
            line_height = 20
            start_y = y_offset + 50

            # System Status Section
            cv2.putText(canvas, "=== System Status ===", (x_offset + 10, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text_bright'], 1)
            start_y += line_height

            # Vision status
            vision_status = "ON" if shared_data.vision_active else "OFF"
            vision_color = self.COLORS['person'] if shared_data.vision_active else self.COLORS['target']
            cv2.putText(canvas, f"Vision: {vision_status} ({shared_data.vision_fps:.1f} FPS)",
                       (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, vision_color, 1)
            start_y += line_height

            # LiDAR status
            lidar_status = "ON" if shared_data.lidar_active else "OFF"
            lidar_color = self.COLORS['person'] if shared_data.lidar_active else self.COLORS['target']
            cv2.putText(canvas, f"LiDAR: {lidar_status} ({shared_data.lidar_fps:.1f} FPS)",
                       (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, lidar_color, 1)
            start_y += int(line_height * 1.2)

            # ========== FUSION Tracking Section (SPACE key) ==========
            cv2.putText(canvas, "=== FUSION Track (SPACE) ===", (x_offset + 10, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            start_y += line_height

            if shared_data.fusion_pair is not None:
                fp = shared_data.fusion_pair
                state_colors = {
                    "MATCHING": (0, 255, 0),
                    "SEARCHING": (0, 165, 255),
                    "VISION_ONLY": (0, 255, 255)
                }
                state_color = state_colors.get(fp.state, (255, 255, 255))

                cv2.putText(canvas, f"[{fp.state}]",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 1)
                start_y += line_height

                cv2.putText(canvas, f"  Vision ID: {fp.vision_id}",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['person'], 1)
                start_y += line_height - 2

                cv2.putText(canvas, f"  LiDAR ID: {fp.lidar_id if fp.lidar_id else 'None'}",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['cluster'], 1)
                start_y += line_height - 2

                cv2.putText(canvas, f"  Distance: {fp.last_lidar_distance:.2f}m",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                start_y += line_height - 2

                cv2.putText(canvas, f"  Confidence: {fp.match_confidence}  Lost: {fp.lidar_lost_count}",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)
                start_y += line_height
            else:
                cv2.putText(canvas, "[NOT ACTIVE]",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)
                start_y += line_height
                cv2.putText(canvas, "Press SPACE to start",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)
                start_y += line_height

            start_y += int(line_height * 0.3)

            # ========== Vision Tracking Section ==========
            cv2.putText(canvas, "=== Vision Track (V key) ===", (x_offset + 10, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['accent'], 1)
            start_y += line_height

            if shared_data.vision_target_id is not None:
                now = time.time()
                lost_time = now - shared_data.vision_last_seen_ts
                if lost_time < 0.1:
                    status_text = f"ID={shared_data.vision_target_id} [LOCKED]"
                    status_color = self.COLORS['person']
                else:
                    remaining = max(0, shared_data.vision_lost_sec - lost_time)
                    status_text = f"ID={shared_data.vision_target_id} [SEARCH {remaining:.1f}s]"
                    status_color = (0, 165, 255)  # Orange

                cv2.putText(canvas, status_text,
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                start_y += line_height

                # Show bbox if available
                if shared_data.vision_target_bbox:
                    x1, y1, x2, y2 = shared_data.vision_target_bbox
                    w = x2 - x1
                    h = y2 - y1
                    cv2.putText(canvas, f"  Box: {w}x{h} @ ({(x1+x2)//2}, {(y1+y2)//2})",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)
                    start_y += line_height
            else:
                cv2.putText(canvas, "Not locked (press V)",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)
                start_y += line_height

            start_y += int(line_height * 0.5)

            # ========== LiDAR Tracking Section ==========
            cv2.putText(canvas, "=== LiDAR Track (L key) ===", (x_offset + 10, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['cluster'], 1)
            start_y += line_height

            if shared_data.lidar_target_id is not None:
                # Find target track
                target_track = None
                for t in shared_data.lidar_tracks:
                    if t.track_id == shared_data.lidar_target_id:
                        target_track = t
                        break

                if target_track:
                    cv2.putText(canvas, f"ID={shared_data.lidar_target_id} [LOCKED]",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['person'], 1)
                    start_y += line_height

                    pos = target_track.pos
                    vel = target_track.vel
                    dist = np.linalg.norm(pos)
                    speed = np.linalg.norm(vel)

                    cv2.putText(canvas, f"  Pos: ({pos[0]:.2f}, {pos[1]:.2f})m  D:{dist:.2f}m",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)
                    start_y += line_height

                    cv2.putText(canvas, f"  Vel: ({vel[0]:.2f}, {vel[1]:.2f})m/s  S:{speed:.2f}m/s",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)
                    start_y += line_height

                    # ========== 클러스터 정보 (튜닝용) ==========
                    cv2.putText(canvas, f"  [Cluster Info]",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                    start_y += line_height

                    cv2.putText(canvas, f"    Points: {target_track.n_points}",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                    start_y += line_height

                    cv2.putText(canvas, f"    Width: {target_track.cluster_width:.3f}m",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                    start_y += line_height

                    # 각도 정보
                    angle_rad = math.atan2(pos[1], pos[0])
                    angle_deg = math.degrees(angle_rad)
                    cv2.putText(canvas, f"    Angle: {angle_deg:.1f}deg ({angle_rad:.3f}rad)",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                    start_y += line_height
                else:
                    max_lost = 30  # default
                    cv2.putText(canvas, f"ID={shared_data.lidar_target_id} [LOST {shared_data.lidar_target_lost_frames}/{max_lost}]",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['target'], 1)
                    start_y += line_height
            else:
                cv2.putText(canvas, "Not locked (press L)",
                           (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)
                start_y += line_height

            start_y += int(line_height * 0.5)

            # ========== Detection Stats ==========
            cv2.putText(canvas, "=== Detection Stats ===", (x_offset + 10, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text_bright'], 1)
            start_y += line_height

            person_det = sum(1 for d in shared_data.vision_detections if d.class_name.lower() == 'person')
            tracked_persons = sum(1 for d in shared_data.vision_detections if d.track_id is not None)
            cv2.putText(canvas, f"Vision: {person_det} persons, {tracked_persons} tracked",
                       (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)
            start_y += line_height

            cv2.putText(canvas, f"LiDAR: {len(shared_data.lidar_clusters)} clusters, {len(shared_data.lidar_tracks)} tracks",
                       (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)
            start_y += int(line_height * 1.2)

            # ========== Cluster Tuning Info ==========
            cv2.putText(canvas, "=== Cluster Settings ===", (x_offset + 10, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            start_y += line_height

            # 현재 설정값 표시
            min_pts = self.config.get('min_points', 5)
            max_pts = self.config.get('max_points', 10)
            w_min = self.config.get('width_min', 0.15)
            w_max = self.config.get('width_max', 0.80)

            cv2.putText(canvas, f"Points: {min_pts} ~ {max_pts}",
                       (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)
            start_y += line_height - 2
            cv2.putText(canvas, f"Width: {w_min:.2f} ~ {w_max:.2f}m",
                       (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)
            start_y += line_height - 2

            # 현재 트랙들의 클러스터 정보 요약
            if shared_data.lidar_tracks:
                pts_list = [t.n_points for t in shared_data.lidar_tracks if t.n_points > 0]
                width_list = [t.cluster_width for t in shared_data.lidar_tracks if t.cluster_width > 0]

                if pts_list:
                    cv2.putText(canvas, f"Current pts: {min(pts_list)}~{max(pts_list)} (avg:{sum(pts_list)/len(pts_list):.1f})",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                    start_y += line_height - 2

                if width_list:
                    cv2.putText(canvas, f"Current width: {min(width_list):.2f}~{max(width_list):.2f}m",
                               (x_offset + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                    start_y += line_height - 2

            start_y += int(line_height * 0.3)

            # Controls help
            cv2.putText(canvas, "=== Controls ===", (x_offset + 10, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text_bright'], 1)
            start_y += line_height

            controls = [
                "V: Lock Vision",
                "L: Lock LiDAR",
                "R: Reset locks",
                "Q: Quit"
            ]
            for ctrl in controls:
                cv2.putText(canvas, ctrl, (x_offset + 15, start_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text'], 1)
                start_y += line_height - 4

    def render(self, shared_data: SharedData) -> np.ndarray:
        """Render complete 4-split view"""
        canvas = self.create_canvas()

        self.draw_vision_view(canvas, shared_data)
        self.draw_lidar_view(canvas, shared_data)
        self.draw_bev_fusion_view(canvas, shared_data)
        self.draw_info_panel(canvas, shared_data)

        # Draw center cross lines
        cv2.line(canvas, (self.view_size, 0), (self.view_size, self.total_height),
                self.COLORS['grid'], 2)
        cv2.line(canvas, (0, self.view_size), (self.total_width, self.view_size),
                self.COLORS['grid'], 2)

        return canvas


# =========================
# Vision Thread
# =========================

def vision_thread_func(shared_data: SharedData, config: dict, stop_event: threading.Event):
    """Vision processing thread with ByteTrack Lock"""
    processor = VisionProcessor(config)

    # Open camera
    cam_index = config.get('cam', 0)
    width = config.get('width', 1280)
    height = config.get('height', 720)
    fps = config.get('fps', 30)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[WARN] Failed to open camera {cam_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    fps_window = deque(maxlen=30)
    last_t = time.time()

    # Local tracking state
    target_id = None
    last_seen_ts = 0.0
    lost_sec = config.get('vision_lost_sec', 2.0)

    print("[INFO] Vision thread started (with ByteTrack)")

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        # 카메라 이미지 좌우 반전
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # Check for lock/reset requests
        with shared_data.lock:
            if shared_data.vision_reset_requested:
                target_id = None
                last_seen_ts = 0.0
                shared_data.vision_reset_requested = False
                print("[INFO] Vision target reset")

            lock_requested = shared_data.vision_lock_requested
            shared_data.vision_lock_requested = False

        # Process frame with tracking
        annotated, detections, target_id, last_seen_ts, target_bbox = processor.process_frame_with_tracking(
            frame, target_id, last_seen_ts, lost_sec
        )

        # Handle lock request
        if lock_requested and target_id is None:
            new_id = processor.pick_center_person_id(detections, w, h)
            if new_id is not None:
                target_id = new_id
                last_seen_ts = time.time()
                print(f"[INFO] Vision target locked: ID {target_id}")

        # Calculate FPS
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps_window.append(1.0 / dt)
        fps_avg = sum(fps_window) / len(fps_window) if fps_window else 0.0

        # Update shared data
        with shared_data.lock:
            shared_data.vision_frame = frame.copy()
            shared_data.vision_annotated = annotated.copy()
            shared_data.vision_detections = detections
            shared_data.vision_fps = fps_avg
            shared_data.vision_active = True
            shared_data.vision_target_id = target_id
            shared_data.vision_last_seen_ts = last_seen_ts
            shared_data.vision_target_bbox = target_bbox
            shared_data.frame_width = w
            shared_data.frame_height = h

    cap.release()
    print("[INFO] Vision thread stopped")


# =========================
# LiDAR Thread (ROS2 or Simulated)
# =========================

if ROS2_AVAILABLE:
    class LidarROS2Node(Node):
        """ROS2 node for LiDAR processing"""

        def __init__(self, shared_data: SharedData, config: dict):
            super().__init__("lidar_fusion_node")
            self.shared_data = shared_data
            self.config = config
            self.processor = LidarProcessor(config)

            scan_topic = config.get('scan_topic', '/scan')
            self.sub = self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)

            self.fps_window = deque(maxlen=30)
            self.last_t = time.time()

            self.get_logger().info(f"LiDAR node started, subscribing to {scan_topic}")

        def on_scan(self, msg: LaserScan):
            now_sec = self.get_clock().now().nanoseconds * 1e-9

            if self.processor.last_stamp_sec is None:
                self.processor.last_stamp_sec = now_sec
                return

            dt = max(1e-3, now_sec - self.processor.last_stamp_sec)
            self.processor.last_stamp_sec = now_sec

            # Check for lock/reset requests
            with self.shared_data.lock:
                if self.shared_data.lidar_reset_requested:
                    self.processor.target_track_id = None
                    self.processor.target_lost_frames = 0
                    self.shared_data.lidar_reset_requested = False
                    self.get_logger().info("LiDAR target reset")

                lock_requested = self.shared_data.lidar_lock_requested
                self.shared_data.lidar_lock_requested = False

            # Process scan
            points_xy, _ = self.processor.preprocess_scan(
                np.array(msg.ranges), msg.angle_min, msg.angle_increment
            )

            if points_xy.shape[0] == 0:
                return

            clusters_pts = self.processor.cluster_points(points_xy)
            candidate_clusters, detections = self.processor.filter_candidates(clusters_pts)
            self.processor.step_tracker(detections, candidate_clusters, dt, now_sec)
            confirmed_tracks = self.processor.get_confirmed_tracks()

            # Handle lock request
            if lock_requested and self.processor.target_track_id is None:
                new_id = self.processor.select_center_track()
                if new_id is not None:
                    self.processor.target_track_id = new_id
                    self.processor.target_lost_frames = 0
                    self.get_logger().info(f"LiDAR target locked: ID {new_id}")

            # Update target tracking status
            self._update_target_status(confirmed_tracks)

            # Calculate FPS
            now = time.time()
            frame_dt = now - self.last_t
            self.last_t = now
            if frame_dt > 0:
                self.fps_window.append(1.0 / frame_dt)
            fps_avg = sum(self.fps_window) / len(self.fps_window) if self.fps_window else 0.0

            # Update shared data
            with self.shared_data.lock:
                self.shared_data.lidar_points_xy = points_xy.copy()
                self.shared_data.lidar_clusters = candidate_clusters.copy()
                self.shared_data.lidar_tracks = [Track(
                    track_id=t.track_id,
                    x=t.x.copy(),
                    P=t.P.copy(),
                    age=t.age,
                    hits=t.hits,
                    misses=t.misses,
                    last_update_time=t.last_update_time,
                    n_points=t.n_points,
                    cluster_width=t.cluster_width
                ) for t in confirmed_tracks]
                self.shared_data.lidar_detections = detections.copy() if len(detections) > 0 else None
                self.shared_data.lidar_fps = fps_avg
                self.shared_data.lidar_active = True

                # Sync target tracking
                self.shared_data.lidar_target_id = self.processor.target_track_id
                self.shared_data.lidar_target_lost_frames = self.processor.target_lost_frames
                self.shared_data.lidar_jumped = self.processor.target_jumped
                self.shared_data.lidar_jumped_ids = self.processor.jumped_track_ids.copy()

                # Sync fusion LiDAR ID (for jump detection)
                if self.shared_data.fusion_pair is not None:
                    self.processor.fusion_lidar_id = self.shared_data.fusion_pair.lidar_id
                else:
                    self.processor.fusion_lidar_id = None

        def _update_target_status(self, confirmed_tracks: List[Track]):
            """Update target tracking status"""
            if self.processor.target_track_id is None:
                return

            target_exists = False
            for tr in confirmed_tracks:
                if tr.track_id == self.processor.target_track_id:
                    target_exists = True
                    self.processor.target_lost_frames = 0
                    break

            if not target_exists:
                self.processor.target_lost_frames += 1
                max_lost = self.config.get('lidar_max_lost_frames', 30)
                if self.processor.target_lost_frames >= max_lost:
                    self.get_logger().warn(f"LiDAR target lost: ID {self.processor.target_track_id}")
                    self.processor.target_track_id = None
                    self.processor.target_lost_frames = 0


def lidar_thread_func(shared_data: SharedData, config: dict, stop_event: threading.Event):
    """LiDAR processing thread"""
    if ROS2_AVAILABLE:
        rclpy.init()
        node = LidarROS2Node(shared_data, config)

        print("[INFO] LiDAR thread started (ROS2)")

        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.01)

        node.destroy_node()
        rclpy.shutdown()
    else:
        # Simulated LiDAR for testing without ROS2
        processor = LidarProcessor(config)
        fps_window = deque(maxlen=30)
        last_t = time.time()
        max_lost_frames = config.get('lidar_max_lost_frames', 30)

        print("[INFO] LiDAR thread started (Simulated)")

        while not stop_event.is_set():
            # Check for lock/reset requests
            with shared_data.lock:
                if shared_data.lidar_reset_requested:
                    processor.target_track_id = None
                    processor.target_lost_frames = 0
                    shared_data.lidar_reset_requested = False
                    print("[INFO] LiDAR target reset")

                lock_requested = shared_data.lidar_lock_requested
                shared_data.lidar_lock_requested = False

            # Generate simulated scan data
            num_points = 360
            angles = np.linspace(-math.pi, math.pi, num_points)
            ranges = np.random.uniform(1.0, 5.0, num_points)

            # Add some "person-like" clusters (2 people for testing)
            t = time.time()
            person1_angle = math.radians(180 + 30 * math.sin(t * 0.5))
            person1_range = 2.0 + 0.5 * math.sin(t * 0.3)
            person2_angle = math.radians(180 + 20 * math.cos(t * 0.3))
            person2_range = 3.0 + 0.3 * math.cos(t * 0.2)

            for i, a in enumerate(angles):
                if abs(wrap_to_pi(a - person1_angle)) < 0.1:
                    ranges[i] = person1_range + np.random.uniform(-0.05, 0.05)
                elif abs(wrap_to_pi(a - person2_angle)) < 0.08:
                    ranges[i] = person2_range + np.random.uniform(-0.05, 0.05)

            # Process
            now_sec = time.time()
            if processor.last_stamp_sec is None:
                processor.last_stamp_sec = now_sec
                time.sleep(0.033)
                continue

            dt = max(1e-3, now_sec - processor.last_stamp_sec)
            processor.last_stamp_sec = now_sec

            points_xy, _ = processor.preprocess_scan(ranges, -math.pi, 2 * math.pi / num_points)

            if points_xy.shape[0] > 0:
                clusters_pts = processor.cluster_points(points_xy)
                candidate_clusters, detections = processor.filter_candidates(clusters_pts)
                processor.step_tracker(detections, candidate_clusters, dt, now_sec)
                confirmed_tracks = processor.get_confirmed_tracks()

                # Handle lock request
                if lock_requested and processor.target_track_id is None:
                    new_id = processor.select_center_track()
                    if new_id is not None:
                        processor.target_track_id = new_id
                        processor.target_lost_frames = 0
                        print(f"[INFO] LiDAR target locked: ID {new_id}")

                # Update target status
                if processor.target_track_id is not None:
                    target_exists = any(tr.track_id == processor.target_track_id for tr in confirmed_tracks)
                    if target_exists:
                        processor.target_lost_frames = 0
                    else:
                        processor.target_lost_frames += 1
                        if processor.target_lost_frames >= max_lost_frames:
                            print(f"[WARN] LiDAR target lost: ID {processor.target_track_id}")
                            processor.target_track_id = None
                            processor.target_lost_frames = 0

                # FPS
                now = time.time()
                frame_dt = now - last_t
                last_t = now
                if frame_dt > 0:
                    fps_window.append(1.0 / frame_dt)
                fps_avg = sum(fps_window) / len(fps_window) if fps_window else 0.0

                with shared_data.lock:
                    shared_data.lidar_points_xy = points_xy.copy()
                    shared_data.lidar_clusters = candidate_clusters.copy()
                    shared_data.lidar_tracks = [Track(
                        track_id=tr.track_id,
                        x=tr.x.copy(),
                        P=tr.P.copy(),
                        age=tr.age,
                        hits=tr.hits,
                        misses=tr.misses,
                        last_update_time=tr.last_update_time,
                        n_points=tr.n_points,
                        cluster_width=tr.cluster_width
                    ) for tr in confirmed_tracks]
                    shared_data.lidar_fps = fps_avg
                    shared_data.lidar_active = True
                    shared_data.lidar_target_id = processor.target_track_id
                    shared_data.lidar_target_lost_frames = processor.target_lost_frames
                    shared_data.lidar_jumped = processor.target_jumped
                    shared_data.lidar_jumped_ids = processor.jumped_track_ids.copy()

                    # Sync fusion LiDAR ID (for jump detection)
                    if shared_data.fusion_pair is not None:
                        processor.fusion_lidar_id = shared_data.fusion_pair.lidar_id
                    else:
                        processor.fusion_lidar_id = None

            time.sleep(0.033)  # ~30 FPS

    print("[INFO] LiDAR thread stopped")


# =========================
# Main Application
# =========================

def main():
    parser = argparse.ArgumentParser(description="Vision-LiDAR Fusion 4-Split View")

    # Vision arguments
    parser.add_argument("--engine", type=str, default="models/yolo11n.engine",
                       help="Path to TensorRT engine or .pt model")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--camera-fov", type=float, default=60.0, help="Camera horizontal FOV (degrees)")

    # LiDAR arguments
    parser.add_argument("--scan-topic", type=str, default="/scan", help="LiDAR scan topic")
    parser.add_argument("--range-min", type=float, default=0.2, help="Min range (m)")
    parser.add_argument("--range-max", type=float, default=6.0, help="Max range (m)")

    # Visualization arguments
    parser.add_argument("--view-size", type=int, default=480, help="Size of each view panel")

    # Feature flags
    parser.add_argument("--no-vision", action="store_true", help="Disable vision processing")
    parser.add_argument("--no-lidar", action="store_true", help="Disable LiDAR processing")

    args = parser.parse_args()

    # Configuration
    config = {
        # Vision
        'engine': args.engine,
        'cam': args.cam,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'width': args.width,
        'height': args.height,
        'fps': args.fps,
        'device': args.device,
        'tracker': 'models/bytetrack.yaml',
        'vision_lost_sec': 2.0,
        'camera_fov': args.camera_fov,  # 카메라 수평 FOV (degrees)

        # LiDAR
        'scan_topic': args.scan_topic,
        'range_min': args.range_min,
        'range_max': args.range_max,
        'roi_angle_deg': 90.0,
        'cluster_a': 0.2,
        'cluster_b': 0.3,
        'cluster_min_thr': 0.5,  # 최소 클러스터링 임계값
        'width_min': 0.10,
        'width_max': 1.5,
        'min_points': 5,
        'max_points': 200,
        'gate_radius': 0.60,
        'hit_min': 3,
        'miss_max': 5,
        'q_pos': 0.5,
        'q_vel': 1.0,
        'r_meas': 0.03,
        'lidar_max_lost_frames': 30,
        'max_move': 0.5,  # 프레임당 최대 이동 거리 (m)
        'jump_threshold': 0.8,  # 튐 감지 임계값 (m)

        # Fusion
        'fusion_angle_tolerance': 0.25,  # 각도 매칭 허용 오차 (~14도)
        'fusion_distance_tolerance': 1.0,  # 거리 매칭 허용 오차 (1m)

        # Visualization
        'view_size': args.view_size,
        'lidar_scale': 60.0,
        'bev_scale': 50.0,
    }

    # Shared data
    shared_data = SharedData()

    # Stop event
    stop_event = threading.Event()

    # Start threads
    threads = []

    if not args.no_vision:
        vision_thread = threading.Thread(
            target=vision_thread_func,
            args=(shared_data, config, stop_event),
            daemon=True
        )
        vision_thread.start()
        threads.append(vision_thread)

    if not args.no_lidar:
        lidar_thread = threading.Thread(
            target=lidar_thread_func,
            args=(shared_data, config, stop_event),
            daemon=True
        )
        lidar_thread.start()
        threads.append(lidar_thread)

    # Renderer
    renderer = FourSplitViewRenderer(config)

    # Create window
    window_name = "Vision-LiDAR Fusion (4-Split View)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config['view_size'] * 2, config['view_size'] * 2)

    # LiDAR processor reference for target locking (when no ROS2)
    lidar_processor_ref = [None]

    print("\n" + "=" * 50)
    print("Vision-LiDAR Fusion 4-Split View (Dual Tracking)")
    print("=" * 50)
    print(f"View size: {config['view_size']}x{config['view_size']} per panel")
    print(f"Vision: {'Enabled' if not args.no_vision else 'Disabled'}")
    print(f"LiDAR: {'Enabled' if not args.no_lidar else 'Disabled'}")
    print("-" * 50)
    print("Controls:")
    print("  SPACE: Start Fusion Tracking (Vision+LiDAR)")
    print("  V: Lock Vision only")
    print("  L: Lock LiDAR only")
    print("  R: Reset all locks")
    print("  Q: Quit")
    print("=" * 50 + "\n")

    # Fusion Manager
    fusion_manager = FusionManager(config)

    try:
        while True:
            # ========== Fusion 업데이트 ==========
            with shared_data.lock:
                # SPACE 키 요청 처리 - 퓨전 시작
                if shared_data.fusion_lock_requested:
                    shared_data.fusion_lock_requested = False

                    # 중앙 Vision 타겟 찾기
                    center_det = fusion_manager.find_center_vision_target(shared_data.vision_detections)
                    if center_det is not None:
                        # 새 퓨전 쌍 생성
                        shared_data.fusion_pair = fusion_manager.create_fusion_pair(
                            center_det,
                            shared_data.lidar_tracks
                        )
                    else:
                        print("[FUSION] 중앙에 Vision 타겟 없음")

                # 퓨전 리셋 요청 처리
                if shared_data.fusion_reset_requested:
                    shared_data.fusion_reset_requested = False
                    if shared_data.fusion_pair is not None:
                        print(f"[FUSION] 퓨전 해제: Vision ID={shared_data.fusion_pair.vision_id}")
                    shared_data.fusion_pair = None

                # 기존 퓨전 업데이트
                if shared_data.fusion_pair is not None:
                    # Fusion LiDAR ID가 튐 목록에 있는지 확인
                    fusion_lidar_jumped = (
                        shared_data.fusion_pair.lidar_id is not None and
                        shared_data.fusion_pair.lidar_id in shared_data.lidar_jumped_ids
                    )
                    shared_data.lidar_jumped = False  # 플래그 리셋
                    shared_data.lidar_jumped_ids = []  # 목록 리셋
                    shared_data.fusion_pair = fusion_manager.update_fusion(
                        shared_data.fusion_pair,
                        shared_data.vision_detections,
                        shared_data.lidar_tracks,
                        fusion_lidar_jumped
                    )

            # Render view
            canvas = renderer.render(shared_data)

            # Show
            cv2.imshow(window_name, canvas)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break

            elif key == ord(' '):
                # SPACE: Fusion Tracking 시작
                with shared_data.lock:
                    shared_data.fusion_lock_requested = True
                print("[INFO] Fusion lock requested (SPACE)")

            elif key == ord('v') or key == ord('V'):
                # Lock Vision target (center person)
                with shared_data.lock:
                    shared_data.vision_lock_requested = True
                print("[INFO] Vision lock requested")

            elif key == ord('l') or key == ord('L'):
                # Lock LiDAR target (center object)
                with shared_data.lock:
                    shared_data.lidar_lock_requested = True
                print("[INFO] LiDAR lock requested")

            elif key == ord('r') or key == ord('R'):
                # Reset all locks
                with shared_data.lock:
                    shared_data.vision_reset_requested = True
                    shared_data.lidar_reset_requested = True
                    shared_data.fusion_reset_requested = True
                print("[INFO] All locks reset requested")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Stop threads
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        cv2.destroyAllWindows()
        print("[INFO] Application closed")


if __name__ == "__main__":
    main()
