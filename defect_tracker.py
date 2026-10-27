"""
Real-Time Defect Detection & Spatial Centroid Tracking Pipeline
================================================================
Dual-Purpose Computer Vision Architecture:
  1. Agricultural Crop Pathology: Rice Leaf Lesion Localization, Multi-Object Tracking & Severity Estimation
  2. Heavy Industrial Safety: Real-Time Conveyor Defect Localization, Surface Anomaly Tracking & Hazard Alerting

Tech Stack: OpenCV, PyTorch, NumPy, Python
Author: Roshan Raj (BIT Mesra)
"""

import cv2
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import os
import sys

# Optional PyTorch bridge for feature extraction
try:
    import torch
    import torchvision.transforms as T
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# ============================================================================
# 1. SPATIAL CENTROID TRACKER (Multi-Object Hungarian / Euclidean Tracking)
# ============================================================================
class CentroidTracker:
    """
    Tracks multiple anomalies/lesions across video frames or moving conveyor feeds.
    Assigns persistent unique IDs based on minimum Euclidean distance matching.
    """
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        self.next_object_id = 0
        self.objects = OrderedDict()        # ID -> centroid (cX, cY)
        self.disappeared = OrderedDict()    # ID -> consecutive frame count missing
        self.bboxes = OrderedDict()         # ID -> (x, y, w, h)
        self.areas = OrderedDict()          # ID -> surface area in pixels
        self.trajectories = OrderedDict()   # ID -> list of past centroid points

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], area: float):
        """Register a newly discovered defect with a unique ID."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.bboxes[self.next_object_id] = bbox
        self.areas[self.next_object_id] = area
        self.trajectories[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, object_id: int):
        """Remove an anomaly when it exits the camera's field of view."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]
        del self.areas[object_id]
        del self.trajectories[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        Takes bounding boxes [(x, y, w, h), ...] in the current frame and
        matches them against existing tracked defects using Euclidean distance.
        """
        # If no detections exist in this frame
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Compute centroids and areas for new detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_areas = []
        for i, (x, y, w, h) in enumerate(rects):
            c_x = int(x + (w / 2.0))
            c_y = int(y + (h / 2.0))
            input_centroids[i] = (c_x, c_y)
            input_areas.append(w * h)

        # If currently tracking zero objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i], input_areas[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Distance matrix between existing centroids and new input centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # Match greedily based on minimum distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = tuple(input_centroids[col])
                self.bboxes[object_id] = rects[col]
                self.areas[object_id] = input_areas[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(tuple(input_centroids[col]))
                if len(self.trajectories[object_id]) > 30:
                    self.trajectories[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Check for lost tracks
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register brand new defects
            for col in unused_cols:
                self.register(input_centroids[col], rects[col], input_areas[col])

        return self.objects


# ============================================================================
# 2. OPENCV DEFECT & LESION SEGMENTATION ENGINE
# ============================================================================
class DefectDetector:
    """
    Classical Computer Vision segmentation engine using HSV/LAB color transforms,
    adaptive morphology, and contour filtering to localize anomalies.
    """
    def __init__(self, min_defect_area: int = 30, max_defect_area: int = 50000):
        self.min_defect_area = min_defect_area
        self.max_defect_area = max_defect_area

    def segment_leaf_and_lesions(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Separates total plant/surface area from diseased/defective surface area.
        Returns: (leaf_mask, lesion_mask, severity_percentage)
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 1. Segment Total Foliage (Green / Plant Tissue)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([88, 255, 255])
        leaf_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 2. Segment Lesions / Defect Spots (Brownish, Necrotic, Yellow halos)
        lower_lesion_1 = np.array([5, 40, 40])     # Brown / reddish tones
        upper_lesion_1 = np.array([24, 255, 220])

        lower_lesion_2 = np.array([89, 50, 40])    # Yellow chlorosis / halo tones
        upper_lesion_2 = np.array([105, 255, 255])

        lesion_mask_1 = cv2.inRange(hsv, lower_lesion_1, upper_lesion_1)
        lesion_mask_2 = cv2.inRange(hsv, lower_lesion_2, upper_lesion_2)
        raw_lesions = cv2.bitwise_or(lesion_mask_1, lesion_mask_2)

        # Morphological Noise Filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_lesions = cv2.morphologyEx(raw_lesions, cv2.MORPH_OPEN, kernel, iterations=1)
        clean_lesions = cv2.morphologyEx(clean_lesions, cv2.MORPH_CLOSE, kernel, iterations=2)

        total_leaf_pixels = max(1, cv2.countNonZero(leaf_mask) + cv2.countNonZero(clean_lesions))
        lesion_pixels = cv2.countNonZero(clean_lesions)

        severity_percentage = (lesion_pixels / total_leaf_pixels) * 100.0

        return leaf_mask, clean_lesions, severity_percentage

    def extract_bounding_boxes(self, lesion_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Finds bounding boxes [x, y, w, h] of localized defect contours."""
        contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_defect_area <= area <= self.max_defect_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h))

        return bounding_boxes


# ============================================================================
# 3. PYTORCH DEEP FEATURE BRIDGE (For JD Compatibility)
# ============================================================================
def extract_pytorch_crops(image_bgr: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> Optional['torch.Tensor']:
    """
    Crops localized defect regions and converts them to a normalized PyTorch tensor
    batch of shape (B, 3, 224, 224), ready for PyTorch ViT or CNN feature extraction.
    """
    if not PYTORCH_AVAILABLE or len(bboxes) == 0:
        return None

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    crops = []
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = image_bgr.shape[:2]

    for (x, y, w, h) in bboxes:
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        crop = rgb[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(transform(crop))

    if len(crops) > 0:
        return torch.stack(crops)  # (Batch, 3, 224, 224)
    return None


# ============================================================================
# 4. INDUSTRIAL HEAD-UP DISPLAY (HUD) & SEVERITY VISUALIZER
# ============================================================================
def render_industrial_hud(
    frame: np.ndarray,
    tracker: CentroidTracker,
    severity: float,
    mode: str = "industrial"
) -> np.ndarray:
    """
    Draws professional real-time analytics overlay:
      - Bounding boxes around tracked defects with persistent IDs
      - Centroid trajectory trails
      - Severity / Surface Wear Gauge
      - Industrial Emergency / Hazard Warning banners
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw Tracked Objects
    for object_id, centroid in tracker.objects.items():
        x, y, bw, bh = tracker.bboxes[object_id]
        c_x, c_y = centroid

        # Draw Bounding Box (Amber / Cyan)
        box_color = (0, 165, 255) if mode == "industrial" else (50, 205, 50)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_color, 2)

        # Draw Centroid and ID tag
        cv2.circle(annotated, (c_x, c_y), 4, (0, 0, 255), -1)
        label = f"ID #{object_id} ({bw*bh}px)"
        cv2.putText(annotated, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

        # Draw motion trajectory trail
        pts = tracker.trajectories[object_id]
        for j in range(1, len(pts)):
            cv2.line(annotated, pts[j - 1], pts[j], (0, 255, 255), 1)

    # Top Analytics Dashboard Bar (Semi-transparent dark header)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Severity Evaluation
    if severity < 3.0:
        status_text = "STAGE 1: NORMAL / MINOR"
        status_color = (0, 255, 0)
    elif severity < 12.0:
        status_text = "STAGE 2: MODERATE DEFECT"
        status_color = (0, 215, 255)
    else:
        status_text = "STAGE 3: CRITICAL HAZARD ALERT"
        status_color = (0, 0, 255)

    # Title & Mode
    title = "HEAVY INDUSTRY DEFECT MONITOR" if mode == "industrial" else "RICE LEAF LESION TRACKER"
    cv2.putText(annotated, title, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    stats = f"Active Anomalies: {len(tracker.objects)} | Total Discovered: {tracker.next_object_id} | Surface Defect: {severity:.2f}%"
    cv2.putText(annotated, stats, (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Status Pill on Top-Right
    cv2.putText(annotated, status_text, (w - 320, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

    # Critical Banner if Hazard Level Exceeded
    if severity >= 12.0:
        cv2.rectangle(annotated, (w // 2 - 200, h - 45), (w // 2 + 200, h - 10), (0, 0, 220), -1)
        cv2.putText(annotated, "[!] EMERGENCY HAZARD THRESHOLD EXCEEDED",
                    (w // 2 - 185, h - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return annotated


# ============================================================================
# 5. SYNTHETIC BENCHMARK & DEMO GENERATOR
# ============================================================================
def generate_sample_industrial_frame(frame_idx: int) -> np.ndarray:
    """
    Generates a realistic synthetic test frame of a moving surface (e.g. leaf/steel conveyor)
    with moving lesion/crack anomalies to verify tracking without requiring video files.
    """
    frame = np.full((480, 640, 3), (35, 120, 45), dtype=np.uint8)  # Foliage base green

    # Add leaf vein / surface texture
    cv2.line(frame, (320, 0), (320, 480), (45, 140, 55), 3)

    # Moving Defect 1 (Drifting down across frames)
    y1 = (80 + frame_idx * 4) % 450
    cv2.ellipse(frame, (280, y1), (18, 9), 35, 0, 360, (25, 45, 115), -1)  # Necrotic brown

    # Moving Defect 2 (Spindle blast lesion)
    y2 = (220 + frame_idx * 3) % 450
    cv2.ellipse(frame, (340, y2), (25, 8), -20, 0, 360, (30, 60, 130), -1)

    # Static small brown spot
    cv2.circle(frame, (250, 340), 7, (20, 40, 100), -1)

    return frame


# ============================================================================
# 6. PIPELINE VERIFICATION RUNNER
# ============================================================================
def run_pipeline_demo(frames_to_run: int = 15, save_demo_image: bool = True):
    print("=" * 70)
    print("REAL-TIME DEFECT DETECTION & CENTROID TRACKING INITIALIZATION")
    print("=" * 70)
    print(f"[*] PyTorch Backend Status: {'ACTIVE (Tensor bridge ready)' if PYTORCH_AVAILABLE else 'OPTIONAL'}")
    print("[*] Initializing CentroidTracker (max_disappeared=25, max_distance=50.0)...")
    tracker = CentroidTracker(max_disappeared=25, max_distance=50.0)

    print("[*] Initializing OpenCV DefectDetector (HSV/LAB Morphology)...")
    detector = DefectDetector(min_defect_area=25, max_defect_area=50000)

    print(f"[*] Simulating real-time video stream inspection ({frames_to_run} frames)...")

    final_hud = None
    for frame_idx in range(frames_to_run):
        # 1. Ingest frame
        frame = generate_sample_industrial_frame(frame_idx)

        # 2. Segment defects
        _, lesion_mask, severity = detector.segment_leaf_and_lesions(frame)

        # 3. Extract bounding boxes
        bboxes = detector.extract_bounding_boxes(lesion_mask)

        # 4. Update Tracker
        tracked_objects = tracker.update(bboxes)

        # 5. Extract PyTorch crops (verifies deep learning bridge)
        if PYTORCH_AVAILABLE and frame_idx == 0:
            torch_batch = extract_pytorch_crops(frame, bboxes)
            if torch_batch is not None:
                print(f"    -> Extracted {torch_batch.shape[0]} PyTorch tensors: Shape {tuple(torch_batch.shape)}")

        # 6. Render HUD
        final_hud = render_industrial_hud(frame, tracker, severity, mode="industrial")

        print(f"  Frame {frame_idx + 1:02d} | BBoxes Detected: {len(bboxes)} | Active Tracks: {len(tracked_objects)} | Defect Surface Area: {severity:.2f}%")

    if save_demo_image and final_hud is not None:
        out_path = "tracked_defect_demo.jpg"
        cv2.imwrite(out_path, final_hud)
        print("=" * 70)
        print(f"[+] SUCCESS: Sample annotated frame exported to: {os.path.abspath(out_path)}")
        print(f"[+] Total Distinct Anomalies Tracked & Indexed: {tracker.next_object_id}")
        print("=" * 70)


if __name__ == "__main__":
    run_pipeline_demo()
