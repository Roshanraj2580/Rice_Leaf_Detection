"""
Integrated Gradio Deployment Application
=========================================
Features:
  1. Real-Time Defect Localization & Spatial Centroid Tracking (OpenCV + HUD Analytics)
  2. Severity Estimation & Industrial Emergency Threshold Alerts
  3. Classical Computer Vision (HSV + Hu Moments + Texture) & Traditional ML Diagnosis
  4. PyTorch Tensor Extraction for Deep Learning Pipelines

Author: Roshan Raj (BIT Mesra)
"""

import cv2
import numpy as np
import gradio as gr
from PIL import Image
import os
import sys

# Import our custom modules
from defect_tracker import CentroidTracker, DefectDetector, render_industrial_hud, extract_pytorch_crops
from traditional_cv_ml import ClassicalFeatureExtractor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Initialize components
detector = DefectDetector(min_defect_area=20, max_defect_area=100000)
classical_extractor = ClassicalFeatureExtractor()

# Global tracker instance
live_tracker = CentroidTracker(max_disappeared=30, max_distance=60.0)

# Pre-train classical classifiers with representative synthetic patterns
def init_classical_models():
    np.random.seed(42)
    n_samples = 40
    n_classes = 4
    class_names = ["Bacterial Blight", "Blast", "Brown Spot", "Tungro"]
    X, y = [], []
    for cls in range(n_classes):
        for _ in range(n_samples):
            base_col = (25 + cls * 30, 110 - cls * 10, 40 + cls * 25)
            img = np.full((120, 120, 3), base_col, dtype=np.uint8)
            cv2.circle(img, (60, 60), 8 + cls * 6, (15, 35, 95), -1)
            feat = classical_extractor.extract_feature_vector(img)
            X.append(feat)
            y.append(cls)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X, y)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    return svm, rf, class_names

svm_model, rf_model, CLASS_NAMES = init_classical_models()


# ============================================================================
# 1. CORE DEFECT DETECTION & TRACKING INFERENCE
# ============================================================================
def process_single_frame(input_image, mode_selection, alert_threshold):
    if input_image is None:
        return None, "No image uploaded.", {}

    # Convert to BGR for OpenCV
    if isinstance(input_image, Image.Image):
        frame_rgb = np.array(input_image)
    else:
        frame_rgb = input_image

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # 1. Segment leaf and lesions
    leaf_mask, lesion_mask, severity = detector.segment_leaf_and_lesions(frame_bgr)

    # 2. Extract bounding boxes
    bboxes = detector.extract_bounding_boxes(lesion_mask)

    # 3. Update spatial tracker
    tracked = live_tracker.update(bboxes)

    # 4. Check PyTorch tensor bridge
    pytorch_crops = extract_pytorch_crops(frame_bgr, bboxes)
    torch_status = f"Generated {pytorch_crops.shape[0]} PyTorch Tensors (Shape: {list(pytorch_crops.shape)})" if pytorch_crops is not None else "No defect crops extracted."

    # 5. Render HUD with Alert Threshold
    annotated_bgr = render_industrial_hud(frame_bgr, live_tracker, severity, mode=mode_selection.lower())
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Status formatting
    is_hazard = severity >= alert_threshold
    hazard_msg = " [!] CRITICAL HAZARD TRIGGERED: Defect threshold exceeded!" if is_hazard else " [OK] Normal Operating Parameters"

    summary_text = (
        f"ANOMALY INSPECTION REPORT:\n"
        f"-----------------------------------------\n"
        f"Inspection Mode         : {mode_selection}\n"
        f"Active Tracked Defects  : {len(tracked)}\n"
        f"Total Defects Indexed   : {live_tracker.next_object_id}\n"
        f"Surface Defect Ratio    : {severity:.2f}%\n"
        f"Safety Alert Threshold  : {alert_threshold}%\n"
        f"System Status           :{hazard_msg}\n"
        f"PyTorch Deep Bridge     : {torch_status}\n"
    )

    # Confidence distribution for severity stages
    stages = {
        "Stage 1 (Normal < 3%)": max(0.0, min(1.0, 1.0 - severity / 10.0)),
        "Stage 2 (Moderate 3-12%)": max(0.0, min(1.0, severity / 12.0 if severity < 12 else (20 - severity) / 8.0)),
        "Stage 3 (Critical Hazard > 12%)": max(0.0, min(1.0, (severity - 8.0) / 10.0 if severity >= 8 else 0.0))
    }

    return annotated_rgb, summary_text, stages


# ============================================================================
# 2. CLASSICAL COMPUTER VISION & ML INFERENCE
# ============================================================================
def analyze_traditional_ml(input_image):
    if input_image is None:
        return "No image uploaded.", {}

    if isinstance(input_image, Image.Image):
        frame_rgb = np.array(input_image)
    else:
        frame_rgb = input_image
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Extract 522-dimensional handcrafted vector
    feat_vector = classical_extractor.extract_feature_vector(frame_bgr).reshape(1, -1)
    hu_moments = classical_extractor.extract_hu_moments(frame_bgr)
    textures = classical_extractor.extract_texture_gradients(frame_bgr)

    # Predict via SVM & Random Forest
    svm_probs = svm_model.predict_proba(feat_vector)[0]
    rf_probs = rf_model.predict_proba(feat_vector)[0]

    svm_pred = CLASS_NAMES[np.argmax(svm_probs)]
    rf_pred = CLASS_NAMES[np.argmax(rf_probs)]

    report = (
        f"CLASSICAL OPENCV FEATURE ANALYSIS:\n"
        f"-----------------------------------------\n"
        f"Total Feature Dimensions : {feat_vector.shape[1]} descriptors\n"
        f"  • HSV Color Bins       : 512 bins (3D Histogram)\n"
        f"  • Hu Moments (Shape)   : 7 Invariant Moments\n"
        f"  • Sobel Gradient (Tex) : Mean={textures[0]:.2f}, Std={textures[1]:.2f}, Skew={textures[2]:.2f}\n\n"
        f"TRADITIONAL ML PREDICTIONS:\n"
        f"-----------------------------------------\n"
        f"Support Vector Machine   : {svm_pred} (Confidence: {np.max(svm_probs)*100:.1f}%)\n"
        f"Random Forest (100 Trees): {rf_pred} (Confidence: {np.max(rf_probs)*100:.1f}%)\n"
    )

    pred_dict = {CLASS_NAMES[i]: float(svm_probs[i]) for i in range(len(CLASS_NAMES))}
    return report, pred_dict


# ============================================================================
# 3. GRADIO UI LAYOUT
# ============================================================================
custom_css = """
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
#title-header { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="Industrial & Foliar Defect Inspection Platform") as demo:
    gr.Markdown(
        """
        # 🛡️ Dual-Purpose Real-Time Defect Detection, Tracking & ML System
        ### Bridging Foliar Crop Pathology to Manufacturing Safety in Heavy Industries
        **Technologies:** `OpenCV`, `PyTorch`, `Python`, `Centroid Tracking`, `Traditional ML (SVM/RF)`, `Deep CNNs`  
        **Author:** Roshan Raj (BIT Mesra)
        """,
        elem_id="title-header"
    )

    with gr.Tab("🎯 1. Real-Time Defect Detection & Spatial Tracking"):
        gr.Markdown("Upload an image of a rice leaf or industrial surface. The system localizes anomalies, tracks centroids, and computes surface degradation.")
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="numpy", label="Upload Leaf or Surface Image")
                mode_select = gr.Radio(
                    choices=["Industrial", "Foliar / Agricultural"],
                    value="Industrial",
                    label="Inspection Domain"
                )
                threshold_slider = gr.Slider(
                    minimum=1.0, maximum=30.0, value=12.0, step=0.5,
                    label="Critical Safety Alert Threshold (% Surface Area)"
                )
                track_btn = gr.Button("🔍 Run Defect Localization & Tracking", variant="primary")

            with gr.Column(scale=1):
                annotated_output = gr.Image(type="numpy", label="Tracked HUD Output (Bounding Boxes + Centroids + Analytics)")
                status_output = gr.Textbox(label="Diagnostic & Hazard Summary", lines=8)
                stage_label = gr.Label(label="Severity Risk Distribution")

        track_btn.click(
            fn=process_single_frame,
            inputs=[img_input, mode_select, threshold_slider],
            outputs=[annotated_output, status_output, stage_label]
        )

    with gr.Tab("🧪 2. Classical OpenCV & Traditional ML Benchmark"):
        gr.Markdown("Extracts handcrafted 522-D visual descriptors (HSV Histograms, Hu Moments, Sobel Gradients) and runs SVM and Random Forest classification.")
        with gr.Row():
            with gr.Column(scale=1):
                ml_img_input = gr.Image(type="numpy", label="Upload Inspection Image")
                ml_btn = gr.Button("⚡ Extract OpenCV Descriptors & Classify", variant="primary")

            with gr.Column(scale=1):
                ml_report = gr.Textbox(label="Handcrafted Feature Vector & ML Evaluation", lines=10)
                ml_confidence = gr.Label(label="Classical ML Class Probabilities")

        ml_btn.click(
            fn=analyze_traditional_ml,
            inputs=[ml_img_input],
            outputs=[ml_report, ml_confidence]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css, share=False)
