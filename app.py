"""Streamlit web interface for GhostVision-AI multi-modal wildlife perception."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import streamlit as st

from ghostvision.adapters import (
    ApplicationSegmentationResult,
    ThermalApplicationResult,
    ThermalDependencyError,
    ThermalResultError,
)
from ghostvision.application import (
    GhostVisionApplication,
    ModalityNotConfiguredError,
    UnsupportedModalityError,
)
from ghostvision.rgb_inference import (
    IncompatibleCheckpointError,
    InferenceDependencyError,
    InvalidImageError,
    MissingCheckpointError,
    RGBInferenceError,
    UnknownModelError,
)
from ghostvision.ui_utils import (
    ENV_RGB_AUGMENTED,
    ENV_RGB_BASELINE,
    ENV_THERMAL_CHECKPOINT,
    RGB_TEST_METRICS,
    THERMAL_TEST_METRICS,
    draw_thermal_detections,
    get_default_checkpoint_paths,
)


def init_page_config() -> None:
    st.set_page_config(
        page_title="GhostVision-AI | Wildlife Perception",
        page_icon="🐾",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    st.title("🐾 GhostVision-AI")
    st.subheader("Dual-Modality AI Perception: RGB Camouflage & Thermal Wildlife Detection")
    st.markdown(
        """
        **GhostVision-AI** is an intelligent vision system designed to detect and localize cryptic 
        and nocturnal wildlife across challenging wilderness environments. It provides two independent, 
        rigorously validated perception pipelines:
        
        - **🌿 RGB Camouflaged Object Segmentation:** Leverages a U-Net architecture with a ResNet34 encoder 
          to segment camouflaged animals blending seamlessly into complex natural surroundings.
        - **🔥 Thermal Wildlife Detection:** Utilizes a lightweight YOLO11n object detector to locate nocturnal 
          and obscured warm-bodied fauna in thermal infrared imagery.
        """
    )
    st.divider()


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("⚙️ Configuration")
    
    defaults = get_default_checkpoint_paths()
    
    st.sidebar.markdown("### 🗂️ Model Checkpoints")
    st.sidebar.caption(
        "Set paths below or configure environment variables (`GHOSTVISION_RGB_BASELINE`, "
        "`GHOSTVISION_RGB_AUGMENTED`, `GHOSTVISION_THERMAL_CHECKPOINT`)."
    )

    rgb_baseline_input = st.sidebar.text_input(
        "RGB Baseline Checkpoint (.pth)",
        value=defaults["rgb_baseline"],
        help="Path to the baseline U-Net ResNet34 epoch 19 checkpoint.",
    ).strip()

    rgb_augmented_input = st.sidebar.text_input(
        "RGB Augmented Checkpoint (.pth)",
        value=defaults["rgb_augmented"],
        help="Path to the augmented U-Net ResNet34 epoch 7 checkpoint.",
    ).strip()

    thermal_input = st.sidebar.text_input(
        "Thermal YOLO11n Checkpoint (.pt)",
        value=defaults["thermal"],
        help="Path to the trained YOLO11n thermal detector weights (best.pt).",
    ).strip()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Inference Settings")
    
    thermal_conf = st.sidebar.slider(
        "Thermal Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required to report a thermal animal detection.",
    )

    # Checkpoint existence status badges
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔎 Checkpoint Status")
    
    col_stat1, col_stat2 = st.sidebar.columns(2)
    with col_stat1:
        st.caption("RGB Baseline")
        if rgb_baseline_input and Path(rgb_baseline_input).is_file():
            st.success("Found", icon="✅")
        else:
            st.warning("Not loaded", icon="⚠️")
            
        st.caption("Thermal")
        if thermal_input and Path(thermal_input).is_file():
            st.success("Found", icon="✅")
        else:
            st.warning("Not loaded", icon="⚠️")

    with col_stat2:
        st.caption("RGB Augmented")
        if rgb_augmented_input and Path(rgb_augmented_path := rgb_augmented_input).is_file():
            st.success("Found", icon="✅")
        else:
            st.warning("Not loaded", icon="⚠️")

    return {
        "rgb_baseline": rgb_baseline_input,
        "rgb_augmented": rgb_augmented_input,
        "thermal": thermal_input,
        "thermal_confidence": thermal_conf,
    }


def render_metrics_tab() -> None:
    st.header("📊 Verified Evaluation Metrics")
    st.markdown(
        """
        The metrics below represent the benchmark results achieved on the **held-out evaluation test splits**.
        Both modalities maintain rigorous separation of test data without leaking or retraining.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌿 RGB Camouflage Segmentation")
        st.caption(f"**Dataset:** {RGB_TEST_METRICS['dataset']} | **Evaluated Images:** {RGB_TEST_METRICS['images_evaluated']:,}")
        
        rgb_data = []
        for m in RGB_TEST_METRICS["models"]:
            rgb_data.append({
                "Model Variant": m["name"],
                "Checkpoint Epoch": m["checkpoint_epoch"],
                "Mean Dice": f"{m['dice']:.4f}",
                "Mean IoU": f"{m['iou']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "Selected Val Dice": f"{m['selected_val_dice']:.4f}",
            })
        st.dataframe(pd.DataFrame(rgb_data), use_container_width=True, hide_index=True)
        st.info(f"💡 **Analysis:** {RGB_TEST_METRICS['notes']}")

    with col2:
        st.subheader("🔥 Thermal Wildlife Detection")
        st.caption(f"**Dataset:** {THERMAL_TEST_METRICS['dataset']} | **Held-out Test Images:** {THERMAL_TEST_METRICS['test_images']:,}")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Precision", f"{THERMAL_TEST_METRICS['precision'] * 100:.2f}%")
        kpi2.metric("Recall", f"{THERMAL_TEST_METRICS['recall'] * 100:.2f}%")
        kpi3.metric("mAP@50", f"{THERMAL_TEST_METRICS['mAP50'] * 100:.2f}%")
        kpi4.metric("mAP@50-95", f"{THERMAL_TEST_METRICS['mAP50_95'] * 100:.2f}%")

        thermal_data = [{
            "Architecture": THERMAL_TEST_METRICS["architecture"],
            "Target Class": THERMAL_TEST_METRICS["classes"],
            "Resolution": THERMAL_TEST_METRICS["input_resolution"],
            "Validation Images": f"{THERMAL_TEST_METRICS['validation_images']:,}",
            "Test Images": f"{THERMAL_TEST_METRICS['test_images']:,}",
        }]
        st.dataframe(pd.DataFrame(thermal_data), use_container_width=True, hide_index=True)
        st.info(f"💡 **Analysis:** {THERMAL_TEST_METRICS['notes']}")


def run_rgb_inference_pipeline(
    uploaded_file: Any,
    model_name: str,
    baseline_path: str,
    augmented_path: str,
) -> None:
    # Validate configuration
    target_path = baseline_path if model_name == "baseline" else augmented_path
    if not target_path or not Path(target_path).is_file():
        st.error(
            f"❌ The checkpoint for **{model_name}** is not configured or does not exist at: "
            f"`{target_path or 'Not specified'}`. Please provide a valid file in the sidebar."
        )
        return

    checkpoint_dict = {}
    if baseline_path and Path(baseline_path).is_file():
        checkpoint_dict["baseline"] = baseline_path
    if augmented_path and Path(augmented_path).is_file():
        checkpoint_dict["augmented"] = augmented_path

    # Safely save uploaded file to a temporary location
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png"]:
        st.error("Invalid file format. Please upload a JPEG or PNG image.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_img_path = Path(temp_dir) / f"upload_{uploaded_file.name}"
        temp_img_path.write_bytes(uploaded_file.getbuffer())

        try:
            with Image.open(temp_img_path) as test_img:
                test_img.verify()
        except Exception:
            st.error("❌ The uploaded file is not a valid, decodable image.")
            return

        # Instantiate GhostVisionApplication
        try:
            app = GhostVisionApplication(
                rgb_checkpoint_paths=checkpoint_dict,
                rgb_model_name=model_name,
            )
        except Exception as exc:
            st.error(f"❌ Application initialization failed: {exc}")
            return

        with st.spinner(f"Running RGB segmentation with {model_name} model..."):
            try:
                result = app.predict("rgb", temp_img_path)
            except InferenceDependencyError as exc:
                st.error(f"❌ Dependency Error: {exc}")
                return
            except (MissingCheckpointError, IncompatibleCheckpointError) as exc:
                st.error(f"❌ Checkpoint Error: {exc}")
                return
            except InvalidImageError as exc:
                st.error(f"❌ Invalid Image: {exc}")
                return
            except Exception as exc:
                st.error(f"❌ Inference Error: {exc}")
                return

        # Render outputs
        if isinstance(result, ApplicationSegmentationResult):
            st.success(f"Segmentation complete using {result.model_name.capitalize()} model!")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### 1. Input Image")
                st.image(uploaded_file, use_container_width=True)
            with c2:
                st.markdown("#### 2. Prediction Mask")
                # Scale mask 0..1 to 0..255 for clean visual display
                mask_display = (result.mask * 255).astype(np.uint8)
                st.image(mask_display, use_container_width=True, clamp=True)
            with c3:
                st.markdown("#### 3. Overlay (Foreground Highlighted)")
                st.image(result.overlay, use_container_width=True)

            with st.expander("🔍 Model & Checkpoint Metadata", expanded=False):
                meta_cols = st.columns(4)
                meta_cols[0].metric("Architecture", str(result.metadata.get("architecture", "Unet-ResNet34")))
                meta_cols[1].metric("Checkpoint Epoch", str(result.metadata.get("checkpoint_epoch", "-")))
                meta_cols[2].metric("Validation Dice", f"{result.metadata.get('validation_dice', 0):.4f}")
                meta_cols[3].metric("Threshold", str(result.metadata.get("threshold", 0.5)))
                st.caption(f"**Loaded Checkpoint:** `{result.metadata.get('checkpoint_path')}`")


def run_thermal_inference_pipeline(
    uploaded_file: Any,
    thermal_path: str,
    confidence: float,
) -> None:
    if not thermal_path or not Path(thermal_path).is_file():
        st.error(
            f"❌ Thermal detector checkpoint is not configured or does not exist at: "
            f"`{thermal_path or 'Not specified'}`. Please specify a valid `.pt` model file in the sidebar."
        )
        return

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png"]:
        st.error("Invalid file format. Please upload a JPEG or PNG image.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_img_path = Path(temp_dir) / f"upload_{uploaded_file.name}"
        temp_img_path.write_bytes(uploaded_file.getbuffer())

        try:
            with Image.open(temp_img_path) as test_img:
                test_img.verify()
        except Exception:
            st.error("❌ The uploaded file is not a valid, decodable image.")
            return

        try:
            app = GhostVisionApplication(
                thermal_model_path=thermal_path,
                thermal_confidence=confidence,
            )
        except Exception as exc:
            st.error(f"❌ Application initialization failed: {exc}")
            return

        with st.spinner("Running YOLO11n thermal inference..."):
            try:
                result = app.predict("thermal", temp_img_path)
            except ThermalDependencyError as exc:
                st.error(f"❌ Thermal Dependency Error: {exc}")
                return
            except FileNotFoundError as exc:
                st.error(f"❌ File Not Found: {exc}")
                return
            except ThermalResultError as exc:
                st.error(f"❌ Thermal Inference Result Error: {exc}")
                return
            except Exception as exc:
                st.error(f"❌ Inference Error: {exc}")
                return

        if isinstance(result, ThermalApplicationResult):
            st.success(f"Thermal detection complete! Identified {result.animal_count} animal(s).")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 1. Original Thermal Image")
                st.image(uploaded_file, use_container_width=True)
            with c2:
                st.markdown(f"#### 2. Detections & Bounding Boxes ({result.animal_count} found)")
                # Draw detections onto the image
                with Image.open(temp_img_path) as raw_img:
                    annotated_img = draw_thermal_detections(raw_img, result.detections)
                st.image(annotated_img, use_container_width=True)

            st.markdown("#### 📋 Detection Summary")
            col_m1, col_m2 = st.columns([1, 3])
            col_m1.metric("Animal Count", result.animal_count)
            
            with col_m2:
                if result.detections:
                    det_data = [
                        {
                            "Detection #": i + 1,
                            "Class": d.class_name,
                            "Confidence": f"{d.confidence:.4f}",
                            "Bounding Box (X1, Y1, X2, Y2)": f"[{d.bbox[0]:.1f}, {d.bbox[1]:.1f}, {d.bbox[2]:.1f}, {d.bbox[3]:.1f}]",
                        }
                        for i, d in enumerate(result.detections)
                    ]
                    st.dataframe(pd.DataFrame(det_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No animals detected above the selected confidence threshold.")


def main() -> None:
    init_page_config()
    render_header()
    config = render_sidebar()

    tab_rgb, tab_thermal, tab_metrics, tab_about = st.tabs([
        "🌿 RGB Camouflage Mode",
        "🔥 Thermal Detection Mode",
        "📊 Evaluation Metrics",
        "ℹ️ About & Docs",
    ])

    with tab_rgb:
        st.header("🌿 RGB Camouflage Segmentation")
        st.markdown(
            "Upload an RGB image to detect and segment camouflaged animals. "
            "Select between the **Baseline** model (trained without augmentation) or "
            "the **Augmented** model (trained with advanced geometric and photometric transformations)."
        )

        col_cfg1, col_cfg2 = st.columns([2, 2])
        with col_cfg1:
            rgb_model_choice = st.selectbox(
                "Select Model Checkpoint Variant",
                options=["augmented", "baseline"],
                format_func=lambda x: "Augmented Model (Epoch 7 - Higher Precision & Dice)" if x == "augmented" else "Baseline Model (Epoch 19)",
                help="Choose which trained checkpoint weights to run inference with.",
            )

        rgb_file = st.file_uploader(
            "Choose an RGB image (JPEG or PNG)",
            type=["jpg", "jpeg", "png"],
            key="rgb_uploader",
        )

        if rgb_file is not None:
            if st.button("Run RGB Segmentation", type="primary", key="btn_run_rgb"):
                run_rgb_inference_pipeline(
                    rgb_file,
                    rgb_model_choice,
                    config["rgb_baseline"],
                    config["rgb_augmented"],
                )

    with tab_thermal:
        st.header("🔥 Thermal Wildlife Detection")
        st.markdown(
            "Upload a thermal infrared image to detect and localize wildlife using the fine-tuned "
            "**YOLO11n** detector."
        )

        thermal_file = st.file_uploader(
            "Choose a Thermal image (JPEG or PNG)",
            type=["jpg", "jpeg", "png"],
            key="thermal_uploader",
        )

        if thermal_file is not None:
            if st.button("Run Thermal Detection", type="primary", key="btn_run_thermal"):
                run_thermal_inference_pipeline(
                    thermal_file,
                    config["thermal"],
                    config["thermal_confidence"],
                )

    with tab_metrics:
        render_metrics_tab()

    with tab_about:
        st.header("ℹ️ Architecture & Deployment Documentation")
        st.markdown(
            """
            ### Unified Routing Pipeline
            GhostVision-AI uses `GhostVisionApplication` from `ghostvision.application` to route requests 
            cleanly without modifying raw model architectures or translating incompatible result shapes:
            
            - **RGB Pipeline:** Output contains segmentation masks and overlays via `ApplicationSegmentationResult`.
            - **Thermal Pipeline:** Output contains discrete bounding boxes and confidence scores via `ThermalApplicationResult`.
            
            ### Checkpoint Configuration
            For production deployments and CI, define the following environment variables:
            ```bash
            export GHOSTVISION_RGB_BASELINE="/path/to/best_resumed_epoch_19.pth"
            export GHOSTVISION_RGB_AUGMENTED="/path/to/best_resumed_epoch_07.pth"
            export GHOSTVISION_THERMAL_CHECKPOINT="/path/to/best.pt"
            ```
            Alternatively, set these paths directly via the Streamlit configuration sidebar.
            
            ### Safety & Privacy Guarantees
            - Checkpoint weights and user uploads are never committed or persisted permanently.
            - All user uploads are processed inside isolated temporary directories and purged immediately.
            - Full validation ensures incompatible or corrupted weights are caught before inference.
            """
        )


if __name__ == "__main__":
    main()

