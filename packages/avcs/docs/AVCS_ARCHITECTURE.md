# Artificial Visual Cortex for Screens (AVCS)

## Overview

**AVCS (Artificial Visual Cortex for Screens)** is a biologically inspired, non-generative, real-time vision-language perception system designed specifically for computer screens.

Unlike standard Vision-Language Models (VLMs) that rely on slow autoregressive token generation (300ms–2000ms), AVCS operates at **true biological visual speeds (sub-40ms on GPU, ~75ms on CPU)** by replacing text generation with dense spatial affordance maps, semantic scene categorization, and top-down cross-attention grounding.

```
                           +----------------------------------------+
                           |     TOP-DOWN LANGUAGE MODULATION       |
                           |   (User Intent / Goal / Query Vector)  |
                           +-------------------+--------------------+
                                               |
                                               v [Text Projection / 256-d]
                                               |
+--------------------------+                   |
|   SCREEN VIDEO STREAM    |                   |
|  (Frames t and t - 1)    |                   |
+------------+-------------+                   |
             |                                 |
             v                                 |
+------------------------------------+         |
| 1. RETINAL MODULE (Multi-Scale)    |         |
|  * Peripheral Stream (Global 256)  |         |
|  * Foveal Crop (High-Res Focus)    |         |
|  * Saccadic Inertial Controller    |         |
+------------+-----------------------+         |
             |                                 |
             v                                 |
+------------------------------------+         |
| 2. CORTICAL NEURAL BACKBONE        |         |
|  * Dual-Path Inverted Residuals    |         |
|  * Squeeze-and-Excitation (SE)     |         |
|  * Multi-Scale Spatial Fusion      |         |
+------------+-----------------------+         |
             |                                 |
             v                                 v
+-------------------------------------------------------------------+
| 3. DUAL-STREAM CORTICAL INTEGRATION                               |
|                                                                   |
|   [Ventral Stream - "WHAT"]       [Dorsal Stream - "WHERE/HOW"]   |
|   * UI Category Identification    * Dense Affordance Probabilities|
|   * Text Density & Scene Latent   * Action Hotspot Coordinates    |
|                                                                   |
|             v                                     v               |
|          +-------------------------------------+                  |
|          |    Cross-Attention with Language    |                  |
|          +-------------------------------------+                  |
+----------------------------------+--------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------+
| 4. COGNITIVE READOUT HEADS (Single-Pass Execution)                |
|  * Affordance Map [32x32x5] : Clickable, Typeable, Scrollable...  |
|  * Semantic Category Logits : Browser, Terminal, Editor, Modal... |
|  * Temporal Surprise & Stability : Stable, Animating, Transition  |
|  * Language Grounding : Target Coordinates & Spatial Heatmap      |
+-------------------------------------------------------------------+
```

---

## Codebase Directory Structure

```
forge/vision/cortex/
├── __init__.py                # Clean high-level API exports
├── config.py                  # CortexConfig, ModelConfig, RetinaConfig, TrainingConfig
├── model.py                   # Unified ArtificialVisualCortex model & CortexState
├── benchmark.py               # Profiling & Latency Verification Suite
│
├── core/                      # Biological Neural Sub-systems
│   ├── __init__.py
│   ├── fovea.py               # Retinal Foveation (Peripheral Global + Foveal High-Res)
│   ├── backbone.py            # Dual-path Inverted Residual Backbone with SE attention
│   ├── dorsal.py              # Dorsal Stream: Spatial Affordance Field (Where/How)
│   ├── ventral.py             # Ventral Stream: Semantic UI Categories & Scene Latent (What)
│   ├── modulation.py          # Top-Down Language Modulation (Zero-Token Grounding)
│   └── predictive.py          # Temporal Predictive Coding & World Model (Surprise)
│
├── data/                      # Screen Data Generation & Pipeline
│   ├── __init__.py
│   ├── synthetic.py           # Procedural Screen Generator with ground-truth affordances
│   └── dataset.py             # PyTorch Dataset for temporal pairs & affordance masks
│
├── training/                  # Cloud GPU Training Engine
│   ├── __init__.py
│   ├── losses.py              # AVCSMultiTaskLoss (Self-supervised V-JEPA + Supervised CE)
│   └── trainer.py             # Cloud GPU Trainer (AMP FP16, CosineAnnealing, AdamW)
│
└── inference/                 # Real-time Streaming & Model Export
    ├── __init__.py
    ├── engine.py              # Asynchronous streaming perception engine with callbacks
    └── export.py              # TorchScript & ONNX exporters for Cloud GPU TensorRT
```

---

## Neurobiological Components

### 1. Retinal Foveation (`core/fovea.py`)
- **Peripheral Stream**: Downscaled global receptive field (`256x256`) that captures overall desktop layout, active windows, and global boundaries.
- **Foveal Stream**: High-resolution foveal crop (`256x256`) centered at the current **fixation point**, capturing small text, fine lines, and small icons.
- **Saccadic Controller**: Automatically shifts gaze to peak temporal motion or saliency hotspots with biological EMA smoothing.

### 2. Dorsal Stream: Where & How (`core/dorsal.py`)
Predicts a dense 5-class interaction affordance tensor:
- `0: static` (Background, non-interactive text)
- `1: clickable` (Buttons, icons, links, tabs)
- `2: typeable` (Text inputs, terminal prompts, search boxes)
- `3: scrollable` (Feeds, long panels, document viewers)
- `4: draggable` (Sliders, splitters, window titlebars)

### 3. Ventral Stream: What (`core/ventral.py`)
Classifies overall UI category (`browser_webpage`, `editor_code`, `terminal_shell`, `modal_dialog`, `form_input`, etc.), estimates text density, and yields a 256-dimensional compact scene latent.

### 4. Language Modulation (`core/modulation.py`)
Accepts an incoming goal or search query (e.g. *"Find close button"*), performs spatial cross-attention with visual tokens, and outputs:
- **Spatial Relevance Heatmap** ($H \times W$)
- **Target Coordinates** $(x, y)$ in normalized $[0, 1]$ range.
- **Zero token generation** required.

### 5. Predictive Coding & World Model (`core/predictive.py`)
- Based on Karl Friston's Free Energy Principle and Yann LeCun's V-JEPA.
- Predicts next latent state $\hat{Z}_{t+1}$ from current state $Z_t$.
- Computes Temporal Surprise: $S_t = 1 - \cos(Z_t, \hat{Z}_t)$.
- Categorizes screen stability: `stable`, `animating`, `transitioning`, `surprise_event`.

---

## Quickstart & CLI Commands

### 1. Run Benchmark (Profile Latency)
```bash
python -m forge.vision.cortex.benchmark --iterations 30
```

### 2. Train on Cloud GPU
```bash
python -m forge.vision.cortex.training.trainer --epochs 20 --batch-size 32 --samples 5000
```

### 3. Export to ONNX / TensorRT
```python
from forge.vision.cortex import ArtificialVisualCortex, export_onnx

model = ArtificialVisualCortex()
export_onnx(model, save_path="runs/avcs_cortex.onnx")
```

### 4. Run Unit Tests
```bash
python -m pytest tests/test_cortex.py -v
```

