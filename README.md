# AI-Powered Facial Expression Recognition System with Non-Manual Feature Interpretation

A complete computer vision system to detect, classify, and explain facial expressions. Designed for semester-level project evaluation, this system combines a **PyTorch Convolutional Neural Network (CNN)** for emotion classification, **MediaPipe Face Mesh** for landmark extraction, **Grad-CAM (Class Activation Mapping)** for explainable decision maps, **FACS Action Unit (AU) mapping**, and **Head Pose Estimation**.

Features a responsive dark-themed dashboard built with **FastAPI, React, Tailwind CSS, and Recharts**.

---

## Technical Upgrades & Academic Features

1. **Real-time Webcam Stream & Upload Overlay**: Captures video frames, processes face coordinates, and renders glowing landmarker traces (eyes, eyebrows, mouth, jawline) on-screen in real-time. The same overlays align automatically on uploaded images/videos.
2. **Head Pose Estimation**: Geometrically calculates **Yaw (rotation)**, **Pitch (elevation)**, and **Roll (tilt)** from key landmark ratio points in real-time, displaying them in a dedicated telemetry card.
3. **FACS Action Units (AUs)**: Computes intensity percentages (0–100%) for standard Action Units including:
   - **AU1 (Inner Brow Raiser)**
   - **AU2 (Outer Brow Raiser)**
   - **AU4 (Brow Lowerer)**
   - **AU12 (Lip Corner Puller - Smile)**
   - **AU25 (Lips Part)**
   - **AU26 (Jaw Drop)**
4. **Grad-CAM Focus Zones**: Maps Class Activation Mapping (Grad-CAM) overlay densities to semantic regions (eyes, eyebrows, mouth, nose) to identify the network's primary focusing region, showing weight bars.
5. **Model Performance Tab**: Contains multiclass Receiver Operating Characteristic (ROC) Curves plotted for all 7 emotions, average macro-averages (Accuracy, Precision, Recall, F1), and a per-class metrics table.
6. **System Architecture Diagram**: An interactive flowchart explaining the data flow:
   `Input Stream -> Face Mesh (Landmarks) -> Feature Pose/FACS -> CNN Classifier & Grad-CAM -> Explanations & Logs`.
7. **Telemetry Exporters**: Export the active session's history logs (capped at 100 entries) as spreadsheet-compatible **CSV** or structured **JSON**. In addition, compile a formal **PDF Evaluation Report** client-side containing prediction logs and pose telemetry.

---

## Directory Layout

```
facial_expression_recognition/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application, routing endpoints
│   │   ├── model.py         # EmotionCNN model and Grad-CAM helpers
│   │   ├── detector.py      # MediaPipe Face Mesh landmark and pose grouping
│   │   └── utils.py         # Base64 serialization, Grad-CAM overlays
│   ├── scripts/
│   │   ├── train.py         # PyTorch dataset, dataloader & training loops
│   │   └── generate_data.py # Synthetic image generation scripts
│   └── requirements.txt     # Python server requirements
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx        # Root tab-layout controller
│   │   │   ├── WebcamViewer.jsx     # Webcam stream and canvas mesh overlays
│   │   │   ├── MediaUploader.jsx    # Image/Video dropzone with canvas overlays
│   │   │   ├── AnalyticsPanel.jsx   # Charts and Confusion Matrix Heatmaps
│   │   │   ├── Explainability.jsx   # Grad-CAM opacity sliders & FACS descriptions
│   │   │   ├── ResearchPanel.jsx    # FACS Action Units, probabilities & coordinates
│   │   │   ├── EvaluationTab.jsx    # ROC curves and Macro metrics
│   │   │   ├── SystemArchitecture.jsx # Flowchart mapping data flow
│   │   │   └── ModelMonitor.jsx     # Hyperparameters forms & training logs
│   │   ├── App.jsx                  # React component shell
│   │   ├── index.css                # Base Tailwind & custom animations
│   │   └── main.jsx                 # Entry point
│   ├── tailwind.config.js           # Layout styles
│   ├── package.json                 # Node packages
│   └── vite.config.js               # Dev server configuration
├── README.md                        # Project documentation
└── run.sh                           # Unified parallel processes launcher
```

---

## Setup & Installation

### Prerequisites
* Python 3.9+
* Anaconda or Miniconda

### Installation & Launch Steps

1. Navigate to the project root directory:
   ```bash
   cd /Users/peeyush/.gemini/antigravity/scratch/facial_expression_recognition
   ```

2. Run the unified launcher script:
   ```bash
   ./run.sh
   ```
   *This script sets up the Conda environment, installs requirements, starts the FastAPI server on `http://localhost:8000`, runs the Vite dev server, and hosts the frontend at `http://localhost:5173`.*

---

## API Documentation

### 1. Analyze Image Frame
* **Endpoint**: `POST /api/analyze-image`
* **Request Payload**:
  ```json
  { "image": "data:image/jpeg;base64,..." }
  ```
* **Response Output**:
  ```json
  {
    "face_detected": true,
    "predicted_emotion": "happy",
    "confidence": 0.985,
    "emotion_confidences": { "happy": 0.985, "sad": 0.005, ... },
    "bbox": [152, 120, 110, 115],
    "landmarks": { "left_eye": [...], ... },
    "pose": { "yaw": 2.4, "pitch": -1.1, "roll": 0.5 },
    "attention_regions": { "eyebrows": 0.12, "eyes": 0.22, "nose": 0.15, "mouth": 0.51 },
    "primary_attention_region": "mouth",
    "crop_image": "data:image/jpeg;base64,...",
    "gradcam_image": "data:image/jpeg;base64,...",
    "timestamp": "2026-05-24T15:16:00Z"
  }
  ```

### 2. Session Analytics
* **Endpoint**: `GET /api/analytics`
* **Response Output**: Includes session distribution, history list, macro averages, and ROC curve plot coordinate coordinates.
