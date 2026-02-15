# RoadSense: DUR Pothole Detection & Metrology Pipeline


## Pipeline Architecture

The solution is divided into three processing "Sprints" or stages:

### Sprint 1: Geometry & Calibration
**Goal**: Transform oblique dashboard footage into a calibrated top-down view (Orthophoto).
- **Script**: `image_calibration.py`
- **Technique**: Uses camera calibration (`iphoneXR_calib.npz`) and Inverse Perspective Mapping (IPM) to remove lens distortion and perspective effects.
- **Output**: Generates `dashboard-IPM-only.mp4` where **10 pixels = 1 cm**.

### Sprint 2: Restoration & Signal Enhancement
**Goal**: Clean up the noisy, blurry footage caused by vehicle vibration and motion.
- **Scripts**: 
  - `extract_frames.py`: Decomposes video into individual frames.
  - `restore_frames.py`: Applies a restoration pipeline including De-noising, LAB Color Space separation, and CLAHE (Contrast Limited Adaptive Histogram Equalization) to recover texture detail lost to motion blur and shadows.
- **Output**: High-quality, flat-lit road textures in `./restored-images`.

### Sprint 3: Segmentation & Metrology
**Goal**: Detect defects and calculate their physical properties.
- **Script**: `segmentation_metrology.py`
- **Technique**: 
  - **Retinex Theory**: Normalizes uneven lighting (shadows).
  - **Dual-Hat Transform**: Combines *BlackHat* (defects darker than road) and *TopHat* (defects lighter than road, e.g., exposed soil) to capture all defect types.
  - **Discrete Green’s Theorem**: Calculates the exact surface area in $cm^2$.
- **Output**: `dur_pothole_report.csv` and visual debug images.

## Installation & Prerequisites

The project requires Python 3.x and the following scientific computing libraries:

```bash
pip install opencv-python numpy pandas
```

*Note: You also need a camera calibration file `iphoneXR_calib.npz` (provided) to correct for lens distortion.*

## Usage Instructions

Follow the pipeline in order:

1.  **Step 1: Calibration & IPM**
    Run the calibration script to process the raw video.
    ```bash
    python image_calibration.py
    ```
    *Input: `./videos/IMG_7372.MOV`*
    *Output: `dashboard-IPM-only.mp4`*

2.  **Step 2: Frame Extraction**
    Extract frames from the rectified video.
    ```bash
    python extract_frames.py
    ```
    *Output: `./extracted-frames/`*

3.  **Step 3: Image Restoration**
    Apply image enhancement to remove blur and normalize lighting.
    ```bash
    python restore_frames.py
    ```
    *Output: `./restored-images/`*

4.  **Step 4: Analysis & Reporting**
    Run the final segmentation and metrology engine.
    ```bash
    python segmentation_metrology.py
    ```
    *Output: `dur_pothole_report.csv` and `./processed_output/`*

## Output Data

The final report `dur_pothole_report.csv` contains the following fields:

| Column | Description |
| :--- | :--- |
| `Frame_ID` | Unique identifier for the analyzed video frame. |
| `Pothole_ID` | Unique ID for the specific defect (e.g., `frame_001_P1`). |
| `Area_cm2` | The Calculated surface area of the defect in square centimeters. |
| `Center_X/Y` | Pixel coordinates of the defect center. |

## References & Research `Sprints`

This project implements concepts from:
- **Burger, W. (2016)**: *Zhang’s camera calibration algorithm*.
- **Land, E.H. & McCann, J.J. (1971)**: *Lightness and Retinex Theory*.
- **Brlek, S. et al. (2005)**: *The Discrete Green Theorem*.


