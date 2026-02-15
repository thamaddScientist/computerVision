import cv2
import numpy as np
import pandas as pd
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
# From Sprint 1: You determined 10 pixels = 1 cm
PIXELS_PER_CM = 10.0  
SCALE_FACTOR = 1.0 / PIXELS_PER_CM  # cm per pixel

# Thresholds
MIN_AREA_CM2 = 150.0  # DUR Standard: Ignore small cracks
DEBUG_MODE = True     # Saves visualized output images
DASHBOARD_MASK_RATIO = 0.15 # Mask bottom 15% to hide dashboard/bonnet

# Paths
INPUT_DIR = './restored-images'
OUTPUT_DIR = './processed_output'
REPORT_FILE = 'dur_pothole_report.csv'

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# CORE ALGORITHMS
# ==========================================

def apply_retinex(image_gray):
    """
    Implements Land & McCann's Retinex Theory.
    Separates Reflectance (R) from Illumination (L).
    """
    img_float = image_gray.astype(float) + 1.0
    
    # Estimate Illumination (Low freq)
    illumination = cv2.GaussianBlur(img_float, (0, 0), sigmaX=51, sigmaY=51)
    
    # Extract Reflectance (High freq)
    reflectance = np.log(img_float) - np.log(illumination)
    
    reflectance = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX)
    return reflectance.astype(np.uint8)

def extract_defects_dual_hat(reflectance_img):
    """
    Uses Morphological Hat Transforms to find BOTH:
    1. Dark holes (BlackHat)
    2. Light exposed soil (TopHat)
    Returns a combined binary mask.
    """
    # Kernel size roughly matches the "texture scale" of the road aggregate
    # A larger kernel (e.g., 15x15) finds larger blobs (potholes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    
    # 1. BlackHat: Returns elements darker than surroundings (Deep holes)
    blackhat = cv2.morphologyEx(reflectance_img, cv2.MORPH_BLACKHAT, kernel)
    
    # 2. TopHat: Returns elements lighter than surroundings (Exposed Soil)
    tophat = cv2.morphologyEx(reflectance_img, cv2.MORPH_TOPHAT, kernel)
    
    # 3. Combine: Add them to get a "Variance/Roughness" Map
    roughness_map = cv2.add(blackhat, tophat)
    
    # 4. Binarize using Otsu's method on the Roughness Map
    # We rely on the fact that potholes are "rougher" than smooth road
    thresh_val, binary = cv2.threshold(roughness_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def calculate_metric_area(contour, scale_cm_per_pixel):
    """
    Implements Discrete Green's Theorem (via Shoelace Formula)
    """
    area_pixels = cv2.contourArea(contour)
    area_cm2 = area_pixels * (scale_cm_per_pixel ** 2)
    return area_cm2

# ==========================================
# MAIN PIPELINE
# ==========================================

def process_frame(filepath, frame_id):
    results = []
    
    # 1. Load Image
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error reading {filepath}")
        return []
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # The IPM warp creates black "funnel" borders. The edges between 
    # road and black border create high contrast that looks like a defect.
    # We must mask out the border artifacts.
    # ---------------------------------------------------------
    # Find all non-black pixels (threshold > 10 to clear compression artifacts)
    _, valid_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Erode mask to shrink it away from the jagged edges
    # 10 pixels = 1cm. We erode ~2cm (20px) to be safe and ignore edge effects.
    mask_kernel = np.ones((21, 21), np.uint8)
    valid_mask = cv2.erode(valid_mask, mask_kernel)
    
    # ---------------------------------------------------------
    # DASHBOARD MASKING
    # The bottom of the frame often contains the car bonnet/dashboard.
    # We explicitly zero out the bottom N% of the mask.
    # ---------------------------------------------------------
    h, w = valid_mask.shape
    cutoff = int(h * (1 - DASHBOARD_MASK_RATIO))
    valid_mask[cutoff:, :] = 0
    # ---------------------------------------------------------

    # 2. Retinex (Normalize Lighting)
    reflectance_map = apply_retinex(gray)
    
    # 3. Segmentation (Dual Hat Transform)
    # This handles the "African Road" context (Dark holes + Red Soil)
    bin_img = extract_defects_dual_hat(reflectance_map)
    
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=valid_mask)
    
    # 4. Morphological Cleanup (Remove noise)
    kernel_clean = np.ones((3,3), np.uint8)
    clean_bin = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_clean)
    # Dilate slightly to connect broken parts of the same pothole
    clean_bin = cv2.dilate(clean_bin, kernel_clean, iterations=2)
    
    # 5. Metrology
    contours, _ = cv2.findContours(clean_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pothole_count = 0
    vis_img = img.copy()
    
    for cnt in contours:
        area_cm2 = calculate_metric_area(cnt, SCALE_FACTOR)
        
        if area_cm2 > MIN_AREA_CM2:
            pothole_count += 1
            unique_id = f"{frame_id}_P{pothole_count}"
            
            # Find Center
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            results.append({
                'Frame_ID': frame_id,
                'Pothole_ID': unique_id,
                'Area_cm2': round(area_cm2, 2),
                'Center_X': cX,
                'Center_Y': cY
            })
            
            # Visualization
            cv2.drawContours(vis_img, [cnt], -1, (0, 0, 255), 2)
            cv2.putText(vis_img, f"{round(area_cm2)}cm2", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if DEBUG_MODE and pothole_count > 0:
        out_path = os.path.join(OUTPUT_DIR, f"vis_{frame_id}.jpg")
        cv2.imwrite(out_path, vis_img)
        
    return results

def main():
    print("Starting DUR Pothole Quantifier (Sprint 3)...")
    print("Using Retinex + Dual-Hat Morphological Segmentation")
    
    report_data = []
    
    # Get all images
    extensions = ['*.jpg', '*.png', '*.jpeg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not files:
        print(f"No images found in {INPUT_DIR}.")
        # Create dummy for testing logic
        dummy = np.full((500, 500, 3), 128, dtype=np.uint8) # Grey road
        cv2.circle(dummy, (150, 150), 40, (50, 50, 50), -1) # Dark Hole
        cv2.circle(dummy, (350, 350), 40, (200, 180, 150), -1) # Light Soil (simulated)
        dummy_path = os.path.join(INPUT_DIR, "test_mixed_defects.jpg")
        cv2.imwrite(dummy_path, dummy)
        files = [dummy_path]
        print("Created dummy frame with Dark and Light defects.")

    for filepath in files:
        filename = os.path.basename(filepath)
        frame_id = os.path.splitext(filename)[0]
        print(f"Processing: {frame_id}")
        frame_results = process_frame(filepath, frame_id)
        report_data.extend(frame_results)
        
    if report_data:
        df = pd.DataFrame(report_data)
        df.to_csv(REPORT_FILE, index=False)
        print(f"Report generated: {REPORT_FILE}")
    else:
        print("No defects found.")

if __name__ == "__main__":
    main()