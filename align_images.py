import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# =================CONFIGURATION=================
INPUT_SOURCE_DIR = "./data/source"   # Original Raw Images
INPUT_TARGET_DIR = "./data/target"   # Professional Edits
OUTPUT_DIR = "./data/aligned_source" # Where to save the fixed images
# ===============================================

def align_images(source_path, target_path, output_path):
    # 1. Load Images
    img_src = cv2.imread(source_path)
    img_tgt = cv2.imread(target_path)

    if img_src is None or img_tgt is None:
        print(f"❌ Error loading: {os.path.basename(source_path)}")
        return

    # 2. Convert to Grayscale for Feature Detection
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)

    # 3. Detect Features (SIFT)
    # SIFT is robust to rotation, scale, and perspective changes
    sift = cv2.SIFT_create()
    keypoints_src, descriptors_src = sift.detectAndCompute(gray_src, None)
    keypoints_tgt, descriptors_tgt = sift.detectAndCompute(gray_tgt, None)

    # 4. Match Features
    # Use FLANN matcher for speed and accuracy
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(descriptors_src, descriptors_tgt, k=2)
    except Exception as e:
        print(f"⚠️ Matching failed for {os.path.basename(source_path)}: {e}")
        return

    # 5. Filter Matches (Lowe's Ratio Test)
    # Only keep "good" matches where the distance is distinct
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Need at least 4 matches to find a Homography
    if len(good_matches) > 10:
        # Extract location of good matches
        src_pts = np.float32([keypoints_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_tgt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 6. Find Homography Matrix
        # RANSAC rejects outliers (bad matches)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 7. Warp Source Image to match Target Geometry
        h, w, c = img_tgt.shape
        aligned_src = cv2.warpPerspective(img_src, M, (w, h))

        # Save result
        cv2.imwrite(output_path, aligned_src)
    else:
        print(f"⚠️ Not enough matches found for {os.path.basename(source_path)} - Skipping alignment.")
        # Fallback: Just copy the original if alignment fails (risky but better than nothing)
        # cv2.imwrite(output_path, img_src)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get list of files
    source_files = sorted(glob.glob(os.path.join(INPUT_SOURCE_DIR, "*")))
    target_files = sorted(glob.glob(os.path.join(INPUT_TARGET_DIR, "*")))

    print(f"🔄 Aligning {len(source_files)} image pairs...")

    for src_path, tgt_path in tqdm(zip(source_files, target_files), total=len(source_files)):
        filename = os.path.basename(src_path)
        out_path = os.path.join(OUTPUT_DIR, filename)
        align_images(src_path, tgt_path, out_path)

    print("✅ Alignment Complete. Use './data/aligned_source' for training.")

if __name__ == "__main__":
    main()