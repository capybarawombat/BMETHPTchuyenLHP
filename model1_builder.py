import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage

# nhớ sửa cái directory nha
DATASET_ROOT = "/home/nh/Downloads/dataset/TRUSTED_dataset_for_nsd"

# sửa cái directory luôn
PROJECT_ROOT = os.getcwd()
MODEL1_OUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/model1_output")

def build_detailed_kidney_model(ct_path, mask_path, output_dir):
    print(f"--- Building Detailed Model from: {os.path.basename(ct_path)} ---")
    
    # 1. Load Data
    try:
        ct = nib.load(ct_path)
        mask = nib.load(mask_path)
    except Exception as e:
        print(f"[ERROR] Could not load files: {e}")
        return None
    
    ct_data = ct.get_fdata()
    mask_data = mask.get_fdata()
    affine = ct.affine

    # 2. Apply Mask (Keep only kidney texture)
    detailed_kidney = ct_data.copy()
    detailed_kidney[mask_data == 0] = -1000 # Air

    # 3. Crop Volume
    r, c, d = np.where(mask_data > 0)
    if len(r) == 0:
        print("[ERROR] Mask is empty.")
        return None
        
    pad = 15
    min_r, max_r = max(0, np.min(r)-pad), min(ct_data.shape[0], np.max(r)+pad)
    min_c, max_c = max(0, np.min(c)-pad), min(ct_data.shape[1], np.max(c)+pad)
    min_d, max_d = max(0, np.min(d)-pad), min(ct_data.shape[2], np.max(d)+pad)

    cropped_volume = detailed_kidney[min_r:max_r, min_c:max_c, min_d:max_d]
    
    # 4. Save
    new_affine = affine.copy()
    new_affine[:3, 3] += np.dot(affine[:3, :3], [min_r, min_c, min_d])
    
    output_filename = os.path.join(output_dir, "detailed_3d_kidney.nii.gz")
    new_img = nib.Nifti1Image(cropped_volume, new_affine)
    nib.save(new_img, output_filename)
    
    print(f"[SUCCESS] Saved detailed model to: {output_filename}")
    return output_filename

if __name__ == "__main__":
    #chỗ này sửa directory input
    CT_FILE = os.path.join(DATASET_ROOT, "CT_DATA/CT_images/200_imgCT.nii.gz")
    MASK_FILE = os.path.join(DATASET_ROOT, "CT_DATA/CT_masks/GT_estimated_masksCT/200_maskCT.nii.gz")
    
    os.makedirs(MODEL1_OUT_DIR, exist_ok=True)
    
    if os.path.exists(CT_FILE) and os.path.exists(MASK_FILE):
        build_detailed_kidney_model(CT_FILE, MASK_FILE, MODEL1_OUT_DIR)
    else:
        print(f"[ERROR] Files not found in {DATASET_ROOT}")