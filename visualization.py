import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- CONFIGURATION ---
FILE_PATH = "data/processed/model1_output/detailed_3d_kidney.nii.gz"

def radiologist_view():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print(f"Loading Model: {FILE_PATH}...")
    img = nib.load(FILE_PATH)
    data = img.get_fdata()
    
    # Rotate to stand upright
    data = np.rot90(data, k=1, axes=(0, 1))

    # Center slices
    d1, d2, d3 = data.shape
    idx1, idx2, idx3 = d1 // 2, d2 // 2, d3 // 2

    # Setup Dark Mode (Looks like professional software)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.15)
    plt.suptitle(f"Model 1 Inspection: {os.path.basename(FILE_PATH)}", fontsize=16, color='white')

    # --- VIEW 1: AXIAL (Top-Down) ---
    ax[0, 0].set_title("Axial (Top-Down)", color='lime')
    im1 = ax[0, 0].imshow(data[:, :, idx3], cmap="gray", vmin=-100, vmax=300)
    ax[0, 0].axis('off')
    v1 = ax[0, 0].axvline(idx2, color='red', alpha=0.5)
    h1 = ax[0, 0].axhline(idx1, color='cyan', alpha=0.5)

    # --- VIEW 2: CORONAL (Front) ---
    ax[0, 1].set_title("Coronal (Front)", color='red')
    im2 = ax[0, 1].imshow(data[:, idx2, :], cmap="gray", vmin=-100, vmax=300, aspect='auto')
    ax[0, 1].axis('off')
    v2 = ax[0, 1].axvline(idx3, color='lime', alpha=0.5)
    h2 = ax[0, 1].axhline(idx1, color='cyan', alpha=0.5)

    # --- VIEW 3: SAGITTAL (Side) ---
    ax[1, 0].set_title("Sagittal (Side)", color='cyan')
    im3 = ax[1, 0].imshow(data[idx1, :, :], cmap="gray", vmin=-100, vmax=300, aspect='auto')
    ax[1, 0].axis('off')
    v3 = ax[1, 0].axvline(idx3, color='lime', alpha=0.5)
    h3 = ax[1, 0].axhline(idx2, color='red', alpha=0.5)

    # --- INFO PANEL ---
    ax[1, 1].axis('off')
    ax[1, 1].text(0.1, 0.7, "INSPECTION CHECKLIST:", fontsize=14, weight='bold', color='yellow')
    ax[1, 1].text(0.1, 0.5, "1. Scroll the sliders below.", fontsize=12)
    ax[1, 1].text(0.1, 0.4, "2. Do you see gray/white texture?", fontsize=12)
    ax[1, 1].text(0.1, 0.3, "3. Is the background black?", fontsize=12)
    ax[1, 1].text(0.1, 0.1, "If YES: Model 1 is READY.", fontsize=14, color='lime')

    # --- SLIDERS ---
    ax_s1 = plt.axes([0.15, 0.08, 0.65, 0.03])
    ax_s2 = plt.axes([0.15, 0.05, 0.65, 0.03])
    ax_s3 = plt.axes([0.15, 0.02, 0.65, 0.03])

    s1 = Slider(ax_s1, 'Sagittal', 0, d1-1, valinit=idx1, valstep=1, color='cyan')
    s2 = Slider(ax_s2, 'Coronal',  0, d2-1, valinit=idx2, valstep=1, color='red')
    s3 = Slider(ax_s3, 'Axial',    0, d3-1, valinit=idx3, valstep=1, color='lime')

    def update(val):
        i1, i2, i3 = int(s1.val), int(s2.val), int(s3.val)
        im1.set_data(data[:, :, i3])
        im2.set_data(data[:, i2, :])
        im3.set_data(data[i1, :, :])
        
        # Update Crosshairs
        v1.set_xdata([i2]); h1.set_ydata([i1])
        v2.set_xdata([i3]); h2.set_ydata([i1])
        v3.set_xdata([i3]); h3.set_ydata([i2])
        fig.canvas.draw_idle()

    s1.on_changed(update)
    s2.on_changed(update)
    s3.on_changed(update)

    print("Opening Radiologist View...")
    plt.show()

if __name__ == "__main__":
    radiologist_view()
