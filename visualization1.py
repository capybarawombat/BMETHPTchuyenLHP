import os
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from skimage import measure

# --- CONFIGURATION ---
FILE_PATH = "data/processed/model1_output/detailed_3d_kidney.nii.gz"
OUTPUT_HTML = "kidney_3d_viewer.html"

def generate_interactive_html():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print(f"Loading Model: {FILE_PATH}...")
    img = nib.load(FILE_PATH)
    data = img.get_fdata()

    # 1. Generate the Smooth Surface (Isosurface)
    # We look for the boundary where density > 50 (Kidney Tissue)
    print("Constructing 3D Mesh (Marching Cubes)... This takes a moment.")
    try:
        # "level" is the threshold. 50 captures the kidney, ignoring air (-1000).
        verts, faces, normals, values = measure.marching_cubes(data, level=50)
    except Exception as e:
        print(f"Error generating mesh: {e}")
        return

    # 2. Simplify for Speed (Optional but recommended)
    # Plotly handles about 100k triangles well. If it's huge, we could decimate.
    # For now, we just pass the data.
    
    print(f"Mesh generated: {len(verts)} vertices, {len(faces)} faces.")
    print("Building Interactive Plot...")

    # 3. Create the 3D Object
    # We swap x, y, z to make it stand upright
    x, y, z = verts.T
    i, j, k = faces.T

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='brown',      # Color of the kidney
            opacity=1.0,        # Solid
            name='Kidney',
            showscale=False,
            lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.1, specular=0.4),
            lightposition=dict(x=100, y=200, z=0)
        )
    ])

    # 4. Add UI Controls
    fig.update_layout(
        title="Interactive Kidney Model (Generated from CT)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data' # Keep real proportions
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="black"
    )

    # 5. Save to HTML
    print(f"Saving to {OUTPUT_HTML}...")
    fig.write_html(OUTPUT_HTML)
    print("Done! Open 'kidney_3d_viewer.html' in your browser.")

if __name__ == "__main__":
    generate_interactive_html()