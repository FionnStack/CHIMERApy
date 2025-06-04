from sunpy.net import Fido, attrs as a
import astropy.units as u
import os
from sunpy.map import Map
import numpy as np
import sys
import chimerapy.chimera as chimera_new
import chimerapy.chimera_original as chimera_old

"""
date = "2016/11/04"
wavelength = 193 * u.Angstrom

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "downloaded_data")
os.makedirs(save_path, exist_ok=True)

results = Fido.search(
    a.Time(f"{date} 00:00:00", f"{date} 00:00:30", near=True),
    a.Instrument("AIA"),
    a.Wavelength(wavelength)
)

if results:
        downloaded_files = Fido.fetch(results[0], path=save_path)
        if downloaded_files:
            print(f"Downloaded file: {downloaded_files[0]}")
        else:
            print("Download failed.")
else:
    print("No files found for the specified date and wavelength.")
"""

def imhmi(m171):
    """Create a dummy HMI map for the original CHIMERA code."""
    imhmi = Map(np.zeros_like(m171.data), m171.meta)
    return imhmi

m171 = Map("C:\\Users\\aoife\\CHIMERApy\\chimerapy\\scripts\\downloaded_data\\aia.lev1.171A_2016_10_31T00_00_10.35Z.image_lev1.fits")
m193 = Map("C:\\Users\\aoife\\CHIMERApy\\chimerapy\\scripts\\downloaded_data\\aia.lev1.193A_2016_10_31T00_00_05.85Z.image_lev1.fits")
m211 = Map("C:\\Users\\aoife\\CHIMERApy\\chimerapy\\scripts\\downloaded_data\\aia.lev1.211A_2016_10_31T00_00_10.62Z.image_lev1.fits")
imhmi_map = imhmi(m171)

labeled_mask, ch_mask, coronal_holes = chimera_new.chimera(m171, m193, m211)

if coronal_holes:
    print("These are the new CHIMERA results")
else:
    print("No coronal holes found for new CHIMERA.")

circ, data, datb, datc, dattoarc, hedb, iarr, props, rs, slate, center, xgrid, ygrid = chimera_old.chimera(m171, m193, m211, imhmi_map)

print("Chimera_original processing complete.")

num_coronal_holes = 0
for i in range(1, props.shape[1]):
    if props[0, i].isdigit():
        num_coronal_holes += 1
print(f"Chimera_original found {num_coronal_holes} coronal holes.")

for i in range(1, num_coronal_holes + 1):
    print(f"Coronal Hole {i}:")
    print(f"  ID: {props[0, i]}")
    print(f"  XCEN: {props[1, i]}")
    print(f"  YCEN: {props[2, i]}")
    print(f"  CENTROID: {props[3, i]}")
    print(f"  X_EB: {props[4, i]}")
    print(f"  Y_EB: {props[5, i]}")
    print(f"  X_WB: {props[6, i]}")
    print(f"  Y_WB: {props[7, i]}")
    print(f"  X_NB: {props[8, i]}")
    print(f"  Y_NB: {props[9, i]}")
    print(f"  X_SB: {props[10, i]}")
    print(f"  Y_SB: {props[11, i]}")
    print(f"  WIDTH: {props[12, i]}")
    print(f"  WIDTH: {props[13, i]}")
    print(f"  AREA: {props[14, i]}")
    print(f"  AREA%: {props[15, i]}")
    print(f"  <B>: {props[16, i]}")
    print(f"  <B+>: {props[17, i]}")
    print(f"  <B->: {props[18, i]}")
    print(f"  BMAX: {props[19, i]}")
    print(f"  BMIN: {props[20, i]}")
    print(f"  TOT_B+: {props[21, i]}")
    print(f"  TOT_B-: {props[22, i]}")
    print(f"  <PHI>: {props[23, i]}")
    print(f"  <PHI+>: {props[24, i]}")
    print(f"  <PHI->: {props[25, i]}")