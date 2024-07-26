import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import pattern_capture
import get_phases
import beta_calib
import alpha_calib
import get_slopes
import crop
import os

# Load initial parametersq
params = {
    "x_cam": 500, # mm
    "y_cam": 80, # mm
    "z_c2s": 160, # mm
    "z_m2s": 337, # mm
    "s_px_size": 0.179, # pixel pitch in mm of Dell screen
    "s_size_X": 1080,
    "s_size_Y": 1080,
    "display_width": 1920,
    "display_height": 1080,
    "N_fringe_x": 100,
    "N_fringe_y": 100,
    "crop": False
}

# Create the different folders paths
Test_path = 'Data/Test 6/'
REF_path = Test_path + "/REF/"
REF_calib_path = REF_path + "/Calibration.png"
REF_pattern_path = REF_path + "/Pattern.png"
REF_CrpPattern_path = REF_path + "/CrpPattern.png"
SUT_path = Test_path + "/SUT/"
SUT_calib_path = SUT_path + "/Calibration.png"
SUT_pattern_path = SUT_path + "/Pattern.png"
SUT_CrpPattern_path = SUT_path + "/CrpPattern.png"
Results_path = Test_path + "/Results/"

if not os.path.exists(Test_path):
    os.makedirs(Test_path)
if not os.path.exists(REF_path):
    os.makedirs(REF_path)
if not os.path.exists(SUT_path):
    os.makedirs(SUT_path)
if not os.path.exists(Results_path):
    os.makedirs(Results_path)

# Impossible de ne pas avoir le bon pattern si je ne fais pas une autre fonction avant de faire le pattern capture
phi1, phi2 = get_phases.phases('Data/Test 1/REF/Pattern.png')


# Capture the reference pattern
pattern_capture.capture(params["display_width"], params["display_height"], params['s_size_X'], params['s_size_Y'], params['N_fringe_x'], params['N_fringe_y'], False, REF_path)

# Crop the reference pattern
if params["crop"]:
    crop.crop(REF_calib_path, REF_pattern_path, REF_CrpPattern_path)

# Enter the scene geometry parameters220
params["x_cam"] = float(input("Enter x_cam: ")) #(params["s_size_X"]/2) * params["s_px_size"]
params["y_cam"] = float(input("Enter y_cam: ")) #(params["display_height"]/2 - params["s_size_Y"]/2) * params["s_px_size"] + 7.5
params["z_c2s"] = float(input("Enter z_c2s: "))
params["z_m2s"] = float(input("Enter z_m2s: "))


# Get the reference phases
phi_ref_x, phi_ref_y = get_phases.phases(REF_pattern_path)

# Beta calibration
phi_x_offset = np.ones_like(phi_ref_x)*100
phi_y_offset = np.ones_like(phi_ref_y)*100
xd = (params["s_px_size"] * params["s_size_X"] * (phi_ref_x + phi_x_offset)) / (2 * np.pi * params["N_fringe_x"])
yd = (params["s_px_size"] * params["s_size_Y"] * (phi_ref_y + phi_y_offset)) / (2 * np.pi * params["N_fringe_y"])

beta_x, beta_y = beta_calib.betas(params["x_cam"], params["y_cam"], xd, yd, params["z_c2s"], params["z_m2s"])


with open('parameters.json', 'w') as f:
    json.dump(params, f)

# Capture the SUT pattern
pattern_capture.capture(1920, 1080, params['s_size_X'], params['s_size_Y'], params['N_fringe_x'], params['N_fringe_y'], False, SUT_path)

# Crop the SUT pattern
if params["crop"]:
    crop.crop(SUT_calib_path, SUT_pattern_path, SUT_CrpPattern_path)

# Get the SUT phases
phi_a_x, phi_a_y = get_phases.phases(SUT_pattern_path)

# Alpha calibration
zm = np.zeros_like(phi_a_x)
for i in range(3):
    alpha_x, alpha_y = alpha_calib.alphas(zm, beta_x, beta_y, xd, yd, phi_a_x, phi_a_y, phi_x_offset, phi_y_offset)

    # Get the slopes
    x_slope, y_slope = get_slopes.get_slopes(alpha_x, alpha_y, beta_x, beta_y)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(x_slope, cmap='jet')
    plt.colorbar()
    plt.title('x_slope')

    plt.subplot(1, 2, 2)
    plt.imshow(y_slope, cmap='jet')
    plt.colorbar()
    plt.title('y_slope')

    plt.show()

    # Get surface
    surface = np.cumsum(x_slope, axis=1) + np.cumsum(y_slope, axis=0)
    plt.imshow(surface, cmap='jet')
    plt.colorbar(label='mm')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(Results_path + f"surface_iter_{i}.png")
    plt.show()
    zm = surface