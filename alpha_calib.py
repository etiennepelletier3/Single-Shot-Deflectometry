import numpy as np
import json

with open('parameters.json') as f:
        params = json.load(f)

def alphas(zm, beta_x, beta_y, xd, yd, phi_a_x, phi_a_y, phi_x_offset, phi_y_offset):
    xb = xd + 2*beta_x*zm
    yb = yd + 2*beta_y*zm

    phi_b_x = (xb * 2 * np.pi * params["N_fringe_x"]) / (params["s_px_size"] * params["s_size_X"]) - phi_x_offset
    phi_b_y = (yb * 2 * np.pi * params["N_fringe_y"]) / (params["s_px_size"] * params["s_size_Y"]) - phi_y_offset

    delta_x_ab = (params["s_px_size"] * (phi_a_x - phi_b_x) * params["s_size_X"]) / (2 * np.pi * params["N_fringe_x"])
    delta_y_ab = (params["s_px_size"] * (phi_a_y - phi_b_y) * params["s_size_Y"]) / (2 * np.pi * params["N_fringe_y"])

    alpha_x = delta_x_ab / (zm - params["z_m2s"]) + beta_x
    alpha_y = delta_y_ab / (zm - params["z_m2s"]) + beta_y

    return alpha_x, alpha_y



