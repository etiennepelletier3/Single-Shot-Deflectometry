import numpy as np
import json


def get_slopes(alpha_x, alpha_y, beta_x, beta_y):
    numx = alpha_x * (alpha_x**2 + alpha_y**2 + 1)**(-1/2) - beta_x * (beta_x**2 + beta_y**2 + 1)**(-1/2)
    denx = (alpha_x**2 + alpha_y**2 + 1)**(-1/2) + (beta_x**2 + beta_y**2 + 1)**(-1/2)

    x_slope = numx / denx

    numy = alpha_y * (alpha_x**2 + alpha_y**2 + 1)**(-1/2) - beta_y * (beta_x**2 + beta_y**2 + 1)**(-1/2)
    deny = (alpha_x**2 + alpha_y**2 + 1)**(-1/2) + (beta_x**2 + beta_y**2 + 1)**(-1/2)

    y_slope = numy / deny
    
    return x_slope, y_slope

