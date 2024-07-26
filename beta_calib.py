import numpy as np
import json

with open('parameters.json') as f:
    params = json.load(f)

def betas(xc, yc, xd, yd, z_c2s, z_m2s):
    beta_x = (xc - xd) / (2*z_m2s + z_c2s)
    beta_y = (yc - yd) / (2*z_m2s + z_c2s)
    return beta_x, beta_y



