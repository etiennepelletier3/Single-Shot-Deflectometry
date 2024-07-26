import numpy as np
import cv2
import os
import json




def capture(dw, dh, sw, sh, N_fringe_x, N_fringe_y, calibrate, output_path):
    # Create the output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initialize the background array
    background_calib = np.zeros((dh, dw, 1), dtype=np.uint8)
    pattern = background_calib.copy()
    # Insert a white square the same size as the pattern into the background for calibration
    white_square = np.ones((sh, sw, 1), dtype=np.uint8) * 255
    background_calib[dh//2 - sh//2:dh//2 + sh//2, dw//2 - sw//2:dw//2 + sw//2] = white_square

    # Get fringe frequencies
    fx = N_fringe_x / sw
    fy = N_fringe_y / sh
    # Initialize the fringe pattern
    fringe_pattern = np.zeros((sh, sw, 1), dtype=np.uint8)

    # Generate the fringe pattern
    for j in range(sh):
        for i in range(sw):
            fringe_pattern[j, i] = (255/2) * (1 + np.cos(2*np.pi * (fx*i + fy*j)))
    fringe_pattern = fringe_pattern.astype(np.uint8)

    pattern[dh//2 - sh//2:dh//2 + sh//2, dw//2 - sw//2:dw//2 + sw//2] = fringe_pattern
    
    


    # Calibrate scene
    # camera = cv2.VideoCapture(0)
    while calibrate:
        # cv2.namedWindow('Fringe Pattern', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('Fringe Pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Calibration', background_calib)
        # ret, frame = camera.read()
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Camera', gray_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # cv2.imwrite(f'{output_path}/Calibration.png', background_calib)
            print("Scene calibrated")
            cv2.destroyAllWindows()
            calibrate = False
        elif key == ord('q'):
            quit()


    # Capture Diagonal Fringe Pattern
    On = True
    while On:
        # cv2.namedWindow('Fringe Pattern', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('Fringe Pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Fringe Pattern', fringe_pattern)
        # ret, frame = camera.read()
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Camera', gray_frame)
        key = cv2.waitKey(1) & 0xFF  # Wait for any key to be pressed
        if key == ord('c'): 
            # cv2.imwrite(f'{output_path}/Pattern.png', pattern)
            print("Pattern captured")
            On = False
    cv2.destroyAllWindows()
    # camera.release()
