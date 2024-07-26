import numpy as np
import cv2

# Define the aruco dictionary and charuco board
ARUCO_DICT = cv2.aruco.DICT_4X4_50
SQUARES_VERTICALLY = 7              # number of squares vertically
SQUARES_HORIZONTALLY = 5            # number of squares horizontally
SQUARE_LENGTH = 0.03                # square side length (m)
MARKER_LENGTH = 0.02               # ArUco marker side length (m)
LENGTH_PX = 1080                    # size of the board in pixels
MARGIN_PX = 50                      # size of the margin in pixels
CALIB_DIR_PATH = 'Data/Test 6/Camera Calibration/' # path to the folder with images

def create_ChArUco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)# getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    return img

while True:
    img = create_ChArUco_board()
    img_mirrored = cv2.flip(img, 1)
    cv2.imshow("ChArUco Mirrored", img_mirrored)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(CALIB_DIR_PATH + 'board.png', img_mirrored)