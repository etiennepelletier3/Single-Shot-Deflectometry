import os
import numpy as np
import cv2
import json
import yaml

# ------------------------------
# ENTER YOUR PARAMETERS HERE:
SENSOR = 'EO-3112C'
LENS = '50mm-C-SERIES'
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # dictionary ID
SQUARES_VERTICALLY = 7              # number of squares vertically
SQUARES_HORIZONTALLY = 5            # number of squares horizontally
SQUARE_LENGTH = 0.03                # square side length (m)
MARKER_LENGTH = 0.020               # ArUco marker side length (m)
LENGTH_PX = 1080                    # size of the board in pixels
MARGIN_PX = 50                      # size of the margin in pixels
CALIB_DIR_PATH = 'Data/Test 6/Camera Calibration/' # path to the folder with images
JSON_PATH = CALIB_DIR_PATH + 'calibration.json' # path to the JSON file with calibration parameters
DICTIONARY_PATH = 'ARTag_02_size16_imd14_ref.yml' # path to the dictionary file
# ------------------------------

save_intrinsics = False

def create_ChArUco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)# getPredefinedDictionary(ARUCO_DICT)
    print(dictionary)
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

def get_intrinsic_parameters():
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Load PNG images from folder
    image_files = [os.path.join(CALIB_DIR_PATH, f) for f in os.listdir(CALIB_DIR_PATH) if f.endswith(".png")]
    image_files.sort()  # Ensure files are in order

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        image_copy = image.copy()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        
        # If at least one marker is detected
        if len(marker_ids) > 0:
            # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            # cv2.imshow('Detected Markers', image_copy)
            # cv2.waitKey(0)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

    # Calibrate camera
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None)

    cv2.destroyAllWindows()
    return camera_matrix, dist_coeffs

if save_intrinsics:
    camera_matrix, dist_coeffs = get_intrinsic_parameters()
    data = {"sensor": SENSOR, "lens": LENS, "camera_matrix": camera_matrix.tolist(), "dist_coeffs": dist_coeffs.tolist()}

    with open(JSON_PATH, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f'Data has been saved to {JSON_PATH}')

# Load calibration parameters from JSON file
with open(JSON_PATH, 'r') as file: # Read the JSON file
    json_data = json.load(file)

camera_matrix = np.array(json_data['camera_matrix']) # Load the camera matrix
dist_coeffs = np.array(json_data['dist_coeffs']) # Load the distortion coefficients

def detect_pose(image, camera_matrix, dist_coeffs):
    rvec, tvec = None, None
    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Detect markers in the undistorted image
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=params)

    # If at least one marker is detected
    if len(marker_ids) > 0:
        # Interpolate CharUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, undistorted_image, board)

        # If enough corners are found, estimate the pose
        if charuco_retval:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)

            # If pose estimation is successful, draw the axis and save the rvec and tvec
            if retval:
                cv2.drawFrameAxes(undistorted_image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=10)
                rvec, tvec = rvec, tvec
    return undistorted_image, rvec, tvec


def check_pose_detection():
    # Iterate through PNG images in the folder
    image_files = [os.path.join(CALIB_DIR_PATH, f) for f in os.listdir(CALIB_DIR_PATH) if f.endswith(".png")]
    image_files.sort()  # Ensure files are in order

    for image_file in image_files:
        # Load an image
        image = cv2.imread(image_file)

        # Detect pose and draw axis
        pose_image, rvec, tvec = detect_pose(image, camera_matrix, dist_coeffs)

        # Show the image
        resized_image = cv2.resize(pose_image, (1000, 800))  # Set the desired size
        cv2.imshow('Pose Image', resized_image)
        cv2.waitKey(0)
    

def get_rmtx_and_tvec():
    image_files = [os.path.join(CALIB_DIR_PATH, f) for f in os.listdir(CALIB_DIR_PATH) if f.startswith("pose")]
    
    for image_file in image_files:
        img = cv2.imread(image_file)
        pose_img, rvec, tvec = detect_pose(img, camera_matrix, dist_coeffs)
        rmtx = cv2.Rodrigues(rvec)[0]
        # Show the image
        resized_image = cv2.resize(pose_img, (800, 600))  # Set the desired size
        cv2.imshow('Pose Image', resized_image)
        cv2.waitKey(0)
        print(f"rmtx: {rmtx}")
        print(f"tvec: {tvec}")
        # return rmtx, tvec

get_rmtx_and_tvec()


