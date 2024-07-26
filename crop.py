from skimage.feature import corner_harris, corner_subpix, corner_peaks
import cv2
import matplotlib.pyplot as plt



def crop(calib_img_path, fringe_img_path, output_img_path):
    # Load the calibration image
    calib_img = cv2.imread(calib_img_path, cv2.IMREAD_GRAYSCALE)
    # Load the fringe image
    fringe_img = cv2.imread(fringe_img_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to binary using a threshold
    _, binary = cv2.threshold(calib_img, 200, 255, cv2.THRESH_BINARY)

    # Crop the pattern image to the white square
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Crop the image to the bounding rectangle
        cropped_img = fringe_img[y:y+h, x:x+w]
    else:
        cropped_img = fringe_img

    # Save the cropped image
    cv2.imwrite(output_img_path, cropped_img)

    # Display the cropped image
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(fringe_img, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    axes[1].imshow(cropped_img, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Cropped Image')
    plt.show()