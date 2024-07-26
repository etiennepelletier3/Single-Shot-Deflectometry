import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase


def phases(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Normalize the image intensity
    # img = img.astype(float)
    # img /= np.max(img)

    # Compute first derivatives using cv2.Sobel
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # First derivative w.r.t x
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # First derivative w.r.t y

    # Compute second derivatives using cv2.Sobel
    Ixx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)  # Second derivative w.r.t x
    Iyy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)  # Second derivative w.r.t y

    # Calculate the diagonal pattern wrapped phase
    num = Ix + Iy
    normalized_num = num / np.sqrt(np.sum(num**2))
    den = Ixx + Iyy
    normalized_den = den / np.sqrt(np.sum(den**2))
    phase = np.arctan2(normalized_num, normalized_den)
    

    # Perform spatial phase unwrapping
    unwrapped_phase = unwrap_phase(phase)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(phase, cmap='jet')
    plt.colorbar()
    plt.title('Phase')

    plt.subplot(1, 2, 2)
    plt.imshow(unwrapped_phase, cmap='jet')
    plt.colorbar()
    plt.title('Unwrapped Phase')

    plt.show()


    # Separate the x and y phases
    Ix = cv2.Sobel(unwrapped_phase, cv2.CV_64F, 1, 0, ksize=3)  # First derivative w.r.t x
    Iy = cv2.Sobel(unwrapped_phase, cv2.CV_64F, 0, 1, ksize=3)  # First derivative w.r.t y
    phi_x = np.cumsum(Ix, axis=1)
    phi_y = np.cumsum(Iy, axis=0)
    

    # Plot the separate phases
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(phi_x, cmap='jet')
    plt.colorbar()
    plt.title('X Phase')

    plt.subplot(1, 2, 2)
    plt.imshow(phi_y, cmap='jet')
    plt.colorbar()
    plt.title('Y Phase')

    plt.show()

    return phi_x, phi_y


# # Display the results
# plt.figure(figsize=(12, 8))

# plt.subplot(2, 3, 1)
# plt.title('Original Image')
# plt.imshow(img, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 2)
# plt.title('First Derivative Ix')
# plt.imshow(Ix, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 3)
# plt.title('First Derivative Iy')
# plt.imshow(Iy, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 5)
# plt.title('Second Derivative Ixx')
# plt.imshow(Ixx, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 6)
# plt.title('Second Derivative Iyy')
# plt.imshow(Iyy, cmap='gray')
# plt.axis('off')

# plt.tight_layout()
# plt.show()