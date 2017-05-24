import cv2
import numpy as np

image = cv2.imread("images/source/your_images/white.jpg", cv2.IMREAD_GRAYSCALE)
mask = np.zeros_like(image)
mask[image.shape[0]/2:, :] = 255
cv2.imwrite("images/source/your_images/mask.jpg", mask)
