from typing import List, Tuple
import numpy as np
from scipy import ndimage
from PIL import Image
import cv2


def calculate_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return 0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity


def extract_tfl_coordinates(image: np.array) -> Tuple[List[int], List[int], List[int], List[int]]:
    # Step 1: Cut off the lower part of the image
    height, width, _ = image.shape
    image = image[:int(height * 0.65)]

    # Step 2: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 3: Apply white top hat morphology
    kernel = np.ones((30, 30), np.uint8)
    top_hat_image = cv2.morphologyEx(grayscale_image, cv2.MORPH_TOPHAT, kernel)

    # Step 4: Select the bright points as markers
    threshold_value = 150  # You can adjust this threshold value
    _, markers = cv2.threshold(top_hat_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 5: Apply a region growing algorithm (watershed)
    markers = ndimage.label(markers)[0]
    labels = cv2.watershed(image, markers)

    # Step 6: Select the bright points that are not a part of a larger object
    red_x, red_y, green_x, green_y = [], [], [], []
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for label in np.unique(labels):
        if label == 0:  # Skip background label
            continue
        mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
        mask[labels == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea(contours[0])
        if contour_area < 2500:  # Adjust this threshold for small object filtering
            if np.any(mask * grayscale_image):  # Check if the mask intersects with the grayscale image
                for contour in contours:
                    circularity = calculate_circularity(contour)
                    if circularity > 0.7:  # Adjust this threshold for circularity filtering
                        y, x = np.where(mask > 0)
                        if len(x) > 0 and len(y) > 0:
                            hue_values = hsv_image[y, x, 0]
                            if np.mean(hue_values) < 40:  # Threshold for red light filtering
                                red_x.extend(x.tolist())
                                red_y.extend(y.tolist())
                            elif np.mean(hue_values) > 50:  # Threshold for green light filtering
                                green_x.extend(x.tolist())
                                green_y.extend(y.tolist())

    return red_x, red_y, green_x, green_y

