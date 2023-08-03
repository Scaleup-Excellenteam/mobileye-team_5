from typing import List, Tuple
import numpy as np
from scipy import ndimage
from PIL import Image
import cv2

# reference for the algorithm used below :
# Blog: https://medium.com/@kenan.r.alkiek/https-medium-com-kenan-r-alkiek-traffic-light-recognition-505d6ab913b1
# GitHub: https://github.com/KenanA95/tl-detector


def calculate_circularity(contour):
    """
    Calculate the circularity of a contour.

    Parameters:
        contour (numpy.ndarray): Contour points as a NumPy array.

    Returns:
        float: The circularity value.
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return 0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity


def apply_white_top_hat(image: np.array) -> np.array:
    """
    Apply white top hat morphology on the input image.

    Parameters:
        image (numpy.ndarray): The input RGB image as a NumPy array.

    Returns:
        numpy.ndarray: The result of white top hat operation.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((30, 30), np.uint8)
    top_hat_image = cv2.morphologyEx(grayscale_image, cv2.MORPH_TOPHAT, kernel)
    return top_hat_image


def apply_watershed(image: np.array, markers: np.array) -> np.array:
    """
    Apply a region growing algorithm (watershed) on the input image.

    Parameters:
        image (numpy.ndarray): The input RGB image as a NumPy array.
        markers (numpy.ndarray): The markers as a binary NumPy array.

    Returns:
        numpy.ndarray: The result of watershed algorithm.
    """
    markers = ndimage.label(markers)[0]
    labels = cv2.watershed(image, markers)
    return labels


def calculate_median(coord_list):
    """
    Calculate the median of coordinates in a cluster.

    Parameters:
        coord_list (list): A list of coordinate tuples.

    Returns:
        tuple: The median x and y coordinates.
    """
    x_coords, y_coords = zip(*coord_list)
    median_x, median_y = int(np.median(x_coords)), int(np.median(y_coords))
    return median_x, median_y


def find_clusters(coord_list, threshold):
    """
    Find clusters of coordinates that are close to each other.

    Parameters:
        coord_list (list): A list of coordinate tuples.
        threshold (int): The minimum number of points required to form a cluster.

    Returns:
        list: A list of clusters, each containing coordinate tuples.
    """
    clusters = []
    coord_list = set(coord_list)

    while coord_list:
        seed = coord_list.pop()
        cluster = {seed}
        stack = [seed]

        while stack:
            x, y = stack.pop()
            neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            for neighbor in neighbors:
                if neighbor in coord_list:
                    stack.append(neighbor)
                    coord_list.remove(neighbor)
                    cluster.add(neighbor)

        if len(cluster) > threshold:
            clusters.append(cluster)

    return clusters


def extract_tfl_coordinates(image: np.array) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Extract red and green traffic light coordinates from the input image.

    Parameters:
        image (numpy.ndarray): The input RGB image as a NumPy array.

    Returns:
        tuple: A tuple containing lists of red_x, red_y, green_x, and green_y coordinates.
    """
    # Step 1: Cut off the lower part of the image
    height, width, _ = image.shape
    image = image[:int(height * 0.65)]

    # Step 2: Apply white top hat morphology
    top_hat_image = apply_white_top_hat(image)

    # Step 3: Select the bright points as markers
    threshold_value = 150  # You can adjust this threshold value
    _, markers = cv2.threshold(top_hat_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 4: Apply a region growing algorithm (watershed)
    labels = apply_watershed(image, markers)

    # Step 5: Select the bright points that are not a part of a larger object
    red_x, red_y, green_x, green_y = [], [], [], []
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for label in np.unique(labels):
        if label == 0:  # Skip background label
            continue
        mask = np.zeros(top_hat_image.shape, dtype=np.uint8)
        mask[labels == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea(contours[0])
        if contour_area < 2500:
            if np.any(mask * top_hat_image):  # Check if the mask intersects with the top hat image
                for contour in contours:
                    circularity = calculate_circularity(contour)
                    if circularity > 0.7:
                        y, x = np.where(mask > 0)
                        if len(x) > 0 and len(y) > 0:
                            hue_values = hsv_image[y, x, 0]
                            if np.mean(hue_values) < 40:  # Threshold for red light filtering
                                red_x.extend(x.tolist())
                                red_y.extend(y.tolist())
                            elif np.mean(hue_values) > 50:  # Threshold for green light filtering
                                green_x.extend(x.tolist())
                                green_y.extend(y.tolist())

    # Step 6: Calculate median of circular shapes for cluster groups
    red_coords = list(zip(red_x, red_y))
    green_coords = list(zip(green_x, green_y))

    red_clusters = find_clusters(red_coords, threshold=5)  # Adjust threshold as needed
    green_clusters = find_clusters(green_coords, threshold=5)  # Adjust threshold as needed

    red_x, red_y = [], []
    for cluster in red_clusters:
        median_x, median_y = calculate_median(cluster)
        red_x.append(median_x)
        red_y.append(median_y)

    green_x, green_y = [], []
    for cluster in green_clusters:
        median_x, median_y = calculate_median(cluster)
        green_x.append(median_x)
        green_y.append(median_y)

    return red_x, red_y, green_x, green_y


