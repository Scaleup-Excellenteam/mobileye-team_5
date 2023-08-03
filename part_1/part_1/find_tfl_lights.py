from typing import List, Optional, Union, Dict, Tuple
import numpy as np
from scipy import signal as sg
from scipy.ndimage import maximum_filter
from PIL import Image
import cv2

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']
POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]

def create_kernel() -> np.array:
    """
    Create a 3x3 kernel that represents a red traffic light.
    """
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1  # Center pixel for the color of traffic light
    kernel[0, 1] = kernel[1, 0] = kernel[1, 2] = kernel[2, 1] = -1  # Surrounding pixels with negative weights
    return kernel


def normalize_kernel(kernel: np.array):
    """
    Normalize the kernel by setting the sum of all the cells to zero.
    :param kernel: The kernel to be normalized.
    """
    kernel_sum = np.sum(kernel)
    kernel_size = kernel.shape[0] * kernel.shape[1]
    neg_value = -(kernel_sum / kernel_size)
    kernel[kernel <= 0] = neg_value


def convolution(image: np.array, kernel: np.array) -> np.array:
    """
    Perform the convolution operation on the image using the given kernel.
    :param image: The input image.
    :param kernel: The kernel for convolution.
    :return: The convolved image.
    """
    return sg.convolve(image.copy(), kernel, mode='same')


def create_red_hsv_mask(image: np.array) -> np.array:
    """
    Create a mask for the red traffic light region in the image in the HSV color space.
    :param image: The input image.
    :return: A binary mask where the red traffic light region is set to 1 and the rest to 0.
    """
    # Define the lower and upper HSV color range for red traffic lights
    lower_red_color = np.array([0, 80, 80])    # Lower boundary for red color in HSV
    upper_red_color = np.array([10, 255, 255])   # Upper boundary for red color in HSV
    # Create the HSV color range mask for red traffic lights
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red_mask = cv2.inRange(hsv_image, lower_red_color, upper_red_color)
    return red_mask


def threshold_image(image: np.array, threshold: int) -> np.array:
    """
    Threshold the image based on the given threshold value.
    :param image: The input image.
    :param threshold: The threshold value.
    :return: The thresholded image.
    """
    thresholded_image = image > threshold
    return thresholded_image


def create_green_hsv_mask(image: np.array) -> np.array:
    """
    Create a mask for the green traffic light region in the image in the HSV color space.
    :param image: The input image.
    :return: A binary mask where the green traffic light region is set to 1 and the rest to 0.
    """
    # Define the lower and upper HSV color range for green traffic lights
    lower_green_color = np.array([50, 110, 110])    # Lower boundary for green color in HSV
    upper_green_color = np.array([150, 255, 255])    # Upper boundary for green color in HSV
    # Create the HSV color range mask for green traffic lights
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv_image, lower_green_color, upper_green_color)
    return green_mask


def extract_tfl_coordinates(image: np.array) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Extract the coordinates of red and green traffic lights from the image using HSV color range.
    :param image: The input image.
    :param red_threshold: The threshold value for red traffic lights.
    :param green_threshold: The threshold value for green traffic lights.
    :return: 4-tuple of x_red, y_red, x_green, y_green.
    """
    # Cut the lower 35% of the image
    height, width, _ = image.shape
    image = image[:int(height * 0.65)]

    red_kernel = create_kernel()
    green_kernel = create_kernel()

    # Normalize the kernels
    normalize_kernel(red_kernel)
    normalize_kernel(green_kernel)

    # Get the red and green channels of the image in the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red_channel_hsv = hsv_image[:, :, 0]
    green_channel_hsv = hsv_image[:, :, 0]

    # Create HSV masks for red and green traffic lights
    red_mask = create_red_hsv_mask(image)
    green_mask = create_green_hsv_mask(image)

    # Apply the HSV masks to the red and green channels
    red_channel_filtered = red_channel_hsv * red_mask
    green_channel_filtered = green_channel_hsv * green_mask

    # Perform convolution for red and green channels
    conv_red = convolution(red_channel_filtered, red_kernel)
    conv_green = convolution(green_channel_filtered, green_kernel)

    # Threshold the convolved images
    red_thresholded = threshold_image(conv_red, 330)
    green_thresholded = threshold_image(conv_green, 180)

    # Apply maximum filter to enhance detection results
    red_filtered = maximum_filter(red_thresholded, size=5)
    green_filtered = maximum_filter(green_thresholded, size=5)

    # Get the coordinates of red and green traffic lights
    red_coordinates = np.argwhere(red_filtered)
    green_coordinates = np.argwhere(green_filtered)

    return red_coordinates[:, 1], red_coordinates[:, 0], green_coordinates[:, 1], green_coordinates[:, 0]