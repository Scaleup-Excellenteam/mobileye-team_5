from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import signal as sg
from scipy.ndimage import maximum_filter, label
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = 'INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE'

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(c_image: np.ndarray, objects: Optional[List[POLYGON_OBJECT]], fig_num: int = None):
    # ensure a fresh canvas for plotting the image and objects.
    plt.figure(fig_num).clf()
    # displays the input image.
    plt.imshow(c_image)
    labels = set()
    if objects:
        for image_object in objects:
            # Extract the 'polygon' array from the image object
            poly: np.array = np.array(image_object['polygon'])
            # Use advanced indexing to create a closed polygon array
            # The modulo operation ensures that the array is indexed circularly, closing the polygon
            polygon_array = poly[np.arange(len(poly)) % len(poly)]
            # gets the x coordinates (first column -> 0) anf y coordinates (second column -> 1)
            x_coordinates, y_coordinates = polygon_array[:, 0], polygon_array[:, 1]
            color = 'r'
            plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            # The legend provides a visual representation of the labels associated with the plotted objects.
            # It helps in distinguishing different objects in the plot based on their labels.
            plt.legend()



def create_kernel() -> np.array:
    """
    Create a 3x3 kernel that represents a red traffic light.
    """
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1  # Center pixel for red traffic light
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


def threshold_image(image: np.array, threshold: int) -> np.array:
    """
    Threshold the image based on the given threshold value.
    :param image: The input image.
    :param threshold: The threshold value.
    :return: The thresholded image.
    """
    thresholded_image = image > threshold
    return thresholded_image


def create_red_hsv_mask(image: np.array) -> np.array:
    """
    Create a mask for the red traffic light region in the image in the HSV color space.
    :param image: The input image.
    :return: A binary mask where the red traffic light region is set to 1 and the rest to 0.
    """
    # Define the lower and upper HSV color range for red traffic lights
    lower_red_color = np.array([0, 100, 100])    # Lower boundary for red color in HSV
    upper_red_color = np.array([10, 255, 255])   # Upper boundary for red color in HSV
    # Create the HSV color range mask for red traffic lights
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red_mask = cv2.inRange(hsv_image, lower_red_color, upper_red_color)
    return red_mask

def create_green_hsv_mask(image: np.array) -> np.array:
    """
    Create a mask for the green traffic light region in the image in the HSV color space.
    :param image: The input image.
    :return: A binary mask where the green traffic light region is set to 1 and the rest to 0.
    """
    # Define the lower and upper HSV color range for green traffic lights
    lower_green_color = np.array([45, 110, 110])    # Lower boundary for green color in HSV
    upper_green_color = np.array([150, 255, 255])    # Upper boundary for green color in HSV
    # Create the HSV color range mask for green traffic lights
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv_image, lower_green_color, upper_green_color)
    return green_mask

def extract_tfl_coordinates(image: np.array, red_threshold: int, green_threshold: int) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
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
    red_thresholded = threshold_image(conv_red, red_threshold)
    green_thresholded = threshold_image(conv_green, green_threshold)

    # Apply maximum filter to enhance detection results
    red_filtered = maximum_filter(red_thresholded, size=3)
    green_filtered = maximum_filter(green_thresholded, size=3)

    # Get the coordinates of red and green traffic lights
    red_coordinates = np.argwhere(red_filtered)
    green_coordinates = np.argwhere(green_filtered)

    return red_coordinates[:, 0], red_coordinates[:, 1], green_coordinates[:, 0], green_coordinates[:, 1]



def test_find_tfl_lights(image_path: str, image_json_path: Optional[str]=None, fig_num=None):
    """
    Run the traffic light detection code and plot the results.
    """
    # using pillow to load the image
    image: Image = Image.open(image_path)
    # converting the image to a numpy ndarray array
    c_image: np.ndarray = np.array(image)

    objects = None
    if image_json_path:
        image_json = json.load(Path(image_json_path).open())
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]

    show_image_and_gt(c_image, objects, fig_num)

    # Set the threshold values for red and green traffic lights
    red_threshold = 330
    green_threshold = 150

    red_x, red_y, green_x, green_y = extract_tfl_coordinates(c_image, red_threshold, green_threshold)
    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_y, red_x, 'ro', markersize=4)
    plt.plot(green_y, green_x, 'go', markersize=4)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results.
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to image json file -> GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    # If you entered a custom dir to run from or the default dir exist in your project then:
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        # gets a list of all the files in the directory that ends with "_leftImg8bit.png".
        file_list: List[Path] = list(directory_path.glob('*_leftImg8bit.png'))

        for image in file_list:
            # Convert the Path object to a string using as_posix() method
            image_path: str = image.as_posix()
            path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            image_json_path: Optional[str] = path if Path(path).exists() else None
            test_find_tfl_lights(image_path, image_json_path)

    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == '__main__':
    main()
