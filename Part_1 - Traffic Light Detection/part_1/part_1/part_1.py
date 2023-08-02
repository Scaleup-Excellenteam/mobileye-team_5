from typing import List, Optional, Union, Dict, Tuple
import json
import networkx as nx
import argparse
from pathlib import Path

import numpy as np
from matplotlib import patches
from scipy import signal as sg
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt

# if you wanna iterate over multiple files and json, the default source folder name is this.
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

DEFAULT_BASE_DIR: str = r"C:\Users\97258\Desktop\cs academy\year 3\Semester B\excelenteam\Check Point\Mobileye-Project\Part_1 - Traffic Light Detection\part_1\part_1"

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


def high_level_conv(image):
    # conv_image = convolve2d(channel_image, high_level_kernel, mode="same")
    high_level_kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                                  [1 / 9, -8 / 9, 1 / 9],
                                  [1 / 9, 1 / 9, 1 / 9]])

    return convolve2d(image, high_level_kernel, mode="same")

def find_tfl_lights(c_image: np.ndarray,
                    **kwargs) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement.

    :param c_image: The image itself as np.uint8, shape of (H, W, 3).
    :param kwargs: Whatever config you want to pass in here.
    :return: 4-tuple of x_red, y_red, x_green, y_green.
    """
    red_part = c_image[:, :, 0]
    green_part = c_image[:, :, 1]


    # Apply high-level kernel for feature detection
    red_conv = high_level_conv(red_part)
    green_part = high_level_conv(green_part)

    # Apply high-level kernel again for feature detection
    red_red_conv = high_level_conv(red_conv)
    green_green_conv = high_level_conv(green_part)

    # Apply max filter to the resulting matrices
    max_filtered_red_red_image = maximum_filter(red_red_conv, size=15)
    max_filtered_red_green_image = maximum_filter(green_green_conv, size=15)

    # Extract Local Maxima area
    red_y, red_x  = np.where(max_filtered_red_red_image > 40)
    green_y, green_x  = np.where(max_filtered_red_green_image > 40)

    return red_x.tolist(), red_y.tolist(),  green_x.tolist(),  green_y.tolist()

# Distance function
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def create_rectangle(group):
    # Find the bounding rectangle
    min_x = min(p[0] for p in group)
    max_x = max(p[0] for p in group)
    min_y = min(p[1] for p in group)
    max_y = max(p[1] for p in group)

    # Find the width and height of the bounding rectangle
    width = max_x - min_x
    height = max_y - min_y

    # Adjust the dimensions according to the given ratio
    adjusted_width = height * 2.5
    adjusted_height = width * 4

    # Calculate the top-left and bottom-right coordinates of the final rectangle
    top_left_x = min_x - (adjusted_width - width) / 2
    top_left_y = min_y - (adjusted_height - height) / 2
    bottom_right_x = max_x + (adjusted_width - width) / 2
    bottom_right_y = max_y + (adjusted_height - height) / 2

    # Ensure the coordinates are within the image bounds
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(1024, bottom_right_x)
    bottom_right_y = min(2048, bottom_right_y)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

# Define a function to create and draw the rectangle
def create_and_draw_rectangle(group, ax):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = create_rectangle(group)
    # Create a rectangle patch
    rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the plot
    ax.add_patch(rect)

def calculate_radius(group):
    if len(group) < 2:
        return 0
    distances = [distance(p1, p2) for i, p1 in enumerate(group) for j, p2 in enumerate(group) if i < j]
    return sum(distances) / len(distances)


def unite_points(x_coords, y_coords, fixed_radius):
    points = [(x, y) for x, y in zip(x_coords, y_coords)]

    groups = []
    grouped_points = set()

    for i, p1 in enumerate(points):
        if i not in grouped_points:
            group = [p1]
            grouped_points.add(i)
            for j, p2 in enumerate(points):
                if j != i and j not in grouped_points and distance(p1, p2) < fixed_radius:
                    group.append(p2)
                    grouped_points.add(j)
            if len(group) > 1:
                groups.append(group)
    return groups


def unite_groups(groups, fixed_radius):
    # Create a graph
    G = nx.Graph()

    # Add a node for each group
    for i in range(len(groups)):
        G.add_node(i)

    # Add an edge between groups if any point in one group is within the fixed_radius of a point in the other group
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j: # Avoid double-checking pairs of groups
                for p1 in group1:
                    for p2 in group2:
                        if distance(p1, p2) < fixed_radius:
                            G.add_edge(i, j)
                            break
                    else:
                        continue
                    break

    # Find connected components
    united_groups = []
    for component in nx.connected_components(G):
        united_group = []
        for index in component:
            united_group.extend(groups[index])
        united_groups.append(united_group)

    return united_groups

def crop_tfl_rect(c_image: np.ndarray, red_x, red_y, green_x, green_y):
    cropped = []
     # Initial fixed radius for grouping points
    fixed_radius = 10  # You can adjust this based on your data

    groups = unite_points(red_x, red_y, fixed_radius)

    groups = unite_groups(groups, fixed_radius)

    fig, ax = plt.subplots()

    for group in groups:
        x = [p[0] for p in group]
        y = [p[1] for p in group]
        plt.scatter(x, y)

        # Create and draw the rectangle for this group using the radius
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = create_rectangle(group)
        rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Cropping the image
        cropped_image = c_image[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
        cropped.append(cropped_image)

    plt.xlim(0, 1024)
    plt.ylim(0, 2048)
    plt.gca().invert_yaxis()
    plt.show()

    return cropped

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


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str]=None, fig_num=None):
    """
    Run the attention code.
    """
    # using pillow to load the image
    image: Image = Image.open(image_path)
    # converting the image to a numpy ndarray array
    c_image: np.ndarray = np.array(image)

    objects = None
    # if image_json_path:
    #     image_json = json.load(Path(image_json_path).open())
    #     objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
    #                                      if image_object['label'] in TFL_LABEL]

    show_image_and_gt(c_image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(c_image)

    guessed_tfl = crop_tfl_rect(c_image, red_x, red_y, green_x, green_y)
    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)


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
