from collections import deque
import numpy as np
from matplotlib import patches, pyplot as plt

'''
Constants:
    - WIDTH_MULTIPLIER (float): Adjusts the width of the bounding rectangle relative to the width of the points group.
    - HEIGHT_MULTIPLIER (float): Adjusts the height of the bounding rectangle relative to the height of the points group.
    - VERTICAL_ADJUSTMENT_GREEN (float): Adjusts the vertical position of the bounding rectangle for green traffic lights.
    - VERTICAL_ADJUSTMENT_RED (float): Adjusts the vertical position of the bounding rectangle for red traffic lights.
'''
# Constants for adjusting the dimensions of the bounding rectangle
WIDTH_MULTIPLIER = 2.5
HEIGHT_MULTIPLIER = 4
VERTICAL_ADJUSTMENT_GREEN = 2 / 3
VERTICAL_ADJUSTMENT_RED = 2 / 3

# Define the threshold for considering points as connected
CONNECTED_THRESHOLD = 5


# Distance function
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def create_rectangle_bound(group, color):
    """
    Create bounding rectangle coordinates for a given group of points.

    This function calculates the bounding rectangle coordinates (top-left and bottom-right) around a group of points.
    The calculated rectangle is adjusted based on the color to fine-tune the position of the rectangle.

    Parameters:
        group (list): A list of tuples representing the points to form the bounding rectangle around.
        color (str): The color of the rectangle. It can be either 'red' or 'green' and is used to adjust the rectangle's
                     vertical position.

    Returns:
        tuple: A tuple containing four elements - the top-left x coordinate, top-left y coordinate, bottom-right x
               coordinate, and bottom-right y coordinate of the bounding rectangle.

    Note:
        - The `group` parameter should be a list of tuples, where each tuple contains the x and y coordinates of a point.
        - The `color` parameter can be either 'red' or 'green', depending on the traffic light color the rectangle
          corresponds to. It is used to adjust the vertical position of the rectangle to better fit the traffic light.

    """
    # Find the bounding rectangle
    min_x = min(p[0] for p in group)
    max_x = max(p[0] for p in group)
    min_y = min(p[1] for p in group)
    max_y = max(p[1] for p in group)

    # Find the width and height of the bounding rectangle
    width = max_x - min_x
    height = max_y - min_y

    # Adjust the dimensions according to the given ratio
    adjusted_width = width * WIDTH_MULTIPLIER
    adjusted_height = height * HEIGHT_MULTIPLIER

    # Calculate the top-left and bottom-right coordinates of the final rectangle
    top_left_x = min_x - (adjusted_width - width) / 2
    top_left_y = min_y - (adjusted_height - height) / 2
    bottom_right_x = max_x + (adjusted_width - width) / 2
    bottom_right_y = max_y + (adjusted_height - height) / 2

    # Adjust starting point based on color
    if color == 'green':
        top_left_y = max(0, top_left_y - height * VERTICAL_ADJUSTMENT_GREEN)
        bottom_right_y = max(0, bottom_right_y - height * VERTICAL_ADJUSTMENT_GREEN)
    elif color == 'red':
        top_left_y = min(1024, top_left_y + height * VERTICAL_ADJUSTMENT_RED)
        bottom_right_y = min(1024, bottom_right_y + height * VERTICAL_ADJUSTMENT_RED)

    # Make sure that the values are also clamped from below:
    top_left_y = max(0, top_left_y)
    bottom_right_y = max(0, bottom_right_y)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def calculate_radius(group):
    """
    Calculate the radius of a group of points.

    This function computes the average distance between all pairs of points in the group to estimate the radius. It
    provides an approximate measure of how spread out the points are from their center.

    Parameters:
        group (list): A list of tuples representing the points for which the radius needs to be calculated.

    Returns:
        float: The estimated radius of the group of points. If the group has less than 2 points, the radius is considered
               as 0.

    Note:
        - The `group` parameter should be a list of tuples, where each tuple contains the x and y coordinates of a point.
        - The radius is computed as the average distance between all possible pairs of points in the group.
    """
    if len(group) < 2:
        return 0

    distances = [distance(p1, p2) for i, p1 in enumerate(group) for j, p2 in enumerate(group) if i < j]
    return sum(distances) / len(distances)


def is_connected(p1, p2):
    """
    Check if two points are connected.

    This function checks if two points are connected, i.e., if they are close to each other within a certain threshold. It
    is used to determine if two points should be considered as part of the same group.

    Parameters:
        p1 (tuple): A tuple representing the first point with x and y coordinates.
        p2 (tuple): A tuple representing the second point with x and y coordinates.

    Returns:
        bool: True if the two points are connected (within a certain threshold), False otherwise.

    Note:
        - The `p1` and `p2` parameters should be tuples with two elements representing the x and y coordinates of the
          respective points.
    """
    x1, y1 = p1
    x2, y2 = p2

    return (abs(x1 - x2) <= CONNECTED_THRESHOLD) and (abs(y1 - y2) <= CONNECTED_THRESHOLD)


def unite_points(x_coords, y_coords):
    """
    Group connected points based on their x and y coordinates.

    This function takes two lists, `x_coords` and `y_coords`, representing the x and y coordinates of points, respectively.
    It groups connected points together and returns a list of groups, where each group is a list of connected points.

    Parameters:
        x_coords (list): A list of integers or floats representing the x coordinates of points.
        y_coords (list): A list of integers or floats representing the y coordinates of points.

    Returns:
        list: A list of groups, where each group is a list of connected points. Each point in the group is represented
              as a tuple (x, y), where x is the x coordinate and y is the y coordinate.

    Example:
        x_coords = [1, 2, 3, 5, 6]
        y_coords = [1, 2, 3, 5, 6]
        result = unite_points(x_coords, y_coords)
        # Result will be: [[(1, 1), (2, 2), (3, 3)], [(5, 5), (6, 6)]]

    Note:
        - The function uses a breadth-first search (BFS) algorithm to find and group connected points efficiently.
        - The `is_connected` function must be defined and provided separately to determine if two points are connected.
        - The function will only consider points connected in a grid-like manner (horizontal and vertical connections).
          If you need to consider diagonal connections as well, you must modify the `is_connected` function accordingly.
    """
    points = [(x, y) for x, y in zip(x_coords, y_coords)]

    groups = []
    grouped_points = set()

    for i, p1 in enumerate(points):
        if i not in grouped_points:
            group = [p1]
            grouped_points.add(i)

            # Use a deque to represent the queue of points to be processed
            queue = deque([i])

            while queue:
                current_point_idx = queue.popleft()
                current_point = points[current_point_idx]

                for j, p2 in enumerate(points):
                    if j != current_point_idx and j not in grouped_points and is_connected(current_point, p2):
                        group.append(p2)
                        grouped_points.add(j)
                        queue.append(j)  # Enqueue the point to be processed later

            if len(group) > 1:
                groups.append(group)

    return groups


def crop_tfl_rect(c_image: np.ndarray, x, y, color):
    """
    Crop suspected Traffic Light (TFL) regions from the given image based on grouped points.

    This function takes an image represented as a NumPy array `c_image`, along with the x and y coordinates of points
    representing a set of grouped points that are suspected to form rectangles around traffic lights. It creates a
    rectangle around each group of points and crops the corresponding regions from the image. The cropped regions are
    collected and returned as a list, where each cropped image potentially contains a traffic light.

    Parameters:
        c_image (np.ndarray): The input image as a NumPy array (e.g., RGB image represented as shape [height, width, 3]).
        x (list): A list of integers representing the x coordinates of the grouped points forming rectangles.
        y (list): A list of integers representing the y coordinates of the grouped points forming rectangles.
        color (str): The color of the rectangle to be drawn. It can be any valid color string supported by Matplotlib.

    Returns:
        list: A list of NumPy arrays, where each array represents a cropped region of the input image corresponding to
              a suspected traffic light. Each cropped region is a NumPy array with shape [height, width, 3].

    Note:
        - The function assumes that the points are already grouped using the `unite_points` function, which must be called
          before using this function to form rectangles around potential traffic lights.
        - The `create_rectangle_bound` function must be defined separately to create a rectangle bound around the group
          of points. The rectangle is specified by its top-left (x, y) coordinates and its bottom-right (x, y) coordinates.
        - The function uses Matplotlib to visualize the points and draw the rectangles on the image (optional). If
          visualization is not required, consider removing the Matplotlib-related code from this function.
        - Make sure that the `color` parameter is a valid Matplotlib color string. For example, 'r' for red, 'g' for green,
          'b' for blue, 'c' for cyan, 'm' for magenta, 'y' for yellow, 'k' for black, 'w' for white, etc.
        - After cropping the images, further processing can be applied to identify and classify the traffic lights within
          each cropped region, depending on the specific use case or application.
    """
    cropped = []

    # grouped point that suspect as TFL for bounding
    groups = unite_points(x, y)

    for group in groups:
        x = [p[0] for p in group]
        y = [p[1] for p in group]
        plt.scatter(x, y)
        plt.scatter(x, y)

        # Create the rectangle for this group
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = create_rectangle_bound(group, color)
        rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the plot
        plt.gca().add_patch(rect)

        # Cropping the image by rectangle bound
        cropped_image = c_image[int(top_left_y):int(bottom_right_y):, int(top_left_x):int(bottom_right_x):]

        cropped.append(cropped_image)
    # display_images(c_image, cropped)
    # display_seq_images(c_image, cropped)
    return cropped


def display_images(original_image, cropped_images):
    # Calculate the number of subplots needed (original image + cropped images)
    n_subplots = len(cropped_images) + 1

    # Create subplots with 1 row and n_subplots columns
    fig, axes = plt.subplots(1, n_subplots, figsize=(100, 100))

    # Plot the original image in the first subplot
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].set_xticks([])  # Turn off x-axis tick labels for original image
    axes[0].set_yticks([])  # Turn off y-axis tick labels for original image

    # Display each cropped image in its own subplot
    for i, cropped_image in enumerate(cropped_images):
        axes[i + 1].imshow(cropped_image)
        axes[i + 1].set_title(f'Cropped {i + 1}')
        axes[i + 1].set_xticks([])  # Turn off x-axis tick labels
        axes[i + 1].set_yticks([])  # Turn off y-axis tick labels

    # Set window size
    plt.gcf().set_size_inches(1000 / plt.gcf().dpi, 1000 / plt.gcf().dpi)

    plt.show()

def display_seq_images(original_image, cropped_images):
    # Display the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.xticks([])  # Turn off x-axis tick labels
    plt.yticks([])  # Turn off y-axis tick labels
    plt.show()

    # Display each cropped image in its own figure
    for i, cropped_image in enumerate(cropped_images):
        plt.figure(figsize=(10, 10))
        plt.imshow(cropped_image)
        plt.title(f'Cropped {i + 1}')
        plt.xticks([])  # Turn off x-axis tick labels
        plt.yticks([])  # Turn off y-axis tick labels
        plt.show()

