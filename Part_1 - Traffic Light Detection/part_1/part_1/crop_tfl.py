from collections import deque

import numpy as np


# Distance function
from matplotlib import patches, pyplot as plt


# Distance function

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def create_rectangle(group, color):
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

    # Adjust starting point based on color
    if color == 'green':
        top_left_y = max(0, top_left_y - height * 2 / 3)
        bottom_right_y = max(0, bottom_right_y - height * 2 / 3)

    elif color == 'red':
        top_left_y = min(1024, top_left_y + height * 2 / 3)
        bottom_right_y = min(1024, bottom_right_y + height * 2 / 3)

    # make sure that the values are also clamped from below:
    top_left_y = max(0, top_left_y)
    bottom_right_y = max(0, bottom_right_y)

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


def is_connected(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (abs(x1 - x2) <= 5) and (abs(y1 - y2) <= 5)



def unite_points(x_coords, y_coords):
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
    cropped = []

    groups = unite_points(x, y)
    # plt.imshow(c_image)

    for group in groups:
        x = [p[0] for p in group]
        y = [p[1] for p in group]
        plt.scatter(x, y)


        # Create and draw the rectangle for this group using the radius
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = create_rectangle(group, color)
        rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the plot
        plt.gca().add_patch(rect)

        # Cropping the image
        # cropped_image = c_image[int(index_top_left_x):int(index_bottom_right_x):,int(index_top_left_y):int(index_bottom_right_y):]
        cropped_image = c_image[int(top_left_y):int(bottom_right_y):, int(top_left_x):int(bottom_right_x):]

        cropped.append(cropped_image)
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

