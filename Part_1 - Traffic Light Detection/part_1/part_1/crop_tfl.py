from collections import deque

import numpy as np


# Distance function
from matplotlib import patches, pyplot as plt


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


def is_connected(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (abs(x1 - x2) <= 2) and (abs(y1 - y2) <= 2)



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

def crop_tfl_rect(c_image: np.ndarray, red_x, red_y, green_x, green_y):
    cropped = []

    groups = unite_points(red_x, red_y)

    for group in groups:
        x = [p[0] for p in group]
        y = [p[1] for p in group]
        plt.scatter(x, y)


        # Create and draw the rectangle for this group using the radius
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = create_rectangle(group)
        rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the plot
        plt.gca().add_patch(rect)
        print(rect)
        # Create and draw the rectangle for this group using the radius
        index_bottom_right_y, index_top_left_x, index_top_left_y, index_bottom_right_x = create_rectangle(group)

        # Cropping the image
        cropped_image = c_image[int(index_top_left_x):int(index_bottom_right_x),int(index_top_left_y):int(index_bottom_right_y)]

        cropped.append(cropped_image)



    return cropped


import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_images_side_by_side(images):
#     # Determine the total number of images
#     n_images = len(images)
#
#     # Determine the maximum height and width
#     max_height = max(img.shape[0] for img in images)
#     max_width = max(img.shape[1] for img in images)
#     channels = images[0].shape[2]
#
#     # Create an empty array to hold all the images side by side, taking channels into account
#     combined_image = np.zeros((max_height, max_width * n_images, channels), dtype=images[0].dtype)
#
#     # Place each image next to the previous one, handling different sizes
#     for i, img in enumerate(images):
#         height, width, _ = img.shape
#         combined_image[:height, i * max_width:(i * max_width) + width, :] = img
#
#     plt.imshow(combined_image.astype(images[0].dtype))
#     plt.axis([0, 500, 500, 0])  # [xmin, xmax, ymax, ymin]
#
#     plt.show()
#
#
#
# def plot_images_sequentially(images):
#     for idx, img in enumerate(images):
#         print(f"Image {idx}:")
#         print(img) # This prints the pixel values of the image
#
#         if img.size > 0: # Check if the image is non-empty
#             plt.imshow(img)
#             plt.axis([0, 500, 500, 0])  # [xmin, xmax, ymax, ymin]
#             plt.show()
#         else:
#             print("Empty or zero-size image, skipping.")

