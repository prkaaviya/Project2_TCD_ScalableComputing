"""

Group 36 - preprocess.py

Command to use:

python3 preprocess.py --input "training/$font/$i" --output "preprocessed/training/$font/$i"
python3 preprocess.py --input "validation/$font/$i" --output "preprocessed/validation/$font/$i"
"""

import os
import sys
import random
import argparse
import numpy as np

import cv2
import matplotlib.pyplot as plt


def load_random_images(directories, num_images):
    """
    Args:
        directories (list): _description_
        num_images (int): _description_

    Returns:
        loaded_images: _description_
    """

    all_image_paths = []

    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    all_image_paths.append(os.path.join(root, file))

    if num_images > len(all_image_paths):
        print(f"Requested {num_images} images, but only found {len(all_image_paths)} images.")
        num_images = len(all_image_paths)

    selected_image_paths = random.sample(all_image_paths, num_images)

    loaded_images = []
    for image_path in selected_image_paths:
        image = cv2.imread(image_path)
        if image is not None:
            loaded_images.append((image_path, image))
        else:
            print(f"Warning: Failed to load image {image_path}")

    return loaded_images


def plot_images(images, grid_shape=(2, 5)):
    """
    Args:
        images (list): _description_
        grid_shape (tuple): _description_

    Returns:
        None
    """

    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(15, 6))
    fig.suptitle('Images to Preprocess', fontsize=16)

    for ax, (img_path, img) in zip(axes.flat, images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')  # Convert BGR to RGB
        ax.set_title(os.path.basename(img_path))
        ax.axis('off')

    for ax in axes.flat[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def remove_noise(orig_img):
    """

    Args:
        orig_img (_type_): _description_

    Returns:
        img: _description_
    """
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img = cv2.medianBlur(img, 5)
    img = remove_circles(img)
    img = remove_small_contours(img, min_area=40)
    img = cv2.bitwise_not(img)

    return img


def remove_circles(img):
    """

    Args:
        img (_type_): _description_

    Returns:
        img: _description_
    """
    hough_circle_locations = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1,
                                minDist=1, param1=50, param2=5, minRadius=0, maxRadius=3)
    if hough_circle_locations is not None:
        circles = np.uint16(np.around(hough_circle_locations[0]))
        mask = np.ones_like(img, dtype=np.uint8) * 255
        for circle in circles:
            x, y, r = circle
            cv2.circle(mask, (x, y), r, 0, -1)
        img = cv2.bitwise_and(img, mask)
    return img


def remove_small_contours(img, min_area=40):
    """

    Args:
        img (_type_): _description_
        min_area (int, optional): _description_. Defaults to 40.

    Returns:
        img: _description_
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(img, [cnt], -1, 0, -1)  # Remove small contours by filling with black
    return img


def main():
    """
    Entrypoint function.
    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='INPUT DIR', type=str)
    parser.add_argument('--output', help='OUTPUT_DIR', type=str)
    args = parser.parse_args()

    input_dir = ""
    output_dir = ""

    if args.input is None:
        print("Please specify input dir")
        sys.exit(1)

    if args.output is None:
        print("Please specify output dir")
        sys.exit(1)

    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        print("Creating output directory " + output_dir)
        os.makedirs(output_dir)


    for f in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, f))
        img = remove_noise(img)
        cv2.imwrite(os.path.join(output_dir, f), img)

if __name__ == "__main__":
    main()
