#!/usr/bin/env python3
# Group 36

import os
import cv2
import argparse
import numpy as np


def apply_gaussian_blur(img, kernel_size=(5, 5)):
    return cv2.GaussianBlur(img, kernel_size, 0)

def apply_adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2)

def apply_morphological_transforms(img, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Removes small noise
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # Closes small holes
    return img

def dilate_and_erode(img, iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=iterations)
    img = cv2.erode(img, kernel, iterations=iterations)
    return img

def to_bw(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1]
    img = apply_gaussian_blur(img)
    img = apply_morphological_transforms(img)
    img = dilate_and_erode(img)

    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='INPUT DIR', type=str)
    parser.add_argument('--output', help='OUTPUT_DIR', type=str)
    args = parser.parse_args()
    
    input_dir = ""
    output_dir = ""
    
    if args.input is None:
        print("Please specify input dir")
        exit(1)
        
    if args.output is None:
        print("Please specify output dir")
        exit(1)
        
    input_dir = args.input
    output_dir = args.output
    
    if not os.path.exists(output_dir):
        print("Creating output directory " + output_dir)
        os.makedirs(output_dir)
       

    for f in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, f))
        img = to_bw(img)
        cv2.imwrite(os.path.join(output_dir, f), img)

if __name__ == "__main__":
    main()
