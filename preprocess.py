#!/usr/bin/env python3
# Group 36

import os
import cv2
import argparse

def to_bw(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1]
    
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
