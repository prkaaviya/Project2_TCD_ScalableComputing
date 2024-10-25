#!/usr/bin/env python3
# Group 36

import os
import numpy
import random
import string
import cv2
import argparse
import captcha.image

class Args:
    width = 192
    height = 96
    length = 4
    count = 12800
    output_dir = "./training"
    symbols = "123456789aBCdeFghjkMnoPQRsTUVwxYZ+%|#][{}\\-"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='', type=int)
    parser.add_argument('--output', help='', type=str)
    args_p = parser.parse_args()
    
    args = Args()
    args.length = args_p.length
    
    if args_p.output != None:
        args.output_dir = args_p.output
    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=["fonts/WildCrazy.ttf"])

    captcha_symbols = args.symbols

    print("Generating captchas with symbol set:" + captcha_symbols)
    
    output_dir = os.path.join(args.output_dir, "wildcrazy", str(args.length))
    
    if not os.path.exists(output_dir):
        print("Creating output directory " + output_dir)
        os.makedirs(output_dir)
        
    args.output_dir = output_dir

    for i in range(args.count):
        random_str = ''.join([random.choice(captcha_symbols) for _ in range(args.length)])
        i_path = random_str.ljust(args.length, "=")
        image_path = os.path.join(args.output_dir, i_path+'.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')

        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)

if __name__ == '__main__':
    main()
