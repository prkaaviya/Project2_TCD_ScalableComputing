"""
Group 36 - preprocess.py

Command to use:
python3 classify.py --model-names try3_wc1,try3_wc2,try3_wc3,try3_wc4,try3_wc5,try3_wc6,try3_wc7 --captcha-dir captchas/ --output test_ans3.csv --symbols symbols.txt
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import argparse
import tensorflow as tf


def decode(characters, y):
    y = np.array(y)
    if len(y.shape) == 3:
        # expected shape (batch_size, sequence_length, num_classes)
        y = np.argmax(y, axis=2)[:, 0]
    elif len(y.shape) == 2:
        # shape (batch_size, num_classes) for single character prediction
        y = np.argmax(y, axis=1)
    else:
        raise ValueError(f"Unexpected prediction shape: {y.shape}")
    
    return ''.join([characters[x] for x in y])

def load_model(model_name, captcha_length, captcha_symbols, input_shape):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)

    # load corresponding weights for the given captcha length
    model.load_weights(f"{model_name}.keras")
    model.compile(loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-3, amsgrad=False),
        metrics=['accuracy'])

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-names', help='Comma-separated model names to use for classification (e.g., wc1,wc2,...wc6)', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_names is None:
        print("Please specify the CNN models to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    model_names = args.model_names.split(',')

    models = {}

    with tf.device('/gpu:0'):
        with open(args.output, 'w') as output_file:
            # load all models for different lengths
            for index, model_name in enumerate(model_names):
                models[index + 1] = load_model(model_name, index + 1, captcha_symbols, (96, 192, 3))

            # classify each captcha
            for file_name in os.listdir(args.captcha_dir):
                # prepare the image
                image_path = os.path.join(args.captcha_dir, file_name)
                raw_data = cv2.imread(image_path)
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = np.array(rgb_data) / 255.0
                image = cv2.resize(image, (192, 96))  # Adjust to model's expected size
                image = np.expand_dims(image, axis=0)  # Shape becomes (1, 96, 192, 3)

                # Try predicting with different models and choose the best one
                best_prediction = None
                best_confidence = -1
                best_decoded = ""

                for length, model in models.items():
                    prediction = model.predict(image)
                    decoded_text = decode(captcha_symbols, prediction)
                    confidence = np.max(prediction)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_prediction = prediction
                        best_decoded = decoded_text

                output_file.write(f"{file_name}, {best_decoded}\n")
                print(f'Classified {file_name} as {best_decoded} with confidence {best_confidence}')

if __name__ == '__main__':
    main()
