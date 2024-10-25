#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import random
import argparse
import tensorflow as tf

# Build a Keras model given some parameters
def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
    input_tensor = tf.keras.Input(input_shape)
    x = input_tensor
    for i, module_length in enumerate([module_size] * model_depth):
        for j in range(module_length):
            x = tf.keras.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.BatchNormalization()(x)   

    x = tf.keras.layers.Flatten()(x)
    x = [tf.keras.layers.Dense(captcha_num_symbols, activation='softmax', name=f'char_{i+1}')(x) for i in range(captcha_length)]
    model = tf.keras.Model(inputs=input_tensor, outputs=x)

    return model

class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(np.ceil(self.count / self.batch_size))

    def __getitem__(self, idx):
        if len(self.files) == 0:
            print("No more files left, resetting files.")
            filex = os.listdir(self.directory_name)
            random.shuffle(filex)
            self.files = dict(zip(map(lambda x: x.split('.')[0], filex), filex))

        actual_batch_size = min(self.batch_size, len(self.files))
        X = np.zeros((actual_batch_size, self.captcha_height, self.captcha_width, 3), dtype=np.float32)
        y = [np.zeros((actual_batch_size, len(self.captcha_symbols)), dtype=np.uint8) for _ in range(self.captcha_length)]

        for i in range(actual_batch_size):
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files.pop(random_image_label)
            
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = np.array(rgb_data) / 255.0
            X[i] = processed_data

            random_image_label = random_image_label.split('_')[0]

            for j, ch in enumerate(random_image_label):
                y[j][i, self.captcha_symbols.find(ch)] = 1

        # Convert y to a tuple to satisfy tf.TypeSpec requirements
        return X, tuple(y)
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--batch-size', help='How many images in training captcha batches', type=int)
    parser.add_argument('--train-dataset', help='Path to the training image dataset', type=str)
    parser.add_argument('--validate-dataset', help='Path to the validation image dataset', type=str)
    parser.add_argument('--output-model-name', help='Name for the trained model', type=str)
    parser.add_argument('--input-model', help='Path for loading an existing model to continue training', type=str)
    parser.add_argument('--epochs', help='Number of training epochs', type=int)
    parser.add_argument('--symbols', help='File with symbols for captchas', type=str)
    args = parser.parse_args()

    required_args = ['width', 'height', 'length', 'batch_size', 'train_dataset', 'validate_dataset', 'output_model_name', 'epochs', 'symbols']
    for arg in required_args:
        if getattr(args, arg) is None:
            print(f"Error: Please specify --{arg.replace('_', '-')}")
            exit(1)

    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found. Falling back to CPU.")

    model = create_model(args.length, len(captcha_symbols), (args.height, args.width, 3))

    if args.input_model:
        model.load_weights(args.input_model)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=False),
        metrics={f'char_{i + 1}': 'accuracy' for i in range(args.length)}
    )

    model.summary()

    training_data = ImageSequence(args.train_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height)
    validation_data = ImageSequence(args.validate_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output_model_name + '.keras', save_best_only=True, monitor='val_loss', mode='min')
    ]

    with open(args.output_model_name + ".json", "w") as json_file:
        json_file.write(model.to_json())

    try:
        model.fit(
            training_data,
            validation_data=validation_data,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt caught, saving current weights as {args.output_model_name}_resume.keras')
        model.save_weights(args.output_model_name + '_resume.keras')

if __name__ == '__main__':
    main()
