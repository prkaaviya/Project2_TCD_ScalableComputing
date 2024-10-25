# Project2_TCD_ScalableComputing

This repository contains the source code, model definitions, and data generation/processing scripts for Project 2 in the Scalable Computing course at Trinity College Dublin. This project focuses on designing a scalable CAPTCHA generation and recognition system, leveraging machine learning and efficient data handling.

## Project Structure

1. training/: Contains images for training dataset.
2. validation/: Contains images for validation dataset.
3. fonts/: Contains the list of fonts given in the test set.
4. preprocessed/: Contains training + validation images preprocessed using OpenCV library.

## Key Scripts

1. fetch.py: This script is used to fetch test dataset to use during classification.
2. generate.py: This script generates CAPTCHA images for training and validation, supporting customizable text length, symbols, and fonts.
3. preprocess.py: This script takes an input of image dataset and applies certain image preprocessing functions to remove noise in the dataset.
4. retrain.py: The main script for training and retraining the CAPTCHA recognition model. It supports GPU/CPU processing and includes argument options for image dimensions, symbol length, batch size, dataset paths, and epochs.

## Installation
### Prerequisites
```
1. Python 3.9 or later
2. TensorFlow
3. OpenCV
```
### Running the Scripts
#### Data Generation

Generate training or validation data:
```
bash
python3 generate.py --width 192 --height 96 --length 6 --symbols symbols.txt --count 100000 --output-dir training/wildcard/6 --font-dir fonts/
python3 generate.py --width 192 --height 96 --length 6 --symbols symbols.txt --count 20000 --output-dir validation/wildcard/6 --font-dir fonts/
```

#### Model Training

To train or retrain the model, run:
```
bash
python3 retrain.py --width 128 --height 64 --length 6 --symbols symbols.txt --batch-size 32 --epochs 10 --output-model model_name --train-dataset training/wildcard/6 --validate-dataset validation/wildcard/6
```

### Repository Features

1. Customizable Training: Parameters for image size, batch size, and training epochs can be specified through the command line.
2. Checkpointing and Early Stopping: The model saves the best weights and implements early stopping to optimize training time.
