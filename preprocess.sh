#! /bin/bash
# Group 36

font=wildcrazy
# data=validation

for i in $(seq 1 6);
do
    echo $i
    python3 preprocess.py --input "training/$font/$i" --output "preprocessed/training/$font/$i"
    python3 preprocess.py --input "validation/$font/$i" --output "preprocessed/validation/$font/$i"
done
