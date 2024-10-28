#! /bin/bash
# Group 36

font=wildcrazy

for i in $(seq 1 7);
do
    echo "Training character set with length = $i"
    python3 retrain.py --width 192 --height 96 --length $i --batch-size 64 --train-dataset ./preprocessed/training/$font/$i/ --validate-dataset ./preprocessed/validation/$font/$i/ --output-model-name try3_wc$i.keras --epochs 10 --symbols symbols.txt
done
