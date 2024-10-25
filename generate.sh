#!/bin/bash
# Group 36

font=wildcrazy

for i in $(seq 1 6);
do
    echo $i
    python3 generate.py --length $i --output "training"
    python3 generate.py --length $i --output "validation"
done
