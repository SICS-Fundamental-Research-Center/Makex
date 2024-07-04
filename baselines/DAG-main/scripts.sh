#!/bin/sh

python3 pipeline_link_prediction.py --dataset yelp --GNN hgt --target-class 1 --n_nodes 10 --num_runs 1000  --lambda_1 0.5 --lambda_2 0.5 --lambda_3 0 --visual False --save True

# for other datasets, or models, you can change the input values of corresponding parameters