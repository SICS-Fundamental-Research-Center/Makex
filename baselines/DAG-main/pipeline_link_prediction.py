import math
import statistics
from util import *
import argparse
import os
import torch
import torch.nn as nn
import json
import numpy as np
from models import *
from explain_link_prediction import *

import time

def arg_parse():
    parser=argparse.ArgumentParser(description="Arguments for DAG-Explainer.")
    parser.add_argument("--dataset", dest="dataset", type=str, help="Dataset to explain: isAcyclic, MUTAG, highschool")
    parser.add_argument("--GNN", dest="gnn", type=str, help="GNN model to be explain: GCN, GIN, GINE (for highschool dataset only).")
    parser.add_argument("--target-class", dest="target_class", type=int, help="Target class to explain.")

    parser.add_argument("--n_nodes", dest="n_nodes", type=str, help="[For isAcyclic dataset only] Number of nodes in the explanation: 3,4,..., 7; or no requirement: all.")

    parser.add_argument("--num_runs", dest="num_runs", type=int, help="How many runs for the RandomGreedy algorithm")

    parser.add_argument("--lambda_1", dest="lambda_1", type=float, help="Hyperparameter lambda1 for Total support in the objective, see Equation (2).")

    parser.add_argument("--lambda_2", dest="lambda_2", type=float, help="Hyperparameter lambda2 for Average denial. in the objective, see Equation (2).")

    parser.add_argument("--lambda_3", dest="lambda_3", type=float, help="Hyperparameter lambda3 for Size in the objective, see Equation (2).")

    parser.add_argument("--visual", dest="visual", type=bool, help="Whether to visualize the output explanation.")

    parser.add_argument("--save", dest="save", type=bool, help="Whether to save the quantitative evaluation results.")


    # TODO: Check argument usage
    parser.set_defaults(
        dataset='isAcyclic',
        target_class=1,
        gnn='GIN',
        n_nodes='all',
        num_runs=1000,
        lambda_1=0.2132,
        lambda_2=7.3742,
        lambda_3=0,
        visual=False,
        save=True
    )

    return parser.parse_args()

def main():

    # Load a configuration
    prog_args = arg_parse()

    dataset = prog_args.dataset
    target_class = prog_args.target_class
    modelname = prog_args.gnn
    print(dataset)

    lambdas = np.array([1, prog_args.lambda_1, prog_args.lambda_2, prog_args.lambda_3])

    explainer = DAG(dataset=dataset, modelname=modelname, lambdas=lambdas)
    print("candidate {}".format(explainer.candidate))

    # 
    n_candidate = len([x for x in explainer.candidate if x[1]==target_class])
    print(n_candidate)
    if n_candidate==0:
        raise ValueError("Empty candidate pool.")
    k = math.ceil((n_candidate+1)/2)
    print("k {}".format(k))

    print('Explaining '+dataset+', '+prog_args.gnn+' model.')

    #### for repeating test

    repeat = int(prog_args.num_runs)
    start = time.time()
    _, output = explainer.repeatExplain(k, repeat, target_class=target_class, save=prog_args.save)
    end = time.time()

    # print('final output')
    # print(output)
    print('\nEvaluation on final explanation set:')
    explainer.evalOutput(output, read_out=True)

    # for e in output:
    #     print(explainer.evalIndividualExp(e))

    # save tested time used
    if prog_args.save:
        with open(explainer.save_path + '/class_' + str(target_class) + '_mulirun_quantitatives.txt', 'a') as f:
            f.write(f"avg time used: {(end - start)/repeat}")

    # if prog_args.visual:
    #     visualResult(output=output, gSpanOutput=explainer.gSpan_output_file, if_isAcyclic=if_isAcyclic, if_highschool= if_highschool, save_path=explainer.save_path)
    print('Average time used for each run:')
    print((end - start) / repeat)

if __name__ == '__main__':
    main()