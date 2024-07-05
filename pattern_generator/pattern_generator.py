import argparse
import pandas as pd
from structure.structure import DataGraph
from Makex import Makex
from Makex import read_subgraph, SubgraphToPattern, Score_Q, random_walk_degree_length, REP_make_hashable
from Makex import REP_make_hashable
from structure.structure import REP
from structure.structure import Pattern


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Makex", usage="run.py [<args>] [-h | --help]")

    parser.add_argument("--cuda", action="store_true", help="Use GPU.")

    parser.add_argument(
        "--subgraph_path",
        type=str,
        default="movielens_1m_subgraph.csv",
        help="explanation model generate subgraphs",
    )

    parser.add_argument(
        "--subgraph_pivot_path",
        type=str,
        default="movielens_1m_subgraph_pivot.csv",
        help="explanation model generate subgraphs id and pivot id",
    )

    parser.add_argument(
        "--max_degree",
        type=int,
        default=2,
        help="subgraph to star pattern max degree",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=2,
        help="subgraph to star pattern max length",
    )

    parser.add_argument(
        "--max_subgraph_to_pattern_num",
        type=int,
        default=5,
        help="max num of star pattern sampled by subgraph",
    )

    parser.add_argument(
        "--hop_decay_factor",
        type=float,
        default=0.8,
        help="pattern node's hop decay factor ",
    )

    return parser.parse_args(args)


def main(args):

    subgraph_path = args.subgraph_path
    subgraph_pivot_path = args.subgraph_pivot_path

    max_degree = args.max_degree
    max_length = args.max_length
    pattern_num = args.max_subgraph_to_pattern_num

    hop_decay_factor = args.hop_decay_factor

    explain_graph_data = read_subgraph(subgraph_path, subgraph_pivot_path)

    gen_pattern_list = []
    Score_Q_dict = {}
    for graph_id in explain_graph_data:
        pivot_x = explain_graph_data[graph_id]['pivot']['pivot_x']
        pivot_y = explain_graph_data[graph_id]['pivot']['pivot_y']
        neighbors_dict = explain_graph_data[graph_id]['neighbors_dict']
        node_type = explain_graph_data[graph_id]['node_type']
        edge_type_dict = explain_graph_data[graph_id]['edge_type_dict']
        edge_type_label_dict = explain_graph_data[graph_id]['edge_type_label_dict']

        for pattern_iter in range(pattern_num):
            star_x, star_y = random_walk_degree_length(neighbors_dict, pivot_x, pivot_y, max_degree, max_length)

            score_Q = Score_Q(star_x, hop_decay_factor) + Score_Q(star_y, hop_decay_factor)
            pattern = SubgraphToPattern(star_x, star_y, node_type, edge_type_dict, edge_type_label_dict, pivot_x, pivot_y)

            if pattern is not None:
                gen_pattern_list.append(pattern)
                key = REP_make_hashable([pattern.vertex_list, pattern.edge_list])
                Score_Q_dict[key] = score_Q
        
    print("gen pattern num:", len(gen_pattern_list))

    
if __name__ == "__main__":
    main(parse_args())
