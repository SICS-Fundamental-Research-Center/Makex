import torch
import torch.nn as nn
import networkx as nx
import csv
import numpy as np

class GraphEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device):
        super(GraphEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).to(device)

    def forward(self, input_sequences):
        embedded = self.embedding(input_sequences)
        mask = input_sequences != 0
        masked_embedded = embedded * mask.unsqueeze(-1).type(torch.float32)
        embedding_sum = torch.sum(masked_embedded, dim=1)
        count_nonzero = torch.sum(mask, dim=1, keepdim=True)
        averaged_embedding = embedding_sum / count_nonzero
        return averaged_embedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

char_set = {'0', '1'}
char_to_int = {char: i + 1 for i, char in enumerate(char_set)}
vocab_size = len(char_set) + 1
embedding_dim = 8

model = GraphEmbeddingModel(vocab_size, embedding_dim, device)
model.to(device)



def _init_node_labels(G, edge_attr, node_attr):
    if node_attr:
        return {u: str(dd[node_attr]) for u, dd in G.nodes(data=True)}
    elif edge_attr:
        return {u: "" for u in G}
    else:
        return {u: str(deg) for u, deg in G.degree()}


def _label_to_embedding(label, model, char_to_int, device):
    int_sequence = [char_to_int[char] for char in label if char in char_to_int]
    input_sequences = torch.tensor([int_sequence], dtype=torch.long, device=device)
    return model(input_sequences).cpu().detach().numpy()

from collections import Counter, defaultdict

def _neighborhood_aggregate(G, node, node_labels, edge_attr=None):
    label_list = []
    for nbr in G.neighbors(node):
        prefix = "" if edge_attr is None else str(G[node][nbr][edge_attr])
        label_list.append(prefix + str(node_labels[nbr]))
    return str(node_labels[node]) + "".join(sorted(label_list))

def weisfeiler_lehman_subgraph_hashes(
    G,
    edge_attr=None,
    node_attr=None,
    iterations=3,
    digest_size=16,
    include_initial_labels=False,
    csv_file_path=None,
    save_iteration=-1
):
    def weisfeiler_lehman_step(G, labels, node_subgraph_hashes,  edge_attr=None, iteration=None, csv_writer=None):
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            
            if iteration == save_iteration:
                hashed_label = _label_to_embedding(label, model, char_to_int, device)
            else:
                hashed_label = _label_to_embedding(label, model, char_to_int, device)
            if csv_writer is not None and iteration == save_iteration:
                csv_writer.writerow([node, hashed_label])
            new_labels[node] = hashed_label
            node_subgraph_hashes[node].append(hashed_label)
        return new_labels

    node_labels = _init_node_labels(G, edge_attr, node_attr)
    if include_initial_labels:
        node_subgraph_hashes = {k: [_label_to_embedding(v, model, char_to_int, device)] for k, v in node_labels.items()}
    else:
        node_subgraph_hashes = defaultdict(list)

    if csv_file_path:
        with open(csv_file_path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Node ID', 'Label'])
            for i in range(iterations):
                node_labels = weisfeiler_lehman_step(
                    G, node_labels, node_subgraph_hashes, edge_attr, iteration=i, csv_writer=csv_writer
                )
    else:
        for i in range(iterations):
            node_labels = weisfeiler_lehman_step(
                G, node_labels, node_subgraph_hashes, edge_attr
            )

    return dict(node_subgraph_hashes)


def process_graph(node_csv, edge_csv, model, device):
    G = nx.Graph()


    with open(node_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            vertex_id, label_id = int(row[0]), row[1]
            G.add_node(vertex_id, label=label_id)

    with open(edge_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            src, tgt = int(row[1]), int(row[2])
            G.add_edge(src, tgt)

    return G




node_csv_path = "../DataSets/Movielens/original_graph/movielens_v.csv"
edge_csv_path = "../DataSets/Movielens/original_graph/movielens_e.csv"
G = process_graph(node_csv_path, edge_csv_path, model, device)

csv_file_path = "../DataSets/Movielens/original_graph/movielens_v_wl.csv"
save_iteration = 10

subgraph_hashes = weisfeiler_lehman_subgraph_hashes(
    G,
    edge_attr=None,
    node_attr="label",
    iterations=save_iteration+1,
    digest_size=16,
    include_initial_labels=False,
    csv_file_path=csv_file_path,
    save_iteration=save_iteration
)