"""Torch Module for SubgraphX"""
import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

import dgl
from dgl import to_heterogeneous, to_homogeneous
from dgl.base import NID
from dgl import to_networkx
from dgl import node_subgraph
from dgl import remove_nodes

from model import ScorePredictor
__all__ = ["KGATSX", "HGTSX", "PinsageSX"]


class MCTSNode:
    r"""Monte Carlo Tree Search Node

    Parameters
    ----------
    nodes : Tensor
        The node IDs of the graph that are associated with this tree node
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.num_visit = 0
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.children = []

    def __repr__(self):
        r"""Get the string representation of the node.

        Returns
        -------
        str
            The string representation of the node
        """
        return str(self.nodes)


class KGATSX(nn.Module):
    def __init__(
        self,
        model,
        num_hops,
        coef=10.0,
        high2low=True,
        num_child=12,
        num_rollouts=20,
        node_min=3,
        shapley_steps=100,
        log=False,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = model

    def shapley(self, subgraph_nodes):
        num_nodes = self.graph.num_nodes()
        subgraph_nodes = subgraph_nodes.tolist()


        # pivots to G_i
        # print(subgraph_nodes)
        # print("starting the checking")
        user_flag = self.user in subgraph_nodes
        item_flag = self.item in subgraph_nodes
        if user_flag == False:
            subgraph_nodes.append(self.user)

        if item_flag == False:
            subgraph_nodes.append(self.item)
        # print(user_flag, item_flag)
        # print("after")
        # print(subgraph_nodes)

        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_region = subgraph_nodes
        # print("current nodes {}".format(local_region))
        for _ in range(self.num_hops - 1):
            in_neighbors, _ = self.graph.in_edges(local_region)
            _, out_neighbors = self.graph.out_edges(local_region)
            neighbors = torch.cat([in_neighbors, out_neighbors]).tolist()
            local_region = list(set(local_region + neighbors))

        split_point = num_nodes
        coalition_space = list(set(local_region) - set(subgraph_nodes)) + [
            split_point
        ]

        marginal_contributions = []
        device = self.feat.device
        for _ in range(self.shapley_steps):
            permuted_space = np.random.permutation(coalition_space)
            split_idx = int(np.where(permuted_space == split_point)[0])

            selected_nodes = permuted_space[:split_idx]

            # print("# selected nodes: {}".format(selected_nodes.shape))

            # Mask for coalition set S_i
            exclude_mask = torch.ones(num_nodes)
            exclude_mask[local_region] = 0.0
            exclude_mask[selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = exclude_mask.clone()
            include_mask[subgraph_nodes] = 1.0

            exclude_feat = self.feat * exclude_mask.unsqueeze(1).to(device)
            include_feat = self.feat * include_mask.unsqueeze(1).to(device)

            with torch.no_grad():
                exclude_probs = self.model.gnn(
                    self.graph, exclude_feat
                ).softmax(dim=-1)
                exclude_value = exclude_probs[:, self.target_class]
                include_probs = self.model.gnn(
                    self.graph, include_feat
                ).softmax(dim=-1)
                include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        if len(mcts_node.children) > 0:
            return mcts_node.children
        
        # print(mcts_node.nodes)
        subg = node_subgraph(self.graph, mcts_node.nodes)
        node_degrees = subg.out_degrees() + subg.in_degrees()
        k = min(subg.num_nodes(), self.num_child)
        chosen_nodes = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices

        mcts_children_maps = dict()

        for node in chosen_nodes:
            new_subg = remove_nodes(subg, node.to(subg.idtype), store_ids=True)
            # Get the largest weakly connected component in the subgraph.
            nx_graph = to_networkx(new_subg.cpu())
            largest_cc_nids = list(
                max(nx.weakly_connected_components(nx_graph), key=len)
            )
            # Map to the original node IDs.
            largest_cc_nids = new_subg.ndata[NID][largest_cc_nids].long()
            largest_cc_nids = subg.ndata[NID][largest_cc_nids].sort().values
            # print("largest cc node ids....")
            # print(largest_cc_nids)
            # maintain the pivots

            if str(largest_cc_nids) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(largest_cc_nids)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(largest_cc_nids)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node, depth):
        if len(mcts_node.nodes) <= self.node_min:
            return mcts_node.immediate_reward
        
        if depth > 800:
            return mcts_node.immediate_reward
        
        # print(mcts_node)
        children_nodes = self.get_mcts_children(mcts_node)
        # print("num nodes:{}".format(len(children_nodes)))
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
            + self.coef
            * c.immediate_reward
            * children_visit_sum_sqrt
            / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child, depth+1)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, pivots, feat, target_class, **kwargs):
        self.model.eval()
        # Get the initial prediction.
        with torch.no_grad():
            print("Compute attention weight in eval func ...")
            A_w = self.model.compute_attention(graph)
            graph.edata['w'] = A_w

        # assert (
        #     graph.num_nodes() > self.node_min
        # ), f"The number of nodes in the\
        #     graph {graph.num_nodes()} should be bigger than {self.node_min}."
        
        if graph.num_nodes() > self.node_min:
            self.graph = graph
            self.feat = feat
            self.target_class = target_class
            self.kwargs = kwargs
            self.user = pivots[0]
            self.item = pivots[1]

            # book all nodes in MCTS
            self.mcts_node_maps = dict()

            root = MCTSNode(graph.nodes())
            self.mcts_node_maps[str(root)] = root

            for i in range(self.num_rollouts):
                # print("roll out {}".format(i))
                if True:
                    print(
                        f"Rollout {i}/{self.num_rollouts}, \
                        {len(self.mcts_node_maps)} subgraphs have been explored."
                    )
                self.mcts_rollout(root, 0)

            best_leaf = None
            best_immediate_reward = float("-inf")
            for mcts_node in self.mcts_node_maps.values():
                
                if len(mcts_node.nodes) > self.node_min:
                    continue

                # print(mcts_node)

                if mcts_node.immediate_reward > best_immediate_reward:
                    best_leaf = mcts_node
                    best_immediate_reward = best_leaf.immediate_reward
            if best_leaf == None:
                return graph.nodes()
            else:
                return best_leaf.nodes
        else:
            return graph.nodes()


class HGTSX(nn.Module):
    def __init__(
        self,
        model,
        num_hops,
        coef=10.0,
        high2low=True,
        num_child=12,
        num_rollouts=20,
        node_min=3,
        shapley_steps=100,
        log=False,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = model

    def shapley(self, subgraph_nodes):
        num_nodes = self.graph.num_nodes()
        subgraph_nodes = subgraph_nodes.tolist()


        # pivots to G_i
        # print(subgraph_nodes)
        # print("starting the checking")
        user_flag = self.user in subgraph_nodes
        item_flag = self.item in subgraph_nodes
        if user_flag == False:
            subgraph_nodes.append(self.user)

        if item_flag == False:
            subgraph_nodes.append(self.item)
        # print(user_flag, item_flag)
        # print("after")
        # print(subgraph_nodes)

        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_region = subgraph_nodes
        # print("current nodes {}".format(local_region))
        for _ in range(self.num_hops - 1):
            in_neighbors, _ = self.graph.in_edges(local_region)
            _, out_neighbors = self.graph.out_edges(local_region)
            neighbors = torch.cat([in_neighbors, out_neighbors]).tolist()
            local_region = list(set(local_region + neighbors))

        split_point = num_nodes
        coalition_space = list(set(local_region) - set(subgraph_nodes)) + [
            split_point
        ]

        marginal_contributions = []
        device = self.feat.device
        for _ in range(self.shapley_steps):
            permuted_space = np.random.permutation(coalition_space)
            split_idx = int(np.where(permuted_space == split_point)[0])

            selected_nodes = permuted_space[:split_idx]

            # print("# selected nodes: {}".format(selected_nodes.shape))

            # Mask for coalition set S_i
            exclude_mask = torch.ones(num_nodes)
            exclude_mask[local_region] = 0.0
            exclude_mask[selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = exclude_mask.clone()
            include_mask[subgraph_nodes] = 1.0

            exclude_feat = self.feat * exclude_mask.unsqueeze(1).to(device)
            include_feat = self.feat * include_mask.unsqueeze(1).to(device)

            
            with torch.no_grad():
                exclude_probs = self.model(
                    self.graph, exclude_feat, self.graph.ndata['ntype'], self.graph.edata['type']
                ).softmax(dim=-1)
                exclude_value = exclude_probs[:, self.target_class]
                include_probs = self.model(
                    self.graph, include_feat,  self.graph.ndata['ntype'], self.graph.edata['type']
                ).softmax(dim=-1)
                include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        if len(mcts_node.children) > 0:
            return mcts_node.children
        
        # print(mcts_node.nodes)
        subg = node_subgraph(self.graph, mcts_node.nodes)
        node_degrees = subg.out_degrees() + subg.in_degrees()
        k = min(subg.num_nodes(), self.num_child)
        chosen_nodes = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices

        mcts_children_maps = dict()

        for node in chosen_nodes:
            new_subg = remove_nodes(subg, node.to(subg.idtype), store_ids=True)
            # Get the largest weakly connected component in the subgraph.
            nx_graph = to_networkx(new_subg.cpu())
            largest_cc_nids = list(
                max(nx.weakly_connected_components(nx_graph), key=len)
            )
            # Map to the original node IDs.
            largest_cc_nids = new_subg.ndata[NID][largest_cc_nids].long()
            largest_cc_nids = subg.ndata[NID][largest_cc_nids].sort().values
            # print("largest cc node ids....")
            # print(largest_cc_nids)
            # maintain the pivots

            if str(largest_cc_nids) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(largest_cc_nids)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(largest_cc_nids)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node, depth):
        if len(mcts_node.nodes) <= self.node_min:
            return mcts_node.immediate_reward
        
        if depth > 800:
            return mcts_node.immediate_reward
        
        # print(mcts_node)
        children_nodes = self.get_mcts_children(mcts_node)
        # print("num nodes:{}".format(len(children_nodes)))
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
            + self.coef
            * c.immediate_reward
            * children_visit_sum_sqrt
            / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child, depth+1)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, pivots, feat, target_class, **kwargs):
        self.model.eval()

        # assert (
        #     graph.num_nodes() > self.node_min
        # ), f"The number of nodes in the\
        #     graph {graph.num_nodes()} should be bigger than {self.node_min}."
        
        if graph.num_nodes() > self.node_min:
            self.graph = graph
            self.feat = feat
            self.target_class = target_class
            self.kwargs = kwargs
            self.user = pivots[0]
            self.item = pivots[1]

            # book all nodes in MCTS
            self.mcts_node_maps = dict()

            root = MCTSNode(graph.nodes())
            self.mcts_node_maps[str(root)] = root

            for i in range(self.num_rollouts):
                # print("roll out {}".format(i))
                if True:
                    print(
                        f"Rollout {i}/{self.num_rollouts}, \
                        {len(self.mcts_node_maps)} subgraphs have been explored."
                    )
                self.mcts_rollout(root, 0)

            best_leaf = None
            best_immediate_reward = float("-inf")
            for mcts_node in self.mcts_node_maps.values():
                
                if len(mcts_node.nodes) > self.node_min:
                    continue

                # print(mcts_node)

                if mcts_node.immediate_reward > best_immediate_reward:
                    best_leaf = mcts_node
                    best_immediate_reward = best_leaf.immediate_reward
            if best_leaf == None:
                return graph.nodes()
            else:
                return best_leaf.nodes
        else:
            return graph.nodes()




class PinsageSX(nn.Module):

    def __init__(
        self,
        model,
        num_hops,
        coef=10.0,
        high2low=True,
        num_child=12,
        num_rollouts=20,
        node_min=3,
        shapley_steps=100,
        log=True,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = model

    def shapley(self, subgraph_nodes):
        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_regions = {
            ntype: nodes.tolist() for ntype, nodes in subgraph_nodes.items()
        }
        for _ in range(self.num_hops - 1):
            for c_etype in self.graph.canonical_etypes:
                src_ntype, _, dst_ntype = c_etype
                if (
                    src_ntype not in local_regions
                    or dst_ntype not in local_regions
                ):
                    continue

                in_neighbors, _ = self.graph.in_edges(
                    local_regions[dst_ntype], etype=c_etype
                )
                _, out_neighbors = self.graph.out_edges(
                    local_regions[src_ntype], etype=c_etype
                )
                local_regions[src_ntype] = list(
                    set(local_regions[src_ntype] + in_neighbors.tolist())
                )
                local_regions[dst_ntype] = list(
                    set(local_regions[dst_ntype] + out_neighbors.tolist())
                )

        split_point = self.graph.num_nodes()
        coalition_space = {
            ntype: list(
                set(local_regions[ntype]) - set(subgraph_nodes[ntype].tolist())
            )
            + [split_point]
            for ntype in subgraph_nodes.keys()
        }

        marginal_contributions = []
        for _ in range(self.shapley_steps):
            selected_node_map = dict()
            for ntype, nodes in coalition_space.items():
                permuted_space = np.random.permutation(nodes)
                split_idx = int(np.where(permuted_space == split_point)[0])
                selected_node_map[ntype] = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = {
                ntype: torch.ones(self.graph.num_nodes(ntype))
                for ntype in self.graph.ntypes
            }
            for ntype, region in local_regions.items():
                exclude_mask[ntype][region] = 0.0
            for ntype, selected_nodes in selected_node_map.items():
                exclude_mask[ntype][selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = {
                ntype: exclude_mask[ntype].clone()
                for ntype in self.graph.ntypes
            }
            for ntype, subgn in subgraph_nodes.items():
                exclude_mask[ntype][subgn] = 1.0

            exclude_feat = {
                ntype: self.feat[ntype]
                * exclude_mask[ntype].unsqueeze(1).to(self.feat[ntype].device)
                for ntype in self.graph.ntypes
            }
            include_feat = {
                ntype: self.feat[ntype]
                * include_mask[ntype].unsqueeze(1).to(self.feat[ntype].device)
                for ntype in self.graph.ntypes
            }

            blocks = [self.graph, self.graph]
            # print(self.pivots)
            num_nodes_dict = {'user': self.graph.num_nodes('user'), 'item': self.graph.num_nodes('item')}
            pivot_g = dgl.heterograph({
                ('user', 'rate', 'item'): (self.pivots['user'], self.pivots['item'])
            }, num_nodes_dict=num_nodes_dict)
            # print("pivot graph {}".format(pivot_g))
            pred = ScorePredictor()
            with torch.no_grad():
                exclude_probs = self.model(
                    blocks, exclude_feat
                )
                exclude_h = {
                    'user': exclude_probs['user'],
                    'item':exclude_probs['item']
                }
                exclude_value = pred(pivot_g.to(self.graph.device), exclude_h)
                # print("exclude value {}".format(exclude_value))


                include_probs = self.model(
                    blocks, include_feat
                )
                include_h = {
                    'user': include_probs['user'],
                    'item':include_probs['item']
                }
                include_value = pred(pivot_g.to(self.graph.device), include_h)
                # print("include value {}".format(include_value))
                # include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = node_subgraph(self.graph, mcts_node.nodes)
        # Choose k nodes based on the highest degree in the subgraph
        node_degrees_map = {
            ntype: torch.zeros(
                subg.num_nodes(ntype), device=subg.nodes(ntype).device
            )
            for ntype in subg.ntypes
        }
        for c_etype in subg.canonical_etypes:
            src_ntype, _, dst_ntype = c_etype
            node_degrees_map[src_ntype] += subg.out_degrees(etype=c_etype)
            node_degrees_map[dst_ntype] += subg.in_degrees(etype=c_etype)

        node_degrees_list = [
            ((ntype, i), degree)
            for ntype, node_degrees in node_degrees_map.items()
            for i, degree in enumerate(node_degrees)
        ]

        # print("node degrees list")
        # print(node_degrees_list)

        node_degrees = torch.stack([v for _, v in node_degrees_list])
        k = min(subg.num_nodes(), self.num_child)
        chosen_node_indicies = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices
        chosen_nodes = [node_degrees_list[i][0] for i in chosen_node_indicies]

        # print("chosen nodes:")
        # print(chosen_nodes)

        # add the privots to the chosen_nodes
        


        mcts_children_maps = dict()

        for ntype, node in chosen_nodes:
            new_subg = remove_nodes(subg, node, ntype, store_ids=True)

            if new_subg.num_edges() > 0:
                new_subg_homo = to_homogeneous(new_subg)
                # Get the largest weakly connected component in the subgraph.
                nx_graph = to_networkx(new_subg_homo.cpu())
                largest_cc_nids = list(
                    max(nx.weakly_connected_components(nx_graph), key=len)
                )
                largest_cc_homo = node_subgraph(new_subg_homo, largest_cc_nids)
                largest_cc_hetero = to_heterogeneous(
                    largest_cc_homo, new_subg.ntypes, new_subg.etypes
                )

                # Follow steps for backtracking to original graph node ids
                # 1. retrieve instanced homograph from connected-component homograph
                # 2. retrieve instanced heterograph from instanced homograph
                # 3. retrieve hetero-subgraph from instanced heterograph
                # 4. retrieve orignal graph ids from subgraph node ids
                cc_nodes = {
                    ntype: subg.ndata[NID][ntype][
                        new_subg.ndata[NID][ntype][
                            new_subg_homo.ndata[NID][
                                largest_cc_homo.ndata[NID][indicies]
                            ]
                        ]
                    ]
                    for ntype, indicies in largest_cc_hetero.ndata[NID].items()
                }
            else:
                available_ntypes = [
                    ntype
                    for ntype in new_subg.ntypes
                    if new_subg.num_nodes(ntype) > 0
                ]
                chosen_ntype = np.random.choice(available_ntypes)
                # backtrack from subgraph node ids to entire graph
                chosen_node = subg.ndata[NID][chosen_ntype][
                    np.random.choice(new_subg.nodes[chosen_ntype].data[NID])
                ]
                cc_nodes = {
                    chosen_ntype: torch.tensor(
                        [chosen_node],
                        device=subg.device,
                    )
                }

            if str(cc_nodes) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(cc_nodes)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(cc_nodes)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node, depth):
        if (
            sum(len(nodes) for nodes in mcts_node.nodes.values())
            <= self.node_min
        ):
            return mcts_node.immediate_reward
        
        if depth > 800:
            return mcts_node.immediate_reward
        
        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
            + self.coef
            * c.immediate_reward
            * children_visit_sum_sqrt
            / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child, depth+1)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, pivots, feat, target_class, **kwargs):
        self.model.eval()
        # assert (
        #     graph.num_nodes() > self.node_min
        # ), f"The number of nodes in the\
        #     graph {graph.num_nodes()} should be bigger than {self.node_min}."

        if graph.num_nodes() > self.node_min:

            self.graph = graph
            self.feat = feat
            self.target_class = target_class
            self.kwargs = kwargs

            self.pivots = pivots # {'user': x, 'item": y}

            # book all nodes in MCTS
            self.mcts_node_maps = dict()

            root_dict = {ntype: graph.nodes(ntype) for ntype in graph.ntypes}
            root = MCTSNode(root_dict)
            self.mcts_node_maps[str(root)] = root

            for i in range(self.num_rollouts):
                if self.log:
                    print(
                        f"Rollout {i}/{self.num_rollouts}, \
                        {len(self.mcts_node_maps)} subgraphs have been explored."
                    )
                self.mcts_rollout(root,0)

            best_leaf = None
            best_immediate_reward = float("-inf")
            for mcts_node in self.mcts_node_maps.values():
                if len(mcts_node.nodes) > self.node_min:
                    continue

                if mcts_node.immediate_reward > best_immediate_reward:
                    best_leaf = mcts_node
                    best_immediate_reward = best_leaf.immediate_reward
        else:
            return graph.nodes()

        return best_leaf.nodes