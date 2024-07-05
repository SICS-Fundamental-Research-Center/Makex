import sys

sys.path.append("..")
from structure.structure import Pattern


def RuleToPattern(x_node_label, y_node_label, encode_map, x_rule, y_rule):
    x_rule_len = x_rule[0]
    vertex_list = []
    edge_list = []
    path_flag = len(encode_map)
    end_flag = path_flag + 1
    vertex_list.append([1, x_node_label])
    vertex_list.append([2, y_node_label])
    vertex_id = 3
    last_src_id = 1
    for i in range(x_rule_len):
        hash_val = x_rule[i + 1]
        if hash_val == path_flag:
            last_src_id = 1
        elif hash_val == end_flag:
            break
        else:
            edge_label, dst_label = encode_map[hash_val]
            dst_id = vertex_id
            vertex_id = vertex_id + 1
            vertex_list.append([dst_id, dst_label])
            edge_list.append([last_src_id, dst_id, edge_label])
            last_src_id = dst_id
    last_src_id = 2
    y_rule_len = y_rule[0]
    for i in range(y_rule_len):
        hash_val = y_rule[i + 1]
        if hash_val == path_flag:
            last_src_id = 2
        elif hash_val == end_flag:
            break
        else:
            edge_label, dst_label = encode_map[hash_val]
            dst_id = vertex_id
            vertex_id = vertex_id + 1
            vertex_list.append([dst_id, dst_label])
            edge_list.append([last_src_id, dst_id, edge_label])
            last_src_id = dst_id
    return Pattern(vertex_list, edge_list)



def PatternToRule(pattern, inv_encode_map, weight):
    def BFSWithRoot(pattern, root, inv_encode_map):
        res_list = []
        q = []
        q.append(root)
        while len(q) > 0:
            u = q[0]
            q.pop()
            for edge in pattern.edge_list:
                if edge[0] == u:
                    v = edge[1]
                    label = edge[2]
                    res_list.append(inv_encode_map[(pattern.vertex_list[v - 1][1], label)])
                    q.append(v)
        return res_list

    tmp_x_list = []
    tmp_y_list = []
    for edge in pattern.edge_list:
        u = edge[0]
        v = edge[1]
        label = edge[2]
        if u == 1:
            val = inv_encode_map[(pattern.vertex_list[v - 1][1], label)]
            tmp_x_list.append(val)
            tmp_list = BFSWithRoot(pattern, v, inv_encode_map)
            for ele in tmp_list:
                tmp_x_list.append(ele)
        elif u == 2:
            val = inv_encode_map[(pattern.vertex_list[v - 1][1], label)]
            tmp_y_list.append(val)
            tmp_list = BFSWithRoot(pattern, v, inv_encode_map)
            for ele in tmp_list:
                tmp_y_list.append(ele)
    res_x_list = []
    res_y_list = []
    res_x_list.append(len(tmp_x_list))
    for ele in tmp_x_list:
        res_x_list.append(ele)
    res_x_list.append(weight)
    res_x_list.append(0)
    res_x_list.append(0)
    res_y_list.append(len(tmp_y_list))
    for ele in tmp_y_list:
        res_y_list.append(ele)
    res_y_list.append(weight)
    res_y_list.append(0)
    res_y_list.append(0)
    return res_x_list, res_y_list
