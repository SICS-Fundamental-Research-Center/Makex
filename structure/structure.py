import pyMakex


class DataGraph:
    def __init__(self, graph_ptr):
        self.graph_ptr = graph_ptr

    def NumOfEncode(self):
        return pyMakex.NumOfEncode(self.graph_ptr)

    def AdjEncodeList(self, vertex_label):
        return pyMakex.AdjEncodeList(self.graph_ptr, vertex_label)

    """
    input:vertex_id
    output:vertex's attr dict, key is attr.key,value is attr.value
    """

    def GetVertexAttribute(self, vertex_id):
        return pyMakex.GetVertexAttribute(self.graph_ptr, vertex_id)

    def GetAllVertexID(self):
        return pyMakex.GetAllVertexID(self.graph_ptr)

    def GetLabelAttributeMap(self, vertex_label):
        return pyMakex.LabelAttributeMap(self.graph_ptr, vertex_label)

    def GetAllEdge(self):
        return pyMakex.GetAllEdge(self.graph_ptr)

    def GetEncodeMap(self):
        return pyMakex.GetEncodeMap(self.graph_ptr)

    def HasOutEdge(self, src_id, dst_id):
        return pyMakex.HasOutEdge(self.graph_ptr, src_id, dst_id)

    def PositivePairNum(self):
        return pyMakex.PositivePairNum(self.graph_ptr)

    def NegativePairNum(self):
        return pyMakex.NegativePairNum(self.graph_ptr)


class Pattern:
    def __init__(self, vertex_list, edge_list):
        self.vertex_list = vertex_list
        self.edge_list = edge_list

    def GetLeaves(self):
        out_degree = {}
        for edge in self.edge_list:
            out_degree[edge[0]] = out_degree[edge[0]] + 1
        leaves_node = []
        for vertex in self.vertex_list:
            if out_degree.get(vertex[0]) == None:
                leaves_node.append(vertex)
        return leaves_node

    def GetHashArg(self):
        copy_vertex_list = self.vertex_list
        copy_edge_list = self.edge_list
        temp_copy_vertex_list = tuple([tuple(vertex) for vertex in copy_vertex_list])
        temp_copy_edge_list = tuple([tuple(edge) for edge in copy_edge_list])
        return (temp_copy_vertex_list, temp_copy_edge_list)


class REP:
    def __init__(self, score, pattern, predicate_list, predicate_score_list, x_id, y_id, q, ml_flag=1):
        self.score = score
        self.pattern = pattern
        self.predicate_list = predicate_list
        self.predicate_score_list = predicate_score_list
        self.x_id = x_id
        self.y_id = y_id
        self.q = q
        self.weight = 0
        self.ml_flag = ml_flag

    def GetREPMatchArg(self):
        return [self.pattern.vertex_list, self.pattern.edge_list, self.predicate_list, self.predicate_score_list, [self.x_id, self.y_id, self.q, self.score]]

    def SetScore(self, score):
        self.weight = score
