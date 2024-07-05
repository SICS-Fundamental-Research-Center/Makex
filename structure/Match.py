import pyMakex


def REPMatch(rep, data_graph, use_ptime=True, use_cache=True, save_result=False):
    ptime_flag = 1
    if use_ptime == False:
        ptime_flag = 0
    cache_flag = 1
    if use_cache == False:
        cache_flag = 0
    result_flag = 0
    if save_result == True:
        result_flag = 1
    ret_result = pyMakex.REPMatch(rep.GetREPMatchArg(), data_graph.graph_ptr, cache_flag, result_flag)

    return ret_result


def REPMatch_U_V(rep, data_graph, user_id, item_id, use_ptime=True, use_cache=True, save_result=False, enable_topk=1, topk_explanation_scores_list = [], topk = 1, rep_id = 0):
    ptime_flag = 1
    if use_ptime == False:
        ptime_flag = 0
    cache_flag = 1
    if use_cache == False:
        cache_flag = 0
    result_flag = 0
    if save_result == True:
        result_flag = 1

    ret_result = pyMakex.REPMatch_U_V(rep.GetREPMatchArg(), data_graph.graph_ptr, user_id, item_id,cache_flag, result_flag, enable_topk, topk_explanation_scores_list, topk, rep_id)

    return ret_result


def PatternMatch(pattern, data_graph, sample_pair=500, max_result=10):
    return pyMakex.PatternMatch(pattern.vertex_list, pattern.edge_list, data_graph.graph_ptr, sample_pair, max_result)


def CheckHasMatch(rep, data_graph, x_id, y_id):
    return pyMakex.CheckHasMatch(rep.GetREPMatchArg(), data_graph.graph_ptr, x_id, y_id)


def REPMatchWithMultiProcess(rep, data_graph, num_process):
    ret_result = pyMakex.REPMatchWithMultiProcess(rep.GetREPMatchArg(), data_graph.graph_ptr, num_process, 1)
    return ret_result


