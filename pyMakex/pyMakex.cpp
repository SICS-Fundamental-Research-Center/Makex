#include <Python.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include "include/algorithm/REPMatch.h"
#include "include/gundam/algorithm/dp_iso.h"
#include "include/gundam/data_type/datatype.h"
#include "include/gundam/graph_type/large_graph.h"
#include "include/gundam/graph_type/large_graph2.h"
#include "include/gundam/graph_type/small_graph.h"
#include "include/gundam/io/csvgraph.h"
#include "include/gundam/type_getter/vertex_handle.h"
#include "include/module/DataGraphWithInformation.h"
#include "include/module/StarMiner.h"
#include "include/structure/PatternWithParameter.h"
#include "include/structure/REP.h"


using Pattern = GUNDAM::LargeGraph2<uint32_t, uint32_t, std::string, uint32_t,
       uint32_t, std::string>;
using PatternVertexIDType = typename Pattern::VertexType::IDType;
using PatternVertexLabelType = typename Pattern::VertexType::LabelType;
using PatternEdgeIDType = typename Pattern::EdgeType::IDType;
using PatternEdgeLabelType = typename Pattern::EdgeType::LabelType;
using PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type;
using PatternVertexConstPtr =
    typename GUNDAM::VertexHandle<const Pattern>::type;
using DataGraph = GUNDAM::LargeGraph2<uint32_t, uint32_t, std::string, uint32_t,
                                      uint32_t, std::string>;
using DataGraphVertexIDType = typename DataGraph::VertexType::IDType;
using DataGraphVertexLabelType = typename DataGraph::VertexType::LabelType;
using DataGraphEdgeIDType = typename DataGraph::EdgeType::IDType;
using DataGraphEdgeLabelType = typename DataGraph::EdgeType::LabelType;
using DataGraphVertexPtr = typename GUNDAM::VertexHandle<DataGraph>::type;
using DataGraphAttrKeyType = typename DataGraph::VertexType::AttributeKeyType;
using DataGraphVertexConstPtr =
    typename GUNDAM::VertexHandle<const DataGraph>::type;
using DataGraphVertexAttributeKeyType =
      typename DataGraph::VertexType::AttributeKeyType;
using StarMiner = Makex::StarMiner<DataGraph>;
using DataGraphAttributePtr = typename DataGraph::VertexType::AttributePtr;
using DataGraphWithInformation = Makex::DataGraphWithInformation<DataGraph>;

using MyCompare = std::function<bool(std::pair<double, std::map<int, int>>, std::pair<double, std::map<int, int>>)>;



void FetchGlobalTopkScoresIntoHeap(PyObject * topk_scores,   
    std::priority_queue<std::pair<double, std::map<int, int>>, 
        std::vector<std::pair<double, std::map<int, int>>>, 
        MyCompare> & topk_min_heap, const int& topk) {
    int scores_size = PyList_Size(topk_scores);
    
    std::map<int, int> default_map;
    for (int i = 0; i < scores_size; ++i) {
        double temp_score = PyFloat_AsDouble(PyList_GetItem(topk_scores, i));
        topk_min_heap.push(std::make_pair(temp_score, default_map));
        if (i >= topk) {
            topk_min_heap.pop();
        }
    }
}


void FetchGlobalTopkScoresIntoHeap_C(const std::vector<double>& topk_scores,
                                   std::priority_queue<std::pair<double, std::map<int, int>>,
                                   std::vector<std::pair<double, std::map<int, int>>>,
                                   MyCompare>& topk_min_heap,
                                   const int& topk) {
    std::map<int, int> default_map;
    for (double temp_score : topk_scores) {
        topk_min_heap.push(std::make_pair(temp_score, default_map));
        if (topk_min_heap.size() > topk) {
            topk_min_heap.pop();
        }
    }
}



static PyObject *ReadDataGraph(PyObject *self, PyObject *args) {
  char *e_file, *v_file;
  if (!PyArg_ParseTuple(args, "s|s", &v_file, &e_file)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  std::cout << "ReadCSVGraph begin" << std::endl;
  std::string v_file_str(v_file), e_file_str(e_file);
  std::cout << "v_file: " << v_file_str << std::endl;
  std::cout << "e_file: " << e_file_str << std::endl;

  DataGraphWithInformation *data_graph_ptr = new DataGraphWithInformation();
  GUNDAM::ReadCSVGraph(data_graph_ptr->data_graph(), v_file_str, e_file_str);

  for (auto vertex_it = data_graph_ptr->data_graph().VertexBegin();
       !vertex_it.IsDone(); vertex_it++) {
    vertex_it->AddAttribute((std::string)("id"), (int)vertex_it->id());
  }

  data_graph_ptr->BuildEncodeHashMap();
  data_graph_ptr->BuildLabelAttributeKeyMap();
  return PyLong_FromLong(long(data_graph_ptr));
}
static PyObject *ReadML(PyObject *self, PyObject *args) {
  long ptr_val;
  char *ml_file;
  double delta_l, delta_r;
  int user_offset;
  if (!PyArg_ParseTuple(args, "l|s|d|d|i", &ptr_val, &ml_file, &delta_l,
                        &delta_r, &user_offset)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  std::string model_file(ml_file);
  data_graph_ptr->BuildMLModel(model_file, delta_l, delta_r, user_offset);
  return PyLong_FromLong(long(data_graph_ptr));
}
static PyObject *PositivePairNum(PyObject *self, PyObject *args) {
  long ptr_val;
  if (!PyArg_ParseTuple(args, "l", &ptr_val)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  return PyLong_FromLong(long(data_graph_ptr->positive_pair_num()));
}
static PyObject *NegativePairNum(PyObject *self, PyObject *args) {
  long ptr_val;
  if (!PyArg_ParseTuple(args, "l", &ptr_val)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  return PyLong_FromLong(long(data_graph_ptr->negative_pair_num()));
}
static PyObject *MLModelSize(PyObject *self, PyObject *args) {
  long ptr_val;
  if (!PyArg_ParseTuple(args, "l", &ptr_val)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  return PyLong_FromLong(long(data_graph_ptr->ml_model_size()));
}
Pattern BuildPattern(PyObject *pattern_vertex_list,
                     PyObject *pattern_edge_list) {
  Pattern pattern;
  size_t pattern_vertex_count = PyList_Size(pattern_vertex_list);
  size_t pattern_edge_count = PyList_Size(pattern_edge_list);
  for (size_t i = 0; i < pattern_vertex_count; i++) {
    PyObject *vertex_information = PyList_GetItem(pattern_vertex_list, i);
    if (!PyList_Check(vertex_information)) {
      printf("vertex information error!\n");
      return pattern;
    }
    size_t vertex_information_size = PyList_Size(vertex_information);
    if (vertex_information_size != 2) {
      printf("vertex information error!\n");
      return pattern;
    }

    PatternVertexIDType vertex_id =
        PyLong_AsLong(PyList_GetItem(vertex_information, 0));
    PatternVertexLabelType vertex_label =
        PyLong_AsLong(PyList_GetItem(vertex_information, 1));
    pattern.AddVertex(vertex_id, vertex_label);
  }

  for (size_t i = 0; i < pattern_edge_count; i++) {
    PyObject *edge_information = PyList_GetItem(pattern_edge_list, i);
    if (!PyList_Check(edge_information)) {
      printf("Input error!\n");
      return pattern;
    }
    size_t edge_information_size = PyList_Size(edge_information);
    if (edge_information_size != 3) {
      printf("Input error!\n");
      return pattern;
    }
    PatternEdgeIDType edge_id = i;
    PatternVertexIDType src_id =
        PyLong_AsLong(PyList_GetItem(edge_information, 0));
    PatternVertexIDType dst_id =
        PyLong_AsLong(PyList_GetItem(edge_information, 1));
    PatternEdgeLabelType edge_label =
        PyLong_AsLong(PyList_GetItem(edge_information, 2));

    pattern.AddEdge(src_id, dst_id, edge_label, edge_id);
  }

  return pattern;
}

Makex::Operation StringToOp(std::string op) {
  if (op == "=") return Makex::Operation::kEqual;
  if (op == "!=") return Makex::Operation::kNotEqual;
  if (op == ">") return Makex::Operation::kGreat;
  if (op == ">=") return Makex::Operation::kGreatEqual;
  if (op == "<") return Makex::Operation::kLess;
  if (op == "<=") return Makex::Operation::kLessEqual;
  if (op == "include") return Makex::Operation::kInclude;
  if (op == "!include") return Makex::Operation::kNotInclude;
  return Makex::Operation::kEqual;
}

Makex::REP<Pattern, DataGraph> BuildREP(PyObject *py_rep) {
  PyObject *pattern_vertex_list = PyList_GetItem(py_rep, 0);
  PyObject *pattern_edge_list = PyList_GetItem(py_rep, 1);
  Pattern rep_pattern(BuildPattern(pattern_vertex_list, pattern_edge_list));
  PyObject *qxylist = PyList_GetItem(py_rep, 4);
  PatternVertexIDType x_id = PyLong_AsLong(PyList_GetItem(qxylist, 0));
  PatternVertexIDType y_id = PyLong_AsLong(PyList_GetItem(qxylist, 1));
  PatternEdgeLabelType q = PyLong_AsLong(PyList_GetItem(qxylist, 2));
  double score = PyFloat_AsDouble(PyList_GetItem(qxylist, 3));

  Makex::REP<Pattern, DataGraph> rep(rep_pattern, x_id, y_id, q, score);

  PyObject *rep_predicate_list = PyList_GetItem(py_rep, 2);
  PyObject *rep_predicate_score_list = PyList_GetItem(py_rep, 3);
  int predicate_size = PyList_Size(rep_predicate_list);
  for (int i = 0; i < predicate_size; i++) {
    PyObject *rep_predicate = PyList_GetItem(rep_predicate_list, i);
    double rep_predicate_score = PyFloat_AsDouble(PyList_GetItem(rep_predicate_score_list, i));
    PyObject *bytes = PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 0));
    char *predicate_type = PyBytes_AS_STRING(bytes);
    std::string predicate_op(predicate_type);

    PatternVertexIDType x_id = PyLong_AsLong(PyList_GetItem(rep_predicate, 1));
    DataGraphAttrKeyType x_a_key = PyBytes_AS_STRING(
        PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 2)));
    if (predicate_op == "Variable") {
      PatternVertexIDType y_id =
          PyLong_AsLong(PyList_GetItem(rep_predicate, 3));
      DataGraphAttrKeyType y_b_key = PyBytes_AS_STRING(
          PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 4)));
      Makex::Operation op = StringToOp(PyBytes_AS_STRING(
          PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 5))));
      rep.Add<Makex::VariablePredicate<Pattern, DataGraph>>(
          x_id, x_a_key, y_id, y_b_key, op, rep_predicate_score);
    } else if (predicate_op == "WL") {
      PatternVertexIDType y_id =
          PyLong_AsLong(PyList_GetItem(rep_predicate, 3));
      DataGraphAttrKeyType y_b_key = PyBytes_AS_STRING(
          PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 4)));
      Makex::Operation op = StringToOp(PyBytes_AS_STRING(
          PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 5))));
      rep.Add<Makex::WLPredicate<Pattern, DataGraph>>(
          x_id, x_a_key, y_id, y_b_key, op, rep_predicate_score);
    } else {
      std::string c_type = (std::string)(PyBytes_AS_STRING(
          PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 4))));
      Makex::Operation op = StringToOp(PyBytes_AS_STRING(
          PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 5))));
      if (c_type == "int") {
        int c = PyLong_AsLong(PyList_GetItem(rep_predicate, 3));
        rep.Add<Makex::ConstantPredicate<Pattern, DataGraph, int>>(
            x_id, x_a_key, c, op, rep_predicate_score);

      } else if (c_type == "double") {
        double c = PyFloat_AsDouble(PyList_GetItem(rep_predicate, 3));
        rep.Add<Makex::ConstantPredicate<Pattern, DataGraph, double>>(
            x_id, x_a_key, c, op, rep_predicate_score);
      } else {
        std::string c = (std::string)(PyBytes_AS_STRING(
            PyUnicode_AsUTF8String(PyList_GetItem(rep_predicate, 3))));
        rep.Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(
            x_id, x_a_key, c, op, rep_predicate_score);
      }
    }
  }
  return rep;
}

static PyObject *REPMatch(PyObject *self, PyObject *args) {
  long ptr_val;
  PyObject *py_rep;
  int use_ptime;
  int use_cache;
  int save_result;
  if (!PyArg_ParseTuple(args, "O|l|i|i", &py_rep, &ptr_val, &use_cache,
                        &save_result)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  Makex::REP<Pattern, DataGraph> rep(BuildREP(py_rep));
  std::vector<std::pair<DataGraphVertexConstPtr, DataGraphVertexConstPtr>>
      positive_result, all_result;
  bool cache_flag = use_cache ? true : false;
  auto t_begin = clock();
  int positive_count, all_count;
  if (save_result) {
    auto ret = Makex::REPMatchBasePTime(
        rep, data_graph_ptr->data_graph(), data_graph_ptr->ml_model(),
        cache_flag, &positive_result, &all_result);
    positive_count = ret.first;
    all_count = ret.second;
  } else {
    auto ret =
        Makex::REPMatchBasePTime(rep, data_graph_ptr->data_graph(),
                                   data_graph_ptr->ml_model(), cache_flag);
    positive_count = ret.first;
    all_count = ret.second;
  }

  auto t_end = clock();
  PyObject *ret = PyList_New(0);

  if (save_result) {
    PyObject *positive_match_result = PyList_New(0);
    for (auto &[x_res, y_res] : positive_result) {
      PyObject *ret_pair = PyList_New(0);
      PyList_Append(ret_pair, PyLong_FromLong(x_res->id()));
      PyList_Append(ret_pair, PyLong_FromLong(y_res->id()));
      PyList_Append(positive_match_result, ret_pair);
    }
    PyObject *all_match_result = PyList_New(0);
    for (auto &[x_res, y_res] : all_result) {
      PyObject *ret_pair = PyList_New(0);
      PyList_Append(ret_pair, PyLong_FromLong(x_res->id()));
      PyList_Append(ret_pair, PyLong_FromLong(y_res->id()));
      PyList_Append(all_match_result, ret_pair);
    }
    PyList_Append(ret, positive_match_result);
    PyList_Append(ret, all_match_result);
  } else {
    PyList_Append(ret, PyLong_FromLong(positive_count));
    PyList_Append(ret, PyLong_FromLong(all_count));
  }

  return ret;
}


static PyObject *REPMatch_U_V(PyObject *self, PyObject *args) {
  long ptr_val;
  PyObject *py_rep;
  int use_ptime;
  int use_cache;
  int save_result;
  int user_id;
  int item_id;
  int enable_topk;
  int topk;
  int rep_id;
  PyObject *topk_scores;


  if (!PyArg_ParseTuple(args, "O|l|i|i|i|i|i|O|i|i", &py_rep, &ptr_val, &user_id, &item_id, &use_cache,
                        &save_result, &enable_topk, &topk_scores, &topk, &rep_id)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  Makex::REP<Pattern, DataGraph> rep(BuildREP(py_rep));
  std::vector<std::pair<DataGraphVertexConstPtr, DataGraphVertexConstPtr>>
      positive_result, all_result;
  bool cache_flag = use_cache ? true : false;
  auto t_begin = clock();
  int positive_count, all_count;

  int match_flag;

  MyCompare cmp_function = [](std::pair<double, std::map<int, int>> a, std::pair<double, std::map<int, int>> b) { return a.first > b.first; };
  std::priority_queue<std::pair<double, std::map<int, int>>, 
                      std::vector<std::pair<double, std::map<int, int>>>, 
                      MyCompare> topk_min_heap(cmp_function);
  FetchGlobalTopkScoresIntoHeap(topk_scores, topk_min_heap, topk);


  if (enable_topk == 1){
    auto ret =
      Makex::REPMatchBasePTime_U_V(
      rep, data_graph_ptr->data_graph(), data_graph_ptr->ml_model(), user_id, item_id, 
      cache_flag, &positive_result, &all_result, &topk_min_heap, topk, rep_id);
    
  positive_count = ret.first;
  all_count = ret.second;
  
  if (positive_count == -1 || all_count == -1) {
    match_flag = -1;
    } else {
        match_flag = 1;
    }
  }

  if (enable_topk == 0){
    auto ret =
      Makex::REPMatchBasePTime_U_V_ALL_Explnantion(
      rep, data_graph_ptr->data_graph(), data_graph_ptr->ml_model(), user_id, item_id, 
      cache_flag, &positive_result, &all_result, &topk_min_heap, topk);
  positive_count = ret.first;
  all_count = ret.second;
  
  if (positive_count == -1 || all_count == -1) {
    match_flag = -1;
    } else {
        match_flag = 1;
    }
  }

  PyObject *ret = PyList_New(0);

  if (save_result) {
    
    PyObject *positive_match_result = PyList_New(0);
    for (auto &[x_res, y_res] : positive_result) {
      PyObject *ret_pair = PyList_New(0);
      PyList_Append(ret_pair, PyLong_FromLong(x_res->id()));
      PyList_Append(ret_pair, PyLong_FromLong(y_res->id()));
      PyList_Append(positive_match_result, ret_pair);
    }
    PyObject *all_match_result = PyList_New(0);
    for (auto &[x_res, y_res] : all_result) {
      PyObject *ret_pair = PyList_New(0);
      PyList_Append(ret_pair, PyLong_FromLong(x_res->id()));
      PyList_Append(ret_pair, PyLong_FromLong(y_res->id()));
      PyList_Append(all_match_result, ret_pair);
    }
    PyList_Append(ret, positive_match_result);
    PyList_Append(ret, all_match_result);

  } else {

    PyList_Append(ret, PyLong_FromLong(positive_count));
    PyList_Append(ret, PyLong_FromLong(all_count));


    if (enable_topk == 1 || enable_topk == 3) {
      PyObject *latest_topk_heap_info = PyList_New(0);
      while (!topk_min_heap.empty()) {
        auto info = topk_min_heap.top();
        double score = info.first;
        std::map<int, int> temp_id_map = info.second;
        PyObject* py_score = PyFloat_FromDouble(score);
        PyObject* vertex_id_map_dict = PyDict_New();

        for (auto& pair: temp_id_map) {
          PyObject* vertex_id_first = PyLong_FromLong(pair.first);
          PyObject* vertex_id_second = PyLong_FromLong(pair.second);
          PyDict_SetItem(vertex_id_map_dict, vertex_id_first, vertex_id_second);
        }
        PyList_Append(latest_topk_heap_info, PyTuple_Pack(2, py_score, vertex_id_map_dict));
        topk_min_heap.pop();
      }

      PyList_Append(ret, latest_topk_heap_info);
    }
    PyList_Append(ret, PyLong_FromLong(match_flag));
  }

  return ret;
}



static PyObject *CheckHasMatch(PyObject *self, PyObject *args) {
  long ptr_val;
  PyObject *py_rep;
  DataGraphVertexIDType x_id, y_id;
  if (!PyArg_ParseTuple(args, "O|l|i|i", &py_rep, &ptr_val, &x_id, &y_id)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  Makex::REP<Pattern, DataGraph> rep(BuildREP(py_rep));
  bool has_match_flag =
      Makex::CheckHasMatch(rep, data_graph_ptr->data_graph(), x_id, y_id);

  return PyBool_FromLong(has_match_flag);
}
static PyObject *NumOfEncode(PyObject *self, PyObject *args) {
  long ptr_val;
  if (!PyArg_ParseTuple(args, "l", &ptr_val)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  return PyLong_FromLong(data_graph_ptr->NumOfEncode());
}

static PyObject *LabelAttributeMap(PyObject *self, PyObject *args) {
  long ptr_val;
  DataGraphVertexLabelType vertex_label;
  if (!PyArg_ParseTuple(args, "l|i", &ptr_val, &vertex_label)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  PyObject *ret_key_list = PyList_New(0);
  std::set<DataGraphAttrKeyType> key_set;
  for (auto vertex_it = data_graph_ptr->data_graph().VertexBegin(vertex_label);
       !vertex_it.IsDone(); vertex_it++) {
    for (auto attr_it = vertex_it->AttributeBegin(); !attr_it.IsDone();
         attr_it++) {
      DataGraphAttrKeyType attr_key = attr_it->key();
      key_set.insert(attr_key);
    }
  }
  for (auto &it : key_set) {
    PyObject *py_val = PyUnicode_FromString(it.c_str());
    PyList_Append(ret_key_list, py_val);
  }
  return ret_key_list;
}

static PyObject *AdjEncodeList(PyObject *self, PyObject *args) {
  long ptr_val;
  DataGraphVertexLabelType vertex_label;

  if (!PyArg_ParseTuple(args, "l|i", &ptr_val, &vertex_label)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  auto adj_encode_set = data_graph_ptr->AdjEncodeList(vertex_label);
  PyObject *ret_adj_encode_list = PyList_New(0);
  for (const auto &it : adj_encode_set) {
    PyList_Append(ret_adj_encode_list, PyLong_FromLong(it));
  }
  return ret_adj_encode_list;
}
static PyObject *GetAllVertexID(PyObject *self, PyObject *args) {
  long ptr_val;
  if (!PyArg_ParseTuple(args, "l", &ptr_val)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  PyObject *ret_vertex_id_list = PyList_New(0);
  for (auto vertex_it = data_graph_ptr->data_graph().VertexBegin();
       !vertex_it.IsDone(); vertex_it++) {
    PyObject *py_val = PyLong_FromLong(vertex_it->id());
    PyList_Append(ret_vertex_id_list, py_val);
  }
  return ret_vertex_id_list;
}
static PyObject *GetAllEdge(PyObject *self, PyObject *args) {
  long ptr_val;
  if (!PyArg_ParseTuple(args, "l", &ptr_val)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  PyObject *ret_edge_list = PyList_New(0);
  for (auto vertex_it = data_graph_ptr->data_graph().VertexBegin();
       !vertex_it.IsDone(); vertex_it++) {
    for (auto edge_it = vertex_it->OutEdgeBegin(); !edge_it.IsDone();
         edge_it++) {
      PyObject *edge_list = PyList_New(0);
      PyObject *src_val = PyLong_FromLong(vertex_it->id());
      PyList_Append(edge_list, src_val);
      PyObject *dst_val = PyLong_FromLong(edge_it->dst_handle()->id());
      PyList_Append(edge_list, dst_val);
      PyObject *label_val = PyLong_FromLong(edge_it->label());
      PyList_Append(edge_list, label_val);
      PyList_Append(ret_edge_list, edge_list);
    }
  }
  return ret_edge_list;
}
static PyObject *HasOutEdge(PyObject *self, PyObject *args) {
  long ptr_val;
  DataGraphVertexIDType src_vertex_id, dst_vertex_id;
  if (!PyArg_ParseTuple(args, "l|i|i", &ptr_val, &src_vertex_id,
                        &dst_vertex_id)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  DataGraphVertexPtr src_vertex_ptr =
      data_graph_ptr->data_graph().FindVertex(src_vertex_id);
  DataGraphVertexPtr dst_vertex_ptr =
      data_graph_ptr->data_graph().FindVertex(dst_vertex_id);
  if (src_vertex_ptr->HasOutEdge(dst_vertex_ptr) > 0) {
    return PyLong_FromLong(1);
  } else {
    return PyLong_FromLong(0);
  }
}
static PyObject *GetVertexAttribute(PyObject *self, PyObject *args) {
  long ptr_val;
  DataGraphVertexIDType vertex_id;
  if (!PyArg_ParseTuple(args, "l|i", &ptr_val, &vertex_id)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  DataGraphVertexPtr vertex_ptr =
      data_graph_ptr->data_graph().FindVertex(vertex_id);
  PyObject *ret_attr_dict = PyDict_New();
  for (auto attr_it = vertex_ptr->AttributeBegin(); !attr_it.IsDone();
       attr_it++) {
    DataGraphAttrKeyType attr_key = attr_it->key();
    GUNDAM::BasicDataType value_type = attr_it->value_type();
    switch (value_type) {
      case GUNDAM::BasicDataType::kTypeInt: {
        int val = attr_it->template value<int>();
        PyObject *py_val = PyLong_FromLong(val);
        PyDict_SetItemString(ret_attr_dict, attr_key.c_str(), py_val);
        break;
      }

      case GUNDAM::BasicDataType::kTypeInt64: {
        int64_t val = attr_it->template value<int64_t>();
        PyObject *py_val = PyLong_FromLongLong(val);
        PyDict_SetItemString(ret_attr_dict, attr_key.c_str(), py_val);
        break;
      }

      case GUNDAM::BasicDataType::kTypeDouble: {
        double val = attr_it->template value<double>();
        PyObject *py_val = PyFloat_FromDouble(val);
        PyDict_SetItemString(ret_attr_dict, attr_key.c_str(), py_val);
        break;
      }

      case GUNDAM::BasicDataType::kTypeString: {
        std::string val = attr_it->template value<std::string>();
        PyObject *py_val = PyUnicode_FromString(val.c_str());
        PyDict_SetItemString(ret_attr_dict, attr_key.c_str(), py_val);
        break;
      }
      default:
        break;
    }
  }

  return ret_attr_dict;
}
static PyObject *GetEncodeMap(PyObject *self, PyObject *args) {
  long ptr_val;
  if (!PyArg_ParseTuple(args, "l", &ptr_val)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  auto inv_encode_map = data_graph_ptr->inv_encode_hash_map();
  PyObject *ret_encode_map = PyList_New(0);
  for (auto &[pos, val] : inv_encode_map) {
    auto &[edge_label, dst_label] = val;
    PyObject *label_pair = PyList_New(0);
    PyList_Append(label_pair, PyLong_FromLong(edge_label));
    PyList_Append(label_pair, PyLong_FromLong(dst_label));
    PyList_Append(ret_encode_map, label_pair);
  }
  return ret_encode_map;
}

static PyObject *PatternMatch(PyObject *self, PyObject *args) {
  long ptr_val;
  PyObject *pattern_vertex_list, *pattern_edge_list;
  int max_result;
  int sample_pair;
  if (!PyArg_ParseTuple(args, "O|O|l|i|i", &pattern_vertex_list,
                        &pattern_edge_list, &ptr_val, &sample_pair,
                        &max_result)) {
    printf("Input error!\n");

    Py_INCREF(Py_None);
    return Py_None;
  }
  if (!PyList_Check(pattern_vertex_list)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

  if (!PyList_Check(pattern_edge_list)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)ptr_val;
  Pattern pattern(BuildPattern(pattern_vertex_list, pattern_edge_list));
  using MatchMap = std::map<PatternVertexPtr, DataGraphVertexPtr>;
  using CandidateSet =
      std::map<PatternVertexPtr, std::vector<DataGraphVertexPtr>>;
  using MatchContainer = std::vector<MatchMap>;
  MatchContainer match_result;
  CandidateSet candidate_set;
  if (!GUNDAM::_dp_iso::InitCandidateSet<GUNDAM::MatchSemantics::kIsomorphism>(
          pattern, data_graph_ptr->data_graph(), candidate_set)) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  if (!GUNDAM::_dp_iso::RefineCandidateSet(
          pattern, data_graph_ptr->data_graph(), candidate_set)) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  auto x_ptr = pattern.FindVertex(1);
  auto y_ptr = pattern.FindVertex(2);
  std::vector<std::pair<DataGraphVertexPtr, DataGraphVertexPtr>> positive_pair,
      negative_pair;

  for (auto x_candidate : candidate_set.find(x_ptr)->second) {
    for (auto y_candidate : candidate_set.find(y_ptr)->second) {
      if (data_graph_ptr->ml_model().size() > 0 &&
          !data_graph_ptr->ml_model().count(
              std::make_pair(x_candidate, y_candidate))) {
        continue;
      }

      bool positive_flag = x_candidate->HasOutEdge(y_candidate) > 0;
      if (positive_flag)
        positive_pair.emplace_back(x_candidate, y_candidate);
      else
        negative_pair.emplace_back(x_candidate, y_candidate);
    }
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> positive_dis(0, positive_pair.size() - 1);
  std::uniform_int_distribution<> negative_dis(0, negative_pair.size() - 1);
  int positive_count = 0, negative_count = 0;
  for (int t = 1;
       t <= std::min(sample_pair,
                     (int)std::min(negative_pair.size(), positive_pair.size()));
       t++) 
  {
    int positive_pos = positive_dis(gen);
    int negative_pos = negative_dis(gen);
    DataGraphVertexPtr target_x_ptr = positive_pair[positive_pos].first;
    DataGraphVertexPtr target_y_ptr = positive_pair[positive_pos].second;
    MatchMap match_state;

    match_state.emplace(x_ptr, target_x_ptr);
    match_state.emplace(y_ptr, target_y_ptr);
    CandidateSet temp_candidate_set(candidate_set);
    MatchContainer single_match_result;

    auto user_callback = [&single_match_result,
                          &max_result](MatchMap &match_state) {
      single_match_result.push_back(match_state);
      if (single_match_result.size() == max_result) return false;
      return true;
    };
    auto prune_callback = [](auto &match_state) { return false; };
    auto t_begin_sample = clock();

    GUNDAM::DPISO<GUNDAM::MatchSemantics::kHomomorphism, Pattern, DataGraph>(
        pattern, data_graph_ptr->data_graph(), temp_candidate_set, match_state,
        user_callback, prune_callback, 1.0);

    match_result.insert(match_result.end(), single_match_result.begin(),
                        single_match_result.end());
    positive_count += single_match_result.size();

    target_x_ptr = negative_pair[negative_pos].first;
    target_y_ptr = negative_pair[negative_pos].second;
    match_state.clear();
    single_match_result.clear();

    temp_candidate_set = candidate_set;
    match_state.emplace(x_ptr, target_x_ptr);
    match_state.emplace(y_ptr, target_y_ptr);
    t_begin_sample = clock();
    GUNDAM::DPISO<GUNDAM::MatchSemantics::kHomomorphism, Pattern, DataGraph>(
        pattern, data_graph_ptr->data_graph(), temp_candidate_set, match_state,
        user_callback, prune_callback, 1.0);
    t_end_sample = clock();

    match_result.insert(match_result.end(), single_match_result.begin(),
                        single_match_result.end());
    negative_count += single_match_result.size();
  }
  t_end = clock();

  PyObject *ret_match_result = PyList_New(0);

  for (auto &match_state : match_result) {
    PyObject *ret_match_state = PyList_New(0);
    for (auto &[query_ptr, target_ptr] : match_state) {
      PyObject *ret_match_pair = PyList_New(0);
      PyList_Append(ret_match_pair, PyLong_FromLong(query_ptr->id()));
      PyList_Append(ret_match_pair, PyLong_FromLong(target_ptr->id()));
      PyList_Append(ret_match_state, ret_match_pair);
    }
    PyList_Append(ret_match_result, ret_match_state);
  }
  return ret_match_result;
}

static PyObject *BuildStarMiner(PyObject *self, PyObject *args) {
  long ptr_val;
  DataGraphVertexLabelType root_label;
  if (!PyArg_ParseTuple(args, "l|i", &ptr_val, &root_label)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)(ptr_val);
  StarMiner *star_miner_ptr = new StarMiner(data_graph_ptr, root_label);
  return PyLong_FromLong(long(star_miner_ptr));
}
static PyObject *MineStar(PyObject *self, PyObject *args) {
  long ptr_val;
  int32_t root_max_outdegree;
  int32_t max_len_of_path;
  int32_t each_vertex_remain_star_number;
  if (!PyArg_ParseTuple(args, "l|i|i|i", &ptr_val, &root_max_outdegree,
                        &max_len_of_path, &each_vertex_remain_star_number)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

  StarMiner *star_miner_ptr = (StarMiner *)(ptr_val);
  typename StarMiner::EncodeStarContainer sample_star;
  star_miner_ptr->GetStar(root_max_outdegree, max_len_of_path,
                          each_vertex_remain_star_number, sample_star);
  PyObject *ret_star_list = PyList_New(0);

  for (const auto &single_star : sample_star) {
    PyObject *py_single_star = PyList_New(0);
    PyList_Append(py_single_star, PyLong_FromLong(single_star.size()));
    for (const auto &ele : single_star) {
      PyList_Append(py_single_star, PyLong_FromLong(ele));
    }
    PyList_Append(py_single_star, PyFloat_FromDouble(0.0));
    PyList_Append(py_single_star, PyFloat_FromDouble(0.0));
    PyList_Append(py_single_star, PyFloat_FromDouble(0.0));
    PyList_Append(ret_star_list, py_single_star);
  }

  return ret_star_list;
}
static PyObject *TestWithLargeFlag(PyObject *self, PyObject *args) {
  long ptr_val;
  char *test_file, *test_negative_file;
  PyObject *py_rep_list, *score_list;
  double delta_l, delta_r;
  int user_offset;
  char *test_result_file;
  int process_num;
  if (!PyArg_ParseTuple(args, "i|l|d|d|i|s|s|s|O|O|", &process_num, &ptr_val, &delta_l, &delta_r,
                        &user_offset, &test_file, &test_negative_file,
                        &test_result_file, &py_rep_list, &score_list)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if (!PyList_Check(py_rep_list)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if (!PyList_Check(score_list)) {
    printf("Input error!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  int sz = PyList_Size(py_rep_list);
  DataGraphWithInformation *data_graph_ptr =
      (DataGraphWithInformation *)ptr_val;
  std::vector<Makex::REP<Pattern, DataGraph>> rep_list;
  std::vector<double> rep_score_list;

  for (size_t i = 0; i < sz; i++) {
    Makex::REP<Pattern, DataGraph> rep(
        BuildREP(PyList_GetItem(py_rep_list, i)));
    rep_list.push_back(rep);
  }


  for (size_t i = 0; i < sz; i++) {
    double rep_score = PyFloat_AsDouble(PyList_GetItem(score_list, i));
    rep_score_list.push_back(rep_score);
  }
  std::ifstream negative_file_stream(test_negative_file);
  std::map<std::pair<DataGraphVertexIDType, DataGraphVertexIDType>, double>
      ml_predict_rating;
  std::map<DataGraphVertexIDType,
           std::vector<std::pair<DataGraphVertexIDType, double>>>
      test_negative_rating;
  using MLModel =
      std::map<std::pair<DataGraphVertexConstPtr, DataGraphVertexConstPtr>,
               std::pair<int, int>>;
  MLModel ml_model;
  while (negative_file_stream) {
    DataGraphVertexIDType user_id, item_id;
    double ml_predict_score;
    negative_file_stream >> user_id >> item_id >> ml_predict_score;
    user_id += user_offset;
    DataGraphVertexConstPtr user_ptr =
        data_graph_ptr->data_graph().FindVertex(user_id);
    DataGraphVertexConstPtr item_ptr =
        data_graph_ptr->data_graph().FindVertex(item_id);
    if (ml_predict_score >= delta_l && ml_predict_score <= delta_r) {
      ml_model[std::make_pair(user_ptr, item_ptr)] = std::make_pair(1, 1);
    }
  }
  std::ofstream test_result_stream(test_result_file);
  std::map<std::pair<DataGraphVertexConstPtr, DataGraphVertexConstPtr>, double>
      predict_score;
  omp_lock_t predict_score_lock;
  omp_init_lock(&predict_score_lock);
  omp_set_num_threads(process_num);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < sz; i++) {
    std::vector<std::pair<DataGraphVertexConstPtr, DataGraphVertexConstPtr>>
        positive_result, all_result;

    std::set<int> pivot_match;
    int who_is_pivot_label = 0;
    std::set<int> pivot_match_;
    int who_is_pivot_label_ = 0;
    int rep_id = i;
    auto ret = Makex::REPMatchBasePTime_Pivot_Match(
        rep_list[i], data_graph_ptr->data_graph(), ml_model, false,&pivot_match, who_is_pivot_label, &pivot_match_, who_is_pivot_label_,
        &positive_result, &all_result, rep_id);
    
    omp_set_lock(&predict_score_lock);
    for (auto &it : all_result) {
      predict_score[it] = std::max(predict_score[it], rep_score_list[i]);
    }
    for (auto &it : positive_result) {
      predict_score[it] = std::max(predict_score[it], rep_score_list[i]);
    }
    omp_unset_lock(&predict_score_lock);
  }

  test_result_stream << predict_score.size() << std::endl;
  std::cout << "test result size: " << predict_score.size() << std::endl;
  for (auto &it : predict_score) {
    DataGraphVertexIDType user_id = it.first.first->id();
    DataGraphVertexIDType item_id = it.first.second->id();
    double rep_predict_rating = it.second;
    test_result_stream << user_id - user_offset << " " << item_id << " "
                       << rep_predict_rating << std::endl;
  }
  return Py_None;
}
static PyMethodDef PyExtMethods[] = {
    {"ReadDataGraph", ReadDataGraph, METH_VARARGS, "ReadDataGraph"},
    {"ReadML", ReadML, METH_VARARGS, "ReadML"},
    {"NumOfEncode", NumOfEncode, METH_VARARGS, "NumOfEncode"},
    {"AdjEncodeList", AdjEncodeList, METH_VARARGS, "AdjEncodeList"},
    {"GetVertexAttribute", GetVertexAttribute, METH_VARARGS,
     "GetVertexAttribute"},
    {"PositivePairNum", PositivePairNum, METH_VARARGS, "PositivePairNum"},
    {"NegativePairNum", NegativePairNum, METH_VARARGS, "NegativePairNum"},
    {"MLModelSize", MLModelSize, METH_VARARGS, "MLModelSize"},
    {"GetEncodeMap", GetEncodeMap, METH_VARARGS, "GetEncodeMap"},
    {"HasOutEdge", HasOutEdge, METH_VARARGS, "HasOutEdge"},
    {"REPMatch", REPMatch, METH_VARARGS, "REPMatch"},
    {"REPMatch_U_V", REPMatch_U_V, METH_VARARGS, "REPMatch_U_V"},
    {"CheckHasMatch", CheckHasMatch, METH_VARARGS, "CheckHasMatch"},
    {"PatternMatch", PatternMatch, METH_VARARGS, "PatternMatch"},
    {"BuildStarMiner", BuildStarMiner, METH_VARARGS, "BuildStarMiner"},
    {"MineStar", MineStar, METH_VARARGS, "MineStar"},
    {"TestWithLargeFlag", TestWithLargeFlag, METH_VARARGS, "TestWithLargeFlag"},
    {"GetAllVertexID", GetAllVertexID, METH_VARARGS, "GetAllVertexID"},
    {"GetAllEdge", GetAllEdge, METH_VARARGS, "GetAllEdge"},
    {"LabelAttributeMap", LabelAttributeMap, METH_VARARGS, "LabelAttributeMap"},
    {NULL, NULL, 0, NULL}};



static struct PyModuleDef pyMakex = {PyModuleDef_HEAD_INIT, "pyMakex", "",
                                       -1, PyExtMethods};


PyMODINIT_FUNC PyInit_pyMakex(void) { return PyModule_Create(&pyMakex); }
