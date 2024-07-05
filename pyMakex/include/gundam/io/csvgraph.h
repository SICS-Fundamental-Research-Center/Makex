#ifndef _GUNDAM_IO_CSVGRAPH_H
#define _GUNDAM_IO_CSVGRAPH_H

#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include "gundam/component/generator.h"
#include "gundam/data_type/datatype.h"
#include "gundam/io/rapidcsv.h"
#include "gundam/type_getter/edge_handle.h"
#include "gundam/type_getter/vertex_handle.h"

namespace GUNDAM {

template <typename AttributeKeyType>
bool GetAttributeInfo(
    const std::vector<std::string>& col_name,
    std::vector<std::pair<AttributeKeyType, enum BasicDataType>>& attr_info) {
  attr_info.clear();

  for (const auto& str : col_name) {
    std::string key_str, type_str;

    bool flag = false;
    for (const auto& c : str) {
      if (c == ':') {
        flag = true;
      } else if (!flag) {
        key_str.push_back(c);
      } else {
        type_str.push_back(c);
      }
    }

    std::stringstream s0(key_str);
    AttributeKeyType attr_key;
    s0 >> attr_key;
    std::stringstream s1;
    std::string str1;
    s1 << attr_key;
    s1 >> str1;
    if (str1 != key_str) {
      return false;
    }

    enum BasicDataType value_type = StringToEnum(type_str.c_str());
    if (value_type == BasicDataType::kTypeUnknown) {
      return false;
    }
    attr_info.emplace_back(std::move(attr_key), std::move(value_type));
  }

  return true;
}

template <bool has_attribute, class GraphType, class VertexEdgePtr,
          class AttributeKeyType,
          typename std::enable_if<!has_attribute, bool>::type = false>
inline bool ReadAttribues(
    GraphType& graph, VertexEdgePtr& vertex_edge_ptr,
    rapidcsv::Document& vertex_edge_file,
    const std::vector<std::pair<AttributeKeyType, enum BasicDataType>>&
        attr_info,
    size_t col_begin, size_t row) {
  return true;
}

template <bool has_attribute, class GraphType, class VertexEdgePtr,
          class AttributeKeyType,
          typename std::enable_if<has_attribute, bool>::type = false>
inline bool ReadAttribues(
    GraphType& graph, VertexEdgePtr& vertex_edge_ptr,
    rapidcsv::Document& vertex_edge_file,
    const std::vector<std::pair<AttributeKeyType, enum BasicDataType>>&
        attr_info,
    size_t col_begin, size_t row) {
  for (size_t col = col_begin; col < attr_info.size(); col++) {
    const AttributeKeyType& attr_key = attr_info[col].first;
    const enum BasicDataType& value_type = attr_info[col].second;

    switch (value_type) {
      case BasicDataType::kTypeString: {
        auto cell = vertex_edge_file.GetCellNew<std::string>(col, row);
        if (cell.second || cell.first.empty()) continue;
        vertex_edge_ptr->AddAttribute(attr_key, cell.first);
        break;
      }
      case BasicDataType::kTypeInt: {
        auto cell = vertex_edge_file.GetCellNew<int>(col, row);
        if (cell.second) continue;
        vertex_edge_ptr->AddAttribute(attr_key, cell.first);
        break;
      }
      case BasicDataType::kTypeInt64: {
        auto cell = vertex_edge_file.GetCellNew<int64_t>(col, row);
        if (cell.second) continue;
        vertex_edge_ptr->AddAttribute(attr_key, cell.first);
        break;
      }
      case BasicDataType::kTypeFloat: {
        auto cell = vertex_edge_file.GetCellNew<float>(col, row);
        if (cell.second) continue;
        vertex_edge_ptr->AddAttribute(attr_key, cell.first);
        break;
      }
      case BasicDataType::kTypeDouble: {
        auto cell = vertex_edge_file.GetCellNew<double>(col, row);
        if (cell.second) continue;
        vertex_edge_ptr->AddAttribute(attr_key, cell.first);
        break;
      }
      case BasicDataType::kTypeDateTime: {
        auto cell = vertex_edge_file.GetCellNew<std::string>(col, row);
        if (cell.second || cell.first.empty()) continue;
        DateTime date_time(cell.first.c_str());
        vertex_edge_ptr->AddAttribute(attr_key, date_time);
        break;
      }
      case BasicDataType::kTypeUnknown:
      default:
        return false;
    }
  }
  return true;
}

template <bool read_attr = true, typename GraphIDType, typename GraphType,
          typename ReadVertexCallback>
int ReadCSVVertexSetFileWithCallback(
    const std::string& v_set_file, std::vector<GraphType>& graph_set,
    std::map<GraphIDType, size_t>& graph_id_to_graph_set_idx,
    ReadVertexCallback callback) {
  std::cout << "read vertex set" << std::endl;
  graph_id_to_graph_set_idx.clear();
  graph_set.clear();
  // read vertex set file(csv)
  // file format: (vertex_id,label_id,graph_id,...)
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using VertexAttributeKeyType =
      typename GraphType::VertexType::AttributeKeyType;
  using VertexHandleType = typename GUNDAM::VertexHandle<GraphType>::type;

  try {
    std::cout << v_set_file << std::endl;

    rapidcsv::Document vertex_set_file(v_set_file,
                                       rapidcsv::LabelParams(0, -1));

    // phase column names
    std::vector<std::string> col_name = vertex_set_file.GetColumnNames();
    std::vector<std::pair<VertexAttributeKeyType, enum BasicDataType>>
        attr_info;

    // erase graph_id column,because graph_id is not attribute
    for (auto it = col_name.begin(); it != col_name.end();) {
      if ((*it).find("graph_id") != (*it).npos) {
        it = col_name.erase(it);
        continue;
      }
      it++;
    }
    if (!GetAttributeInfo(col_name, attr_info)) {
      std::cout << "Attribute key type is not correct!" << std::endl;
      return -1;
    }

    size_t col_num = attr_info.size();
    // check col num >= 2
    if (col_num < 2 || attr_info[0].first != "vertex_id" ||
        attr_info[1].first != "label_id") {
      std::cout << "Vertex file does not have vertex_id or label_id!"
                << std::endl;
      return -1;
    }
    // check attributes
    if constexpr (read_attr && !GraphType::vertex_has_attribute) {
      if (col_num >= 3) {
        std::cout << "Vertex file has attribute but graph does not support!"
                  << std::endl;
        return -1;
      }
    }

    const std::vector<VertexIDType> vertex_id =
        vertex_set_file.GetColumn<VertexIDType>(0);
    const std::vector<VertexLabelType> label_id =
        vertex_set_file.GetColumn<VertexLabelType>(1);
    const std::vector<GraphIDType> graph_id =
        vertex_set_file.GetColumn<GraphIDType>(2);

    int count_success = 0;
    int count_fail = 0;
    size_t sz = vertex_id.size();
    for (size_t row = 0; row < sz; row++) {
      GraphIDType id = graph_id[row];
      graph_id_to_graph_set_idx.emplace(id, graph_id_to_graph_set_idx.size());
    }
    graph_set.resize(graph_id_to_graph_set_idx.size());
    for (size_t row = 0; row < sz; row++) {
      assert(row < graph_id.size());
      GraphIDType id = graph_id[row];
      auto graph_id_to_graph_set_idx_it = graph_id_to_graph_set_idx.find(id);
      assert(graph_id_to_graph_set_idx_it != graph_id_to_graph_set_idx.end());
      size_t graph_idx = graph_id_to_graph_set_idx_it->second;
      assert(graph_idx >= 0 && graph_idx < graph_set.size());
      auto& graph_ref = graph_set[graph_idx];
      auto [vertex_handle, add_ret] =
          graph_ref.AddVertex(vertex_id[row], label_id[row]);

      if (add_ret) {
        // vertex added successfully
        if constexpr (read_attr) {
          add_ret = ReadAttribues<GraphType::vertex_has_attribute>(
              graph_ref, vertex_handle, vertex_set_file, attr_info, 2, row);
        }
      }

      if (!add_ret) {
        // add vertex failed, what ever in AddVertex or ReadAttribues
        count_fail++;
        continue;
      }

      if constexpr (!std::is_null_pointer_v<ReadVertexCallback>) {
        if (!callback(vertex_handle)) return -2;
      }
      count_success++;
    }
    if (count_fail > 0) {
      std::cout << "Failed: " << count_fail << std::endl;
    }
    return count_success;
  } catch (...) {
    return -1;
  }
}

template <bool read_attr = true, class GraphType, class ReadVertexCallback>
int ReadCSVVertexFileWithCallback(const std::string& v_file, GraphType& graph,
                                  ReadVertexCallback callback) {
  // read vertex file(csv)
  // file format: (vertex_id,label_id,......)
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using VertexAttributeKeyType =
      typename GraphType::VertexType::AttributeKeyType;
  using VertexHandleType = typename GUNDAM::VertexHandle<GraphType>::type;

  try {
    std::cout << v_file << std::endl;

    rapidcsv::Document vertex_file(v_file, rapidcsv::LabelParams(0, -1));

    // phase column names
    std::vector<std::string> col_name = vertex_file.GetColumnNames();
    std::vector<std::pair<VertexAttributeKeyType, enum BasicDataType>>
        attr_info;
    if (!GetAttributeInfo(col_name, attr_info)) {
      std::cout << "Attribute key type is not correct!" << std::endl;
      return -1;
    }

    size_t col_num = attr_info.size();

    for (auto& attribute : attr_info) {
        std::string attr_name = attribute.first;
        std::cout << "Attributes name: " << attr_name << std::endl;
      }
    
    // check col num >= 2
    if (col_num < 2 || attr_info[0].first != "vertex_id" ||
        attr_info[1].first != "label_id") {
      std::cout << "Vertex file does not have vertex_id or label_id!"
                << std::endl;
      return -1;
    }
    // check attributes
    if constexpr (read_attr && !GraphType::vertex_has_attribute) {
      if (col_num >= 3) {
        std::cout << "vertex file has attribute but graph does not support!"
                  << std::endl;
        return -1;
      }
    }

    const std::vector<VertexIDType> vertex_id =
        vertex_file.GetColumn<VertexIDType>(0);
    const std::vector<VertexLabelType> label_id =
        vertex_file.GetColumn<VertexLabelType>(1);

    int count_success = 0;
    int count_fail = 0;
    size_t sz = vertex_id.size();
    std::cout << "vertex_id size: " << sz << std::endl;
    for (size_t row = 0; row < sz; row++) {

      auto [vertex_handle, r] = graph.AddVertex(vertex_id[row], label_id[row]);
      if (r) {
        if constexpr (read_attr) {
          r = ReadAttribues<GraphType::vertex_has_attribute>(
              graph, vertex_handle, vertex_file, attr_info, 2, row);
        }
      }

      if (r) {
        if constexpr (!std::is_null_pointer_v<ReadVertexCallback>) {
          if (!callback(vertex_handle)) return -2;
        }
        ++count_success;
      } else {
        ++count_fail;
      }
    }
    if (count_fail > 0) {
      std::cout << "Failed: " << count_fail << std::endl;
    }
    return count_success;
  } catch (...) {
    return -1;
  }
}

template <bool read_attr = true, typename GraphIDType, typename GraphType,
          typename ReadEdgeCallback>
int ReadCSVEdgeSetFileWithCallback(
    const std::string& e_set_file, std::vector<GraphType>& graph_set,
    const std::map<GraphIDType, size_t>& graph_id_to_graph_set_idx,
    ReadEdgeCallback callback) {
  std::cout << "read edge set" << std::endl;
  assert(graph_set.size() == graph_id_to_graph_set_idx.size());
  // read edge set file(csv)
  // file format: (edge_id,source_id,target_id,label_id,graph_id,......)
  using VertexIDType = typename GraphType::VertexType::IDType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;
  using EdgeAttributeKeyType = typename GraphType::EdgeType::AttributeKeyType;
  using EdgeHandleType = typename GUNDAM::EdgeHandle<GraphType>::type;

  try {
    std::cout << e_set_file << std::endl;

    rapidcsv::Document edge_set_file(e_set_file, rapidcsv::LabelParams(0, -1));

    // phase column names
    std::vector<std::string> col_name = edge_set_file.GetColumnNames();

    std::vector<std::pair<EdgeAttributeKeyType, enum BasicDataType>> attr_info;

    for (auto it = col_name.begin(); it != col_name.end();) {
      if ((*it).find("graph_id") != (*it).npos) {
        it = col_name.erase(it);
      } else {
        it++;
      }
    }

    if (!GetAttributeInfo(col_name, attr_info)) {
      std::cout << "Attribute key type is not correct!" << std::endl;
      return -1;
    }

    size_t col_num = attr_info.size();
    // check col_num >= 4
    if (col_num < 4 || attr_info[0].first != "edge_id" ||
        attr_info[1].first != "source_id" ||
        attr_info[2].first != "target_id" || attr_info[3].first != "label_id") {
      std::cout << "Edge file is not correct!(col num must >=4)" << std::endl;
      return -1;
    }
    if constexpr (read_attr && !GraphType::edge_has_attribute) {
      if (col_num >= 5) {
        std::cout << "Edge file has attribute but graph does not support!"
                  << std::endl;
        return -1;
      }
    }

    const std::vector<EdgeIDType> edge_id =
        edge_set_file.GetColumn<EdgeIDType>(0);
    const std::vector<VertexIDType> source_id =
        edge_set_file.GetColumn<VertexIDType>(1);
    const std::vector<VertexIDType> target_id =
        edge_set_file.GetColumn<VertexIDType>(2);
    const std::vector<EdgeLabelType> label_id =
        edge_set_file.GetColumn<EdgeLabelType>(3);
    const std::vector<GraphIDType> graph_id =
        edge_set_file.GetColumn<GraphIDType>(4);

    int count_success = 0;
    int count_fail = 0;
    size_t sz = edge_id.size();
    for (size_t row = 0; row < sz; row++) {
      GraphIDType id = graph_id[row];
      auto graph_idx_it = graph_id_to_graph_set_idx.find(id);
      if (graph_idx_it == graph_id_to_graph_set_idx.end()) {
        // this graph_id is not contained in graph_id_to_graph_set_idx
        // i.e. does not contained in the v_set_file
        count_fail++;
        continue;
      }

      size_t graph_idx = graph_idx_it->second;
      assert(graph_idx >= 0 && graph_idx < graph_set.size());
      auto& graph_ref = graph_set[graph_idx];
      auto [edge_handle, add_ret] = graph_ref.AddEdge(
          source_id[row], target_id[row], label_id[row], edge_id[row]);
      if (add_ret) {
        if constexpr (read_attr) {
          add_ret = ReadAttribues<GraphType::edge_has_attribute>(
              graph_ref, edge_handle, edge_set_file, attr_info, 4, row);
        }
      }

      if (!add_ret) {
        count_fail++;
        continue;
      }

      if constexpr (!std::is_null_pointer_v<ReadEdgeCallback>) {
        if (!callback(edge_handle)) {
          return -2;
        }
      }
      ++count_success;
    }
    if (count_fail > 0) {
      std::cout << "Failed: " << count_fail << std::endl;
    }
    return count_success;
  } catch (...) {
    return -1;
  }
}

template <bool read_attr = true, class GraphType, class ReadEdgeCallback>
int ReadCSVEdgeFileWithCallback(const std::string& e_file, GraphType& graph,
                                ReadEdgeCallback callback) {
  // read edge file(csv)
  // file format: (edge_id,source_id,target_id,label_id,......)
  using VertexIDType = typename GraphType::VertexType::IDType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;
  using EdgeAttributeKeyType = typename GraphType::EdgeType::AttributeKeyType;
  using EdgeHandleType = typename GUNDAM::EdgeHandle<GraphType>::type;
  try {
    std::cout << e_file << std::endl;
    rapidcsv::Document edge_file(e_file, rapidcsv::LabelParams(0, -1));

    // phase column names
    std::vector<std::string> col_name = edge_file.GetColumnNames();
    std::vector<std::pair<EdgeAttributeKeyType, enum BasicDataType>> attr_info;
    if (!GetAttributeInfo(col_name, attr_info)) {
      std::cout << "Attribute key type is not correct!" << std::endl;
      return -1;
    }

    size_t col_num = attr_info.size();
    // check col_num >= 4
    if (col_num < 4 || attr_info[0].first != "edge_id" ||
        attr_info[1].first != "source_id" ||
        attr_info[2].first != "target_id" || attr_info[3].first != "label_id") {
      std::cout << "Edge file is not correct!(col num must >=4)" << std::endl;
      return -1;
    }
    if constexpr (read_attr && !GraphType::edge_has_attribute) {
      if (col_num >= 5) {
        std::cout << "Edge file has attribute but graph does not support!"
                  << std::endl;
        return -1;
      }
    }


    const std::vector<EdgeIDType> edge_id = edge_file.GetColumn<EdgeIDType>(0);
    const std::vector<VertexIDType> source_id =
        edge_file.GetColumn<VertexIDType>(1);
    const std::vector<VertexIDType> target_id =
        edge_file.GetColumn<VertexIDType>(2);
    const std::vector<EdgeLabelType> label_id =
        edge_file.GetColumn<EdgeLabelType>(3);

    int count_success = 0;
    int count_fail = 0;
    size_t sz = edge_id.size();
    for (size_t row = 0; row < sz; row++) {
      bool r;
      EdgeHandleType edge_handle;
      std::tie(edge_handle, r) = graph.AddEdge(source_id[row], target_id[row],
                                               label_id[row], edge_id[row]);
      if (r) {
        if constexpr (read_attr) {
          r = ReadAttribues<GraphType::edge_has_attribute>(
              graph, edge_handle, edge_file, attr_info, 4, row);
        }
      }

      if (r) {
        if constexpr (!std::is_null_pointer_v<ReadEdgeCallback>) {
          if (!callback(edge_handle)) return -2;
        }
        ++count_success;
      } else {
        ++count_fail;
      }
    }
    if (count_fail > 0) {
      std::cout << "Failed: " << count_fail << std::endl;
    }
    return count_success;
  } catch (...) {
    return -1;
  }
}

template <class GraphType, class ReadVertexCallback, class ReadEdgeCallback>
int ReadCSVGraphSetWithCallback(std::vector<GraphType>& graph_set,
                                std::vector<std::string>& graph_name_set,
                                const std::vector<std::string>& v_list,
                                const std::vector<std::string>& e_list,
                                ReadVertexCallback rv_callback,
                                ReadEdgeCallback re_callback) {
  graph_set.clear();

  std::map<std::string, size_t> graph_id_to_graph_set_idx;
  int count_v = 0;
  for (const auto& v_file : v_list) {
    int res = ReadCSVVertexSetFileWithCallback(
        v_file, graph_set, graph_id_to_graph_set_idx, rv_callback);
    if (res < 0) return res;
    count_v += res;
  }

  int count_e = 0;
  for (const auto& e_file : e_list) {
    int res = ReadCSVEdgeSetFileWithCallback(
        e_file, graph_set, graph_id_to_graph_set_idx, re_callback);
    if (res < 0) return res;
    count_e += res;
  }

  std::cout << " Vertex: " << count_v << std::endl;
  std::cout << "   Edge: " << count_e << std::endl;

  assert(graph_set.size() == graph_id_to_graph_set_idx.size());
  graph_name_set.resize(graph_set.size());

  for (const auto& graph_id_idx_pair : graph_id_to_graph_set_idx) {
    assert(graph_id_idx_pair.second >= 0 &&
           graph_id_idx_pair.second < graph_name_set.size());
    graph_name_set[graph_id_idx_pair.second] = graph_id_idx_pair.first;
  }

  return count_v + count_e;
}

template <class GraphType, class ReadVertexCallback, class ReadEdgeCallback>
int ReadCSVGraphSetWithCallback(std::vector<GraphType>& graph_set,
                                const std::vector<std::string>& v_list,
                                const std::vector<std::string>& e_list,
                                ReadVertexCallback rv_callback,
                                ReadEdgeCallback re_callback) {
  std::vector<std::string> graph_name_set;
  return ReadCSVGraphSetWithCallback(graph_set, graph_name_set, v_list, e_list,
                                     rv_callback, re_callback);
}

template <class GraphType, class ReadVertexCallback, class ReadEdgeCallback>
int ReadCSVGraphWithCallback(GraphType& graph,
                             const std::vector<std::string>& v_list,
                             const std::vector<std::string>& e_list,
                             ReadVertexCallback rv_callback,
                             ReadEdgeCallback re_callback) {
  graph.Clear();
  int count_v = 0;

  for (const auto& v_file : v_list) {
    int res = ReadCSVVertexFileWithCallback(v_file, graph, rv_callback);
    if (res < 0) return res;
    count_v += res;
  }

  int count_e = 0;
  for (const auto& e_file : e_list) {
    int res = ReadCSVEdgeFileWithCallback(e_file, graph, re_callback);
    if (res < 0) return res;
    count_e += res;
  }
  return count_v + count_e;
}

template <class GraphType>
inline int ReadCSVGraphSet(std::vector<GraphType>& graph_set,
                           const std::vector<std::string>& v_list,
                           const std::vector<std::string>& e_list) {
  return ReadCSVGraphSetWithCallback(graph_set, v_list, e_list, nullptr,
                                     nullptr);
}

template <class GraphType>
inline int ReadCSVGraphSet(std::vector<GraphType>& graph_set,
                           std::vector<std::string>& graph_name_set,
                           const std::vector<std::string>& v_list,
                           const std::vector<std::string>& e_list) {
  return ReadCSVGraphSetWithCallback(graph_set, graph_name_set, v_list, e_list,
                                     nullptr, nullptr);
}

template <class GraphType>
inline int ReadCSVGraph(GraphType& graph,
                        const std::vector<std::string>& v_list,
                        const std::vector<std::string>& e_list) {
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;

  return ReadCSVGraphWithCallback(graph, v_list, e_list, nullptr, nullptr);
}

template <class GraphType>
inline int ReadCSVGraphSet(std::vector<GraphType>& graph_set,
                           const std::string& v_file,
                           const std::string& e_file) {
  std::vector<std::string> v_list, e_list;
  v_list.push_back(v_file);
  e_list.push_back(e_file);
  return ReadCSVGraphSet(graph_set, v_list, e_list);
}

template <class GraphType>
inline int ReadCSVGraphSet(std::vector<GraphType>& graph_set,
                           std::vector<std::string>& graph_name_set,
                           const std::string& v_file,
                           const std::string& e_file) {
  std::vector<std::string> v_list, e_list;
  v_list.push_back(v_file);
  e_list.push_back(e_file);
  return ReadCSVGraphSet(graph_set, graph_name_set, v_list, e_list);
}

template <class GraphType>
inline int ReadCSVGraph(GraphType& graph, const std::string& v_file,
                        const std::string& e_file) {
  std::vector<std::string> v_list, e_list;
  v_list.push_back(v_file);
  e_list.push_back(e_file);
  return ReadCSVGraph(graph, v_list, e_list);
}

template <class GraphType, class VertexIDGenerator, class EdgeIDGenerator>
inline int ReadCSVGraph(GraphType& graph,
                        const std::vector<std::string>& v_list,
                        const std::vector<std::string>& e_list,
                        VertexIDGenerator& vidgen, EdgeIDGenerator& eidgen) {
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;

  return ReadCSVGraphWithCallback(
      graph, v_list, e_list,
      [&vidgen](auto& vertex) -> bool {
        vidgen.UseID(vertex->id());
        return true;
      },
      [&eidgen](auto& edge) -> bool {
        eidgen.UseID(edge->id());
        return true;
      });
}

template <class GraphType, class VertexIDGenerator, class EdgeIDGenerator>
inline int ReadCSVGraph(GraphType& graph, const std::string& v_file,
                        const std::string& e_file, VertexIDGenerator& vidgen,
                        EdgeIDGenerator& eidgen) {
  std::vector<std::string> v_list, e_list;
  v_list.push_back(v_file);
  e_list.push_back(e_file);
  return ReadCSVGraph(graph, v_list, e_list, vidgen, eidgen);
}

// Write CSV
constexpr char CsvSeparator = ',';
template <typename T>
std::string ProtectedSeparatorVal(T&& val) {
  //':' is used in CSV head,so cannot used in Separator
  assert(CsvSeparator != ':');
  std::stringstream s_stream;
  s_stream << val;
  std::string stream_str = s_stream.str();
  if (stream_str.find(CsvSeparator) != stream_str.npos &&
      !(stream_str[0] == '"' && stream_str.back() == '"')) {
    return (std::string) "\"" + s_stream.str() + (std::string) "\"";
  }
  return s_stream.str();
}
// Write CSV columns
template <typename StreamType>
inline void WriteCSVColumns(StreamType& s, std::vector<std::string>& key_str,
                            std::vector<std::string>& type_str) {
  assert(CsvSeparator != ':');
  for (size_t i = 0; i < key_str.size(); i++) {
    if (i > 0) s << CsvSeparator;
    s << ProtectedSeparatorVal(key_str[i]) << ":"
      << ProtectedSeparatorVal(type_str[i]);
  }
  s << std::endl;
}

// Write CSV line
template <typename StreamType>
inline void WriteCSVLine(StreamType& s, std::vector<std::string>& cols) {
  assert(CsvSeparator != ':');
  for (size_t i = 0; i < cols.size(); i++) {
    if (i > 0) s << CsvSeparator;
    s << ProtectedSeparatorVal(cols[i]);
  }
  s << std::endl;
}

template <bool write_attr, class VertexEdgePtr, class AttributeKeyType,
          typename std::enable_if<!write_attr, bool>::type = false>
void WriteAttributes(VertexEdgePtr& vertex_edge_ptr,
                     const std::map<AttributeKeyType, size_t>& attr_pos,
                     std::vector<std::string>& line) {}

template <bool write_attr, class VertexEdgePtr, class AttributeKeyType,
          typename std::enable_if<write_attr, bool>::type = false>
void WriteAttributes(VertexEdgePtr& vertex_edge_ptr,
                     const std::map<AttributeKeyType, size_t>& attr_pos,
                     std::vector<std::string>& line) {
  for (auto attr_it = vertex_edge_ptr->AttributeBegin(); !attr_it.IsDone();
       ++attr_it) {
    size_t pos = attr_pos.find(attr_it->key())->second;
    line[pos] = attr_it->value_str();
  }
}

template <bool write_attr, class VertexEdgePtr, class AttributeKeyType,
          typename std::enable_if<!write_attr, bool>::type = false>
void GetWriteAttributeInfo(VertexEdgePtr ptr, std::vector<std::string>& key_str,
                           std::vector<std::string>& type_str,
                           std::map<AttributeKeyType, size_t>& attr_pos) {}

template <bool write_attr, class VertexEdgePtr, class AttributeKeyType,
          typename std::enable_if<write_attr, bool>::type = false>
void GetWriteAttributeInfo(VertexEdgePtr ptr, std::vector<std::string>& key_str,
                           std::vector<std::string>& type_str,
                           std::map<AttributeKeyType, size_t>& attr_pos) {
  for (auto attr_it = ptr->AttributeBegin(); !attr_it.IsDone(); ++attr_it) {
    auto attr_key = attr_it->key();
    if (attr_pos.emplace(attr_key, key_str.size()).second) {
      key_str.emplace_back(ToString(attr_key));
      type_str.emplace_back(EnumToString(attr_it->value_type()));
    }
  }
}

template <bool write_attr = true, class GraphType, class WriteVertexCallback>
int WriteCSVVertexSetFileWithCallback(
    const std::vector<GraphType>& graph_set,
    const std::vector<std::string>& graph_name_set,
    const std::string& v_set_file, WriteVertexCallback wv_callback) {
  // write vertex set file(csv)
  // file format: (vertex_id,label_id,graph_id,......)
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using VertexAttributeKeyType =
      typename GraphType::VertexType::AttributeKeyType;

  assert(graph_set.size() == graph_name_set.size());

  // get columns
  std::vector<std::string> key_str, type_str;
  std::map<VertexAttributeKeyType, size_t> attr_pos;
  const size_t kNumberOfGraph = graph_set.size();
  for (size_t i = 0; i < kNumberOfGraph; i++) {
    auto& graph = graph_set[i];
    for (auto vertex_it = graph.VertexBegin(); !vertex_it.IsDone();
         ++vertex_it) {
      if constexpr (!std::is_null_pointer_v<WriteVertexCallback>) {
        if (!wv_callback(vertex_it)) continue;
      }
      if (key_str.empty()) {
        key_str.emplace_back("vertex_id");
        key_str.emplace_back("label_id");
        // add graph_id
        key_str.emplace_back("graph_id");
        type_str.emplace_back(TypeToString<VertexIDType>());
        type_str.emplace_back(TypeToString<VertexLabelType>());
        type_str.emplace_back(TypeToString<std::string>());
      }
      if constexpr (write_attr) {
        GetWriteAttributeInfo<GraphType::vertex_has_attribute>(
            vertex_it, key_str, type_str, attr_pos);
      }
    }
    if (key_str.empty()) {
      return 0;
    }
  }

  std::cout << v_set_file << std::endl;

  std::ofstream vertex_set_file(v_set_file);

  WriteCSVColumns(vertex_set_file, key_str, type_str);

  // write each vertex
  int count = 0;
  for (size_t i = 0; i < kNumberOfGraph; i++) {
    auto& graph = graph_set[i];
    for (auto vertex_it = graph.VertexBegin(); !vertex_it.IsDone();
         ++vertex_it) {
      if constexpr (!std::is_null_pointer_v<WriteVertexCallback>) {
        if (!wv_callback(vertex_it)) continue;
      }
      std::vector<std::string> line;
      line.resize(key_str.size());
      line[0] = ToString(vertex_it->id());
      line[1] = ToString(vertex_it->label());
      line[2] = graph_name_set[i];
      if constexpr (write_attr) {
        WriteAttributes<GraphType::vertex_has_attribute>(vertex_it, attr_pos,
                                                         line);
      }
      WriteCSVLine(vertex_set_file, line);
      ++count;
    }
  }

  return count;
}

template <bool write_attr = true, class GraphType, class WriteVertexCallback>
int WriteCSVVertexFileWithCallback(const GraphType& graph,
                                   const std::string& v_file,
                                   WriteVertexCallback wv_callback) {
  // write vertex file(csv)
  // file format: (vertex_id,label_id,......)
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using VertexAttributeKeyType =
      typename GraphType::VertexType::AttributeKeyType;

  // get columns
  std::vector<std::string> key_str, type_str;
  std::map<VertexAttributeKeyType, size_t> attr_pos;
  for (auto vertex_it = graph.VertexBegin(); !vertex_it.IsDone(); ++vertex_it) {
    if constexpr (!std::is_null_pointer_v<WriteVertexCallback>) {
      if (!wv_callback(vertex_it)) continue;
    }
    if (key_str.empty()) {
      key_str.emplace_back("vertex_id");
      key_str.emplace_back("label_id");
      type_str.emplace_back(TypeToString<VertexIDType>());
      type_str.emplace_back(TypeToString<VertexLabelType>());
    }
    if constexpr (write_attr) {
      GetWriteAttributeInfo<GraphType::vertex_has_attribute>(
          vertex_it, key_str, type_str, attr_pos);
    }
  }
  if (key_str.empty()) {
    return 0;
  }

  std::cout << v_file << std::endl;

  std::ofstream vertex_file(v_file);

  WriteCSVColumns(vertex_file, key_str, type_str);

  // write each vertex
  int count = 0;
  for (auto vertex_it = graph.VertexBegin(); !vertex_it.IsDone(); ++vertex_it) {
    if constexpr (!std::is_null_pointer_v<WriteVertexCallback>) {
      if (!wv_callback(vertex_it)) continue;
    }
    std::vector<std::string> line;
    line.resize(key_str.size());
    line[0] = ToString(vertex_it->id());
    line[1] = ToString(vertex_it->label());
    if constexpr (write_attr) {
      WriteAttributes<GraphType::vertex_has_attribute>(vertex_it, attr_pos,
                                                       line);
    }
    WriteCSVLine(vertex_file, line);
    ++count;
  }

  return count;
}

template <bool write_attr = true, class GraphType, class WriteEdgeCallback>
int WriteCSVEdgeSetFileWithCallback(
    const std::vector<GraphType>& graph_set,
    const std::vector<std::string>& graph_name_set, const std::string& e_file,
    WriteEdgeCallback we_callback) {
  // write edge set file(csv)
  // file format: (edge_id,source_id,target_id,label_id,graph_id,......)
  using VertexIDType = typename GraphType::VertexType::IDType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;
  using EdgeAttributeKeyType = typename GraphType::EdgeType::AttributeKeyType;

  assert(graph_set.size() == graph_name_set.size());

  // get columns
  std::vector<std::string> key_str, type_str;
  std::map<EdgeAttributeKeyType, size_t> attr_pos;

  const size_t kNumberOfGraph = graph_set.size();
  for (int i = 0; i < kNumberOfGraph; i++) {
    auto& graph = graph_set[i];
    for (auto vertex_cit = graph.VertexBegin(); !vertex_cit.IsDone();
         ++vertex_cit) {
      for (auto edge_it = vertex_cit->OutEdgeBegin(); !edge_it.IsDone();
           ++edge_it) {
        if constexpr (!std::is_null_pointer_v<WriteEdgeCallback>) {
          if (!we_callback(edge_it)) continue;
        }
        if (key_str.empty()) {
          assert(type_str.empty());
          key_str.emplace_back("edge_id");
          key_str.emplace_back("source_id");
          key_str.emplace_back("target_id");
          key_str.emplace_back("label_id");
          key_str.emplace_back("graph_id");
          type_str.emplace_back(TypeToString<EdgeIDType>());
          type_str.emplace_back(TypeToString<VertexIDType>());
          type_str.emplace_back(TypeToString<VertexIDType>());
          type_str.emplace_back(TypeToString<EdgeLabelType>());
          type_str.emplace_back(TypeToString<int>());
        }
        if constexpr (write_attr) {
          GetWriteAttributeInfo<GraphType::edge_has_attribute>(
              edge_it, key_str, type_str, attr_pos);
        }
      }
    }
  }

  if (key_str.empty()) {
    std::cout << "empty" << std::endl;
    return 0;
  }

  std::cout << e_file << std::endl;

  std::ofstream edge_set_file(e_file);

  WriteCSVColumns(edge_set_file, key_str, type_str);

  // write each edge
  int count = 0;

  for (int i = 0; i < kNumberOfGraph; i++) {
    auto& graph = graph_set[i];
    for (auto vertex_cit = graph.VertexBegin(); !vertex_cit.IsDone();
         ++vertex_cit) {
      for (auto edge_it = vertex_cit->OutEdgeBegin(); !edge_it.IsDone();
           ++edge_it) {
        if constexpr (!std::is_null_pointer_v<WriteEdgeCallback>) {
          if (!we_callback(edge_it)) continue;
        }
        std::vector<std::string> line;
        line.resize(key_str.size());
        line[0] = ToString(edge_it->id());
        line[1] = ToString(edge_it->const_src_handle()->id());
        line[2] = ToString(edge_it->const_dst_handle()->id());
        line[3] = ToString(edge_it->label());
        line[4] = graph_name_set[i];
        if constexpr (write_attr) {
          WriteAttributes<GraphType::edge_has_attribute>(edge_it, attr_pos,
                                                         line);
        }
        WriteCSVLine(edge_set_file, line);
        ++count;
      }
    }
  }

  return count;
}

template <bool write_attr = true, class GraphType, class WriteEdgeCallback>
int WriteCSVEdgeFileWithCallback(const GraphType& graph,
                                 const std::string& e_file,
                                 WriteEdgeCallback we_callback) {
  // write edge file(csv)
  // file format: (edge_id,source_id,target_id,label_id,......)
  using VertexIDType = typename GraphType::VertexType::IDType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;
  using EdgeAttributeKeyType = typename GraphType::EdgeType::AttributeKeyType;

  // get columns
  std::vector<std::string> key_str, type_str;
  std::map<EdgeAttributeKeyType, size_t> attr_pos;

  for (auto vertex_cit = graph.VertexBegin(); !vertex_cit.IsDone();
       ++vertex_cit) {
    for (auto edge_it = vertex_cit->OutEdgeBegin(); !edge_it.IsDone();
         ++edge_it) {
      if constexpr (!std::is_null_pointer_v<WriteEdgeCallback>) {
        if (!we_callback(edge_it)) continue;
      }
      if (key_str.empty()) {
        assert(type_str.empty());
        key_str.emplace_back("edge_id");
        key_str.emplace_back("source_id");
        key_str.emplace_back("target_id");
        key_str.emplace_back("label_id");
        type_str.emplace_back(TypeToString<EdgeIDType>());
        type_str.emplace_back(TypeToString<VertexIDType>());
        type_str.emplace_back(TypeToString<VertexIDType>());
        type_str.emplace_back(TypeToString<EdgeLabelType>());
      }
      if constexpr (write_attr) {
        GetWriteAttributeInfo<GraphType::edge_has_attribute>(
            edge_it, key_str, type_str, attr_pos);
      }
    }
  }
  if (key_str.empty()) {
    return 0;
  }

  std::cout << e_file << std::endl;

  std::ofstream edge_file(e_file);

  WriteCSVColumns(edge_file, key_str, type_str);

  // write each edge
  int count = 0;

  for (auto vertex_cit = graph.VertexBegin(); !vertex_cit.IsDone();
       ++vertex_cit) {
    for (auto edge_it = vertex_cit->OutEdgeBegin(); !edge_it.IsDone();
         ++edge_it) {
      if constexpr (!std::is_null_pointer_v<WriteEdgeCallback>) {
        if (!we_callback(edge_it)) continue;
      }
      std::vector<std::string> line;
      line.resize(key_str.size());
      line[0] = ToString(edge_it->id());
      line[1] = ToString(edge_it->const_src_handle()->id());
      line[2] = ToString(edge_it->const_dst_handle()->id());
      line[3] = ToString(edge_it->label());
      if constexpr (write_attr) {
        WriteAttributes<GraphType::edge_has_attribute>(edge_it, attr_pos, line);
      }
      WriteCSVLine(edge_file, line);
      ++count;
    }
  }

  return count;
}

template <bool write_attr = true, class GraphType>
int WriteCSVGraphSet(const std::vector<GraphType>& graph_set,
                     const std::string& v_file, const std::string& e_file) {
  std::vector<std::string> graph_name_set;
  graph_name_set.reserve(graph_set.size());
  for (size_t i = 0; i < graph_set.size(); i++) {
    graph_name_set.emplace_back(std::to_string(i));
  }
  return WriteCSVGraphSet(graph_set, graph_name_set, v_file, e_file);
}

template <bool write_attr = true, class GraphType>
int WriteCSVGraphSet(const std::vector<GraphType>& graph_set,
                     const std::vector<std::string>& graph_name_set,
                     const std::string& v_file, const std::string& e_file) {
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using VertexAttributeKeyType =
      typename GraphType::VertexType::AttributeKeyType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;
  using EdgeAttributeKeyType = typename GraphType::EdgeType::AttributeKeyType;

  int res;
  int count_v, count_e;

  res = WriteCSVVertexSetFileWithCallback<write_attr>(graph_set, graph_name_set,
                                                      v_file, nullptr);
  if (res < 0) return res;

  count_v = res;

  res = WriteCSVEdgeSetFileWithCallback<write_attr>(graph_set, graph_name_set,
                                                    e_file, nullptr);
  if (res < 0) return res;

  count_e = res;

  return count_v + count_e;
}

template <bool write_attr = true, class GraphType>
int WriteCSVGraph(const GraphType& graph, const std::string& v_file,
                  const std::string& e_file) {
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using VertexAttributeKeyType =
      typename GraphType::VertexType::AttributeKeyType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;
  using EdgeAttributeKeyType = typename GraphType::EdgeType::AttributeKeyType;

  int res;
  int count_v, count_e;

  res = WriteCSVVertexFileWithCallback<write_attr>(graph, v_file, nullptr);
  if (res < 0) return res;

  count_v = res;

  res = WriteCSVEdgeFileWithCallback<write_attr>(graph, e_file, nullptr);
  if (res < 0) return res;

  count_e = res;

  return count_v + count_e;
}

template <bool write_attr = true, class GraphType, class VertexSet,
          class EdgeSet>
int WriteCSVParticalGraph(const GraphType& graph, const std::string& v_file,
                          const std::string& e_file,
                          const VertexSet& vertex_set,
                          const EdgeSet& edge_set) {
  using VertexIDType = typename GraphType::VertexType::IDType;
  using VertexLabelType = typename GraphType::VertexType::LabelType;
  using VertexAttributeKeyType =
      typename GraphType::VertexType::AttributeKeyType;
  using EdgeIDType = typename GraphType::EdgeType::IDType;
  using EdgeLabelType = typename GraphType::EdgeType::LabelType;
  using EdgeAttributeKeyType = typename GraphType::EdgeType::AttributeKeyType;

  int res;
  int count_v, count_e;

  res = WriteCSVVertexFileWithCallback<write_attr>(
      graph, v_file, [&vertex_set](auto& vertex) -> bool {
        return (vertex_set.find(vertex->id()) != vertex_set.end());
      });
  if (res < 0) return res;

  count_v = res;

  res = WriteCSVEdgeFileWithCallback<write_attr>(
      graph, e_file, [&edge_set](auto& edge) -> bool {
        return (edge_set.find(edge->id()) != edge_set.end());
      });
  if (res < 0) return res;

  count_e = res;

  return count_v + count_e;
}
}  // namespace GUNDAM

#endif