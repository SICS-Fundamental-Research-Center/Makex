#include <omp.h>

#include <chrono>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <string>
#include <regex>
#include <sstream>
#include <cmath>
#include <mutex>
#include <thread>
#include <map>
#include <queue>
#include <functional>
#include <filesystem>

#include "../pyMakex/include/algorithm/REPMatch.h"
#include "../pyMakex/include/gundam/algorithm/dp_iso.h"
#include "../pyMakex/include/gundam/data_type/datatype.h"
#include "../pyMakex/include/gundam/graph_type/large_graph.h"
#include "../pyMakex/include/gundam/graph_type/large_graph2.h"
#include "../pyMakex/include/gundam/graph_type/small_graph.h"
#include "../pyMakex/include/gundam/io/csvgraph.h"
#include "../pyMakex/include/gundam/type_getter/vertex_handle.h"
#include "../pyMakex/include/module/DataGraphWithInformation.h"
#include "../pyMakex/include/module/StarMiner.h"
#include "../pyMakex/include/structure/PatternWithParameter.h"
#include "../pyMakex/include/structure/Predicate.h"
#include "../pyMakex/include/structure/REP.h"

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
using StarMiner = Makex::StarMiner<DataGraph>;
using DataGraphAttributePtr = typename DataGraph::VertexType::AttributePtr;
using DataGraphWithInformation = Makex::DataGraphWithInformation<DataGraph>;


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
template <typename T>
inline constexpr double CalTime(T &begin, T &end) {
  return double(
             std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                 .count()) *
         std::chrono::microseconds::period::num /
         std::chrono::microseconds::period::den;
}


std::mutex write_mutex;

void parseLine(const std::string& line, std::vector<std::vector<int>> &vector_vertex,
              std::vector<std::vector<int>> &vector_edges,
              std::vector<std::vector<std::string>> &vector_predicates,
              std::vector<double> &vector_predicatesScore,
              std::vector<double> &vector_RepScore) {

    std::regex repRegex("rep (\\d+): \\[\\[(.*?\\])\\], \\[(\\[.*?\\])\\], \\[(\\[.*?\\])\\], (\\[.*?\\]), (\\[.*?\\])\\]");

    std::smatch match;
    
    if (std::regex_search(line, match, repRegex)) {

        std::string verticesStr = match.str(2);

        std::regex pattern("\\[(\\d+), (\\d+)\\]");

        auto words_begin = std::sregex_iterator(verticesStr.begin(), verticesStr.end(), pattern);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            std::vector<int> temp;
            temp.push_back(std::stoi(match[1]));
            temp.push_back(std::stoi(match[2]));
            vector_vertex.push_back(temp);
        }

        std::string edgesStr = match.str(3);

        std::regex edge_pattern("\\[(\\d+), (\\d+), (\\d+)\\]");


        auto edges_begin = std::sregex_iterator(edgesStr.begin(), edgesStr.end(), edge_pattern);
        auto edges_end = std::sregex_iterator();

        for (std::sregex_iterator i = edges_begin; i != edges_end; ++i) {
            std::smatch match = *i;
            std::vector<int> temp;
            temp.push_back(std::stoi(match[1]));
            temp.push_back(std::stoi(match[2]));
            temp.push_back(std::stoi(match[3]));
            vector_edges.push_back(temp);
        }


        std::string predicatesStr = match.str(4);

        std::regex predicates_pattern("\\[(.*?), (.*?), (.*?), (.*?), (.*?), (.*?)\\]");
 
        auto predicates_begin = std::sregex_iterator(predicatesStr.begin(), predicatesStr.end(), predicates_pattern);
        auto predicates_end = std::sregex_iterator();

        for (std::sregex_iterator i = predicates_begin; i != predicates_end; ++i) {
            std::smatch match = *i;
            std::vector<std::string> temp;
            temp.push_back(match[1]);
            temp.push_back(match[2]);
            temp.push_back(match[3]);
            temp.push_back(match[4]);
            temp.push_back(match[5]);
            temp.push_back(match[6]);
            vector_predicates.push_back(temp);
        }

        std::string predicatesScore = match.str(5);
        std::regex predicates_score_pattern("\\d+\\.\\d+");
 
        auto predicates_score_begin = std::sregex_iterator(predicatesScore.begin(), predicatesScore.end(), predicates_score_pattern);
        auto predicates_score_end = std::sregex_iterator();

        for (std::sregex_iterator i = predicates_score_begin; i != predicates_score_end; ++i) {
            std::smatch match = *i;
            double number = std::stod(match.str());
            vector_predicatesScore.push_back(number);
        }
        std::string RepScore = match.str(6);
        std::regex Rep_score_pattern("\\d+\\.\\d+");
 
        auto Rep_score_begin = std::sregex_iterator(RepScore.begin(), RepScore.end(), Rep_score_pattern);
        auto Rep_score_end = std::sregex_iterator();

        for (std::sregex_iterator i = Rep_score_begin; i != Rep_score_end; ++i) {
            std::smatch match = *i;
            double number = std::stod(match.str());
            vector_RepScore.push_back(number);
        }
    } else {
        std::cerr << "Failed to parse line: " << line << std::endl;
    }
}

template <class Pattern, class DataGraph>
void addVerticesAndEdges(Makex::REP<Pattern, DataGraph> &rep, const std::vector<std::vector<int>>& vertices, const std::vector<std::vector<int>>& edges) {
    Pattern &rep_pattern = rep.pattern();
    for (const auto& vertex : vertices) {
        int vertexId = vertex[0];
        int vertexLabel = vertex[1];
        rep_pattern.AddVertex(vertexId, vertexLabel);
    }
    int edgeId = 1;
    for (const auto& edge : edges) {
          int sourceId = edge[0];
          int targetId = edge[1];
          int edgeLabel = edge[2]; 
          rep_pattern.AddEdge(sourceId, targetId, edgeLabel, edgeId);
          edgeId += 1;
    }

}


template <class Pattern, class DataGraph>
void addVerticesAndEdges_Path(Makex::REP<Pattern, DataGraph> &rep, const std::vector<std::pair<int, int>>& vertices, const std::vector<std::tuple<int, int, int>>& edges) {
    Pattern &rep_pattern = rep.pattern();
    for (const auto& vertex : vertices) {
        int vertexId = vertex.first;
        int vertexLabel = vertex.second;
        rep_pattern.AddVertex(vertexId, vertexLabel);
    }

    
    int edgeId = 1;
    for (const auto& edge : edges) {
        int sourceId = std::get<0>(edge);
        int targetId = std::get<1>(edge);    
        int edgeLabel = std::get<2>(edge); 
        rep_pattern.AddEdge(sourceId, targetId, edgeLabel, edgeId);
        edgeId += 1;
    }
}




template <class Pattern, class DataGraph>
void AddPredicates(Makex::REP<Pattern, DataGraph> &rep, std::vector<std::vector<std::string>> &data) {
  for (const auto& sublist : data) {
        std::string predicate_type = sublist[0];
        uint32_t value1 = std::stoi(sublist[1]);
        if (predicate_type == "Constant") {
          if (sublist[4] == "int"){
            int value3 = std::stoi(sublist[3]);
            rep.template Add<Makex::ConstantPredicate<Pattern, DataGraph, int>>(value1, std::string(sublist[2]), value3, StringToOp(sublist[5]));
          }
          if (sublist[4] == "double"){
            double value3 = std::stoi(sublist[3]);
            rep.template Add<Makex::ConstantPredicate<Pattern, DataGraph, double>>(value1, std::string(sublist[2]), value3, StringToOp(sublist[5]));
          }
          if (sublist[4] == "string"){
            rep.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value1, std::string(sublist[2]), std::string(sublist[3]), StringToOp(sublist[5]));
          }
        } else if (predicate_type == "Variable") {
            uint32_t id2 = std::stoi(sublist[3]);
            rep.template Add<Makex::VariablePredicate<Pattern, DataGraph>>(value1, std::string(sublist[2]), id2, std::string(sublist[2]), StringToOp(sublist[5]));
        }
  }
}



void removeQuotes(std::string& str) {
    if (!str.empty() && str.front() == '\'' && str.back() == '\'') {
        str = str.substr(1, str.size() - 2);
    }
}


template <class Pattern, class DataGraph>
void BuildMovielensREP(Makex::REP<Pattern, DataGraph> &rep, std::vector<std::vector<int>> &vertices, std::vector<std::vector<int>> &edges, std::vector<std::vector<std::string>> &predicates, std::vector<double> &predicates_score) {
  addVerticesAndEdges(rep, vertices, edges);
  int predicates_ = 0;
  for (const auto& sublist : predicates) {
        std::string predicate_type = sublist[0];
        removeQuotes(predicate_type);
        uint32_t value1 = std::stoi(sublist[1]);
        std::string operate = sublist[5];
        removeQuotes(operate);

        if (predicate_type == "Constant") {
          std::string value_type = sublist[4];
          removeQuotes(value_type);
          std::string attributes = sublist[2];
          removeQuotes(attributes);
          if (value_type == "int"){
            int value3 = std::stoi(sublist[3]);
            
            rep.template Add<Makex::ConstantPredicate<Pattern, DataGraph, int>>(value1, std::string(attributes), value3, StringToOp(operate),predicates_score[predicates_]);
          }
          if (value_type == "double"){
            double value3 = std::stoi(sublist[3]);
            rep.template Add<Makex::ConstantPredicate<Pattern, DataGraph, double>>(value1, std::string(attributes), value3, StringToOp(operate),predicates_score[predicates_]);
          }
          if (value_type == "string"){
            std::string attributes_value = sublist[3];
            removeQuotes(attributes_value);
            rep.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value1, std::string(attributes), std::string(attributes_value), StringToOp(operate),predicates_score[predicates_]);
          }
        } else if (predicate_type == "Variable") {
            uint32_t id2 = std::stoi(sublist[3]);
            std::string attributes = sublist[2];
            removeQuotes(attributes);
            rep.template Add<Makex::VariablePredicate<Pattern, DataGraph>>(value1, std::string(attributes), id2, std::string(attributes), StringToOp(operate),predicates_score[predicates_]);
        }
    predicates_ += 1;
  }
}


template <class Pattern, class DataGraph>
void GetRepsFromFile(std::string rep_file, std::vector<Makex::REP<Pattern, DataGraph>> &rep_set) {
  std::ifstream file(rep_file);
  std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            if (line.substr(0, 3) == "rep") {
              std::vector<std::vector<int>> vector_vertex;
              std::vector<std::vector<int>> vector_edges;
              std::vector<std::vector<std::string>> vector_predicates;
              std::vector<double> vector_predicatesScore;
              std::vector<double> vector_RepScore;
              parseLine(line,vector_vertex,vector_edges,vector_predicates,vector_predicatesScore,vector_RepScore);
              Makex::REP<Pattern, DataGraph> rep(1, 0, 1, vector_RepScore[0]);
              BuildMovielensREP(rep, vector_vertex, vector_edges, vector_predicates, vector_predicatesScore);
              rep_set.push_back(rep);
            }
        }
        file.close();
    }
  
}



using MyCompare = std::function<bool(std::pair<double, std::map<int, int>>, std::pair<double, std::map<int, int>>)>;

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



void GetTestPairFromFile(std::string test_pair_file, std::vector<std::pair<int, int>> &test_pair){

  std::ifstream file(test_pair_file);
    std::string line;
bool firstLine = true;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            if (firstLine) {
                firstLine = false;
                continue;
            }

            std::istringstream iss(line);
            std::string token;
            std::getline(iss, token, ',');
            int user_id = std::stoi(token);
            std::getline(iss, token, ',');
            int item_id = std::stoi(token);

            test_pair.emplace_back(user_id, item_id);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }

}


void ReadPattern(const std::string& line, std::vector<std::vector<int>> &vector_vertex,
    std::vector<std::vector<int>> &vector_edges) {
    std::regex repRegex("pattern (\\d+): \\[(\\[.*?\\])\\]  \\[(\\[.*?\\])\\]");
    std::smatch match_pattern;
    
    if (std::regex_search(line, match_pattern, repRegex)) {
        std::string verticesStr = match_pattern.str(2);

        std::regex vertex_pattern("\\[(\\d+), (\\d+)\\]");

        auto words_begin = std::sregex_iterator(verticesStr.begin(), verticesStr.end(), vertex_pattern);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            std::vector<int> temp;
            temp.push_back(std::stoi(match[1]));
            temp.push_back(std::stoi(match[2]));
            vector_vertex.push_back(temp);
        }
        for (const auto& vec : vector_vertex) {
            std::cout << "[" << vec[0] << ", " << vec[1] << "]" << std::endl;
        }
        
        std::string edgesStr = match_pattern.str(3);

        std::regex edge_pattern("\\[(\\d+), (\\d+), (\\d+)\\]");


        auto edges_begin = std::sregex_iterator(edgesStr.begin(), edgesStr.end(), edge_pattern);
        auto edges_end = std::sregex_iterator();

        for (std::sregex_iterator i = edges_begin; i != edges_end; ++i) {
            std::smatch match = *i;
            std::vector<int> temp;
            temp.push_back(std::stoi(match[1]));
            temp.push_back(std::stoi(match[2]));
            temp.push_back(std::stoi(match[3]));
            vector_edges.push_back(temp);
        }
        
        for (const auto& vec : vector_edges) {
            std::cout << "[" << vec[0] << ", " << vec[1] << ", " << vec[2]  << "]" << std::endl;
        }

    }
}

bool replicate_pattern(const std::string& line, std::unordered_set<std::string>& uniquePatterns) {
    size_t pos = line.find(':');
    if (pos != std::string::npos) {
        std::string content = line.substr(pos + 2);
        if (uniquePatterns.insert(content).second) {
            return false;
        } else {
            return true;
        }
    }
    return true; 
}



template <class Pattern, class DataGraph>
void GetAllPatternFromFile(const std::string& pattern_file, std::vector<Makex::REP<Pattern, DataGraph>> &rep_pattern_set) {
  std::ifstream file(pattern_file);
  std::string line;

  if (file.is_open()) {
    std::unordered_set<std::string> uniquePatterns;
    while (std::getline(file, line)) {
      bool replicate = replicate_pattern(line, uniquePatterns);
      if(replicate){
        continue;
      }
      std::vector<std::vector<int>> vector_vertex;
      std::vector<std::vector<int>> vector_edges;
      ReadPattern(line, vector_vertex, vector_edges);

      for (const auto& vertex : vector_vertex) {
          std::cout << "[";
          for (size_t i = 0; i < vertex.size(); ++i) {
              std::cout << vertex[i];
              if (i != vertex.size() - 1) {
                  std::cout << ", ";
              }
          }
          std::cout << "]" << std::endl;
      }

      for (const auto& edge : vector_edges) {
          std::cout << "[";
          for (size_t i = 0; i < edge.size(); ++i) {
              std::cout << edge[i];
              if (i != edge.size() - 1) {
                  std::cout << ", ";
              }
          }
          std::cout << "]" << std::endl;
      }
      Makex::REP<Pattern, DataGraph> rep_pattern(1, 0, 1, 1.0);
      addVerticesAndEdges(rep_pattern, vector_vertex, vector_edges);
      rep_pattern_set.push_back(rep_pattern);
    }
  }
}






template <class Pattern, class DataGraph>
void GetAllPattern_Path_FromFile(const std::string& pattern_file, std::vector<Makex::REP<Pattern, DataGraph>> &rep_pattern_set, std::map<int, std::vector<int>> &rep_id_to_path_id, std::map<int, std::vector<int>> &all_paths) {
  std::ifstream file(pattern_file);
  std::string line;

  int rep_id = 0;
  int max_path_id = -1;

  
  if (file.is_open()) {
    std::unordered_set<std::string> uniquePatterns;
    while (std::getline(file, line)) {
      bool replicate = replicate_pattern(line, uniquePatterns);
      if(replicate){
        continue;
      }
      std::vector<std::vector<int>> vector_vertex;
      std::vector<std::vector<int>> vector_edges;
      ReadPattern(line, vector_vertex, vector_edges);

      for (const auto& vertex : vector_vertex) {
          std::cout << "[";
          for (size_t i = 0; i < vertex.size(); ++i) {
              std::cout << vertex[i];
              if (i != vertex.size() - 1) {
                  std::cout << ", ";
              }
          }
          std::cout << "]" << std::endl;
      }

      int pivot_x_path = 0;
      int pivot_y_path = 0;
      for (const auto& edge : vector_edges) {
        std::cout << "[";
        for (size_t i = 0; i < edge.size(); ++i) {
            std::cout << edge[i];
            if(edge[i] == 1){
            pivot_x_path += 1;
            }
            if(edge[i] == 2){
                pivot_y_path += 1;
            }
            if (i != edge.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
      }

      if(pivot_x_path <=0 || pivot_y_path <=0){
        continue;
      }

        std::map<int, std::vector<int>> current_paths;
        int current_path_id = -1;
        

        for (const auto& edge : vector_edges) {
            int start_vertex = edge[0];

            if (start_vertex == 1 || start_vertex == 2) {
                current_path_id += 1;
                current_paths[current_path_id].push_back(edge[2]);
            } else {
                current_paths[current_path_id].push_back(edge[2]);
            }
        }

        if(current_paths.empty()){
            std::cout << "paths are empty" << std::endl;
        }
        if(current_paths.size() == 1){
            continue;
            std::cout << "paths size is 1" << std::endl;
        }

        for(auto &path : current_paths) {
            std::cout << "path id: " << path.first << " path: ";
            for(auto &label : path.second){
                std::cout << label << " ";
            }
            std::cout << std::endl;
        }

        for (const auto& path : current_paths) {
            auto it = std::find_if(all_paths.begin(), all_paths.end(), 
                                [&path](const auto& p) { return p.second == path.second; });
            if (it != all_paths.end()) {
                rep_id_to_path_id[rep_id].push_back(it->first);
            } 
            else {
                if (all_paths.empty()) {
                    all_paths[0] = path.second;
                    rep_id_to_path_id[rep_id].push_back(0);
                    max_path_id = 0; 
                    }
                    else{
                    int new_key = max_path_id + 1;
                    all_paths[new_key] = path.second;
                    rep_id_to_path_id[rep_id].push_back(new_key);
                    max_path_id = new_key;
                }
            }
        }

        Makex::REP<Pattern, DataGraph> rep_pattern(1, 0, 1, 1.0);
        addVerticesAndEdges(rep_pattern, vector_vertex, vector_edges);
        rep_pattern_set.push_back(rep_pattern);
        rep_id += 1;
    }
  }
}


void GetPredicatesFromFile(const std::string& candidate_predicates_file, std::map<int, std::vector<std::pair<std::string, std::string>>>& node_label_predicates) {
    std::ifstream file(candidate_predicates_file);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << candidate_predicates_file << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() != 3) {
            std::cerr << "Error: Invalid line format: " << line << std::endl;
            continue;
        }

        int node_label_id = std::stoi(tokens[0]);
        std::string attribute = tokens[1];
        std::string value = tokens[2];

        node_label_predicates[node_label_id].push_back(std::make_pair(attribute, value));
    }

    file.close();
}



typedef std::tuple<PatternVertexPtr, std::string, std::string> KeyType_Predicates_VertexPtr;
typedef std::pair<std::string, std::string> KeyType_Predicates;
typedef std::pair<int, double> ValueType_Support_Conf;
typedef std::tuple<int, double, std::set<int>> ValueType_Support_Conf_Pair_Match;
typedef std::tuple<int, std::string, std::string> KeyType_Predicates_VertexID;


struct Compare_Support {
    bool operator()(const ValueType_Support_Conf& a, const ValueType_Support_Conf& b) const {
        return a.first > b.first;    }
};



std::vector<std::pair<KeyType_Predicates, ValueType_Support_Conf>> getTopN(const std::map<KeyType_Predicates, ValueType_Support_Conf>& data, int topn) {
    std::vector<std::pair<KeyType_Predicates, ValueType_Support_Conf>> result;
    for (const auto& entry : data) {
        result.push_back(entry);
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return Compare_Support()(a.second, b.second);
    });
    if (result.size() > topn) {
        result.resize(topn);
    }
    return result;
}



std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> getTopN_VertexPtr(const std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>& data, int topn) {
    std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> result;
    for (const auto& entry : data) {
        result.push_back(entry);    
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return Compare_Support()(a.second, b.second);
    });
    if (result.size() > topn) {
        result.resize(topn);
    }
    return result;
}

void sortAndKeepTopN(std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& vectorMap, size_t topn) {
    std::vector<std::pair<size_t, int>> sumSupports(vectorMap.size());
    for (size_t i = 0; i < vectorMap.size(); ++i) {
        int sum = 0;
        for (const auto& pair : vectorMap[i]) {
            sum += pair.second.first;
        }
            sumSupports[i] = {i, sum};
    }

    std::partial_sort(sumSupports.begin(), sumSupports.begin() + topn, sumSupports.end(), 
                      [](const std::pair<size_t, int>& a, const std::pair<size_t, int>& b) {
        return a.second > b.second;});

    std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> topNVectorMap;
    for (size_t i = 0; i < topn && i < sumSupports.size(); ++i) {
        topNVectorMap.push_back(vectorMap[sumSupports[i].first]);
    }

    vectorMap = std::move(topNVectorMap);
}


template <class Pattern, class DataGraph>
std::vector<std::pair<Makex::REP<Pattern, DataGraph>, ValueType_Support_Conf>> getTopN_rep(const std::map<Makex::REP<Pattern, DataGraph>, ValueType_Support_Conf>& data, int topn) {
    std::vector<std::pair<Makex::REP<Pattern, DataGraph>, ValueType_Support_Conf>> result;
    for (const auto& entry : data) {
        result.push_back(entry);    
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return Compare_Support()(a.second, b.second);
    });
    if (result.size() > topn) {
        result.resize(topn);
    }
    return result;
}


void combinePathsRecursive(const std::map<int, std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& rep_all_predicates,
                           std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& currentCombination,
                           std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& allCombinations,
                           std::map<int, std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>::const_iterator it) {
    if (it == rep_all_predicates.end()) {
        allCombinations.push_back(currentCombination);
        return;
    }

    const auto& pathSets = it->second;
    for (const auto& path : pathSets) {
        for (const auto& pair : path) { 
            currentCombination.push_back(pair);
            combinePathsRecursive(rep_all_predicates, currentCombination, allCombinations, std::next(it));
            currentCombination.pop_back();
        }
    }
}

void combinePaths(
    const std::map<int, std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& rep_all_predicates,
    std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& allCombinations){

      std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> currentCombination;
      combinePathsRecursive(rep_all_predicates, currentCombination, allCombinations, rep_all_predicates.begin());

    }



void combinePathsRecursive_map(const std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& rep_all_predicates,
                           std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& currentCombination,
                           std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& allCombinations,
                           std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>::const_iterator it) {
    if (it == rep_all_predicates.end()) {
        allCombinations.push_back(currentCombination);
        return;
    }

    const auto& pathSets = it->second;
    for (const auto& path : pathSets) {
        for (const auto& pair : path) {            
            currentCombination.push_back(pair);
            combinePathsRecursive_map(rep_all_predicates, currentCombination, allCombinations, std::next(it));
            currentCombination.pop_back();        
        }
    }
}



void combinePaths_map(
    const std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& rep_all_predicates,
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& allCombinations){

      std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> currentCombination;
      combinePathsRecursive_map(rep_all_predicates, currentCombination, allCombinations, rep_all_predicates.begin());

}



template <class Pattern>
void writeVertices(std::ofstream& outFile, const Pattern& pattern) {
    outFile << "[";

    for (auto pattern_vertex_iter = pattern.VertexBegin(); !pattern_vertex_iter.IsDone(); ++pattern_vertex_iter) {
        int id = pattern_vertex_iter->id();
        int label = pattern_vertex_iter->label();
        auto next = pattern_vertex_iter;
        ++next;
        if (next.IsDone()) {
            outFile << "[" << id << ", " << label << "]";
        }
        else{
            outFile << "[" << id << ", " << label << "], ";
        }   
    }
    outFile << "], ";
}

void writeEdges(std::ofstream& outFile, const std::vector<std::vector<std::tuple<int, int, int>>>& edges_pattern) {

    outFile << "[";
    for (size_t j = 0; j < edges_pattern.size(); ++j) {
        const auto& edges_path = edges_pattern[j];
        for (size_t i = 0; i < edges_path.size(); ++i) {
            if(j == edges_pattern.size() - 1 && i == edges_path.size() - 1){
                outFile << "[" << std::get<0>(edges_path[i]) << ", " << std::get<1>(edges_path[i]) << ", " << std::get<2>(edges_path[i]) << "]";
            }
            else{
                outFile << "[" << std::get<0>(edges_path[i]) << ", " << std::get<1>(edges_path[i]) << ", " << std::get<2>(edges_path[i]) << "], ";
            }
        }
    }
    outFile << "], ";
}





void printPivotMatch(std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>> match_vector2_path_each_vertex_predicates_VertexPtr){
    for (const auto& candidate_predicates : match_vector2_path_each_vertex_predicates_VertexPtr) {
        std::cout << "path match:" << std::endl;
        for (const auto& predicates : candidate_predicates) {
            for (const auto& map : predicates) {
                auto vertexPtr = std::get<0>(map.first);
                std::string predicate1 = std::get<1>(map.first);
                std::string predicate2 = std::get<2>(map.first);
                int support = std::get<0>(map.second);
                double conf = std::get<1>(map.second);
                std::set<int> pivot_match = std::get<2>(map.second);
                
                std::cout << "      Vertex ID: " << vertexPtr->id()
                            << ", Vertex Label: " << vertexPtr->label()
                            << ", Attributes: " << predicate1
                            << ", Attributes Value: " << predicate2
                            << ", Support: " << support
                            << ", Confidence: " << conf 
                            << ", Pivot match number: " << pivot_match.size() << std::endl;
            }
        }
    }
}




void printPivotMatch_ID(std::vector<std::vector<std::map<KeyType_Predicates_VertexID, ValueType_Support_Conf_Pair_Match>>> match_vector2_path_each_vertex_predicates_VertexPtr){
    for (const auto& candidate_predicates : match_vector2_path_each_vertex_predicates_VertexPtr) {
        std::cout << "path match:" << std::endl;
        for (const auto& predicates : candidate_predicates) {
            for (const auto& map : predicates) {
                int vertexid = std::get<0>(map.first);
                std::string predicate1 = std::get<1>(map.first);
                std::string predicate2 = std::get<2>(map.first);
                int support = std::get<0>(map.second);
                double conf = std::get<1>(map.second);
                std::set<int> pivot_match = std::get<2>(map.second);
                
                std::cout << "      Vertex ID: " << vertexid
                            << ", Attributes: " << predicate1
                            << ", Attributes Value: " << predicate2
                            << ", Support: " << support
                            << ", Confidence: " << conf 
                            << ", Pivot match number: " << pivot_match.size() << std::endl;
            }
        }
    }
}





void printRepAllPredicatesTest_Match(std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>>& rep_all_predicates_test) {
    for (const auto& entry : rep_all_predicates_test) {
        std::cout << "Key path id : " << entry.first << std::endl;
        for (const auto& vecVecMap : entry.second) {
            std::cout << "  this path each candidate predicates:" << std::endl;
            for (const auto& vecMap : vecVecMap) {
                for (const auto& map : vecMap) {
                    auto vertexPtr = std::get<0>(map.first);
                    std::string predicate1 = std::get<1>(map.first);
                    std::string predicate2 = std::get<2>(map.first);
                    int support = std::get<0>(map.second);
                    double conf = std::get<1>(map.second);
                    std::set<int> pivot_match = std::get<2>(map.second);

                    
                    std::cout << "      Vertex ID: " << vertexPtr->id()
                              << ", Vertex Label: " << vertexPtr->label()
                              << ", Attributes: " << predicate1
                              << ", Value: " << predicate2
                              << ", Support: " << support
                              << ", Confidence: " << conf 
                              << ", pivot_match.size(): " << pivot_match.size() << std::endl;
                }
            }
        }
    }
}


template <class Pattern>
void writeRepToFile_vector(std::string& rep_file_generate, const Pattern& query_graph, std::vector<std::vector<std::tuple<int, int, int>>>& edges_pattern, const std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& allCombinations) {
    std::lock_guard<std::mutex> lock(write_mutex);
    std::ofstream outFile(rep_file_generate, std::ios::app);
    if (!outFile) {
        std::cerr << "Unable to open file: " << rep_file_generate << std::endl;
        return;
    }

    for (const auto& combinationSet : allCombinations) {
        outFile << "[";
        writeVertices(outFile, query_graph);
    
        writeEdges(outFile, edges_pattern);
        std::vector<ValueType_Support_Conf> Support_Conf;
        outFile << "[";
        std::set<std::tuple<int, std::string, std::string>> seenPredicates;
        for (size_t j = 0; j < combinationSet.size(); ++j) {
            const auto&  all_predicates = combinationSet[j];
            for (size_t i = 0; i < all_predicates.size(); ++i) {
                const auto& predicate = all_predicates[i];
                for(const auto& pair : predicate){
                    PatternVertexPtr vertex = std::get<0>(pair.first);
                    std::string attribute = std::get<1>(pair.first);
                    std::string attribute_value = std::get<2>(pair.first);

                    auto predicateIdentifier = std::make_tuple(vertex->id(), attribute, attribute_value);
                    if (seenPredicates.find(predicateIdentifier) != seenPredicates.end()) {
                        continue;
                    }
                    seenPredicates.insert(predicateIdentifier);
                    Support_Conf.push_back(pair.second);
                    std::string operate = "=";

                    if(attribute.find("wllabel_select") != std::string::npos){

                        std::string delimiter = "**";
                        size_t first_pos = attribute_value.find(delimiter);
                        std::string value_before = attribute_value.substr(0, first_pos);

                        size_t second_pos = attribute_value.find(delimiter, first_pos + delimiter.length());
                        std::string value_between = attribute_value.substr(first_pos + delimiter.length(), second_pos - first_pos - delimiter.length());
                        
                        std::string value_after = attribute_value.substr(second_pos + delimiter.length());

                        int value_before_num = std::stoi(value_before);
                        int value_between_num = std::stoi(value_between);
                        std::string wl_attributes_value = value_after;

                        std::string new_attribute_value = std::to_string(vertex->id()) + "**" + std::to_string(value_between_num) + "**" + wl_attributes_value;
                        outFile << "['Constant', " << vertex->id() << ", '" << attribute << "', '" << new_attribute_value << "', 'string', '" << operate << "'";
                    }
                    else{
                        if(attribute_value == "Children's"){
                            outFile << "['Constant', " << vertex->id() << ", '" << attribute << "', '" << "Children\\'s" << "', 'string', '" << operate << "'";
                        }
                        else {
                            outFile << "['Constant', " << vertex->id() << ", '" << attribute << "', '" << attribute_value << "', 'string', '" << operate << "'";
                        }
                    }

                    if(j == combinationSet.size() - 1 && i == all_predicates.size() - 1){
                        outFile << "]";
                    }
                    else{
                        outFile << "], ";
                    }
                }
            }
        }

        outFile << "], [";

        for(size_t i = 0; i < Support_Conf.size(); ++i){
            if(i != Support_Conf.size() - 1){
                outFile << Support_Conf[i].second << ", ";
            }
            else{
                outFile << Support_Conf[i].second;
            }
            
        }
        outFile << "], [1, 2, 1, 1.0]]" << std::endl;

    }
    outFile.close();
}



template <class Pattern>
void writeRepToFile_support_conf(std::string rep_file_generate, const Pattern& query_graph, std::vector<std::vector<std::tuple<int, int, int>>>& edges_pattern, const std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& allCombinations, std::vector<ValueType_Support_Conf>& final_rep_support_conf,
                                std::vector<int>& sortedVector) {
    std::lock_guard<std::mutex> lock(write_mutex);
    std::ofstream outFile(rep_file_generate, std::ios::app);
    if (!outFile) {
        std::cerr << "Unable to open file: " << rep_file_generate << std::endl;
        return;
    }

    int rep_id = 0;
    for (size_t index = 0; index < sortedVector.size(); ++index) {
        const auto& combinationSet = allCombinations[sortedVector[index]];
    
        outFile << "[";
        writeVertices(outFile, query_graph);
    
        writeEdges(outFile, edges_pattern);
        std::vector<ValueType_Support_Conf> Support_Conf;
        outFile << "[";
        std::set<std::tuple<int, std::string, std::string>> seenPredicates;

        for (size_t j = 0; j < combinationSet.size(); ++j) {
            const auto&  all_predicates = combinationSet[j];
            for (size_t i = 0; i < all_predicates.size(); ++i) {
                const auto& predicate = all_predicates[i];
                for(const auto& pair : predicate){
                    PatternVertexPtr vertex = std::get<0>(pair.first);
                    std::string attribute = std::get<1>(pair.first);
                    std::string attribute_value = std::get<2>(pair.first);
                    auto predicateIdentifier = std::make_tuple(vertex->id(), attribute, attribute_value);
                    if (seenPredicates.find(predicateIdentifier) != seenPredicates.end()) {
                        continue;
                    }
                    seenPredicates.insert(predicateIdentifier);


                    Support_Conf.push_back(pair.second);
                    std::string operate = "=";
                    if(attribute.find("wllabel_select") != std::string::npos){

                        std::string delimiter = "**";
                        size_t first_pos = attribute_value.find(delimiter);
                        std::string value_before = attribute_value.substr(0, first_pos);

                        size_t second_pos = attribute_value.find(delimiter, first_pos + delimiter.length());
                        std::string value_between = attribute_value.substr(first_pos + delimiter.length(), second_pos - first_pos - delimiter.length());
                        
                        std::string value_after = attribute_value.substr(second_pos + delimiter.length());

                        int value_before_num = std::stoi(value_before);
                        int value_between_num = std::stoi(value_between);
                        std::string wl_attributes_value = value_after;

                        std::string new_attribute_value = std::to_string(vertex->id()) + "**" + std::to_string(value_between_num) + "**" + wl_attributes_value;

                        outFile << "['Constant', " << vertex->id() << ", '" << attribute << "', '" << new_attribute_value << "', 'string', '" << operate << "'";
                    }
                    else{
                        if(attribute_value == "Children's"){
                            outFile << "['Constant', " << vertex->id() << ", '" << attribute << "', '" << "Children\\'s" << "', 'string', '" << operate << "'";
                        }
                        else {
                            outFile << "['Constant', " << vertex->id() << ", '" << attribute << "', '" << attribute_value << "', 'string', '" << operate << "'";
                        }
                    }
                     

                    if(j == combinationSet.size() - 1 && i == all_predicates.size() - 1){
                        outFile << "]";
                    }
                    else{
                        outFile << "], ";
                    }
                }
            }
        }

        outFile << "], [";

        for(size_t i = 0; i < Support_Conf.size(); ++i){
            if(i != Support_Conf.size() - 1){
                outFile << Support_Conf[i].second << ", ";
            }
            else{
                outFile << Support_Conf[i].second;
            }
            
        }
        int sup_rep = final_rep_support_conf[rep_id].first;
        double conf_rep = final_rep_support_conf[rep_id].second;
        rep_id++;
        outFile << "], [1, 2, 1, 1.0], "<< "[" << sup_rep << ", "<< conf_rep <<  "]]" << std::endl;

    }
    outFile.close();
}




void printTopkPathPredicates_map(const std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& topk_path_predicates) {
    for (const auto& path : topk_path_predicates) {        
        std::cout << "[";
            
        for (size_t mapIndex = 0; mapIndex < path.size(); ++mapIndex) {            
            const auto& map = path[mapIndex];
            for (const auto& pair : map) {
                PatternVertexPtr vertex = std::get<0>(pair.first);const std::string& predicate1 = std::get<1>(pair.first);const std::string& predicate2 = std::get<2>(pair.first);                
                int support = pair.second.first;double conf = pair.second.second;                
                    std::cout << "{Vertex ID: " << vertex->id() << ", Label: " << vertex->label()
                            << ", Attributes: " << predicate1 << ", Attributes value: " << predicate2
                            << ", Support: " << support << ", Conf: " << conf << "}";
                    
                    if (!(mapIndex == path.size() - 1 && &pair == &*map.rbegin())) std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}




void change_wllabel(std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& allCombinations, std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& newAllCombinations) {

    std::vector<std::string> vector_rep_predicates;
    for (auto& combo : allCombinations) {
        std::vector<int> wllabel_id;
        for (auto& vec : combo) {
            for (auto& vecMap : vec) {
                int vertex_id = std::get<0>(vecMap.begin()->first)->id();
                std::string currentWllabel = std::get<1>(vecMap.begin()->first);                
                if(currentWllabel == "wllabel_select"){
                    wllabel_id.push_back(vertex_id);
                }
            }
        }

       int seed = 1996;
        std::mt19937 gen(seed);

        std::uniform_int_distribution<> distrib(0, wllabel_id.size() - 1);

        int randomIndex = distrib(gen);
        int randomValue = wllabel_id[randomIndex];

        std::vector<int> visited_node;
        std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>> newCombo;

        std::string rep_predicates;
        int wl_pivot_id = 1;
        bool wl_pivot = false;
        for (auto& vec : combo) {
            std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> newComboPath;
            for (auto& vecMap : vec) {
                int vertex_id = std::get<0>(vecMap.begin()->first)->id();
                if(std::find(visited_node.begin(), visited_node.end(), vertex_id) != visited_node.end()){
                    continue;
                }
                visited_node.push_back(vertex_id);
                std::string currentWllabel = std::get<1>(vecMap.begin()->first);
                std::string currentWllabel_value = std::get<2>(vecMap.begin()->first);
                if(currentWllabel == "wllabel_select"){
                    std::string delimiter = "**";
                    size_t first_pos = currentWllabel_value.find(delimiter);
                    std::string value_before = currentWllabel_value.substr(0, first_pos);

                    size_t second_pos = currentWllabel_value.find(delimiter, first_pos + delimiter.length());
                    std::string value_between = currentWllabel_value.substr(first_pos + delimiter.length(), second_pos - first_pos - delimiter.length());
                    
                    std::string value_after = currentWllabel_value.substr(second_pos + delimiter.length());

                    int value_before_num = std::stoi(value_before);
                    int value_between_num = std::stoi(value_between);
                    std::string wl_attributes_value = value_after;

                    wl_pivot_id = value_between_num;
                }

                if(currentWllabel == "wllabel_select"){
                    if(vertex_id == randomValue){
                        wl_pivot = true;
                        newComboPath.push_back(vecMap);
                        rep_predicates += std::to_string(vertex_id) + "**" + currentWllabel + "**" + currentWllabel_value + "**";
                    }
                }
                else{
                    if(!wl_pivot){
                        if(vertex_id != randomValue){
                            newComboPath.push_back(vecMap);
                            rep_predicates += std::to_string(vertex_id) + "**" + currentWllabel + "**" + currentWllabel_value + "**";
                        }
                    }
                    else{
                        if(vertex_id != wl_pivot_id){
                            newComboPath.push_back(vecMap);
                            rep_predicates += std::to_string(vertex_id) + "**" + currentWllabel + "**" + currentWllabel_value + "**";
                        }
                    }
                }
            }
            newCombo.push_back(newComboPath);
        }


        if(vector_rep_predicates.size() == 0){
            vector_rep_predicates.push_back(rep_predicates);
            newAllCombinations.push_back(newCombo);
        }
        else{
            if(std::find(vector_rep_predicates.begin(), vector_rep_predicates.end(), rep_predicates) != vector_rep_predicates.end()){
                continue;
            }
            else{
                vector_rep_predicates.push_back(rep_predicates);
                newAllCombinations.push_back(newCombo);
            }
        }

        std::vector<int> visited_node_no_wl;
        std::string rep_predicates_no_wl;
        std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>> newCombo_constant;
        for (auto& vec : combo) {
            std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> newComboPath_constant;
            for (auto& vecMap : vec) {
                int vertex_id = std::get<0>(vecMap.begin()->first)->id();
                if(std::find(visited_node_no_wl.begin(), visited_node_no_wl.end(), vertex_id) != visited_node_no_wl.end()){
                    continue;
                }
                visited_node_no_wl.push_back(vertex_id);
                std::string currentWllabel = std::get<1>(vecMap.begin()->first);
                std::string currentWllabel_value = std::get<2>(vecMap.begin()->first);
                if(currentWllabel != "wllabel_select"){
                    newComboPath_constant.push_back(vecMap);
                    rep_predicates_no_wl += std::to_string(vertex_id) + "**" + currentWllabel + "**" + currentWllabel_value + "**";
                }
            }
            newCombo_constant.push_back(newComboPath_constant);
        }


        if(vector_rep_predicates.size() == 0){
            vector_rep_predicates.push_back(rep_predicates_no_wl);
            newAllCombinations.push_back(newCombo_constant);
        }
        else{
            if(std::find(vector_rep_predicates.begin(), vector_rep_predicates.end(), rep_predicates_no_wl) != vector_rep_predicates.end()){
                continue;
            }
            else{
                vector_rep_predicates.push_back(rep_predicates_no_wl);
                newAllCombinations.push_back(newCombo_constant);
            }
        }
    }
}





void cartesianHelper(const std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& vecs,
                     std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& result,
                     std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& temp,
                     int depth) {
    if (depth == vecs.size()) {
        result.push_back(temp);
        return;
    }
    
    for (const auto& item : vecs[depth]) {
        temp.push_back(item);
        cartesianHelper(vecs, result, temp, depth + 1);
        temp.pop_back();
    }
}





void cartesianHelper_Match(const std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& vecs,
                     std::vector<std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& result,
                     std::vector<std::pair<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& temp,
                     int depth) {
    if (depth == vecs.size()) {
        result.push_back(temp);
        return;
    }
    
    for (const auto& item : vecs[depth]) {
        temp.push_back(item);
        cartesianHelper(vecs, result, temp, depth + 1);
        temp.pop_back();
    }
}




int calculateTotalSupport(const std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& maps) {
    int totalSupport = 0;
    for (const auto& map : maps) {
        for (const auto& entry : map) {
            totalSupport += entry.second.first;        
        }
    }
    return totalSupport;
}

void sortVectorBySupportSum(std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& outerVector, size_t topn) {
    std::sort(outerVector.begin(), outerVector.end(), 
              [](const std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& a,
                 const std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& b) {
                  return calculateTotalSupport(a) > calculateTotalSupport(b);
              });

    if (outerVector.size() > topn) {
        outerVector.resize(topn);
    }
}



std::set<int> ExtractSets(const std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>& vectorMap, PatternVertexPtr& query_vertex) {
    std::set<int> result;
    for (const auto& map : vectorMap) {
        for (const auto& kv : map) {
            if (std::get<0>(kv.first)->id() == query_vertex->id()) {
                const std::set<int>& currentSet = std::get<2>(kv.second);
                result.insert(currentSet.begin(), currentSet.end());
            }
        }
    }
    return result;
}

int SetDifference(const std::set<int>& a, const std::set<int>& b) {
    std::vector<int> diff;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
    std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(diff));
    return diff.size();
}

void sortVectorByPivotMatchDifference_(std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>& outerVector, size_t topn, PatternVertexPtr& query_vertex) {
    std::vector<std::pair<std::set<int>, std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>*>> sets;
    for (auto& item : outerVector) {
        sets.emplace_back(ExtractSets(item, query_vertex), &item);
    }

    std::sort(sets.begin(), sets.end(), [&](const auto& lhs, const auto& rhs) {
        return SetDifference(lhs.first, rhs.first) > SetDifference(rhs.first, lhs.first);
    });

    if (sets.size() > topn) {
        sets.resize(topn);
    }
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>> sortedVector;
    for (auto& set : sets) {
        sortedVector.push_back(*set.second);
    }
    outerVector.swap(sortedVector);
}






void sortVectorByPivotMatchDifference(std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>& outerVector, size_t topn, PatternVertexPtr& query_vertex) {
    std::vector<std::pair<std::set<int>, size_t>> sets;

    for (size_t i = 0; i < outerVector.size(); ++i) {
        sets.emplace_back(ExtractSets(outerVector[i], query_vertex), i);
    }
    std::sort(sets.begin(), sets.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first.empty() && rhs.first.empty()) return false;
        if (lhs.first.empty()) return false;
        if (rhs.first.empty()) return true;
        return lhs.first.size() > rhs.first.size();
    });

    std::vector<std::set<int>> selected_sets;
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>> sortedVector;

    if (!sets.empty()) {
        selected_sets.push_back(sets[0].first);
        sortedVector.push_back(outerVector[sets[0].second]);
        sets.erase(sets.begin());
    }
    while (!sets.empty()) {
        auto max_diff_it = std::max_element(sets.begin(), sets.end(), [&](const auto& lhs, const auto& rhs) {
            int lhs_diff = 0, rhs_diff = 0;
            for (auto& selected_set : selected_sets) {
               lhs_diff += SetDifference(lhs.first, selected_set);
               rhs_diff += SetDifference(rhs.first, selected_set);

            }
            return lhs_diff < rhs_diff;
        });

        selected_sets.push_back(max_diff_it->first);
        sortedVector.push_back(outerVector[max_diff_it->second]);
        sets.erase(max_diff_it);
    }

    if (sortedVector.size() > topn) {
        sortedVector.resize(topn);
    }
    outerVector.swap(sortedVector);
}





void sortVectorByPivotMatchDifference_Consider_Support(std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>& outerVector, size_t topn, PatternVertexPtr& query_vertex, double sort_by_support_weights) {
    std::vector<std::tuple<std::set<int>, size_t, int>> sets;

    for (size_t i = 0; i < outerVector.size(); ++i) {
        int support_tp = 0;
        for (const auto& map : outerVector[i]) {
            for (const auto& kv : map) {
                if (std::get<0>(kv.first)->id() == query_vertex->id()) {
                    support_tp = std::get<0>(kv.second);
                }
            }
        }
        sets.emplace_back(std::make_tuple(ExtractSets(outerVector[i], query_vertex), i, support_tp));
    }
    std::sort(sets.begin(), sets.end(), [](const auto& lhs, const auto& rhs) {
        if (std::get<0>(lhs).empty() && std::get<0>(rhs).empty()) return false;
        if (std::get<0>(lhs).empty()) return false;
        if (std::get<0>(rhs).empty()) return true;
        return std::get<0>(lhs).size() > std::get<0>(rhs).size();
    });

    std::vector<std::set<int>> selected_sets;
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>> sortedVector;

    if (!sets.empty()) {
        selected_sets.push_back(std::get<0>(sets[0]));
        sortedVector.push_back(outerVector[std::get<1>(sets[0])]);
        sets.erase(sets.begin());
    }
    while (!sets.empty()) {
        auto max_diff_it = std::max_element(sets.begin(), sets.end(), [&](const auto& lhs, const auto& rhs) {
            int lhs_diff = 0, rhs_diff = 0;
            std::set<int> merged_set;

            for (const auto& s : selected_sets) {
                merged_set.insert(s.begin(), s.end());
            }

            lhs_diff = SetDifference(std::get<0>(lhs), merged_set);
            rhs_diff = SetDifference(std::get<0>(rhs), merged_set);



            int lhs_support_tp = std::get<0>(lhs).size();
            int rhs_support_tp = std::get<0>(rhs).size();

            lhs_diff = sort_by_support_weights * lhs_support_tp + (1-sort_by_support_weights) * lhs_diff;
            rhs_diff = sort_by_support_weights * rhs_support_tp + (1-sort_by_support_weights) * rhs_diff;


            return lhs_diff < rhs_diff;
        });

        selected_sets.push_back(std::get<0>(*max_diff_it));
        sortedVector.push_back(outerVector[std::get<1>(*max_diff_it)]);
        sets.erase(max_diff_it);
    }

    if (sortedVector.size() > topn) {
        sortedVector.resize(topn);
    }

    outerVector.swap(sortedVector);
}




void sortVectorByPivotMatchDifference_Consider_Support_Conf(std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>& outerVector, size_t topn, PatternVertexPtr& query_vertex, double sort_by_support_weights, double conf_limit) {
    std::vector<std::tuple<std::set<int>, size_t, int>> sets;

    for (size_t i = 0; i < outerVector.size(); ++i) {
        int support_tp = 0;
        for (const auto& map : outerVector[i]) {
            for (const auto& kv : map) {
                if (std::get<0>(kv.first)->id() == query_vertex->id()) {
                    support_tp = std::get<0>(kv.second);
                }
            }
        }
        sets.emplace_back(std::make_tuple(ExtractSets(outerVector[i], query_vertex), i, support_tp));
    }
    std::sort(sets.begin(), sets.end(), [](const auto& lhs, const auto& rhs) {
        if (std::get<0>(lhs).empty() && std::get<0>(rhs).empty()) return false;
        if (std::get<0>(lhs).empty()) return false;
        if (std::get<0>(rhs).empty()) return true;
        return std::get<0>(lhs).size() > std::get<0>(rhs).size();
    });

    std::vector<std::set<int>> selected_sets;
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>> sortedVector;
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>> sortedVector2;

    if (!sets.empty()) {
        selected_sets.push_back(std::get<0>(sets[0]));
        sortedVector.push_back(outerVector[std::get<1>(sets[0])]);
        sortedVector2.push_back(outerVector[std::get<1>(sets[0])]);
        sets.erase(sets.begin());
    }
    int select_predicate_size = 0;
    while (!sets.empty()) {
        if(select_predicate_size > topn){
            break;
        }
        auto max_diff_it = std::max_element(sets.begin(), sets.end(), [&](const auto& lhs, const auto& rhs) {
            int lhs_diff = 0, rhs_diff = 0;

            std::set<int> merged_set;

            for (const auto& s : selected_sets) {
                merged_set.insert(s.begin(), s.end());
            }

            lhs_diff = SetDifference(std::get<0>(lhs), merged_set);
            rhs_diff = SetDifference(std::get<0>(rhs), merged_set);

            int lhs_support_tp = std::get<0>(lhs).size();
            int rhs_support_tp = std::get<0>(rhs).size();

            lhs_diff = sort_by_support_weights * lhs_support_tp + (1-sort_by_support_weights) * lhs_diff;
            rhs_diff = sort_by_support_weights * rhs_support_tp + (1-sort_by_support_weights) * rhs_diff;


            return lhs_diff < rhs_diff;
        });
        sortedVector2.push_back(outerVector[std::get<1>(*max_diff_it)]);
        selected_sets.push_back(std::get<0>(*max_diff_it));
        sets.erase(max_diff_it);


        double min_conf = std::numeric_limits<double>::max();

        for (const auto& map : outerVector[std::get<1>(*max_diff_it)]) {
            for (const auto& pair : map) {
                double conf = std::get<1>(pair.second);
                if (conf < min_conf) {
                    min_conf = conf;
                }
            }
        }
        if (min_conf < conf_limit) {
            continue;
        }
        sortedVector.push_back(outerVector[std::get<1>(*max_diff_it)]);
        
        select_predicate_size += 1;
    }

    
    if (sortedVector.size() >= topn) {
        sortedVector.resize(topn);
        outerVector.swap(sortedVector);
    }
    else{
        int less = topn - sortedVector.size();
        sortedVector.swap(sortedVector2);
        sortedVector.resize(topn+less);
        outerVector.swap(sortedVector);
    }

}





void print2VectorMap(const std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& outerVector) {
    for (const auto& innerVector : outerVector) {
        std::cout << "[";
        for (const auto& map : innerVector) {
            std::cout << "{";
            for (const auto& entry : map) {
                PatternVertexPtr vertex = std::get<0>(entry.first);
                const std::string& predicate1 = std::get<1>(entry.first);
                const std::string& predicate2 = std::get<2>(entry.first);
                std::cout << "Vertex ID: " << vertex->id() << "Vertex Label: " << vertex->label()<< ", Predicates: [" << predicate1 << ", " << predicate2 << "], ";
                std::cout << "Support: " << entry.second.first << ", Confidence: " << entry.second.second << "; ";
            }
            std::cout << "}, ";
        }
        std::cout << "]" << std::endl;
    }
}






void generateCartesian_test(
    const std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& rep_all_predicates_test,
    std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& allCombinations,
std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& currentPath,    std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>::const_iterator it) {
    
    if (it == rep_all_predicates_test.end()) {
        allCombinations.push_back(currentPath);        
        return;
    }
    
    for (const auto& pathOptions : it->second) {
        currentPath.push_back(pathOptions);        
        generateCartesian_test(rep_all_predicates_test, allCombinations, currentPath, std::next(it));
        currentPath.pop_back();    
    }
}


void generateCartesian_Match(
    const std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>>& rep_all_predicates_test,
    std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& allCombinations,
std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& currentPath,    std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>>::const_iterator it) {

    if (it == rep_all_predicates_test.end()) {
        allCombinations.push_back(currentPath);        
        return;
    }
    
    for (const auto& pathOptions : it->second) {        
        std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> convertedPath;
        for (const auto& mapMatch : pathOptions) {
            std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf> convertedMap;
            for (const auto& kv : mapMatch) {
                convertedMap[kv.first] = ValueType_Support_Conf{std::get<0>(kv.second), std::get<1>(kv.second)};            
            }
            convertedPath.push_back(convertedMap);
        }
        currentPath.push_back(convertedPath);        
        generateCartesian_Match(rep_all_predicates_test, allCombinations, currentPath, std::next(it));
        currentPath.pop_back();
    }
}




void generateCartesian(
    const std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& rep_all_predicates_test,
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& allCombinations,
    std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>& currentCombination,
    std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>::const_iterator it) {
    if (it == rep_all_predicates_test.end()) {
        allCombinations.push_back(currentCombination);
        return;
    }
    
for (const auto& pathOptions : it->second) {
currentCombination = pathOptions;        generateCartesian(rep_all_predicates_test, allCombinations, currentCombination, std::next(it));
currentCombination.pop_back();    }
}


void generateCartesian2(
    const std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& options,
    std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& results,
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& current,
    size_t index = 0) {
    if (index == options.size()) {
        results.push_back(current);
        return;
    }
    for (const auto& option : options[index]) {
        current.push_back(option);
        generateCartesian2(options, results, current, index + 1);
        current.pop_back();
    }
}



void CalculateCartesianProduct(
    const std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>>& options,
    size_t depth,
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& result,
    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>& current) {
    
    if (depth == options.size()) {
        result = current;
        
        return;
    }

    for (const auto& option : options[depth]) {
        current.push_back(option);
        CalculateCartesianProduct(options, depth + 1, result, current);
        current.pop_back();
    }
}





void GetEdgeLabelToNodeLabel(const std::string& edge_label_reverse_csv, std::map<int, std::pair<int, int>>& edge_label_to_node_label) {
    std::ifstream file(edge_label_reverse_csv);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
    }

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        int source_node_label, target_node_label, edge_label;
        char delimiter;

        ss >> source_node_label >> delimiter >> target_node_label >> delimiter >> edge_label;
        edge_label_to_node_label[edge_label] = std::make_pair(source_node_label, target_node_label);

        std::cout << "source_node_label: " << source_node_label 
                  << ", target_node_label: " << target_node_label 
                  << ", Edge: " << edge_label << std::endl;
    }

    file.close();
}

void filter_rep_by_supp_conf(const std::string& file_path, const std::string& output_file_path, int rep_support, double rep_conf) {


    if (!inputFile.is_open() || !outputFile.is_open()) {
        std::cerr << "Can't open the file!" << std::endl;
        return 1;
    }
    
    std::string line;
    while (std::getline(inputFile, line)) {
        if (check_rep(line)) {
            outputFile << line << std::endl;
        }
    }
    
    inputFile.close();
    outputFile.close();
}


bool check_rep(const std::string& line, int rep_support, double rep_conf) {
    std::size_t lastOpenBracket = line.rfind('[');
    std::size_t lastCloseBracket = line.rfind(']');
    
    if (lastOpenBracket != std::string::npos && lastCloseBracket != std::string::npos && lastOpenBracket < lastCloseBracket) {
        std::string lastList = line.substr(lastOpenBracket + 1, lastCloseBracket - lastOpenBracket - 1);
        std::istringstream iss(lastList);
        std::string value;
        std::vector<double> values;
        
        while (std::getline(iss, value, ',')) {
            values.push_back(std::stod(value));
        }
        
        if (values.size() == 2) {
            return values[0] >= rep_support && values[1] >= rep_conf;
        }
    }
    return false;
}




namespace fs = std::filesystem;

template <class Pattern, class DataGraph>
void makex_rep_discovery(const std::string& pattern_file, const std::string& candidate_predicates_file, int rep_support, double rep_conf, double rep_to_path_ratio, int topn_each_node_predicates, double sort_by_support_weights, 
                    std::string v_file, std::string e_file, std::string ml_train_file,
                    double delta_l, double delta_r, int user_offset, std::string rep_file_generate, 
                    std::string rep_file_generate_support_conf, std::string edge_label_reverse_csv, 
                    std::string rep_file_generate_support_conf_none_support, int num_process) {

    bool using_cache = false;

    DataGraphWithInformation *data_graph_ptr = new DataGraphWithInformation();
    GUNDAM::ReadCSVGraph(data_graph_ptr->data_graph(), v_file, e_file);
    for (auto vertex_it = data_graph_ptr->data_graph().VertexBegin();
        !vertex_it.IsDone(); vertex_it++) {
    vertex_it->AddAttribute((std::string)("id"), (int)vertex_it->id());
    }
    data_graph_ptr->BuildEncodeHashMap();
    data_graph_ptr->BuildLabelAttributeKeyMap();
    data_graph_ptr->BuildMLModel(ml_train_file, delta_l, delta_r, user_offset);


    std::map<int, std::vector<std::pair<std::string, std::string>>> node_label_predicates;
    GetPredicatesFromFile(candidate_predicates_file, node_label_predicates);

    for (const auto& entry : node_label_predicates) {
        int node_label = entry.first;
        const std::vector<std::pair<std::string, std::string>>& predicates = entry.second;
        for (const auto& predicate : predicates) {
            std::cout << predicate.first << " " << predicate.second << " ";
        }
        std::cout << std::endl;
    }

    std::map<int, std::vector<std::vector<std::string>>> node_predicates;

    std::vector<Makex::REP<Pattern, DataGraph>> rep_pattern_set;

    std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>> path_id_to_predicates;

    std::map<int, std::vector<int>> rep_id_to_path_id;
    std::map<int, std::vector<int>> all_paths;

    GetAllPattern_Path_FromFile(pattern_file, rep_pattern_set, rep_id_to_path_id, all_paths);

    for (const auto& entry : rep_id_to_path_id) {
        int rep_id = entry.first;
        const std::vector<int>& path_ids = entry.second;
        std::cout << "rep_id: " << rep_id << ", path_ids: ";
        for (const auto& path_id : path_ids) {
            std::cout << path_id << " ";
        }
        std::cout << std::endl;
    }

    for (const auto& entry : all_paths) {
        int path_id = entry.first;
        const std::vector<int>& paths = entry.second;
        std::cout << "path_id: " << path_id << ", paths: ";
        for (const auto& path : paths) {
            std::cout << path << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Pattern number: " << rep_pattern_set.size() << std::endl;

    std::map<int, std::pair<int, int>> edge_label_to_node_label;


    GetEdgeLabelToNodeLabel(edge_label_reverse_csv, edge_label_to_node_label);

    std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexID, ValueType_Support_Conf_Pair_Match>>>> match_path_all_vertex_topn_predicates;




    auto start = std::chrono::high_resolution_clock::now();

    auto path_predicates_start = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(num_process);
    #pragma omp parallel for schedule(dynamic)

    for (size_t path_node_index_ = 0; path_node_index_ < all_paths.size(); ++path_node_index_) {

        int path_id =  path_node_index_;
        std::map<PatternVertexPtr, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>> current_path_predicates;

        std::vector<std::pair<int, int>> vertices_path;
        std::vector<std::tuple<int, int, int>> edges_path;

        
        int node_id = 0;
        bool source_id_ = true;
        int who_is_pivot_label = 0;
        int who_is_add_pivot = 0;
        for(auto& edge_label_id : all_paths[path_node_index_]) {
            int source_node_label = edge_label_to_node_label[edge_label_id].first;
            int target_node_label = edge_label_to_node_label[edge_label_id].second;
            if(source_id_){
                source_id_ = false;
                if(source_node_label == 0){
                    who_is_pivot_label  = 0;
                    who_is_add_pivot = 1;
                    vertices_path.push_back(std::make_pair(1,1));
                    vertices_path.push_back(std::make_pair(2, source_node_label));
                    vertices_path.push_back(std::make_pair(3, target_node_label));
                    edges_path.push_back(std::make_tuple(2, 3, edge_label_id));
                }
                if(source_node_label == 1) {
                    who_is_pivot_label  = 1;
                    who_is_add_pivot = 2;
                    vertices_path.push_back(std::make_pair(1, source_node_label));
                    vertices_path.push_back(std::make_pair(2,0));
                    vertices_path.push_back(std::make_pair(3, target_node_label));
                    edges_path.push_back(std::make_tuple(1, 3, edge_label_id));
                }
                node_id = 3;
                continue;
            }
            if(who_is_pivot_label == 0 && !source_id_){
                vertices_path.push_back(std::make_pair(node_id + 1, target_node_label));
                edges_path.push_back(std::make_tuple(node_id, node_id+1, edge_label_id));
                node_id += 1;
            }
            if(who_is_pivot_label == 1 && !source_id_){
                vertices_path.push_back(std::make_pair(node_id + 1, target_node_label));
                edges_path.push_back(std::make_tuple(node_id, node_id+1, edge_label_id));
                node_id += 1;
            }
        }

        Makex::REP<Pattern, DataGraph> rep_path_original(1, 0, 1, 1.0);
                        
        addVerticesAndEdges_Path(rep_path_original, vertices_path, edges_path);

        Pattern &rep_pattern = rep_path_original.pattern();
        std::vector<std::vector<std::pair<PatternVertexPtr, int>>> pattern_to_path_result;    
        GUNDAM::_dp_iso::_DAGDP::GetAllPathsFromRootsToLeaves_with_edge_label(rep_pattern, pattern_to_path_result);

        PatternVertexPtr visited_node;
        bool visited_wllabel_select = false;
        PatternVertexPtr last_not_kg_node;
        for(auto& path : pattern_to_path_result) {
            for (size_t path_node_index = 0; path_node_index < path.size(); ++path_node_index) {

                int node_predicates_num = topn_each_node_predicates;
                std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>> each_vertex_predicates;
                PatternVertexPtr query_vertex = path[path_node_index].first;
                int node_id = query_vertex->id();
                int node_label = query_vertex->label();
                if(node_label == 2 || node_id == who_is_add_pivot) {
                    current_path_predicates[query_vertex] = std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>();
                    continue;
                }

                if(node_label != 2){
                    last_not_kg_node = query_vertex;
                }
                if( path_node_index == 0){
                    if (node_label_predicates.find(node_label) != node_label_predicates.end()) {
                        const auto&  candidate_predicates = node_label_predicates[node_label];
                        for (const auto& predicate : candidate_predicates) {
                            Makex::REP<Pattern, DataGraph> rep_path(1, 0, 1, 1.0);
                            addVerticesAndEdges_Path(rep_path, vertices_path, edges_path);
                            std::string operate = "=";
                            removeQuotes(operate);
                            std::string attributes = predicate.first;
                            removeQuotes(attributes);
                            if(attributes == "wllabel"){
                                continue;
                            }

                            std::string id_str = std::to_string(node_id);
                            int value1_id = std::stoi(id_str);
                            std::string attributes_value = predicate.second;
                            removeQuotes(attributes_value);
                            rep_path.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value1_id, std::string(attributes), std::string(attributes_value), StringToOp(operate),1.0);

                            std::set<int> pivot_match;
                            std::set<int> pivot_match_;
                            std::pair<int, int> match_result = Makex::REPMatchBasePTime_Pivot_Match(rep_path, data_graph_ptr->data_graph(), data_graph_ptr->ml_model(),using_cache, &pivot_match, who_is_pivot_label, &pivot_match_, who_is_pivot_label);
                            int support_tp_match = match_result.first;
                            int support_all_match = match_result.second;
                            double conf_match = 0.0;
                            if(support_tp_match == 0){
                                conf_match = 0.0;
                            }
                            else{
                                conf_match = static_cast<double>(support_tp_match) / support_all_match;
                            }

                            int pivot_match_size = pivot_match.size();

                            std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>> match_temp_predicates_VertexPtr;
                            KeyType_Predicates_VertexPtr match_key_current = std::make_tuple(query_vertex, attributes, attributes_value);
                            ValueType_Support_Conf_Pair_Match match_value_current = std::make_tuple(support_tp_match, conf_match, pivot_match);
                            std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match> match_newMap_current;
                            match_newMap_current[match_key_current] = match_value_current;
                            match_temp_predicates_VertexPtr.push_back(match_newMap_current);
                            if(support_tp_match>0){
                                each_vertex_predicates.push_back(match_temp_predicates_VertexPtr);
                            }
                        }
                    }

                    //printPivotMatch(each_vertex_predicates);
                    int this_node_topn_predicates_num = topn_each_node_predicates + 1;
                    sortVectorByPivotMatchDifference_Consider_Support_Conf(each_vertex_predicates, this_node_topn_predicates_num, query_vertex, sort_by_support_weights, rep_conf);
                    
                    if(each_vertex_predicates.size()>topn_each_node_predicates){
                        int less = each_vertex_predicates.size() - topn_each_node_predicates;
                        node_predicates_num = topn_each_node_predicates +less;
                    }
                    
                    //printPivotMatch(each_vertex_predicates);
                    current_path_predicates[query_vertex] = each_vertex_predicates;
                    if(!current_path_predicates[query_vertex].empty()){
                        visited_node = query_vertex;
                    }
                    else{
                        visited_node = query_vertex;
                    }

                }
                else{
                    
                    std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>> candidate_path_predicates;
                    candidate_path_predicates = current_path_predicates[visited_node];
                    //printPivotMatch(candidate_path_predicates);

                    if (node_label_predicates.find(query_vertex->label()) != node_label_predicates.end()) {
                        auto&  candidate_predicates = node_label_predicates[query_vertex->label()];

                        std::set<std::pair<std::string, std::string>> unique_predicates;

                        PatternVertexPtr source_node_query_vertex = path[path_node_index - 1].first;
                        int source_node_id = source_node_query_vertex->id();
                        int source_node_label = source_node_query_vertex->label();
                        int target_node_label = query_vertex->label();
                        int target_node_id = query_vertex->id();
                        if(source_node_id == 2 && source_node_label == 0 && target_node_label == 1){
                            unique_predicates.emplace("wllabel_variable", std::to_string(target_node_id) + "**1");
                        }

                        if(source_node_id == 1 && source_node_label == 1 && target_node_label == 0){
                            unique_predicates.emplace("wllabel_variable", std::to_string(target_node_id) + "**2");
                        }
                        /*
                        for (const auto& predicate : candidate_predicates) {
                            std::string attributes = predicate.first;
                            std::string attributes_value = predicate.second;
                            if(attributes.find("wllabel_variable") != std::string::npos) {
                                std::string delimiter = "**";
                                size_t pos = attributes_value.find(delimiter);
                                std::string value_before = attributes_value.substr(0, pos);
                                std::string value_after = attributes_value.substr(pos + delimiter.length());
                                int value_before_num = std::stoi(value_before);
                                int value_after_num = std::stoi(value_after);
                            }
                            else {
                                std::string id_str = std::to_string(query_vertex->id());
                                int value1_id = std::stoi(id_str);
                                std::string attributes = predicate.first;
                                std::string operate = "=";                                
                                removeQuotes(operate);
                                removeQuotes(attributes);
                                std::string attributes_value = predicate.second;
                                removeQuotes(attributes_value);
                            }
                        }

                        for (const auto& predicate : unique_predicates) {
                            std::string attributes = predicate.first;
                            std::string attributes_value = predicate.second;
                            if(attributes.find("wllabel_variable") != std::string::npos) {
                                std::string delimiter = "**";
                                size_t pos = attributes_value.find(delimiter);
                                std::string value_before = attributes_value.substr(0, pos);
                                std::string value_after = attributes_value.substr(pos + delimiter.length());
                                int value_before_num = std::stoi(value_before);
                                int value_after_num = std::stoi(value_after);
                            }
                            else {
                                std::string id_str = std::to_string(query_vertex->id());
                                int value1_id = std::stoi(id_str);
                                std::string attributes = predicate.first;
                                std::string operate = "=";
                                
                                removeQuotes(operate);
                                removeQuotes(attributes);
                                std::string attributes_value = predicate.second;
                                removeQuotes(attributes_value);
                            }
                        }
                        */


                        for (const auto& predicate : candidate_predicates) {
                            
                            std::string test_attributes = predicate.first;
                            std::string id_str = std::to_string(query_vertex->id());
                            int value1_id = std::stoi(id_str);

                            if(test_attributes.find("wllabel") != std::string::npos){
                                bool this_wl_added = false;
                                for(const auto& wlpredicate : unique_predicates){

                                    std::string attributes_value = wlpredicate.second;
                                    removeQuotes(attributes_value);

                                    std::string delimiter = "**";
                                    size_t pos = attributes_value.find(delimiter);
                                    std::string value_before = attributes_value.substr(0, pos);
                                    std::string value_after = attributes_value.substr(pos + delimiter.length());
                                    int value_before_num = std::stoi(value_before);
                                    int value_after_num = std::stoi(value_after);
                                    if(value_before_num == value1_id) {
                                        this_wl_added = true;
                                    }
                                    
                                }
                                if(!this_wl_added){
                                    continue;
                                }
                            }

                            for (const auto& match_combinations : candidate_path_predicates) {
                                Makex::REP<Pattern, DataGraph> rep_path(1, 0, 1, 1.0);
                                addVerticesAndEdges_Path(rep_path, vertices_path, edges_path);

                                
                                std::string attributes = predicate.first;
                                std::string operate = "=";
                                
                                removeQuotes(operate);
                                removeQuotes(attributes);
                                std::string attributes_value = predicate.second;
                                removeQuotes(attributes_value);

                                
                                if(attributes.find("wllabel") != std::string::npos){
                                    for(const auto& wlpredicate : unique_predicates){
                                        std::string wl_attributes_value = wlpredicate.second;
                                        removeQuotes(wl_attributes_value);
                                        std::string delimiter = "**";

                                        size_t pos = wl_attributes_value.find(delimiter);
                                        std::string value_before = wl_attributes_value.substr(0, pos);
                                        std::string value_after = wl_attributes_value.substr(pos + delimiter.length());
                                        int value_before_num = std::stoi(value_before);
                                        int value_after_num = std::stoi(value_after);
                                        if(value_before_num != value1_id && value_after_num != value1_id) {
                                            continue;
                                        }
                                        rep_path.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value_before_num, std::string(attributes), std::string(attributes_value), StringToOp(operate),1.0);
                                        rep_path.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value_after_num, std::string(attributes), std::string(attributes_value), StringToOp(operate),1.0);
                                        
                                    }
                                }
                                else{
                                    rep_path.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value1_id, std::string(attributes), std::string(attributes_value), StringToOp(operate),1.0);
                                }

                                
                                std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>> match_temp_predicates_VertexPtr;
                                for (const auto& predicates : match_combinations) {
                                    for (const auto& pair : predicates) {
                                        auto farther_vertex = std::get<0>(pair.first);
                                        std::string farther_id_str = std::to_string(farther_vertex->id());
                                        int farther_id = std::stoi(farther_id_str);
                                        std::string farther_attributes = std::get<1>(pair.first);
                                        removeQuotes(farther_attributes);
                                        std::string farther_attributes_value = std::get<2>(pair.first);
                                        removeQuotes(farther_attributes_value);

                                        std::string operate = "=";
                                        

                                        int farther_supp = std::get<0>(pair.second);double farther_conf = std::get<1>(pair.second);
                                        std::set<int> match_pivot = std::get<2>(pair.second);

                                                        
                                        if(farther_attributes.find("wllabel_select") != std::string::npos){
                                            std::string delimiter = "**";
                                            size_t first_pos = farther_attributes_value.find(delimiter);
                                            std::string value_before = farther_attributes_value.substr(0, first_pos);

                                            size_t second_pos = farther_attributes_value.find(delimiter, first_pos + delimiter.length());
                                            std::string value_between = farther_attributes_value.substr(first_pos + delimiter.length(), second_pos - first_pos - delimiter.length());
                                            
                                            std::string value_after = farther_attributes_value.substr(second_pos + delimiter.length());

                                            int value_before_num = std::stoi(value_before);
                                            int value_between_num = std::stoi(value_between);
                                            std::string wl_attributes_value = value_after;

                                            rep_path.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value_before_num, std::string("wllabel"), std::string(wl_attributes_value), StringToOp(operate),1.0);
                                            rep_path.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value_between_num, std::string("wllabel"), std::string(wl_attributes_value), StringToOp(operate),1.0);

                                        }
                                        else{
                                            rep_path.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(farther_id, std::string(farther_attributes), std::string(farther_attributes_value), StringToOp(operate),1.0);
                                        }
                                        
                                        KeyType_Predicates_VertexPtr key_ = std::make_tuple(farther_vertex, farther_attributes, farther_attributes_value);

                                        std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match> match_newMap_current;
                                        match_newMap_current[key_] = std::make_tuple(farther_supp, farther_conf, match_pivot);
                                        match_temp_predicates_VertexPtr.push_back(match_newMap_current);
                                    }
                                }
                                std::set<int> pivot_match;
                                std::set<int> pivot_match_;
            
                                std::pair<int, int> match_result = Makex::REPMatchBasePTime_Pivot_Match(rep_path, data_graph_ptr->data_graph(), data_graph_ptr->ml_model(),using_cache, &pivot_match, who_is_pivot_label, &pivot_match_, who_is_pivot_label);
                                int support_tp_match = match_result.first;
                                int support_all_match = match_result.second;
                                double conf_match = 0.0;
                                if(support_tp_match == 0){
                                    conf_match = 0.0;
                                }
                                else{
                                    conf_match = static_cast<double>(support_tp_match) / support_all_match;
                                }

                                int pivot_match_size = pivot_match.size();

                                KeyType_Predicates_VertexPtr match_key_current;
                                if(attributes.find("wllabel") != std::string::npos){
                                    std::string current_attributes = "wllabel_select";
                                    int value_before_num = 0;
                                    int value_after_num = 0;
                                    for(const auto& wlpredicate : unique_predicates){
                                        std::string delimiter = "**";
                                        std::string wl_attributes_value = wlpredicate.second;

                                        size_t pos = wl_attributes_value.find(delimiter);
                                        std::string value_before = wl_attributes_value.substr(0, pos);
                                        std::string value_after = wl_attributes_value.substr(pos + delimiter.length());
                                        value_before_num = std::stoi(value_before);
                                        value_after_num = std::stoi(value_after);
                                    }
                                    std::string current_attributes_value = std::to_string(value_before_num) + "**" + std::to_string(value_after_num) + "**" + attributes_value;
                                    match_key_current = std::make_tuple(query_vertex, current_attributes, current_attributes_value);
                                }
                                else{
                                    match_key_current = std::make_tuple(query_vertex, attributes, attributes_value);
                                }
                                
                                ValueType_Support_Conf_Pair_Match match_value_current = std::make_tuple(support_tp_match, conf_match, pivot_match);
                                std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match> match_newMap_current;
                                match_newMap_current[match_key_current] = match_value_current;
                                match_temp_predicates_VertexPtr.push_back(match_newMap_current);
                                if(support_tp_match>0){
                                    each_vertex_predicates.push_back(match_temp_predicates_VertexPtr);
                                }
                            }
                        }
                    }


                    //printPivotMatch(each_vertex_predicates);
                    int this_node_topn_predicates_num = std::pow(node_predicates_num, 1);

                    sortVectorByPivotMatchDifference_Consider_Support_Conf(each_vertex_predicates, this_node_topn_predicates_num, query_vertex, sort_by_support_weights,rep_conf);

                    if(each_vertex_predicates.size()>topn_each_node_predicates){
                        int less = each_vertex_predicates.size() - topn_each_node_predicates;
                        node_predicates_num = topn_each_node_predicates +less;
                    }
                    
                    //printPivotMatch(each_vertex_predicates);

                    current_path_predicates[query_vertex] = each_vertex_predicates;
                    for (const auto& candidate_predicates : each_vertex_predicates) {
                        for (const auto& predicates : candidate_predicates) {
                            for (const auto& map : predicates) {
                                std::string attributes = std::get<1>(map.first);
                                if(attributes == "wllabel_select"){
                                    visited_wllabel_select = true;
                                }
                            }
                        }
                    }


                }            
                if(!current_path_predicates[query_vertex].empty()){
                    visited_node = query_vertex;
                }
            }
        }


        std::vector<std::vector<std::map<KeyType_Predicates_VertexID, ValueType_Support_Conf_Pair_Match>>> match_path_all_vertex_topn_predicates_path_id;
        for(auto &candidate_predicates : current_path_predicates[last_not_kg_node]){
            std::vector<std::map<KeyType_Predicates_VertexID, ValueType_Support_Conf_Pair_Match>> test;
            for (const auto& predicates : candidate_predicates) {
                for (const auto& map : predicates) {
                    auto vertex_ptr = std::get<0>(map.first);
                    int vertex_ID = vertex_ptr->id();
                    std::string attributes = std::get<1>(map.first);
                    std::string attributes_value = std::get<2>(map.first);
                    int support_tp = std::get<0>(map.second);
                    double conf = std::get<1>(map.second);
                    std::set<int> pivot_match = std::get<2>(map.second);

                    KeyType_Predicates_VertexID match_key_current = std::make_tuple(vertex_ID, attributes, attributes_value);
                    ValueType_Support_Conf_Pair_Match match_value_current = std::make_tuple(support_tp, conf, pivot_match);
                    std::map<KeyType_Predicates_VertexID, ValueType_Support_Conf_Pair_Match> match_newMap_current;
                    match_newMap_current[match_key_current] = match_value_current;
                    test.push_back(match_newMap_current);
                }
            }
            match_path_all_vertex_topn_predicates_path_id.push_back(test);
        }

        match_path_all_vertex_topn_predicates[path_id] = match_path_all_vertex_topn_predicates_path_id;
        //printPivotMatch_ID(match_path_all_vertex_topn_predicates[path_id]);
        std::cout << std::endl << std::endl;


    }

   auto path_predicates_stop = std::chrono::high_resolution_clock::now();
   auto duration_path_predicates = std::chrono::duration<double>(path_predicates_stop - path_predicates_start);

   std::cout << "Time taken by get all path candidate preconditions: "
         << duration_path_predicates.count() << " seconds" << std::endl;


    int rep_id = -1;

    auto pattern_predicates_start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_process);
    #pragma omp parallel for schedule(dynamic)
    for (size_t rep_index_ = 0; rep_index_ < rep_pattern_set.size(); ++rep_index_) {
        auto rep = rep_pattern_set[rep_index_];
        rep_id = rep_index_;
        std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexID, ValueType_Support_Conf_Pair_Match>>>> each_path_all_predicates;
        std::map<int, std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>>>> match_rep_all_predicates_test;

        const std::vector<int>& path_ids = rep_id_to_path_id[rep_id];
        for(auto& path_id: path_ids){
            std::cout << "rep id:  " << rep_id << " contained path_id: " << path_id << ", ";
        }


        Pattern &rep_pattern = rep.pattern();
        std::vector<std::vector<std::tuple<int, int, int>>> edges_pattern;
        std::vector<std::pair<int, int>> vertices_pattern_all;
        std::vector<std::tuple<int, int, int>> edges_pattern_all;

        std::vector<std::vector<std::pair<PatternVertexPtr, int>>> pattern_to_path_result;
        GUNDAM::_dp_iso::_DAGDP::GetAllPathsFromRootsToLeaves_with_edge_label(rep_pattern, pattern_to_path_result);



        for(auto& path : pattern_to_path_result){
            std::vector<std::pair<int, int>> vertices_path;
            std::vector<std::tuple<int, int, int>> edges_path;
            std::vector<int> edges_label;
            for(auto& node: path){
                int node_id = node.first->id();
                int node_label = node.first->label();

                edges_label.push_back(node.second);
                vertices_path.push_back(std::make_pair(node_id, node_label));

                auto add_vertex_if_not_exists = [&](int node_id, int node_label) {
                    std::pair<int, int> vertex = {node_id, node_label};
                    if (std::find(vertices_pattern_all.begin(), vertices_pattern_all.end(), vertex) == vertices_pattern_all.end()) {
                        vertices_pattern_all.push_back(vertex);                        
                    }
                };
                add_vertex_if_not_exists(node_id, node_label);

            }


            for (size_t i = 0; i < vertices_path.size() - 1; ++i) {
                int sourceId = vertices_path[i].first;
                int targetId = vertices_path[i+1].first;
                int edgeLabel = edges_label[i];
                edges_path.push_back(std::make_tuple(sourceId, targetId, edgeLabel));
                edges_pattern_all.push_back(std::make_tuple(sourceId, targetId, edgeLabel));
            }
            edges_pattern.push_back(edges_path);

        }

        int this_rep_path_id = 0;
        for(auto & path_id: path_ids){
            each_path_all_predicates[this_rep_path_id] = match_path_all_vertex_topn_predicates[path_id];
            this_rep_path_id += 1;
        }


        int current_pattern_path_id = 0;
        for(auto& path : pattern_to_path_result){
            for(auto &each_vertex_predicates : each_path_all_predicates[current_pattern_path_id]){
                std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match>> match_temp_predicates_VertexPtr;
                for(auto &each_vertex_predicates_VertexID : each_vertex_predicates){
                    for(auto &each_vertex_predicates_VertexID_pair : each_vertex_predicates_VertexID){
                        auto vertex_id = std::get<0>(each_vertex_predicates_VertexID_pair.first);
                        auto attributes = std::get<1>(each_vertex_predicates_VertexID_pair.first);
                        auto attributes_value = std::get<2>(each_vertex_predicates_VertexID_pair.first);
                        auto support_tp = std::get<0>(each_vertex_predicates_VertexID_pair.second);
                        auto conf = std::get<1>(each_vertex_predicates_VertexID_pair.second);
                        auto pivot_match = std::get<2>(each_vertex_predicates_VertexID_pair.second);
                        PatternVertexPtr query_vertex;
                        if(vertex_id == 1){
                            query_vertex = path[vertex_id - 1].first;
                        }
                        else if(vertex_id == 2){
                            query_vertex = path[vertex_id - 2].first;
                        }
                        else{
                            if(vertex_id > 10){
                                query_vertex = path.back().first;
                            }
                            else{
                                query_vertex = path[vertex_id - 2].first;
                            }
                        }
                        KeyType_Predicates_VertexPtr match_key_current = std::make_tuple(query_vertex, attributes, attributes_value);
                        ValueType_Support_Conf_Pair_Match match_value_current = std::make_tuple(support_tp, conf, pivot_match);
                        std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf_Pair_Match> match_newMap_current;
                        match_newMap_current[match_key_current] = match_value_current;
                        if(support_tp>0){
                            match_temp_predicates_VertexPtr.push_back(match_newMap_current);
                        }

                    }
                }
                match_rep_all_predicates_test[current_pattern_path_id].push_back(match_temp_predicates_VertexPtr);
            }
            current_pattern_path_id = current_pattern_path_id + 1;
        }

        std::cout << std::endl;
        //printRepAllPredicatesTest_Match(match_rep_all_predicates_test);

        std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>> currentPath;
        std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>> old_allCombinations;
        std::vector<std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>>> allCombinations;

        generateCartesian_Match(match_rep_all_predicates_test, old_allCombinations, currentPath, match_rep_all_predicates_test.begin());

        std::vector<std::string> vector_rep_predicates;
        for (auto& combo : old_allCombinations) {
            std::vector<int> wllabel_id;
            for (auto& vec : combo) {
                for (auto& vecMap : vec) {
                    int vertex_id = std::get<0>(vecMap.begin()->first)->id();
                    std::string currentWllabel = std::get<1>(vecMap.begin()->first);                    
                    if(currentWllabel == "wllabel_select"){
                        wllabel_id.push_back(vertex_id);
                    }
                }
            }

            if(wllabel_id.size() != 0){

                int seed = 1996;

                std::mt19937 gen(seed);

                std::uniform_int_distribution<> distrib(0, wllabel_id.size() - 1);

                int randomIndex = distrib(gen);
                int randomValue = wllabel_id[randomIndex];

                std::vector<int> visited_node;
                visited_node.push_back(1);
                visited_node.push_back(2);
                std::vector<std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>>> newCombo;

                std::string rep_predicates;
                int wl_pivot_id = 1;
                bool wl_pivot = false;
                for (auto& vec : combo) {
                    std::vector<std::map<KeyType_Predicates_VertexPtr, ValueType_Support_Conf>> newComboPath;
                    for (auto& vecMap : vec) {
                        int vertex_id = std::get<0>(vecMap.begin()->first)->id();
                        if(std::find(visited_node.begin(), visited_node.end(), vertex_id) != visited_node.end()){
                            continue;
                        }
                        visited_node.push_back(vertex_id);
                        std::string currentWllabel = std::get<1>(vecMap.begin()->first);
                        std::string currentWllabel_value = std::get<2>(vecMap.begin()->first);
                        if(currentWllabel == "wllabel_select"){
                            std::string delimiter = "**";
                            size_t first_pos = currentWllabel_value.find(delimiter);
                            std::string value_before = currentWllabel_value.substr(0, first_pos);

                            size_t second_pos = currentWllabel_value.find(delimiter, first_pos + delimiter.length());
                            std::string value_between = currentWllabel_value.substr(first_pos + delimiter.length(), second_pos - first_pos - delimiter.length());
                            
                            std::string value_after = currentWllabel_value.substr(second_pos + delimiter.length());

                            int value_before_num = std::stoi(value_before);
                            int value_between_num = std::stoi(value_between);
                            std::string wl_attributes_value = value_after;

                            wl_pivot_id = value_between_num;
                        }



                        if(currentWllabel == "wllabel_select"){
                            if(vertex_id == randomValue){
                                wl_pivot = true;
                                newComboPath.push_back(vecMap);
                                rep_predicates += std::to_string(vertex_id) + "**" + currentWllabel + "**" + currentWllabel_value + "**";
                            }
                        }
                        else{
                            if(!wl_pivot){
                                
                                if(vertex_id != randomValue){
                                    newComboPath.push_back(vecMap);
                                    rep_predicates += std::to_string(vertex_id) + "**" + currentWllabel + "**" + currentWllabel_value + "**";
                                }
                            }
                            else{
                                if(vertex_id != wl_pivot_id){
                                    newComboPath.push_back(vecMap);
                                    rep_predicates += std::to_string(vertex_id) + "**" + currentWllabel + "**" + currentWllabel_value + "**";
                                }
                            }
                        }
                    }
                    newCombo.push_back(newComboPath);
                
                }



                if(vector_rep_predicates.size() == 0){
                    vector_rep_predicates.push_back(rep_predicates);
                    allCombinations.push_back(newCombo);
                    printTopkPathPredicates_map(newCombo);
                }
                else{
                    if(std::find(vector_rep_predicates.begin(), vector_rep_predicates.end(), rep_predicates) != vector_rep_predicates.end()){
                        continue;
                    }
                    else{
                        vector_rep_predicates.push_back(rep_predicates);
                        allCombinations.push_back(newCombo);
                        printTopkPathPredicates_map(newCombo);
                    }
                }
            }

        }

        writeRepToFile_vector(rep_file_generate, rep_pattern, edges_pattern, allCombinations);

        std::vector<ValueType_Support_Conf> final_rep_support_conf;
        std::vector<std::tuple<std::set<int>, int, int>> final_rep_pivot_support;

        int pivot_match_union_index = 0;
        for (const auto& combinationSet : allCombinations) {
            bool variables_used = false;
            int variables = 0;
            Makex::REP<Pattern, DataGraph> rep_calculate_support(1, 0, 1, 1.0);
            addVerticesAndEdges_Path(rep_calculate_support, vertices_pattern_all, edges_pattern_all);

            for (size_t j = 0; j < combinationSet.size(); ++j) {
                const auto&  all_predicates = combinationSet[j];
                for (size_t i = 0; i < all_predicates.size(); ++i) {
                    const auto& predicate = all_predicates[i];
                    for(const auto& pair : predicate){
                        auto farther_vertex = std::get<0>(pair.first);
                        std::string farther_id_str = std::to_string(farther_vertex->id());
                        int farther_id = std::stoi(farther_id_str);
                        std::string farther_attributes = std::get<1>(pair.first);
                        removeQuotes(farther_attributes);
                        std::string farther_attributes_value = std::get<2>(pair.first);
                        removeQuotes(farther_attributes_value);
                        std::string operate = "=";
                        
                        removeQuotes(operate);

                        int farther_supp = pair.second.first;
                        double farther_conf = pair.second.second;

                        if(farther_attributes.find("wllabel_select") != std::string::npos){

                            std::string delimiter = "**";
                            size_t first_pos = farther_attributes_value.find(delimiter);
                            std::string value_before = farther_attributes_value.substr(0, first_pos);

                            size_t second_pos = farther_attributes_value.find(delimiter, first_pos + delimiter.length());
                            std::string value_between = farther_attributes_value.substr(first_pos + delimiter.length(), second_pos - first_pos - delimiter.length());
                            
                            std::string value_after = farther_attributes_value.substr(second_pos + delimiter.length());

                            int value_before_num = std::stoi(value_before);
                            int value_between_num = std::stoi(value_between);
                            std::string wl_attributes_value = value_after;


                            rep_calculate_support.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(farther_id, std::string("wllabel"), std::string(wl_attributes_value), StringToOp(operate),1.0);
                            rep_calculate_support.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(value_between_num, std::string("wllabel"), std::string(wl_attributes_value), StringToOp(operate),1.0);

                            variables += 1;

                        }
                        else{
                            rep_calculate_support.template Add<Makex::ConstantPredicate<Pattern, DataGraph, std::string>>(farther_id, std::string(farther_attributes), std::string(farther_attributes_value), StringToOp(operate),1.0);
                        }

                    }
                }
            }


            std::set<int> pivot_match_item;
            int who_is_pivot_label_item = 0;
            std::set<int> pivot_match_user;
            int who_is_pivot_label_user = 1;
            std::pair<int, int> result_rep = Makex::REPMatchBasePTime_Pivot_Match(rep_calculate_support, data_graph_ptr->data_graph(), data_graph_ptr->ml_model(),false, &pivot_match_item, who_is_pivot_label_item, &pivot_match_user, who_is_pivot_label_user);

            std::set<int> pivot_match_union;
            pivot_match_union.insert(pivot_match_item.begin(), pivot_match_item.end());
            pivot_match_union.insert(pivot_match_user.begin(), pivot_match_user.end());
            
            int support_tp = result_rep.first;
            int support_all = result_rep.second;
            double conf = 0.0;
            if(support_tp == 0){
                conf = 0.0;
            }
            else{
                conf = static_cast<double>(support_tp) / support_all;
            }
            final_rep_support_conf.push_back({support_tp, conf});

            final_rep_pivot_support.push_back({pivot_match_union, pivot_match_union_index, support_tp});
            pivot_match_union_index += 1;
        }


       sort(final_rep_pivot_support.begin(), final_rep_pivot_support.end(), [](const auto& a, const auto& b) {
            return std::get<2>(a) > std::get<2>(b);
        });

        std::vector<std::set<int>> selected_sets;
        std::vector<int> sortedVector;

        if (!final_rep_pivot_support.empty()) {
            selected_sets.push_back(std::get<0>(final_rep_pivot_support[0]));
            sortedVector.push_back(std::get<1>(final_rep_pivot_support[0]));
            final_rep_pivot_support.erase(final_rep_pivot_support.begin());
        }


        while (!final_rep_pivot_support.empty()) {
            auto max_diff_it = std::max_element(final_rep_pivot_support.begin(), final_rep_pivot_support.end(), [&](const auto& lhs, const auto& rhs) {
                int lhs_diff = 0, rhs_diff = 0;
                
                std::set<int> merged_set;

                for (const auto& s : selected_sets) {
                    merged_set.insert(s.begin(), s.end());
                }

                lhs_diff = SetDifference(std::get<0>(lhs), merged_set);
                rhs_diff = SetDifference(std::get<0>(rhs), merged_set);

                int lhs_support_tp = std::get<0>(lhs).size();
                int rhs_support_tp = std::get<0>(rhs).size();


                lhs_diff = sort_by_support_weights * lhs_support_tp + (1-sort_by_support_weights) * lhs_diff;
                rhs_diff = sort_by_support_weights * rhs_support_tp + (1-sort_by_support_weights) * rhs_diff;


                return lhs_diff < rhs_diff;
            });

            selected_sets.push_back(std::get<0>(*max_diff_it));
            sortedVector.push_back(std::get<1>(*max_diff_it));
            final_rep_pivot_support.erase(max_diff_it);

        }
        writeRepToFile_support_conf(rep_file_generate_support_conf, rep_pattern, edges_pattern, allCombinations, final_rep_support_conf, sortedVector);
    }


    auto pattern_predicates_end = std::chrono::high_resolution_clock::now();
    auto duration_pattern = std::chrono::duration<double>(pattern_predicates_end - pattern_predicates_start);

    std::cout << "Time taken by generate REP: " << duration_pattern.count() << " seconds" << std::endl;

    remove_zero_rows_and_save(rep_file_generate_support_conf, rep_file_generate_support_conf_none_support, rep_support, rep_conf);


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(stop - start);
    std::cout << "Time taken by all: " << duration.count() << " seconds" << std::endl;

}





void clearFile(const std::string& filename) {
    std::ofstream file(filename, std::ofstream::out | std::ofstream::trunc);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file for clearing: " << filename << std::endl;
    } else {
        std::cout << "File cleared successfully: " << filename << std::endl;
file.close();    }
}


int main(int argc, char* argv[]) {

    std::string pattern_file = argv[1];
    std::string candidate_predicates_file = argv[2];

    int rep_support = std::stoi(argv[3]);

    double rep_conf = std::stod(argv[4]);
    double rep_to_path_ratio = std::stod(argv[5]);
    int each_node_predicates = std::stoi(argv[6]);
    double sort_by_support_weights = std::stod(argv[7]);
    std::string v_file = argv[8];
    std::string e_file = argv[9];
    std::string ml_file = argv[10];
    double delta_l = std::stod(argv[11]);
    double delta_r = std::stod(argv[12]);
    int user_offset = std::stoi(argv[13]);
    std::string rep_file_generate = argv[14];
    std::string rep_file_generate_support_conf = argv[15];
    std::string edge_label_reverse_csv = argv[16];
    std::string rep_file_generate_support_conf_none_support = argv[17];
    int num_process = std::stoi(argv[18]);


    clearFile(rep_file_generate);
    clearFile(rep_file_generate_support_conf);
    clearFile(rep_file_generate_support_conf_none_support);

    makex_rep_discovery<Pattern,DataGraph>(pattern_file, candidate_predicates_file, rep_support, rep_conf, rep_to_path_ratio, each_node_predicates, sort_by_support_weights, v_file, e_file, ml_file, delta_l, delta_r, user_offset, rep_file_generate, rep_file_generate_support_conf, edge_label_reverse_csv, rep_file_generate_support_conf_none_support, num_process);

    return 0;
}





