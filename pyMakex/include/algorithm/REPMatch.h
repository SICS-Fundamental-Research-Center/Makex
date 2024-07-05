#ifndef _REPMATCH_H
#define _REPMATCH_H

#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <string>
#include <queue>
#include <set>
#include <algorithm>
#include <limits>
#include <numeric>
#include <chrono>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include "../include/gundam/algorithm/dp_iso.h"
#include "../include/gundam/type_getter/vertex_handle.h"
#include "../structure/REP.h"
#include "../include/structure/Predicate.h"
#include "../include/gundam/match/matchresult.h"
#include "../include/gundam/component/util.h"
#include "../include/gundam/algorithm/vf2.h"

using MyCompare = std::function<bool(std::pair<double, std::map<int, int>>, std::pair<double, std::map<int, int>>)>;

namespace Makex {

inline double sigmoid(double x, double k = 1.0) {
    return 1.0 / (1.0 + exp(-k * x));
}

inline double score_transform(double ori_score) {
  double new_score = (1.0 - (1.0 / (1.0 + ori_score))) + 1e-6;
  return new_score;
}

template <typename Container>struct container_hash {
    std::size_t operator()(Container const& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};
template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline void TransformIndexToTargetVector(
  const std::map<PatternVertexPtr,
          std::vector<DataVertexPtr>> &candidate_set,
  const std::vector<PatternVertexPtr> &query_vertex_vector,
  const std::vector<int> node_match_group_info,
  std::vector<DataVertexPtr>& target_vertex_vector
) {
    for (int idx = 0; idx < node_match_group_info.size(); ++idx) {
        PatternVertexPtr current_query_vertex = query_vertex_vector.at(idx);
        int node_index = node_match_group_info.at(idx);
        DataVertexPtr current_target_vertex = candidate_set.at(current_query_vertex).at(node_index);
        target_vertex_vector.emplace_back(current_target_vertex);
    }
}

template <
    class Pattern,
    class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline void PredicateCheckDataVertex(
  const Predicate<Pattern, DataGraph>* single_predicate,
  const std::vector<std::vector<std::pair<double, std::map<PatternVertexPtr, DataVertexPtr>>>>& all_path_topk_matches_ptr_result,
  const std::map<const PatternVertexPtr, int>& pv_ptr_to_path_id_map,
  std::vector<std::vector<int>>& predicate_check_false_record_vector,
  std::unordered_set<std::vector<int>, container_hash<std::vector<int>>>& predicate_check_false_record_set
) {
  const PatternVertexPtr x = single_predicate->GetX().first;
  const PatternVertexPtr y = single_predicate->GetY().first;
  int x_path_id = pv_ptr_to_path_id_map.at(x); 
  int y_path_id = pv_ptr_to_path_id_map.at(y); 

  std::map<PatternVertexPtr, std::vector<DataVertexPtr>> fake_candidate_set;


  int x_rank = 0;
  for (auto x_match_instance: all_path_topk_matches_ptr_result[x_path_id]) {
    DataVertexPtr x_path_data_vertex;
    DataVertexPtr y_path_data_vertex;


    for (const auto& pair : x_match_instance.second) {
      if ((pair.first)->id() == x->id()) {
        x_path_data_vertex = x_match_instance.second[x];
        break;
      }
    }



    int y_rank = 0;
    for (auto y_match_instance: all_path_topk_matches_ptr_result[y_path_id]) {


      for (const auto& pair : y_match_instance.second) {
        if ((pair.first)->id() == y->id()) {
          y_path_data_vertex = y_match_instance.second[y];
          fake_candidate_set[x] = std::vector<DataVertexPtr>({x_path_data_vertex});
          fake_candidate_set[y] = std::vector<DataVertexPtr>({y_path_data_vertex});

          if (!single_predicate->PredicateCheck(fake_candidate_set)) {
            predicate_check_false_record_vector.emplace_back(std::vector<int>({x_path_id, y_path_id, x_rank, y_rank}));
            predicate_check_false_record_set.insert(std::vector<int>({x_path_id, y_path_id, x_rank, y_rank}));
            predicate_check_false_record_set.insert(std::vector<int>({y_path_id, x_path_id, y_rank, x_rank}));
          }
        }
      }

      ++y_rank;
    }
    ++x_rank;
  }
}

template <
    class Pattern,
    class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline bool PathEdgeMatches(
    std::vector<PatternVertexPtr> query_vertex_vector,
    std::vector<DataVertexPtr> target_vertex_vector) {
    if (target_vertex_vector.size() == 1){
      return true;
    }
    
    for(int i = 0; i < target_vertex_vector.size()-1 && i < query_vertex_vector.size() - 1; i++){
      bool find_flag = false;

      auto query_found_edge_ite = query_vertex_vector[0]->OutEdgeBegin();
      for (auto query_edge_it = query_vertex_vector[i]->OutEdgeBegin(); !query_edge_it.IsDone(); query_edge_it++) {
        if (query_edge_it->dst_handle() == query_vertex_vector[i+1]) {
            query_found_edge_ite = query_edge_it;
            find_flag = true;
            break;
        }
      }
      if(!find_flag) {
        return false;
      }

      for (auto target_edge_it = target_vertex_vector[i]->OutEdgeBegin(); !target_edge_it.IsDone(); target_edge_it++) {
        if (target_edge_it->label() == query_found_edge_ite->label() && target_edge_it->dst_handle() == target_vertex_vector[i+1]) {
            find_flag = true;
            break;
        }
      }
      if(!find_flag){
        return false;
      }
    }

    return true;
}



template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline bool EdgeMatches(
    const REP<Pattern, DataGraph> &rep,
    PatternVertexPtr query_vertex, 
    DataVertexPtr target_vertex, 
    bool is_out,
    const std::map<PatternVertexPtr, DataVertexPtr>& current_match) {

    for (auto edge_it = (is_out ? query_vertex->OutEdgeBegin() : query_vertex->InEdgeBegin()); !edge_it.IsDone(); edge_it++) {
        PatternVertexPtr next_query_vertex = is_out ? edge_it->dst_handle() : edge_it->src_handle();
        if (current_match.find(next_query_vertex) == current_match.end()) continue;

        DataVertexPtr next_target_vertex = current_match.at(next_query_vertex);
        bool find_flag = false;

        for (auto target_edge_it = is_out ? target_vertex->OutEdgeBegin() : target_vertex->InEdgeBegin(); !target_edge_it.IsDone(); target_edge_it++) {
            if (target_edge_it->label() == edge_it->label() && 
                (is_out ? target_edge_it->dst_handle() : target_edge_it->src_handle()) == next_target_vertex) {
                find_flag = true;
                break;
            }
        }

        if (!find_flag) return false;
    }
    
    return true;
}


struct Tuple {
    double sum;
    std::vector<int> indices;

    bool operator<(const Tuple& other) const {
return sum > other.sum;    }
};

double get_difference(const std::vector<double>& arr, int idx1, int idx2) {
    if ((idx1 + 1) < arr.size() && (idx2 + 1) < arr.size()) {
        return arr[idx1] - arr[idx2];
    }
    return std::numeric_limits<double>::infinity();
}


struct PathMatchingResult {
    std::vector<double> values;
    std::vector<int> indices;
};






void InterPathTopkCombinations(const std::vector<std::vector<double>>& arrays, const int& topk, std::vector<PathMatchingResult>& results) {
    int num_arrays = arrays.size();
    Tuple initial;
    initial.sum = 0;
    for (const auto& arr : arrays) {
        initial.sum -= arr[0];
        initial.indices.push_back(0);
    }

    std::priority_queue<Tuple> max_heap;
    max_heap.push(initial);

    std::set<std::vector<int>> seen;
    seen.insert(initial.indices);

    while (!max_heap.empty() && results.size() < topk) {
        Tuple current = max_heap.top();
        max_heap.pop();

        PathMatchingResult result;
        result.values.push_back(-current.sum);
        for (int i = 0; i < num_arrays; ++i) {
            result.values.push_back(arrays[i][current.indices[i]]);
        }
        result.indices = current.indices;
        results.push_back(result);

        std::vector<double> differences(num_arrays);
        for (int i = 0; i < num_arrays; ++i) {
            differences[i] = get_difference(arrays[i], current.indices[i], current.indices[i] + 1);
        }
        std::vector<int> sorted_diff_indices(num_arrays);
        std::iota(sorted_diff_indices.begin(), sorted_diff_indices.end(), 0);
        std::sort(sorted_diff_indices.begin(), sorted_diff_indices.end(),
             [&](int a, int b) { return differences[a] < differences[b]; });

        for (int diff_idx : sorted_diff_indices) {
            if (current.indices[diff_idx] + 1 < arrays[diff_idx].size()) {
                Tuple new_tuple = current;
                new_tuple.indices[diff_idx]++;
                new_tuple.sum += differences[diff_idx];
                if (seen.find(new_tuple.indices) == seen.end()) {
                    max_heap.push(new_tuple);
                    seen.insert(new_tuple.indices);
                }
            }
        }
    }
}







double JaccardSimilarity(const std::map<int, int>& map1, const std::map<int, int>& map2) {
    std::unordered_set<int> set1, set2;

    for (const auto& pair : map1) {
        set1.insert(pair.second);
    }

    for (const auto& pair : map2) {
        set2.insert(pair.second);
    }

    size_t intersection_size = 0;
    for (const auto& item : set1) {
        if (set2.find(item) != set2.end()) {
            intersection_size++;
        }
    }

    size_t union_size = set1.size() + set2.size() - intersection_size;

    return static_cast<double>(intersection_size) / union_size;
}




template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
void InnerPathTopkCalculation_no_Jaccard(const std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set,
                              const std::vector<PatternVertexPtr>& query_vertex_vector, 
                              const std::vector<std::vector<double>>& arrays, 
                              const int& topk, 
                              std::vector<std::pair<double, std::map<int, int>>>& path_topk_matches_result,
                              std::vector<std::pair<double, std::map<PatternVertexPtr, DataVertexPtr>>>& path_topk_matches_ptr_result,
                              const bool& record_path_topk_matches_ptr_result = false
                              ) {

    int num_arrays = arrays.size();

    Tuple initial;
    initial.sum = 0;
    for (const auto& arr : arrays) {
        initial.sum -= arr[0];
        initial.indices.push_back(0);
    }

    std::priority_queue<Tuple> max_heap;
    max_heap.push(initial);

std::set<std::vector<int>> seen;    seen.insert(initial.indices);

    std::vector<PathMatchingResult> results;
    while (!max_heap.empty() && results.size() < topk) {
        Tuple current = max_heap.top();
        max_heap.pop();

        PathMatchingResult result;
        result.values.push_back(-current.sum);
        for (int i = 0; i < num_arrays; ++i) {
            result.values.push_back(arrays[i][current.indices[i]]);
        }
        result.indices = current.indices;

        std::vector<DataVertexPtr> target_vertex_vector;
        TransformIndexToTargetVector<Pattern, DataGraph, PatternVertexPtr, DataVertexPtr>(
            candidate_set, query_vertex_vector, result.indices, target_vertex_vector
        );
        if (PathEdgeMatches<Pattern, DataGraph, PatternVertexPtr, DataVertexPtr>(query_vertex_vector, target_vertex_vector)) {
            results.push_back(result);
        }

        std::vector<double> differences(num_arrays);
        for (int i = 0; i < num_arrays; ++i) {
            differences[i] = get_difference(arrays[i], current.indices[i], current.indices[i] + 1);
        }

        std::vector<int> sorted_diff_indices(num_arrays);
        std::iota(sorted_diff_indices.begin(), sorted_diff_indices.end(), 0);
        std::sort(sorted_diff_indices.begin(), sorted_diff_indices.end(),
             [&](int a, int b) { return differences[a] < differences[b]; });

        for (int diff_idx : sorted_diff_indices) {
            if (current.indices[diff_idx] + 1 < arrays[diff_idx].size()) {
                Tuple new_tuple = current;
                new_tuple.indices[diff_idx]++;
                new_tuple.sum += differences[diff_idx];
                if (seen.find(new_tuple.indices) == seen.end()) {
                    max_heap.push(new_tuple);
                    seen.insert(new_tuple.indices);
                }
            }
        }
    }

    {
        for (const auto & node_match_group_info: results) {
            std::map<int, int> temp_match;
            std::map<PatternVertexPtr, DataVertexPtr> temp_ptr_match;

            for (int idx = 0; idx < node_match_group_info.indices.size(); ++idx) {
                PatternVertexPtr current_query_vertex = query_vertex_vector.at(idx);
                int node_index = node_match_group_info.indices.at(idx);
                DataVertexPtr current_target_vertex = candidate_set.at(current_query_vertex).at(node_index);

                temp_match[current_query_vertex->id()] = current_target_vertex->id();
                temp_ptr_match[current_query_vertex] = current_target_vertex;
            }

            path_topk_matches_result.emplace_back(std::make_pair(node_match_group_info.values[0], temp_match));
            if (record_path_topk_matches_ptr_result) {
              path_topk_matches_ptr_result.emplace_back(std::make_pair(node_match_group_info.values[0], temp_ptr_match));
            }
        }
    }
}





template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
void InnerPathTopkCalculation(const std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set,
                              const std::vector<PatternVertexPtr>& query_vertex_vector, 
                              const std::vector<std::vector<double>>& arrays, 
                              const int& topk, 
                              std::vector<std::pair<double, std::map<int, int>>>& path_topk_matches_result,
                              std::vector<std::pair<double, std::map<PatternVertexPtr, DataVertexPtr>>>& path_topk_matches_ptr_result,
                              std::vector<std::vector<std::pair<double, std::map<int, int>>>>& all_path_topk_matches_result,
                              const bool& record_path_topk_matches_ptr_result = false
                              ) {


    int num_arrays = arrays.size();

    Tuple initial;
    initial.sum = 0;
    for (const auto& arr : arrays) {
        initial.sum -= arr[0];
        initial.indices.push_back(0);
    }

    std::priority_queue<Tuple> max_heap;
    max_heap.push(initial);

std::set<std::vector<int>> seen;    seen.insert(initial.indices);


    std::vector<PathMatchingResult> results;
    while (!max_heap.empty() && results.size() < topk*50) {
        Tuple current = max_heap.top();
        max_heap.pop();

        PathMatchingResult result;
        result.values.push_back(-current.sum);
        for (int i = 0; i < num_arrays; ++i) {
            result.values.push_back(arrays[i][current.indices[i]]);
        }
        result.indices = current.indices;

        std::vector<DataVertexPtr> target_vertex_vector;
        TransformIndexToTargetVector<Pattern, DataGraph, PatternVertexPtr, DataVertexPtr>(
            candidate_set, query_vertex_vector, result.indices, target_vertex_vector
        );
        if (PathEdgeMatches<Pattern, DataGraph, PatternVertexPtr, DataVertexPtr>(query_vertex_vector, target_vertex_vector)) {
            results.push_back(result);
        }

        std::vector<double> differences(num_arrays);
        for (int i = 0; i < num_arrays; ++i) {
            differences[i] = get_difference(arrays[i], current.indices[i], current.indices[i] + 1);
        }

        std::vector<int> sorted_diff_indices(num_arrays);
        std::iota(sorted_diff_indices.begin(), sorted_diff_indices.end(), 0);
        std::sort(sorted_diff_indices.begin(), sorted_diff_indices.end(),
             [&](int a, int b) { return differences[a] < differences[b]; });

        for (int diff_idx : sorted_diff_indices) {
            if (current.indices[diff_idx] + 1 < arrays[diff_idx].size()) {
                Tuple new_tuple = current;
                new_tuple.indices[diff_idx]++;
                new_tuple.sum += differences[diff_idx];
                if (seen.find(new_tuple.indices) == seen.end()) {
                    max_heap.push(new_tuple);
                    seen.insert(new_tuple.indices);
                }
            }
        }
    }

    if(results.size() == 0){
      return;
    }

    
    int add_num = 0;
    for (const auto & node_match_group_info: results) {
        std::map<int, int> temp_match;
        std::map<PatternVertexPtr, DataVertexPtr> temp_ptr_match;             

        for (int idx = 0; idx < node_match_group_info.indices.size(); ++idx) {
            PatternVertexPtr current_query_vertex = query_vertex_vector.at(idx);
            int node_index = node_match_group_info.indices.at(idx);
            DataVertexPtr current_target_vertex = candidate_set.at(current_query_vertex).at(node_index);

            temp_match[current_query_vertex->id()] = current_target_vertex->id();
            temp_ptr_match[current_query_vertex] = current_target_vertex;
        }
        bool add_ = true;
        
        for (const auto& single_path_all_match : all_path_topk_matches_result) {
          if (!add_) {
            break;
          }
          for (const auto& single_path_match : single_path_all_match) {
            double similarity = JaccardSimilarity(temp_match, single_path_match.second);


            if(similarity > 0.55){
              add_ = false;
              break;
            }
          }
        }

        if(add_ && add_num < topk){
          add_num += 1;
          path_topk_matches_result.emplace_back(std::make_pair(node_match_group_info.values[0], temp_match));
          if (record_path_topk_matches_ptr_result) {
            path_topk_matches_ptr_result.emplace_back(std::make_pair(node_match_group_info.values[0], temp_ptr_match));
          }
        }
        if(add_num > topk){
          break;
        }

    }

    while(path_topk_matches_result.size() < topk){

      for (const auto & node_match_group_info: results) {
        std::map<int, int> temp_match;
        std::map<PatternVertexPtr, DataVertexPtr> temp_ptr_match;             

        for (int idx = 0; idx < node_match_group_info.indices.size(); ++idx) {
            PatternVertexPtr current_query_vertex = query_vertex_vector.at(idx);
            int node_index = node_match_group_info.indices.at(idx);
            DataVertexPtr current_target_vertex = candidate_set.at(current_query_vertex).at(node_index);

            temp_match[current_query_vertex->id()] = current_target_vertex->id();
            temp_ptr_match[current_query_vertex] = current_target_vertex;
        }

        if(path_topk_matches_result.size() < topk){
          path_topk_matches_result.emplace_back(std::make_pair(node_match_group_info.values[0], temp_match));
          if (record_path_topk_matches_ptr_result) {
              path_topk_matches_ptr_result.emplace_back(std::make_pair(node_match_group_info.values[0], temp_ptr_match));
          }
        }
        else{
          break;
        }
      }
    }

}



void MergeMatchingIds_pattern_node_num(int pattern_nodes_num, const std::vector<PathMatchingResult>& path_merge_matching_result, 
    const std::vector<std::vector<std::pair<double, std::map<int, int>>>> all_path_topk_matches_result,
    const int& topk,
    const double& rep_score,
    std::priority_queue<std::pair<double, std::map<int, int>>, 
        std::vector<std::pair<double, std::map<int, int>>>, 
        MyCompare> * topk_min_heap_ptr,
    std::vector<std::pair<double, std::map<int, int>>>& matches_with_score,
    double& max_score) {
    if (path_merge_matching_result.size() == 0) {
        return ;
    }
    for (auto & path_matching_result: path_merge_matching_result) {
        std::vector<int> inner_path_index = path_matching_result.indices;

        double merge_score = 0.0;
        std::map<int, int> merge_matches;
        for (int path_num = 0; path_num < inner_path_index.size(); ++path_num) {
            double path_score = all_path_topk_matches_result[path_num][inner_path_index[path_num]].first;
            std::map<int, int> path_match = all_path_topk_matches_result.at(path_num).at(inner_path_index[path_num]).second;

            merge_score += path_score;
            merge_matches.insert(path_match.begin(), path_match.end());
        }


        if(merge_matches.size() != pattern_nodes_num){
          std::cout << "Error Match, matches.size(): " << merge_matches.size() << " pattern_nodes_num: "<< pattern_nodes_num << std::endl;
        }
        
        matches_with_score.emplace_back(std::make_pair(rep_score * score_transform(merge_score), merge_matches));
    }
    std::sort(matches_with_score.begin(), matches_with_score.end(), [](const auto& a, const auto& b) {
       return a.first > b.first;    
    });


    if (matches_with_score.size() >= topk) {
        matches_with_score.resize(topk);
    }

    max_score = matches_with_score[0].first;

    for (auto & match: matches_with_score) {
        topk_min_heap_ptr->push(match);
    }
}


void MergeMatchingIds(const std::vector<PathMatchingResult>& path_merge_matching_result, 
    const std::vector<std::vector<std::pair<double, std::map<int, int>>>> all_path_topk_matches_result,
    const int& topk,
    const double& rep_score,
    std::priority_queue<std::pair<double, std::map<int, int>>, 
        std::vector<std::pair<double, std::map<int, int>>>, 
        MyCompare> * topk_min_heap_ptr,
    std::vector<std::pair<double, std::map<int, int>>>& matches_with_score,
    double& max_score) {
    if (path_merge_matching_result.size() == 0) {
        return ;
    }
    for (auto & path_matching_result: path_merge_matching_result) {
        std::vector<int> inner_path_index = path_matching_result.indices;

        double merge_score = 0.0;
        std::map<int, int> merge_matches;
        for (int path_num = 0; path_num < inner_path_index.size(); ++path_num) {
            double path_score = all_path_topk_matches_result[path_num][inner_path_index[path_num]].first;
            std::map<int, int> path_match = all_path_topk_matches_result.at(path_num).at(inner_path_index[path_num]).second;

            merge_score += path_score;
            merge_matches.insert(path_match.begin(), path_match.end());
        }


        
        matches_with_score.emplace_back(std::make_pair(rep_score * score_transform(merge_score), merge_matches));
    }
    std::sort(matches_with_score.begin(), matches_with_score.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;    
    });


    if (matches_with_score.size() >= topk) {
        matches_with_score.resize(topk);
    }

    max_score = matches_with_score[0].first;

    for (auto & match: matches_with_score) {
        topk_min_heap_ptr->push(match);
    }
}



void MergeMatchingIds_C(const std::vector<PathMatchingResult>& path_merge_matching_result, 
    const std::vector<std::vector<std::pair<double, std::map<int, int>>>> all_path_topk_matches_result,
    const int& topk,
    const double& rep_score,
    std::priority_queue<std::pair<double, std::map<int, int>>, 
        std::vector<std::pair<double, std::map<int, int>>>, 
        MyCompare> * topk_min_heap_ptr,
    std::vector<std::pair<double, std::map<int, int>>>& matches_with_score,
    double& max_score, int& rep_id) {
    if (path_merge_matching_result.size() == 0) {
        return ;
    }
    for (auto & path_matching_result: path_merge_matching_result) {
        std::vector<int> inner_path_index = path_matching_result.indices;

        double merge_score = 0.0;
        std::map<int, int> merge_matches;
        for (int path_num = 0; path_num < inner_path_index.size(); ++path_num) {
            double path_score = all_path_topk_matches_result[path_num][inner_path_index[path_num]].first;
            std::map<int, int> path_match = all_path_topk_matches_result.at(path_num).at(inner_path_index[path_num]).second;

            merge_score += path_score;
            merge_matches.insert(path_match.begin(), path_match.end());
        }

        matches_with_score.emplace_back(std::make_pair(rep_score * score_transform(merge_score), merge_matches));
    }
    std::sort(matches_with_score.begin(), matches_with_score.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;    
    });

    if (matches_with_score.size() >= topk) {
        matches_with_score.resize(topk);
    }

    max_score = matches_with_score[0].first;

    for (auto & match: matches_with_score) {
          double min_score = 0.0;
          if (!topk_min_heap_ptr->empty()) {
              min_score = topk_min_heap_ptr->top().first;
          }
        if (match.first > min_score){

          topk_min_heap_ptr->push(match);
        }
    }

    while (topk_min_heap_ptr->size() > topk) {
      topk_min_heap_ptr->pop();
    }
}




template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline void GenerateAllMatches(
    const REP<Pattern, DataGraph> &rep,
    const Pattern &query_graph,
    const DataGraph &target_graph,
    std::vector<PatternVertexPtr> &topu_seq,
    typename std::vector<PatternVertexPtr>::iterator current,
    std::map<PatternVertexPtr,
             std::vector<DataVertexPtr>> &candidate_set,
    std::map<PatternVertexPtr, DataVertexPtr>& current_match,
    std::vector<std::map<PatternVertexPtr, DataVertexPtr>>& all_matches) {

    if (current == topu_seq.end()) {
        all_matches.push_back(current_match);
        return;
    }

    PatternVertexPtr query_vertex = *current;
    auto &candidates = candidate_set[query_vertex];

    for (DataVertexPtr target_vertex : candidates) {
        if (EdgeMatches(rep, query_vertex, target_vertex, true, current_match) &&
            EdgeMatches(rep, query_vertex, target_vertex, false, current_match)) {

            current_match[query_vertex] = target_vertex;
            GenerateAllMatches(rep, query_graph, target_graph, topu_seq, std::next(current), candidate_set, current_match, all_matches);
            current_match.erase(query_vertex);
        }
    }
}





template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type
    >
inline void GenerateAllMatches_Node_Score(
    const REP<Pattern, DataGraph> &rep,
    const Pattern &query_graph,
    const DataGraph &target_graph,
    std::vector<PatternVertexPtr> &topu_seq,
    typename std::vector<PatternVertexPtr>::iterator current,
     std::map<PatternVertexPtr,
             std::vector<DataVertexPtr>> &candidate_set,
    std::map<PatternVertexPtr, DataVertexPtr>& current_match,
    std::map<int, int>& current_match_id_info,
    const std::map<PatternVertexPtr, std::map<DataVertexPtr,double>>& current_match_instance_node_score,
    std::vector<std::map<PatternVertexPtr, DataVertexPtr>>& all_matches,
    std::vector<std::map<PatternVertexPtr, std::map<DataVertexPtr,double>>>& all_matches_node_score,
    std::vector<std::pair<double, std::map<int, int>>> &matches_with_score,
    std::priority_queue<std::pair<double, std::map<int, int>>, 
        std::vector<std::pair<double, std::map<int, int>>>, 
        MyCompare> * topk_min_heap_ptr, 
    const int& topk,
    double& max_score){

    if (current == topu_seq.end()) {
        all_matches.push_back(current_match);

        double accu_score = 0.0;
        for (const auto &[query_vertex, target_vertex] : current_match) {
            accu_score += current_match_instance_node_score.at(query_vertex).at(target_vertex);
        }

        double transform_score = score_transform(accu_score);
        double final_score = rep.get_rep_score() * transform_score;
        if (final_score > max_score) {
          max_score = final_score;
        }

        if (topk_min_heap_ptr->size() < topk || final_score > topk_min_heap_ptr->top().first) {
          topk_min_heap_ptr->push(std::make_pair(final_score, current_match_id_info));
        
          if (topk_min_heap_ptr->size() > topk) {
            topk_min_heap_ptr->pop();
          }
        }

        return;
    }


    PatternVertexPtr query_vertex = *current;
    auto &candidates = candidate_set[query_vertex];

    for (DataVertexPtr target_vertex : candidates) {
        if (EdgeMatches(rep, query_vertex, target_vertex, true, current_match) &&
          EdgeMatches(rep, query_vertex, target_vertex, false, current_match)) {

          current_match[query_vertex] = target_vertex;
          current_match_id_info[query_vertex->id()] = target_vertex->id();
      
          GenerateAllMatches_Node_Score(rep, query_graph, target_graph, topu_seq, std::next(current), candidate_set, current_match, current_match_id_info, current_match_instance_node_score, all_matches, all_matches_node_score, matches_with_score, topk_min_heap_ptr, topk, max_score);

          current_match.erase(query_vertex);
          current_match_id_info.erase(query_vertex->id());
        }
    }

}



template <class Pattern, class DataGraph>
class CandidateSorter {
 private:
  using PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type;
  using DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type;

  std::unordered_map<PatternVertexPtr, std::unordered_map<DataVertexPtr, double>> precomputed_scores;

  public:
    void UpdateScores(const std::unordered_map<PatternVertexPtr, std::unordered_map<DataVertexPtr, double>>& scores) {
      precomputed_scores = scores;
    }



    void SortCandidateSet(std::map<PatternVertexPtr, std::vector<DataVertexPtr>> *candidate_set) {
      for (auto& it : *candidate_set) {
          PatternVertexPtr query_vertex = it.first;
          std::vector<DataVertexPtr>& waiting_for_sort_vector = it.second;

          std::sort(waiting_for_sort_vector.begin(), waiting_for_sort_vector.end(),
                    [this, &query_vertex](DataVertexPtr a, DataVertexPtr b) {
              auto score_a = this->precomputed_scores[query_vertex][a];
              auto score_b = this->precomputed_scores[query_vertex][b];
              if (score_a != score_b) {
                return score_a > score_b;                      
              } else {
                return a->id() < b->id();
              }
          });
      }
    }

    std::unordered_map<PatternVertexPtr, double> GetMaxScorePerVertex(std::map<PatternVertexPtr, std::vector<DataVertexPtr>>& candidate_set) {
      std::unordered_map<PatternVertexPtr, double> each_node_max_scores;

      for (const auto& vertex_candidates : candidate_set) {
          PatternVertexPtr query_vertex = vertex_candidates.first;
          if (!vertex_candidates.second.empty()) {
            each_node_max_scores[query_vertex] = this->precomputed_scores[query_vertex][vertex_candidates.second[0]];
          } else {
            each_node_max_scores[query_vertex] = 0;
          }
      }
      return each_node_max_scores;
    }


    double GetTotalMaxScore(std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set) {
      double total_max_score = 0.0;
      auto max_scores = GetMaxScorePerVertex(candidate_set);
      for (const auto& score : max_scores) {
        total_max_score += score.second;
      }
      return total_max_score;
    }

};





template <typename T>
void CartesianProduct(const std::vector<std::vector<T>>& input, std::vector<std::vector<T>>& output, std::vector<T> current, int depth) {
    if (depth == input.size()) {
        output.push_back(current);
        return;
    }
    for (const auto& item : input[depth]) {
        std::vector<T> next_current = current;
        next_current.push_back(item);
        CartesianProduct(input, output, next_current, depth + 1);
    }
}

template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline void PrintAllMatches_ALL_Explanation(const REP<Pattern, DataGraph> &rep, const std::vector<std::vector<std::map<PatternVertexPtr,DataVertexPtr>>>& all_path_matches) {
    std::vector<std::vector<std::map<PatternVertexPtr,DataVertexPtr>>> cartesian_results;
    CartesianProduct(all_path_matches, cartesian_results, {}, 0);
    int match_number = 1;
    for (const auto &result : cartesian_results) {
      int path_number = 1;
      for (const auto &path_match : result){
        int path_length = 0;
        for (const auto &[query_vertex, target_vertex] : path_match) {
          path_length++;
        }
        path_number++;
      }
      match_number++;
    }
    //std::cout << "all explanation number: " << match_number << std::endl;
}




template <typename T>
void CartesianProduct_No_save_Results(const std::vector<std::vector<T>>& input, std::vector<std::vector<T>>& output, std::vector<T> current, int depth, int &match_number) {
    if (depth == input.size()) {
        match_number++;
        return;
    }
    for (const auto& item : input[depth]) {
        if (match_number <= static_cast<long long>(std::numeric_limits<int>::max())) {
          std::vector<T> next_current = current;
          next_current.push_back(item);
          CartesianProduct_No_save_Results(input, output, next_current, depth + 1, match_number);
        }
        else{
          return;
        }
    }
}

template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline void PrintAllMatches_ALL_Explanation_No_save_Results(const REP<Pattern, DataGraph> &rep, const std::vector<std::vector<std::map<PatternVertexPtr,DataVertexPtr>>>& all_path_matches) {
    int match_number = 0;
    
    std::vector<std::vector<std::map<PatternVertexPtr,DataVertexPtr>>> cartesian_results;
    int vector_number = 1;
    int total_match = 1;
    for (const auto &result : all_path_matches) {
      //std::cout << "Path #" << vector_number << " has " << result.size() << " number of results." << std::endl;
      total_match = total_match * result.size();
      vector_number++;
    }

    //std::cout << "This REP path number: " << all_path_matches.size() << " and this path all match number is: " << total_match << std::endl;
    auto print_start = std::chrono::high_resolution_clock::now();
    CartesianProduct_No_save_Results(all_path_matches, cartesian_results, {}, 0, match_number);
    //std::cout << "all explanation number: " << match_number << std::endl;
}



template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline void PrintAllMatches_Node_Score(const REP<Pattern, DataGraph> &rep, 
  const std::vector<std::map<PatternVertexPtr,DataVertexPtr>>& all_matches,
  const std::map<PatternVertexPtr, std::map<DataVertexPtr,double>>& current_match_instance_node_score) {
    
    std::vector<std::pair<double, const std::map<PatternVertexPtr, DataVertexPtr>*>> matches_with_score;

    for (const auto &match : all_matches) {
        double accu_score = 0.0;
        for (const auto &[query_vertex, target_vertex] : match) {
            accu_score += current_match_instance_node_score.at(query_vertex).at(target_vertex);
        }
        matches_with_score.emplace_back(accu_score, &match);
    }

    std::sort(matches_with_score.begin(), matches_with_score.end(), 
              [](const auto &a, const auto &b) {
      return a.first > b.first;              
    });

  int match_number = 1;    
  for (const auto &[score, match] : matches_with_score) {
    std::cout << "Match #" << match_number << ":" << std::endl;
    for (const auto &[query_vertex, target_vertex] : *match) {
        std::cout << "Query Vertex ID: " << query_vertex->id()
                  << " -> Target Vertex ID: " << target_vertex->id() << std::endl;
    }
    std::cout << "Score: " << score << std::endl;
    match_number++;
  }
}









template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
double CalculateMatchScore(const REP<Pattern, DataGraph> &rep, const std::map<PatternVertexPtr, std::map<DataVertexPtr, double>> &match) {
    double total_score = 0.0;
    
    for (const auto &[query_vertex, target_vertex_score_map] : match) {
      double w_score = 0.0;
        for (const auto &[target_vertex, score_] : target_vertex_score_map) {
          w_score += score_;
        
        size_t out_degree = 0;
        for (auto edge_it = target_vertex->OutEdgeBegin(); !edge_it.IsDone(); edge_it++) {
              out_degree++;
        }

        if (out_degree == 0){
            total_score += sigmoid(0.00001 * w_score);
        }
        else{
            total_score += sigmoid(log(out_degree) * w_score);
        }
      }

    }
    return total_score;
}




template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline void PrintAllMatches_Node_Score(const REP<Pattern, DataGraph> &rep, std::vector<std::map<PatternVertexPtr, std::map<DataVertexPtr, double>>> &all_matches_node_score) {
    std::sort(all_matches_node_score.begin(), all_matches_node_score.end(),
              [&rep](const auto& match1, const auto& match2) {
return CalculateMatchScore(rep, match1) > CalculateMatchScore(rep, match2);              });

    int match_number = 1;
    for (const auto &match : all_matches_node_score) {
        double rep_total_score = CalculateMatchScore(rep, match);
        std::cout << "Match #" << match_number << ":" << std::endl;
        for (const auto &[query_vertex, target_vertex_score_map] : match) {
            for (const auto &[target_vertex, score_] : target_vertex_score_map) {
                std::cout << "Query Vertex ID: " << query_vertex->id()
                          << " -> Target Vertex ID: " << target_vertex->id() << " Target Vertex Score: " << score_ << std::endl;
            }
        }
        std::cout << "This Match Score: " << rep_total_score << std::endl;
        std::cout << "------------------------" << std::endl;
        match_number++;
    }
}






template <
    class Pattern, class DataGraph,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline int REPMatchBaseBacktrack(
    const REP<Pattern, DataGraph> &rep, const DataGraph &data_graph,
    std::vector<std::pair<DataVertexPtr, DataVertexPtr>> &supp_result) {
  using PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type;
  using MatchMap = std::map<PatternVertexPtr, DataVertexPtr>;
  using CandidateSet = std::map<PatternVertexPtr, std::vector<DataVertexPtr>>;
  CandidateSet candidate_set;
  const Pattern &rep_pattern = rep.pattern();
  auto x_ptr = rep.x_ptr();
  auto y_ptr = rep.y_ptr();
  if (!GUNDAM::_dp_iso::InitCandidateSet<GUNDAM::MatchSemantics::kHomomorphism>(
          rep_pattern, data_graph, candidate_set)) {
    return 0;
  }
  if (!GUNDAM::_dp_iso::RefineCandidateSet(rep_pattern, data_graph,
                                           candidate_set)) {
    return 0;
  }

  auto prune_callback = [&rep](MatchMap &match_state) {
    bool satisfy_flag = true;
    for (auto &single_predicate : rep.x_prediate()) {
      satisfy_flag &= single_predicate->Satisfy(match_state);
      if (!satisfy_flag) {
        break;
      }
    }
    return !satisfy_flag;
  };
  auto user_callback = [&rep, &supp_result](MatchMap &match_state) {
    auto x_ptr = match_state.find(rep.x_ptr())->second;
    auto y_ptr = match_state.find(rep.y_ptr())->second;
    supp_result.emplace_back(x_ptr, x_ptr);
    return false;
  };
  int total_call = 0;
  for (auto &target_x_ptr : candidate_set.find(x_ptr)->second) {
    for (auto &target_y_ptr : candidate_set.find(y_ptr)->second) {
      MatchMap match_state;
      match_state.emplace(x_ptr, target_x_ptr);
      match_state.emplace(y_ptr, target_y_ptr);
      CandidateSet temp_candidate_set(candidate_set);
      GUNDAM::DPISO<GUNDAM::MatchSemantics::kHomomorphism, const Pattern,
                    const DataGraph>(rep_pattern, data_graph,
                                     temp_candidate_set, match_state,
                                     user_callback, prune_callback, -1.0);
    }
  }
  return 1;
}
template <class Pattern, class PatternVertexPtr =
                             typename GUNDAM::VertexHandle<const Pattern>::type>
inline void BuildIndegree(const Pattern &pattern,
                          std::map<PatternVertexPtr, int> &indegree) {
  for (auto vertex_it = pattern.VertexBegin(); !vertex_it.IsDone();
       vertex_it++) {
    for (auto edge_it = vertex_it->OutEdgeBegin(); !edge_it.IsDone();
         edge_it++) {
      PatternVertexPtr dst_ptr = edge_it->dst_handle();
      indegree[dst_ptr]++;
    }
  }
}
template <class Pattern, class DataGraph,
          class PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type,
          class DataVertexPtr = typename GUNDAM::VertexHandle<DataGraph>::type>
inline bool UpdateCandidateSetPush(
    PatternVertexPtr query_vertex_ptr,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set) {
  for (auto edge_it = query_vertex_ptr->OutEdgeBegin(); !edge_it.IsDone();
       edge_it++) {
    PatternVertexPtr dst_ptr = edge_it->dst_handle();
    std::vector<DataVertexPtr> before_dst_candidate(candidate_set[dst_ptr]);
    auto edge_label = edge_it->label();
    std::vector<DataVertexPtr> dst_candidate_res;
    for (auto &target_ptr : candidate_set[query_vertex_ptr]) {
      std::vector<DataVertexPtr> adj_list, intersection_result, union_result;
      for (auto edge_it = target_ptr->OutEdgeBegin(edge_label);
           !edge_it.IsDone(); edge_it++) {
        adj_list.emplace_back(edge_it->dst_handle());
      }
      std::sort(adj_list.begin(), adj_list.end());
      std::set_intersection(
          adj_list.begin(), adj_list.end(), before_dst_candidate.begin(),
          before_dst_candidate.end(),
          inserter(intersection_result, intersection_result.begin()));
      std::set_union(intersection_result.begin(), intersection_result.end(),
                     dst_candidate_res.begin(), dst_candidate_res.end(),
                     inserter(union_result, union_result.begin()));
      std::swap(union_result, dst_candidate_res);
      if (dst_candidate_res.size() == before_dst_candidate.size()) break;
    }
    if (dst_candidate_res.empty()) return false;
    std::swap(candidate_set[dst_ptr], dst_candidate_res);
  }
  return true;
}
template <class Pattern, class DataGraph,
          class PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type,
          class DataVertexPtr = typename GUNDAM::VertexHandle<DataGraph>::type>
inline bool UpdateCandidateSetPull(
    PatternVertexPtr query_vertex_ptr,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set) {
  for (auto edge_it = query_vertex_ptr->OutEdgeBegin(); !edge_it.IsDone();
       edge_it++) {
    PatternVertexPtr dst_ptr = edge_it->dst_handle();
    std::vector<DataVertexPtr> before_dst_candidate(candidate_set[dst_ptr]);
    auto edge_label = edge_it->label();
    std::vector<DataVertexPtr> dst_candidate_res;
    for (auto &target_ptr : candidate_set[dst_ptr]) {
      for (auto vertex_it = target_ptr->InVertexBegin(edge_label);
           !vertex_it.IsDone(); vertex_it++) {
        DataVertexPtr src_ptr = vertex_it;
        bool find_flag =
            std::binary_search(candidate_set[query_vertex_ptr].begin(),
                               candidate_set[query_vertex_ptr].end(), src_ptr);
        if (find_flag) {
          dst_candidate_res.emplace_back(target_ptr);
          break;
        }
      }
    }
    if (dst_candidate_res.empty()) return false;
    std::swap(candidate_set[dst_ptr], dst_candidate_res);
  }
  return true;
}
template <class Pattern, class DataGraph,
          class PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type,
          class DataVertexPtr = typename GUNDAM::VertexHandle<DataGraph>::type>
inline bool UpdateCandidateSet(
    PatternVertexPtr query_vertex_ptr,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set) {
  int push_state = 0, pull_state = 0;
  for (auto edge_it = query_vertex_ptr->OutEdgeBegin(); !edge_it.IsDone();
       edge_it++) {
    PatternVertexPtr dst_ptr = edge_it->dst_handle();
    for (auto &target_ptr : candidate_set[query_vertex_ptr]) {
      push_state += target_ptr->CountOutEdge(edge_it->label());
    }
    for (auto &target_ptr : candidate_set[dst_ptr]) {
      pull_state += target_ptr->CountInEdge(edge_it->label());
    }
  }
  if (pull_state < push_state) {
    return UpdateCandidateSetPull<Pattern, DataGraph>(query_vertex_ptr,
                                                      candidate_set);
  } else {
    return UpdateCandidateSetPush<Pattern, DataGraph>(query_vertex_ptr,
                                                      candidate_set);
  }
}
template <class Pattern, class DataGraph,
          class PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type,
          class DataVertexPtr = typename GUNDAM::VertexHandle<DataGraph>::type>
inline bool RefineCandidateSet(
    Pattern &pattern, DataGraph &data_graph,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set) {
  std::map<PatternVertexPtr, int> indegree;
  BuildIndegree(pattern, indegree);
  std::vector<PatternVertexPtr> topu_seq;
  GUNDAM::_dp_iso::_DAGDP::TopuSort(pattern, indegree, topu_seq);
  for (auto &query_vertex_ptr : topu_seq) {
    if (!UpdateCandidateSet<Pattern, DataGraph>(query_vertex_ptr,
                                                candidate_set)) {
      return false;
    }
  }
  return true;
}
template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline bool RefineCandidateSetFirstRound(
    const Pattern &pattern, const DataGraph &data_graph,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set) {
  std::map<PatternVertexPtr, int> indegree;
  BuildIndegree(pattern, indegree);
  std::vector<PatternVertexPtr> topu_seq;
  GUNDAM::_dp_iso::_DAGDP::TopuSort(pattern, indegree, topu_seq);
  if (!GUNDAM::_dp_iso::DAGDP(pattern, data_graph, topu_seq, candidate_set))
    return false;
  constexpr int loop_num = 2;
  for (int i = 1; i <= loop_num; i++) {
    std::reverse(topu_seq.begin(), topu_seq.end());
    if (!GUNDAM::_dp_iso::DAGDP(pattern, data_graph, topu_seq, candidate_set))
      return false;
  };
  return true;
}


template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline bool RefineCandidateSetFirstRound_Topu(
    const Pattern &pattern, const DataGraph &data_graph,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set, std::vector<PatternVertexPtr> &topu_seq) {
  std::map<PatternVertexPtr, int> indegree;
  BuildIndegree(pattern, indegree);
  GUNDAM::_dp_iso::_DAGDP::TopuSort(pattern, indegree, topu_seq);
  if (!GUNDAM::_dp_iso::DAGDP(pattern, data_graph, topu_seq, candidate_set))
    return false;
  constexpr int loop_num = 2;
  for (int i = 1; i <= loop_num; i++) {
    std::reverse(topu_seq.begin(), topu_seq.end());
    if (!GUNDAM::_dp_iso::DAGDP(pattern, data_graph, topu_seq, candidate_set))
      return false;
  };
  return true;
}





template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline bool RefineCandidateSetFirstRound_Topu_U_V(
    const Pattern &pattern, const DataGraph &data_graph,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set, std::vector<PatternVertexPtr> &topu_seq, CandidateSorter<Pattern, DataGraph> &sorter, double rep_score, double existing_heap_min_score) {
  std::map<PatternVertexPtr, int> indegree;
  BuildIndegree(pattern, indegree);
  GUNDAM::_dp_iso::_DAGDP::TopuSort(pattern, indegree, topu_seq);

  std::reverse(topu_seq.begin(), topu_seq.end());


  if (!GUNDAM::_dp_iso::DAGDP(pattern, data_graph, topu_seq, candidate_set)){
    return false;
  }
      
  double total_max_score_candidate_set_refine = 0.0;
  total_max_score_candidate_set_refine = sorter.GetTotalMaxScore(candidate_set);

  double total_max_score_rep_refine = rep_score * score_transform(total_max_score_candidate_set_refine);

  if (total_max_score_rep_refine <= existing_heap_min_score) {
    return false;
  }


  std::reverse(topu_seq.begin(), topu_seq.end());
  if (!GUNDAM::_dp_iso::DAGDP(pattern, data_graph, topu_seq, candidate_set)){
    return false;
  }
    

  double total_max_score_candidate_set_refine2 = 0.0;
  total_max_score_candidate_set_refine2 = sorter.GetTotalMaxScore(candidate_set);

  double total_max_score_rep_refine2 = rep_score * score_transform(total_max_score_candidate_set_refine2);

  if (total_max_score_rep_refine2 <= existing_heap_min_score) {
    return false;
  }
  

  std::reverse(topu_seq.begin(), topu_seq.end());
  if (!GUNDAM::_dp_iso::DAGDP(pattern, data_graph, topu_seq, candidate_set)){
    return false;
  }
    
  double total_max_score_candidate_set_refine3 = 0.0;
  total_max_score_candidate_set_refine3 = sorter.GetTotalMaxScore(candidate_set);
  double total_max_score_rep_refine3 = rep_score * score_transform(total_max_score_candidate_set_refine3);

  if (total_max_score_rep_refine3 <= existing_heap_min_score) {
    return false;
  }
    
  return true;
}



template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline bool PredicateCheck(
    const REP<Pattern, DataGraph> &rep,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set) {
  for (auto &single_predicate : rep.x_prediate()) {

    if (!single_predicate->PredicateCheck(candidate_set)) {
      return false;
    }
  }
  return true;
}
template <class PatternVertexPtr>
inline int GetLeaves(PatternVertexPtr root_ptr,
                     std::vector<PatternVertexPtr> &leaves_set) {
  std::queue<PatternVertexPtr> bfs_queue;
  bfs_queue.push(root_ptr);
  while (!bfs_queue.empty()) {
    PatternVertexPtr now_ptr = bfs_queue.front();
    bfs_queue.pop();
    if (now_ptr->CountOutEdge() == 0) {
      leaves_set.emplace_back(now_ptr);
    } else {
      for (auto edge_it = now_ptr->OutEdgeBegin(); !edge_it.IsDone();
           edge_it++) {
        bfs_queue.push(edge_it->const_dst_handle());
      }
    }
  }
  return 1;
}
template <class PatternVertexPtr>
inline int LeafMapDFS(PatternVertexPtr now_ptr,
                      std::map<PatternVertexPtr, PatternVertexPtr> &leaf_map) {
  if (now_ptr->CountOutEdge() == 0) {
    leaf_map.emplace(now_ptr, now_ptr);
  }
  for (auto edge_it = now_ptr->OutEdgeBegin(); !edge_it.IsDone(); edge_it++) {
    PatternVertexPtr next_ptr = edge_it->dst_handle();
    LeafMapDFS(next_ptr, leaf_map);
    leaf_map.emplace(now_ptr, leaf_map[next_ptr]);
  }
  return 1;
}
template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline int BuildLeafMap(
    const REP<Pattern, DataGraph> &rep,
    std::map<PatternVertexPtr, PatternVertexPtr> &leaf_map) {
  auto BuildLeafMapFunc = [&leaf_map](PatternVertexPtr now_ptr) {
    for (auto edge_it = now_ptr->OutEdgeBegin(); !edge_it.IsDone(); edge_it++) {
      PatternVertexPtr dst_ptr = edge_it->const_dst_handle();
      LeafMapDFS(dst_ptr, leaf_map);
    }
  };
  BuildLeafMapFunc(rep.x_ptr());
  BuildLeafMapFunc(rep.y_ptr());
  return 1;
}
template <class PatternVertexPtr, class DataVertexPtr>
inline int CacheDFS(
    PatternVertexPtr now_ptr, DataVertexPtr now_data_ptr,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set,
    std::map<std::pair<PatternVertexPtr, DataVertexPtr>,
             std::vector<DataVertexPtr>> &cache) {
  if (now_ptr->CountOutEdge() == 0) {
    std::vector<DataVertexPtr> cache_vertex_list{now_data_ptr};
    cache.emplace(std::make_pair(now_ptr, now_data_ptr),
                  std::move(cache_vertex_list));
    return 1;
  }
  std::vector<DataVertexPtr> union_result;
  for (auto pattern_edge_it = now_ptr->OutEdgeBegin();
       !pattern_edge_it.IsDone(); pattern_edge_it++) {
    auto edge_label = pattern_edge_it->label();
    PatternVertexPtr now_dst_ptr = pattern_edge_it->dst_handle();
    std::vector<DataVertexPtr> &now_dst_ptr_candidate =
        candidate_set[now_dst_ptr];
    for (auto data_vertex_it = now_data_ptr->OutVertexBegin(edge_label);
         !data_vertex_it.IsDone(); data_vertex_it++) {
      DataVertexPtr data_dst_ptr = data_vertex_it;
      if (!std::binary_search(now_dst_ptr_candidate.begin(),
                              now_dst_ptr_candidate.end(), data_dst_ptr)) {
        continue;
      }
      if (!cache.count(std::make_pair(now_dst_ptr, data_dst_ptr))) {
        CacheDFS(now_dst_ptr, data_dst_ptr, candidate_set, cache);
      }
      auto &cache_list = cache[std::make_pair(now_dst_ptr, data_dst_ptr)];
      std::vector<DataVertexPtr> temp_union_result;
      std::set_union(union_result.begin(), union_result.end(),
                     cache_list.begin(), cache_list.end(),
                     inserter(temp_union_result, temp_union_result.begin()));
      std::swap(union_result, temp_union_result);
    }
  }
  cache.emplace(std::make_pair(now_ptr, now_data_ptr), std::move(union_result));
  return 1;
}

template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline int BuildCache(
    const REP<Pattern, DataGraph> &rep, const DataGraph &data_graph,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set,
    std::map<std::pair<PatternVertexPtr, DataVertexPtr>,
             std::vector<DataVertexPtr>> &cache,
    std::map<PatternVertexPtr, PatternVertexPtr> &leaf_map) {
  using MatchMap = std::map<PatternVertexPtr, DataVertexPtr>;
  using CandidateSet = std::map<PatternVertexPtr, std::vector<DataVertexPtr>>;
  const Pattern &rep_pattern = rep.pattern();
  auto x_ptr = rep.x_ptr();
  auto y_ptr = rep.y_ptr();
  auto BuildCacheFunc = [&candidate_set, &cache](PatternVertexPtr now_ptr) {
    for (auto edge_it = now_ptr->OutEdgeBegin(); !edge_it.IsDone(); edge_it++) {
      auto dst_ptr = edge_it->const_dst_handle();
      for (auto &target_ptr : candidate_set.find(dst_ptr)->second) {
        CacheDFS(dst_ptr, target_ptr, candidate_set, cache);
      }
    }
  };
  BuildCacheFunc(x_ptr);
  BuildCacheFunc(y_ptr);
  return 1;
}
template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type,
    class CandidateSet = std::map<PatternVertexPtr, std::vector<DataVertexPtr>>,
    class CacheType = std::map<std::pair<PatternVertexPtr, DataVertexPtr>,
                               std::vector<DataVertexPtr>>>
int BuildCandidateSetCache(
    const REP<Pattern, DataGraph> &rep, CandidateSet candidate_set,
    std::map<PatternVertexPtr, PatternVertexPtr> &leaf_map, CacheType &cache,
    std::map<DataVertexPtr, CandidateSet> &x_candidate_set_cache,
    std::map<DataVertexPtr, CandidateSet> &y_candidate_set_cache) {
  auto BuildCandidateSetCacheFunc =
      [&candidate_set, &leaf_map, &cache](
          PatternVertexPtr now_ptr,
          std::map<DataVertexPtr, CandidateSet> &candidate_set_cache) {
        for (auto &match_now_ptr : candidate_set.find(now_ptr)->second) {
          CandidateSet now_candidate_set;
          for (auto edge_it = now_ptr->OutEdgeBegin(); !edge_it.IsDone();
               edge_it++) {
            auto edge_label = edge_it->label();
            auto now_dst_ptr = edge_it->const_dst_handle();
            auto leaf_ptr = leaf_map[now_dst_ptr];
            auto &l = candidate_set[now_dst_ptr];
            std::vector<DataVertexPtr> union_result;
            for (auto vertex_it = match_now_ptr->OutVertexBegin(edge_label);
                 !vertex_it.IsDone(); vertex_it++) {
              DataVertexPtr match_now_dst_ptr = vertex_it;
              if (!std::binary_search(l.begin(), l.end(), match_now_dst_ptr)) {
                continue;
              }
              auto &l1 = cache[std::make_pair(now_dst_ptr, match_now_dst_ptr)];
              std::vector<DataVertexPtr> temp_union_result;
              std::set_union(
                  union_result.begin(), union_result.end(), l1.begin(),
                  l1.end(),
                  inserter(temp_union_result, temp_union_result.begin()));
              std::swap(union_result, temp_union_result);
              if (union_result.size() == candidate_set[leaf_ptr].size()) {
                break;
              }
            }
            now_candidate_set.emplace(leaf_ptr, std::move(union_result));
          }
          if (now_candidate_set.empty()) {
            std::vector<DataVertexPtr> union_result{match_now_ptr};
            now_candidate_set.emplace(now_ptr, std::move(union_result));
          }
          candidate_set_cache.emplace(match_now_ptr,
                                      std::move(now_candidate_set));
        }
      };
  BuildCandidateSetCacheFunc(rep.x_ptr(), x_candidate_set_cache);
  BuildCandidateSetCacheFunc(rep.y_ptr(), y_candidate_set_cache);
  return 1;
}





template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type,
    class DataVertexIDType = typename DataGraph::VertexType::IDType,
    class PatternVertexIDType = typename Pattern::VertexType::IDType>
inline std::pair<int, int> REPMatchBasePTime(
    const REP<Pattern, DataGraph> &rep, const DataGraph &data_graph,
    std::map<std::pair<DataVertexPtr, DataVertexPtr>, std::pair<int, int>>
        &ml_model,
    bool using_cache = true,
    std::vector<std::pair<DataVertexPtr, DataVertexPtr>> *positive_result =
        nullptr,
    std::vector<std::pair<DataVertexPtr, DataVertexPtr>> *all_result =
        nullptr) {
  using MatchMap = std::map<PatternVertexPtr, DataVertexPtr>;
  using CandidateSet = std::map<PatternVertexPtr, std::vector<DataVertexPtr>>;
  CandidateSet candidate_set;
  const Pattern &rep_pattern = rep.pattern();
  auto x_ptr = rep.x_ptr();
  auto y_ptr = rep.y_ptr();
  auto filter_callback = [&rep](PatternVertexPtr pattern_vertex_ptr,
                                DataVertexPtr data_vertex_ptr) {
    MatchMap match_state;
    match_state.emplace(pattern_vertex_ptr, data_vertex_ptr);
    bool satisfy_flag = true;
    for (auto &single_predicate : rep.x_prediate()) {
      satisfy_flag &= single_predicate->Satisfy1(match_state);
      if (!satisfy_flag) {
        break;
      }
    }
    return satisfy_flag;
  };





  std::vector<std::vector<std::pair<PatternVertexPtr, int>>> pattern_to_path_result;
  GUNDAM::_dp_iso::_DAGDP::GetAllPathsFromRootsToLeaves_with_edge_label(rep_pattern, pattern_to_path_result);



  

  std::set<std::string> info_set;
  for (auto &single_predicate : rep.x_prediate()) {
    info_set.insert(single_predicate->info());

  }


  

  

  if (!GUNDAM::_dp_iso::InitCandidateSet<GUNDAM::MatchSemantics::kHomomorphism>(
          rep_pattern, data_graph, candidate_set, filter_callback)) {
    std::cout << "return after InitCandidateSet" << std::endl;
    return {0, 0};
  }
  if (!RefineCandidateSetFirstRound(rep_pattern, data_graph, candidate_set)) {
    std::cout << "return after RefineCandidateSetFirstRound" << std::endl;
    return {0, 0};
  }
  if (!PredicateCheck(rep, candidate_set)) {
    std::cout << "return after PredicateCheck" << std::endl;
    return {0, 0};
  }





  

  std::map<int,std::vector<int>> x_y_match;


  if (!info_set.count("Variable")) {
    

    std::map<int, std::vector<int>> pattern_positive_match;

    int all_count = 0, positive_count = 0;
    auto like_label = rep.q_label();
    for (auto x_candidate : candidate_set.find(x_ptr)->second) {
      for (auto y_candidate : candidate_set.find(y_ptr)->second) {
        
        if (!ml_model.empty() &&
            !ml_model.count(std::make_pair(x_candidate, y_candidate))) {
          continue;
        }
        bool positive_flag = x_candidate->HasOutEdge(y_candidate) > 0;
        if (positive_flag) {
          positive_count++;
          pattern_positive_match[x_candidate->id()].push_back(y_candidate->id());
          if (positive_result != nullptr) {
            positive_result->emplace_back(x_candidate, y_candidate);
            
          }
          
          all_count++;
        } else {
          all_count++;
          if (all_result != nullptr) {
            all_result->emplace_back(x_candidate, y_candidate);
          }
          
        }
      }
    }


     std::cout << "REPMatch.h count is " << positive_count << " " << all_count << std::endl; 
    return {positive_count, all_count};
  }

  int all_count = 0, positive_count = 0;
  if (using_cache) {
    std::cout << "using cache" << std::endl;
    using CacheType = std::map<std::pair<PatternVertexPtr, DataVertexPtr>,
                               std::vector<DataVertexPtr>>;

    auto t_begin = clock();
    std::map<PatternVertexPtr, PatternVertexPtr> leaf_map;
    BuildLeafMap<Pattern, DataGraph>(rep, leaf_map);
    CacheType cache;
    BuildCache(rep, data_graph, candidate_set, cache, leaf_map);
    std::map<DataVertexPtr, CandidateSet> x_candidate_set_cache,
        y_candidate_set_cache;
    BuildCandidateSetCache<Pattern, DataGraph>(rep, candidate_set, leaf_map,
                                               cache, x_candidate_set_cache,
                                               y_candidate_set_cache);
    auto t_end = clock();
    int x_size = 0;
    int y_size = 0;
    for (const auto &[x_candidate, x_cache] : x_candidate_set_cache) {
      x_size += 1;
      for (const auto &[y_candidate, y_cache] : y_candidate_set_cache) {
        y_size += 1;

        
        if (!ml_model.empty() &&
            !ml_model.count(std::make_pair(x_candidate, y_candidate))) {
          continue;
        }
        
        
        
        CandidateSet refine_candidate_set(x_cache);
        for (const auto &it : y_cache) {
          refine_candidate_set.emplace(it.first, it.second);
        }
        if (!PredicateCheck(rep, refine_candidate_set)) {
          continue;
        }
        if (x_candidate->HasOutEdge(y_candidate) > 0) {
          positive_count++;
        
          if (positive_result != nullptr) {
            positive_result->emplace_back(x_candidate, y_candidate);
          }
          all_count++;
        } else {
          all_count++;
          
          if (all_result != nullptr) {
            all_result->emplace_back(x_candidate, y_candidate);
          }
        }
        
        
      }
    }
  } else {
    std::cout << "not using cache" << std::endl;
    std::vector<PatternVertexPtr> x_leaves, y_leaves;
    GetLeaves(x_ptr, x_leaves);
    GetLeaves(y_ptr, y_leaves);
    std::map<DataVertexPtr, CandidateSet> x_candidate_set_cache,
        y_candidate_set_cache;
    auto y_first_candidate = candidate_set[y_ptr][0];
    auto x_first_candidate = candidate_set[x_ptr][0];
    auto t_begin = clock();
    for (auto x_candidate : candidate_set.find(x_ptr)->second) {
      CandidateSet refine_candidate_set(candidate_set);
      refine_candidate_set[x_ptr] = {x_candidate};
      refine_candidate_set[y_ptr] = {y_first_candidate};
      if (!RefineCandidateSet(rep_pattern, data_graph, refine_candidate_set)) {
        continue;
      }
      x_candidate_set_cache[x_candidate] = refine_candidate_set;
    }
    for (auto y_candidate : candidate_set.find(y_ptr)->second) {
      CandidateSet refine_candidate_set(candidate_set);
      refine_candidate_set[x_ptr] = {x_first_candidate};
      refine_candidate_set[y_ptr] = {y_candidate};
      if (!RefineCandidateSet(rep_pattern, data_graph, refine_candidate_set)) {
        continue;
      }
      y_candidate_set_cache[y_candidate] = refine_candidate_set;
    }
    auto t_end = clock();
    for (auto x_candidate : candidate_set.find(x_ptr)->second) {
      if (!x_candidate_set_cache.count(x_candidate)) continue;
      for (auto y_candidate : candidate_set.find(y_ptr)->second) {
        if (!y_candidate_set_cache.count(y_candidate)) continue;
        
        if (!ml_model.empty() &&
            !ml_model.count(std::make_pair(x_candidate, y_candidate))) {
          continue;
        }
        


        if (x_candidate->HasOutEdge(y_candidate) <= 0) {
          all_count++;
          continue;
        }
        else{
          all_count++;
          auto &x_cache = x_candidate_set_cache[x_candidate];
          auto &y_cache = y_candidate_set_cache[y_candidate];
          CandidateSet refine_candidate_set;
          for (auto &x_leaf : x_leaves) {
            refine_candidate_set[x_leaf] = x_cache[x_leaf];
          }
          for (auto &y_leaf : y_leaves) {
            refine_candidate_set[y_leaf] = y_cache[y_leaf];
          }
          if (!PredicateCheck(rep, refine_candidate_set)) {
            continue;
          }
          positive_count++;
        }
        continue;
        auto &x_cache = x_candidate_set_cache[x_candidate];
        auto &y_cache = y_candidate_set_cache[y_candidate];
        CandidateSet refine_candidate_set;
        for (auto &x_leaf : x_leaves) {
          refine_candidate_set[x_leaf] = x_cache[x_leaf];
        }
        for (auto &y_leaf : y_leaves) {
          refine_candidate_set[y_leaf] = y_cache[y_leaf];
        }
        if (!PredicateCheck(rep, refine_candidate_set)) {
          continue;
        }
        if (x_candidate->HasOutEdge(y_candidate) > 0) {
          positive_count++;
          
          if (positive_result != nullptr) {
            positive_result->emplace_back(x_candidate, y_candidate);
          }
          
        }
        all_count++;
        
        if (all_result != nullptr) {
          all_result->emplace_back(x_candidate, y_candidate);
        }
        
      }
    }
  }
  std::cout << "REPMatch.h supp is " << positive_count << " " << all_count << std::endl;
  return {positive_count, all_count};
}








template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type,
    class DataVertexIDType = typename DataGraph::VertexType::IDType>
inline std::pair<int, int> REPMatchBasePTime_U_V(
    const REP<Pattern, DataGraph> &rep, const DataGraph &data_graph,
    std::map<std::pair<DataVertexPtr, DataVertexPtr>, std::pair<int, int>>
        &ml_model,
    const DataVertexIDType x_id_,
    const DataVertexIDType y_id_, 
    bool using_cache = true,
    std::vector<std::pair<DataVertexPtr, DataVertexPtr>> *positive_result =
        nullptr,
    std::vector<std::pair<DataVertexPtr, DataVertexPtr>> *all_result =
        nullptr,
    std::priority_queue<std::pair<double, std::map<int, int>>, 
        std::vector<std::pair<double, std::map<int, int>>>, 
        MyCompare> * topk_min_heap_ptr = nullptr, 
    int& topk = 1,
    int& rep_id = 0) {

  using MatchMap = std::map<PatternVertexPtr, DataVertexPtr>;
  using CandidateSet = std::map<PatternVertexPtr, std::vector<DataVertexPtr>>;
  CandidateSet candidate_set;
  const Pattern &rep_pattern = rep.pattern();
  auto x_ptr = rep.x_ptr();
  auto y_ptr = rep.y_ptr();
  auto x_id = x_id_;
  auto y_id = y_id_;

  auto filter_callback = [&rep](PatternVertexPtr pattern_vertex_ptr,
                                DataVertexPtr data_vertex_ptr) {
    MatchMap match_state;
    match_state.emplace(pattern_vertex_ptr, data_vertex_ptr);
    bool satisfy_flag = true;
    for (auto &single_predicate : rep.x_prediate()) {
      satisfy_flag &= single_predicate->Satisfy1(match_state);
      if (!satisfy_flag) {
        break;
      }
    }
    return satisfy_flag;
  };

  std::map<std::string, double> attributes_frequency;
  attributes_frequency["age"] = 20;
  attributes_frequency["avgrating"] = 5.0;


  attributes_frequency["item_review_count"] = 7968;
  attributes_frequency["user_review_count"] = 11942;
  attributes_frequency["fans"] = 2785;
  attributes_frequency["average_stars"] = 5.0;
  

  if (!GUNDAM::_dp_iso::InitCandidateSet_U_V<GUNDAM::MatchSemantics::kHomomorphism>(
          rep_pattern, data_graph, x_id_, y_id_, x_ptr, y_ptr, candidate_set, filter_callback)) {
    return std::make_pair(-1, -1);
  }


  int pattern_nodes_num = 0;
  for (auto query_vertex_iter = rep_pattern.VertexBegin();
       !query_vertex_iter.IsDone(); query_vertex_iter++) {
        pattern_nodes_num += 1;
  }

  std::unordered_map<PatternVertexPtr, std::unordered_map<DataVertexPtr,double>> current_match_instance_node_score;
  for (const auto& pair : candidate_set) {
      for (const auto& value : pair.second) {
          size_t out_degree = value->CountOutEdge();          
          double graph_node_score_inX = 0.0;
          
          for (auto &single_predicate : rep.x_prediate()) {
              if (pair.first == single_predicate->GetX().first || pair.first == single_predicate->GetY().first) {
                  double predicate_score = single_predicate->GetPredicateScore();
                  if (pair.first != single_predicate->GetY().first) {
                      graph_node_score_inX += predicate_score;
                  } else {
                      double graph_node_score = single_predicate->makex_cs_score(value, attributes_frequency);
                      graph_node_score_inX += predicate_score * graph_node_score; 
                  }
              }
          }

          double w_node_score = std::log1p(out_degree * 0.1) / std::log(10);
          if (graph_node_score_inX != 0.0) {
              w_node_score *= graph_node_score_inX;
          }

          current_match_instance_node_score[pair.first][value] = w_node_score;

      }
  }



  double rep_original_score = rep.get_rep_score();

  CandidateSorter<Pattern, DataGraph> sorter;
  sorter.UpdateScores(current_match_instance_node_score);
  sorter.SortCandidateSet(&candidate_set);
  double total_max_score_candidate_set = 0.0;
  total_max_score_candidate_set = sorter.GetTotalMaxScore(candidate_set);

  double total_max_score_rep = rep_original_score * score_transform(total_max_score_candidate_set);
  double existing_heap_min_score = 0.0;

  if (topk_min_heap_ptr->size() > 0) {
    existing_heap_min_score = topk_min_heap_ptr->top().first;
  }


  if (total_max_score_rep <= existing_heap_min_score) {
    return std::make_pair(-1, -1);
  }


  std::vector<PatternVertexPtr> topu_seq;

  if (!RefineCandidateSetFirstRound_Topu_U_V(rep_pattern, data_graph, candidate_set, topu_seq, sorter, rep_original_score, existing_heap_min_score)) {
    return std::make_pair(-1, -1);
  }

  if (!PredicateCheck(rep, candidate_set)) {
    return std::make_pair(-1, -1);
  }


  std::set<std::string> info_set;
  for (auto &single_predicate : rep.x_prediate()) {
    info_set.insert(single_predicate->info());
  }

  double total_max_score_candidate_set_refine = 0.0;
  total_max_score_candidate_set_refine = sorter.GetTotalMaxScore(candidate_set);


  double total_max_score_rep_refine = rep_original_score * score_transform(total_max_score_candidate_set_refine);
  if (total_max_score_rep_refine <= existing_heap_min_score) {
    return std::make_pair(-1, -1);
  }

  int all_count = 1, positive_count = 1;

  std::map<PatternVertexPtr,DataVertexPtr> current_match_instance;

  std::vector<std::map<PatternVertexPtr,DataVertexPtr>> all_matches;
  std::vector<std::map<PatternVertexPtr, std::map<DataVertexPtr,double>>> all_matches_node_score;

  std::vector<std::pair<double, std::map<int, int>>> matches_with_score;
  std::map<int, int> current_match_id_info;

  std::vector<std::vector<PatternVertexPtr>> pattern_to_path_result;
  GUNDAM::_dp_iso::_DAGDP::GetAllPathsFromRootsToLeaves(rep_pattern, pattern_to_path_result);


  std::vector<std::vector<double>> all_path_topk_scores;
  std::vector<std::vector<std::pair<double, std::map<int, int>>>> all_path_topk_matches_result;


  for (auto &path_vec: pattern_to_path_result) {
    std::vector<std::pair<double, std::map<int, int>>> path_topk_matches_result;
    std::vector<std::pair<double, std::map<PatternVertexPtr, DataVertexPtr>>> path_topk_matches_ptr_result;
    std::vector<double> current_path_topk_scores;
    std::vector<std::vector<double>> candidate_score_arrays;
    for (const auto& query_vertex : path_vec) {
      std::vector<double> query_vertex_score_array;
      for (const auto& target_vertex: candidate_set.at(query_vertex)) {
        query_vertex_score_array.emplace_back(current_match_instance_node_score.at(query_vertex).at(target_vertex));
      }
      candidate_score_arrays.emplace_back(query_vertex_score_array);
      
    }
 
    InnerPathTopkCalculation<Pattern, DataGraph, PatternVertexPtr, DataVertexPtr>(candidate_set, path_vec, candidate_score_arrays, topk, path_topk_matches_result,
      path_topk_matches_ptr_result, all_path_topk_matches_result, false);

    if (path_topk_matches_result.size() == 0) {
      return std::make_pair(-1, -1);
    }

    path_topk_matches_result.resize(topk);

    for (auto& pair: path_topk_matches_result) {
      current_path_topk_scores.emplace_back(pair.first);
    }
    all_path_topk_scores.emplace_back(current_path_topk_scores);
    all_path_topk_matches_result.emplace_back(path_topk_matches_result);
  }

  std::vector<PathMatchingResult> path_merge_matching_result;
  InterPathTopkCombinations(all_path_topk_scores, topk, path_merge_matching_result);

  double max_score = 0.0;

  MergeMatchingIds_pattern_node_num(pattern_nodes_num, path_merge_matching_result, all_path_topk_matches_result, topk, rep.get_rep_score(), topk_min_heap_ptr, matches_with_score, max_score);

  return {positive_count, all_count};
}











template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type>
inline bool InitCandidateSet_First(
    const Pattern &pattern, const DataGraph &data_graph,
    std::map<PatternVertexPtr, std::vector<DataVertexPtr>> &candidate_set, std::vector<PatternVertexPtr> &topu_seq) {
  std::map<PatternVertexPtr, int> indegree;
  BuildIndegree(pattern, indegree);
  GUNDAM::_dp_iso::_DAGDP::TopuSort(pattern, indegree, topu_seq);
  if (!GUNDAM::_dp_iso::DAGDP(pattern, data_graph, topu_seq, candidate_set))
    return false;
  return true;
}




template <
    class Pattern, class DataGraph,
    class PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type,
    class DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type,
    class DataVertexIDType = typename DataGraph::VertexType::IDType>
inline std::pair<int, int> REPMatchBasePTime_U_V_ALL_Explnantion(
    const REP<Pattern, DataGraph> &rep, const DataGraph &data_graph,
    std::map<std::pair<DataVertexPtr, DataVertexPtr>, std::pair<int, int>>
        &ml_model,
    const DataVertexIDType x_id_,
    const DataVertexIDType y_id_, 
    bool using_cache = true,
    std::vector<std::pair<DataVertexPtr, DataVertexPtr>> *positive_result =
        nullptr,
    std::vector<std::pair<DataVertexPtr, DataVertexPtr>> *all_result =
        nullptr,
    std::priority_queue<std::pair<double, std::map<int, int>>, 
        std::vector<std::pair<double, std::map<int, int>>>, 
        MyCompare> * topk_min_heap_ptr = nullptr, 
    int& topk = 1) {
  using MatchMap = std::map<PatternVertexPtr, DataVertexPtr>;
  using CandidateSet = std::map<PatternVertexPtr, std::vector<DataVertexPtr>>;
  CandidateSet candidate_set;
  const Pattern &rep_pattern = rep.pattern();
  auto x_ptr = rep.x_ptr();
  auto y_ptr = rep.y_ptr();
  auto x_id = x_id_;
  auto y_id = y_id_;

  auto filter_callback = [&rep](PatternVertexPtr pattern_vertex_ptr,
                                DataVertexPtr data_vertex_ptr) {
    MatchMap match_state;
    match_state.emplace(pattern_vertex_ptr, data_vertex_ptr);
    bool satisfy_flag = true;
    for (auto &single_predicate : rep.x_prediate()) {
      satisfy_flag &= single_predicate->Satisfy1(match_state);
      if (!satisfy_flag) {
        break;
      }
    }
    return satisfy_flag;
  };


  if (!GUNDAM::_dp_iso::InitCandidateSet_U_V<GUNDAM::MatchSemantics::kHomomorphism>(
          rep_pattern, data_graph, x_id_, y_id_, x_ptr, y_ptr, candidate_set, filter_callback)) {
    return std::make_pair(-1, -1);
  }


  std::vector<PatternVertexPtr> topu_seq;
  
  if (!RefineCandidateSetFirstRound_Topu(rep_pattern, data_graph, candidate_set,topu_seq)) {
    return std::make_pair(-1, -1);
  }

  if (!PredicateCheck(rep, candidate_set)) {
    return std::make_pair(-1, -1);
  }
  std::set<std::string> info_set;
  for (auto &single_predicate : rep.x_prediate()) {
    info_set.insert(single_predicate->info());
  }


  std::vector<std::vector<PatternVertexPtr>> pattern_to_path_result;
  GUNDAM::_dp_iso::_DAGDP::GetAllPathsFromRootsToLeaves(rep_pattern, pattern_to_path_result);


  std::vector<std::vector<std::map<PatternVertexPtr,DataVertexPtr>>> all_path_matches;
  for (auto &path_vec: pattern_to_path_result) {
    
    std::map<PatternVertexPtr,DataVertexPtr> current_match_instance;

    std::vector<std::map<PatternVertexPtr,DataVertexPtr>> all_matches;

    GenerateAllMatches(rep, rep_pattern, data_graph, path_vec, path_vec.begin(), candidate_set, current_match_instance, all_matches);
    all_path_matches.emplace_back(all_matches);
  }

  PrintAllMatches_ALL_Explanation_No_save_Results(rep, all_path_matches);
  return {0,0};
}



template <class Pattern, class DataGraph,
          class DataVertexIDType = typename DataGraph::VertexType::IDType>
bool CheckHasMatch(const REP<Pattern, DataGraph> &rep,
                   const DataGraph &data_graph, const DataVertexIDType x_id,
                   const DataVertexIDType y_id) {
  using DataVertexPtr = typename GUNDAM::VertexHandle<const DataGraph>::type;
  using PatternVertexPtr = typename GUNDAM::VertexHandle<const Pattern>::type;
  using MatchMap = std::map<PatternVertexPtr, DataVertexPtr>;
  using CandidateSet = std::map<PatternVertexPtr, std::vector<DataVertexPtr>>;
  CandidateSet candidate_set;
  const Pattern &rep_pattern = rep.pattern();
  auto x_ptr = rep.x_ptr();
  auto y_ptr = rep.y_ptr();
  auto filter_callback = [&rep](PatternVertexPtr pattern_vertex_ptr,
                                DataVertexPtr data_vertex_ptr) {
    MatchMap match_state;
    match_state.emplace(pattern_vertex_ptr, data_vertex_ptr);
    bool satisfy_flag = true;
    for (auto &single_predicate : rep.x_prediate()) {
      satisfy_flag &= single_predicate->Satisfy1(match_state);
      if (!satisfy_flag) {
        break;
      }
    }
    return satisfy_flag;
  };
  if (!GUNDAM::_dp_iso::InitCandidateSet<GUNDAM::MatchSemantics::kHomomorphism>(
          rep_pattern, data_graph, candidate_set, filter_callback)) {
    return 0;
  }
  if (!RefineCandidateSetFirstRound(rep_pattern, data_graph, candidate_set)) {
    return 0;
  }
  if (!PredicateCheck(rep, candidate_set)) {
    return 0;
  }
  DataVertexPtr target_x_ptr = data_graph.FindVertex(x_id);
  DataVertexPtr target_y_ptr = data_graph.FindVertex(y_id);
  CandidateSet refine_candidate_set(candidate_set);
  refine_candidate_set[x_ptr] = {target_x_ptr};
  refine_candidate_set[y_ptr] = {target_y_ptr};
  if (!RefineCandidateSet(rep_pattern, data_graph, refine_candidate_set)) {
    return false;
  }
  if (!PredicateCheck(rep, refine_candidate_set)) {
    return false;
  }
  return true;
}

}#endif
