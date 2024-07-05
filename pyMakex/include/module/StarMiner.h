#ifndef _STARMINER_H
#define _STARMINER_H
#include <map>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "DataGraphWithInformation.h"
#include "gundam/type_getter/vertex_handle.h"
namespace CFLogic {
// Mine star to get RNN train data with root label is root_label
template <class DataGraph>
class StarMiner {
 public:
  using VertexLabelType = typename DataGraph::VertexType::LabelType;
  using EdgeLabelType = typename DataGraph::EdgeType::LabelType;
  using DataGraphVertexPtr = typename GUNDAM::VertexHandle<DataGraph>::type;
  using HashValueType = int32_t;
  using EncodeType = std::pair<EdgeLabelType, VertexLabelType>;
  using EncodeHashMap = std::map<EncodeType, HashValueType>;
  using InvEncodeHashMap = std::map<HashValueType, EncodeType>;
  /*
   a single star is
   hash((r1,dst_label1)),hash((r2,dst_label2))...max_size,hash((r1,dst_label1)),hash((r2,dst_label2))...max_size+1.
   max_size is num of encode.encode begin with 0.
   max_size is path end wildcard ,max_size+1 is star end wildcard.
   first vertex label of star is root_label.
   */
  using EncodeStarType = std::vector<HashValueType>;
  using EncodeStarContainer = std::vector<EncodeStarType>;

 private:
  DataGraphWithInformation<DataGraph>* data_graph_ptr_;
  VertexLabelType root_label_;

 public:
  // construct function and deconstruct function
  StarMiner() {
    this->data_graph_ptr_ = nullptr;
    this->encode_hash_map.clear();
  }
  StarMiner(DataGraphWithInformation<DataGraph>* data_graph_ptr,
            VertexLabelType root_label)
      : data_graph_ptr_(data_graph_ptr), root_label_(root_label) {}
  StarMiner(const StarMiner<DataGraph>& b)
      : data_graph_ptr_(b.data_graph_ptr_), root_label_(b.root_label_) {}
  StarMiner(StarMiner<DataGraph>&& b) = default;
  ~StarMiner() {}
  // get path ,using dfs to find all path
  bool GetPath(DataGraphVertexPtr vertex_ptr, EncodeStarType& now_path,
               const int32_t max_len_of_path,
               EncodeStarContainer& path_result) {
    if (!now_path.empty()) {
      path_result.emplace_back(now_path);
    }
    if (now_path.size() == max_len_of_path) return true;
    for (auto edge_it = vertex_ptr->OutEdgeBegin(); !edge_it.IsDone();
         edge_it++) {
      DataGraphVertexPtr dst_ptr = edge_it->dst_handle();
      HashValueType hash_val = this->data_graph_ptr_->encode_hash_map()
                                   .find({edge_it->label(), dst_ptr->label()})
                                   ->second;
      now_path.push_back(hash_val);
      if (!GetPath(dst_ptr, now_path, max_len_of_path, path_result)) {
        return false;
      }
      now_path.pop_back();
    }
    return true;
  }
  // sample star ,now method is rand
  void SampleStar(const EncodeStarContainer& path_result,
                  const int32_t root_max_outdegree,
                  const int32_t each_vertex_remain_star_number,
                  EncodeStarContainer& star_container) {
    // using mt19937  rand path to construct star
    std::random_device rd;
    std::mt19937 gen(rd());
    // rand outdegree
    std::uniform_int_distribution<> outdegree_dis(1, root_max_outdegree);
    // rand select path
    std::uniform_int_distribution<> select_path_dis(0, path_result.size() - 1);
    std::set<std::set<int>> gen_star_set;

    for (int i = 0; i < each_vertex_remain_star_number; i++) {
      int degree = outdegree_dis(gen);
      degree = std::min(degree, (int)path_result.size());
      std::set<int> select_path_set;
      while (select_path_set.size() < degree) {
        int path_id = select_path_dis(gen);
        if (select_path_set.count(path_id)) continue;
        select_path_set.insert(path_id);
      }
      if (gen_star_set.count(select_path_set)) continue;
      if (select_path_set.empty()) continue;
      gen_star_set.insert(select_path_set);
      EncodeStarType now_star;
      for (auto& path_id : select_path_set) {
        now_star.insert(now_star.end(), path_result[path_id].begin(),
                        path_result[path_id].end());
        // path end
        now_star.emplace_back(this->data_graph_ptr_->NumOfEncode());
      }
      now_star.pop_back();
      star_container.emplace_back(std::move(now_star));
    }
  }
  // using random walk to get star
  void RandomWalk(DataGraphVertexPtr vertex_ptr,
                  const int32_t root_max_outdegree,
                  const int32_t max_len_of_path,
                  const int32_t each_vertex_remain_star_number,
                  EncodeStarContainer& star_container) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // rand outdegree
    std::uniform_int_distribution<> outdegree_dis(1, root_max_outdegree);
    std::uniform_int_distribution<> path_len_dis(1, max_len_of_path);
    for (int star_id = 0; star_id < each_vertex_remain_star_number; star_id++) {
      EncodeStarType now_star;
      int out_degree = outdegree_dis(gen);
      for (int degree_iter = 0; degree_iter < out_degree; degree_iter++) {
        int path_len = path_len_dis(gen);
        DataGraphVertexPtr pre_ptr = vertex_ptr, now_ptr = vertex_ptr;
        for (int len_iter = 0; len_iter < path_len; len_iter++) {
          size_t num_out_edge = now_ptr->CountOutEdge();
          if (num_out_edge ==0){
            break;
          }
          std::uniform_int_distribution<> dst_edge_dis(0, num_out_edge - 1);
          size_t select_edge_pos = dst_edge_dis(gen);
          auto edge_it = now_ptr->OutEdgeBegin();
          int cnt = 0;
          for (; !edge_it.IsDone(); edge_it++) {
            if (cnt == select_edge_pos) break;
            cnt++;
          }
          auto edge_label = edge_it->label();
          auto dst_ptr = edge_it->dst_handle();
          // auto [edge_label, dst_ptr] = dst_edge[select_edge_pos];
          HashValueType hash_val = this->data_graph_ptr_->encode_hash_map()
                                       .find({edge_label, dst_ptr->label()})
                                       ->second;
          now_star.emplace_back(hash_val);
          pre_ptr = now_ptr;
          now_ptr = dst_ptr;
        }
        now_star.emplace_back(this->data_graph_ptr_->NumOfEncode());
      }
      now_star.pop_back();
      star_container.emplace_back(std::move(now_star));
    }
  }
  // mine star ,because may be too much star, sample some star
  void GetStar(const int32_t root_max_outdegree, const int32_t max_len_of_path,
               const int32_t each_vertex_remain_star_number,
               EncodeStarContainer& star_container) {
    // std::cout << "using random walk" << std::endl;
    constexpr int max_star_num = 500000;
    for (auto vertex_it =
             this->data_graph_ptr_->data_graph().VertexBegin(this->root_label_);
         !vertex_it.IsDone(); vertex_it++) {
      EncodeStarType now_path;
      EncodeStarContainer path_result;
      DataGraphVertexPtr vertex_ptr = vertex_it;

      this->RandomWalk(vertex_ptr, root_max_outdegree, max_len_of_path,
                       each_vertex_remain_star_number, star_container);
      if (star_container.size() > max_star_num) {
        return;
      }
      /*
      // dfs to get all path
      this->GetPath(vertex_ptr, now_path, max_len_of_path, path_result);
      // unique path ,because using homo
      path_result.erase(std::unique(path_result.begin(), path_result.end()),
                        path_result.end());
      // sample some star
      this->SampleStar(path_result, root_max_outdegree,
                       each_vertex_remain_star_number, star_container);
      */
    }
  }
};
}  // namespace CFLogic
#endif
