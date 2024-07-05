#ifndef _DATA_GRAPH_WITH_INFORMATION_H
#define _DATA_GRAPH_WITH_INFORMATION_H
#include <map>
#include <set>

#include "gundam/type_getter/vertex_handle.h"
namespace CFLogic {
template <class DataGraph>
class DataGraphWithInformation {
 public:
  using VertexIDType = typename DataGraph::VertexType::IDType;
  using VertexLabelType = typename DataGraph::VertexType::LabelType;
  using EdgeLabelType = typename DataGraph::EdgeType::LabelType;
  using VertexAttributeKeyType =
      typename DataGraph::VertexType::AttributeKeyType;
  using DataGraphVertexPtr =
      typename GUNDAM::VertexHandle<const DataGraph>::type;

  using HashValueType = int32_t;
  using EncodeType = std::pair<EdgeLabelType, VertexLabelType>;
  using EncodeHashMap = std::map<EncodeType, HashValueType>;
  using InvEncodeHashMap = std::map<HashValueType, EncodeType>;
  using LabelAttributeKeyMap =
      std::map<VertexLabelType, std::set<VertexAttributeKeyType>>;
  using MLModel = std::map<std::pair<DataGraphVertexPtr, DataGraphVertexPtr>,
                           std::pair<int, int>>;

  DataGraphWithInformation() {
    this->encode_hash_map_.clear();
    this->inv_encode_hash_map_.clear();
    this->label_attribute_key_map_.clear();
    this->small_ml_model_.clear();
    this->big_ml_model_.clear();
    this->has_ml_model_ = false;
    this->positive_pair_num_ = 0;
    this->negative_pair_num_ = 0;
    this->ml_model_size_ = 0;
  }
  DataGraphWithInformation(DataGraph& data_graph) : data_graph_(data_graph) {
    this->encode_hash_map_.clear();
    this->inv_encode_hash_map_.clear();
    this->label_attribute_key_map_.clear();
    this->small_ml_model_.clear();
    this->big_ml_model_.clear();
    this->has_ml_model_ = false;
    this->positive_pair_num_ = 0;
    this->negative_pair_num_ = 0;
    this->ml_model_size_ = 0;
  }
  DataGraphWithInformation(const DataGraphWithInformation<DataGraph>& b)
      : data_graph_(b.data_graph_),
        encode_hash_map_(b.encode_hash_map_),
        inv_encode_hash_map_(b.inv_encode_hash_map_),
        label_attribute_key_map_(b.label_attribute_key_map_),
        small_ml_model_(b.small_ml_model_),
        big_ml_model_(b.big_ml_model_),
        has_ml_model_(b.has_ml_model_),
        positive_pair_num_(b.positive_pair_num_),
        negative_pair_num_(b.negative_pair_num_),
        ml_model_size_(b.ml_model_size_) {}
  DataGraphWithInformation(DataGraphWithInformation<DataGraph>&& b) = default;
  ~DataGraphWithInformation();
  // build encodetype to unique hash value
  void BuildEncodeHashMap() {
    HashValueType last_hash_value = 0;
    for (auto vertex_it = this->data_graph_.VertexBegin(); !vertex_it.IsDone();
         vertex_it++) {
      for (auto edge_it = vertex_it->OutEdgeBegin(); !edge_it.IsDone();
           edge_it++) {
        VertexLabelType dst_label = edge_it->dst_handle()->label();
        EdgeLabelType edge_label = edge_it->label();
        if (this->encode_hash_map_.count({edge_label, dst_label})) {
          continue;
        }
        this->encode_hash_map_.emplace(std::make_pair(edge_label, dst_label),
                                       last_hash_value++);
        this->inv_encode_hash_map_.emplace(
            last_hash_value, std::make_pair(edge_label, dst_label));
      }
    }
  }
  void BuildLabelAttributeKeyMap() {
    for (auto vertex_it = this->data_graph_.VertexBegin(); !vertex_it.IsDone();
         vertex_it++) {
      VertexLabelType vertex_label = vertex_it->label();
      for (auto attr_it = vertex_it->AttributeBegin(); !attr_it.IsDone();
           attr_it++) {
        VertexAttributeKeyType attr_key = attr_it->key();
        this->label_attribute_key_map_[vertex_label].insert(attr_key);
      }
    }
  }
  void BuildMLModel(std::string& model_file, double delta_l, double delta_r,
                    int user_offset) {
    std::ifstream cf_model(model_file);
    std::cout << "BuildMLModel begin" << std::endl;
    this->has_ml_model_ = true;

    std::cout << "user_offset: " << user_offset << std::endl;
    while (cf_model) {
      VertexIDType user_id, item_id;
      double rating;
      cf_model >> user_id >> item_id >> rating;
      user_id += user_offset;
      //std::cout << "user_id" << user_id << std::endl;
      //std::cout << "item_id" << item_id << std::endl;
      //std::cout << "rating" << rating << std::endl;
      //std::cout << "new +++++++*****************++++++++++++++&&&&&&&&&&&&&&&&" << std::endl;
      DataGraphVertexPtr user_ptr = this->data_graph().FindVertex(user_id);
      if (user_ptr == nullptr) {
        std::cout << "user_ptr == nullptr user_id " << user_id << std::endl;
        std::cout << "user_ptr == nullptr user_id_ori " << user_id - user_offset << std::endl;
        std::cout << "user_ptr == nullptr item id " << item_id << std::endl;
        std::cout << "user_ptr == nullptr rating " << rating << std::endl;
        std::cout << model_file << std::endl;
      }
      //std::cout << "user_ptr" << user_ptr << std::endl;
      DataGraphVertexPtr item_ptr = this->data_graph().FindVertex(item_id);
      //std::cout << "item_ptr" << item_ptr << std::endl;
      if (item_ptr == nullptr) {
        std::cout << "item_id" << item_id << std::endl;
      }
      if (rating >= delta_l && rating <= delta_r) {
        this->ml_model_[std::make_pair(user_ptr, item_ptr)] =
            std::make_pair(1, 1);
        if (user_ptr->HasOutEdge(item_ptr) > 0) {
          // positive pair
          this->positive_pair_num_++;
        } else {
          this->negative_pair_num_++;
        }
      }
    }
    this->ml_model_size_ = this->ml_model_.size();
    std::cout << "ml size = " << this->ml_model_.size() << std::endl;
  }
  DataGraph& data_graph() { return this->data_graph_; }
  const DataGraph& data_graph() const { return this->data_graph_; }
  EncodeHashMap& encode_hash_map() { return this->encode_hash_map_; }
  const EncodeHashMap& encode_hash_map() const {
    return this->encode_hash_map_;
  }
  InvEncodeHashMap& inv_encode_hash_map() { return this->inv_encode_hash_map_; }
  const InvEncodeHashMap& inv_encode_hash_map() const {
    return this->inv_encode_hash_map_;
  }
  LabelAttributeKeyMap& label_attribute_key_map() {
    return this->label_attribute_key_map_;
  }
  const LabelAttributeKeyMap& label_attribute_key_map() const {
    return this->label_attribute_key_map_;
  }
  MLModel& ml_model() { return this->ml_model_; }
  const MLModel& ml_model() const { return this->ml_model_; }
  bool has_ml_model() const { return this->has_ml_model_; }
  size_t positive_pair_num() const { return this->positive_pair_num_; }
  size_t negative_pair_num() const { return this->negative_pair_num_; }
  size_t ml_model_size() const { return this->ml_model_size_; }
  size_t NumOfEncode() const { return this->encode_hash_map_.size(); }
  std::set<HashValueType> AdjEncodeList(VertexLabelType vertex_label) const {
    // using RVO,so using vector as return val
    std::set<HashValueType> adj_encode_set;
    for (auto vertex_it = this->data_graph_.VertexBegin(vertex_label);
         !vertex_it.IsDone(); vertex_it++) {
      for (auto edge_it = vertex_it->OutEdgeBegin(); !edge_it.IsDone();
           edge_it++) {
        adj_encode_set.insert(
            this->encode_hash_map_
                .find({edge_it->label(), edge_it->dst_handle()->label()})
                ->second);
      }
    }
    return adj_encode_set;
  }

 private:
  DataGraph data_graph_;
  EncodeHashMap encode_hash_map_;
  InvEncodeHashMap inv_encode_hash_map_;
  LabelAttributeKeyMap label_attribute_key_map_;
  std::map<std::pair<DataGraphVertexPtr, DataGraphVertexPtr>,
           std::pair<int, int>>
      small_ml_model_, big_ml_model_, ml_model_;
  bool has_ml_model_ = false;
  size_t positive_pair_num_ = 0, negative_pair_num_ = 0, ml_model_size_ = 0;
};

}  // namespace CFLogic
#endif
