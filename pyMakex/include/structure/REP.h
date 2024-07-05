#ifndef _REP_H
#define _REP_H
#include "BaseContainer.h"
#include "Predicate.h"
namespace Makex {

template <class Pattern, class DataGraph>
class REP {
 public:
  typedef Pattern value_type;
  using VertexType = typename Pattern::VertexType;
  using VertexIDType = typename VertexType::IDType;
  using VertexLabelType = typename VertexType::LabelType;
  using EdgeType = typename Pattern::EdgeType;
  using EdgeIDType = typename EdgeType::IDType;
  using EdgeLabelType = typename EdgeType::LabelType;
  using VertexPtr = typename GUNDAM::VertexHandle<Pattern>::type;
  using VertexConstPtr = typename GUNDAM::VertexHandle<const Pattern>::type;
  using EdgePtr = typename GUNDAM::EdgeHandle<Pattern>::type;
  using EdgeConstPtr = typename GUNDAM::EdgeHandle<const Pattern>::type;
  using VertexSizeType = size_t;
  using DataGraphVertexPtr =
      typename GUNDAM::VertexHandle<const DataGraph>::type;
  using SuppType = std::vector<DataGraphVertexPtr>;
  using PredicateType = Predicate<Pattern, DataGraph>;
  using PredicateList = BaseContainer<PredicateType>;

 private:
  double score_ = 0.0;
  Pattern pattern_;
  PredicateList x_prediate_;
  VertexPtr x_ptr_, y_ptr_;
  EdgeLabelType q_label_;
  std::vector<VertexPtr> x_leaves, y_leaves;

 public:
  REP(VertexLabelType x_label, VertexLabelType y_label, EdgeLabelType q_label) {
    this->x_ptr_ = this->pattern_.AddVertex(1, x_label).first;
    this->y_ptr_ = this->pattern_.AddVertex(2, y_label).first;
    this->q_label_ = q_label;
  }

  REP(VertexLabelType x_label, VertexLabelType y_label, EdgeLabelType q_label,const double score) {
    this->x_ptr_ = this->pattern_.AddVertex(1, x_label).first;
    this->y_ptr_ = this->pattern_.AddVertex(2, y_label).first;
    this->q_label_ = q_label;
    this->score_ = score;
  }

  ~REP() {}
  REP(Pattern& pattern, const VertexIDType x_node_id,
      const VertexIDType y_node_id, const EdgeLabelType q, const double score) {
    this->pattern_ = pattern;
    this->x_ptr_ = this->pattern_.FindVertex(x_node_id);
    this->y_ptr_ = this->pattern_.FindVertex(y_node_id);
    this->q_label_ = q;
    this->score_ = score;
  }
  REP(const REP<Pattern, DataGraph>& b) {
    this->score_ = b.score_;
    this->pattern_ = b.pattern_;
    this->x_prediate_ = b.x_prediate_;
    this->q_label_ = b.q_label_;
    this->x_ptr_ = this->pattern_.FindVertex(b.x_ptr_->id());
    this->y_ptr_ = this->pattern_.FindVertex(b.y_ptr_->id());
    for (auto& ptr : x_prediate_) {
      ptr = ptr->Copy(pattern_);
    }
  }
  template <class T, typename... Args>
  void Add(Args... args) {
    this->x_prediate_.template Add<T>(this->pattern_, args...);
  }
  Pattern& pattern() { return this->pattern_; }
  const Pattern& pattern() const { return this->pattern_; }
  VertexPtr x_ptr() { return this->x_ptr_; }
  VertexConstPtr x_ptr() const {
    return this->pattern_.FindVertex(this->x_ptr_->id());
  }
  VertexPtr y_ptr() { return this->y_ptr_; }
  VertexConstPtr y_ptr() const {
    return this->pattern_.FindVertex(this->y_ptr_->id());
  }
  EdgeLabelType q_label() const { return this->q_label_; }
  PredicateList& x_prediate() { return x_prediate_; }
  const PredicateList& x_prediate() const { return x_prediate_; }
  double get_rep_score() const { return score_; }
};

}
#endif