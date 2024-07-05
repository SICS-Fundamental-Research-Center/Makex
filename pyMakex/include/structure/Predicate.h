#ifndef _PREDICATE_H
#define _PREDICATE_H
#include <variant>

#include "../gundam/data_type/datatype.h"
#include "../gundam/type_getter/vertex_handle.h"
namespace Makex {
inline double score_transform_predicate(double ori_score) {
    return (1.0 - (1.0 / (1.0 + ori_score))) + 1e-6;
}

enum Operation {
  kEqual,
  kNotEqual,
  kLess,
  kLessEqual,
  kGreat,
  kGreatEqual,
  kInclude,
  kNotInclude
};
std::string ToString(std::string a) { return a; }
template <typename T>
std::string ToString(T &&a) {
  return std::to_string(a);
}
template <typename T>
bool CheckOp(enum Operation op, T &&a, T &&b) {
  switch (op) {
    case Operation::kEqual:
      return a == b;
      break;
    case Operation::kNotEqual:
      return a != b;
      break;
    case Operation::kLess:
      return a < b;
      break;
    case Operation::kLessEqual:
      return a <= b;
      break;
    case Operation::kGreat:
      return a > b;
      break;
    case Operation::kGreatEqual:
      return a >= b;
      break;
    case Operation::kInclude: {
      std::string a_string = ToString(a);
      std::string b_string = ToString(b);
      if (a_string.find(b_string) != a_string.npos) {
        return true;
      }
      return false;
    }
    case Operation::kNotInclude: {
      std::string a_string = ToString(a);
      std::string b_string = ToString(b);
      if (a_string.find(b_string) == a_string.npos) {
        return true;
      }
      return false;
    }
    default:
      return false;
      break;
  }
}
template <class Pattern, class DataGraph>
class Predicate {
 public:
  using PatternVertexType = typename Pattern::VertexType;
  using PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type;
  using PatternVertexIDType = typename Pattern::VertexType::IDType;
  using PatternVertexConstPtr =
      typename GUNDAM::VertexHandle<const Pattern>::type;

  using DataGraphIDType = typename DataGraph::VertexType::IDType;
  using DataGraphVertexType = typename DataGraph::VertexType;
  using DataGraphVertexIDType = typename DataGraph::VertexType::IDType;
  using DataGraphVertexPtr = typename GUNDAM::VertexHandle<DataGraph>::type;
  using DataGraphVertexConstPtr =
      typename GUNDAM::VertexHandle<const DataGraph>::type;
  using DataGraphEdgeType = typename DataGraph::EdgeType;
  using DataGraphEdgeIDType = typename DataGraph::EdgeType::IDType;
  using DataGraphEdgeLabelType = typename DataGraphEdgeType::LabelType;
  using DataGraphAttributeConstPtr =
      typename DataGraph::VertexType::AttributeConstPtr;
  using DataGraphAttributePtr = typename DataGraph::VertexType::AttributePtr;
  using DataGraphVertexAttributeKeyType =
      typename DataGraph::VertexType::AttributeKeyType;
  using DataGraphEdgePtr = typename GUNDAM::EdgeHandle<const DataGraph>::type;
  virtual ~Predicate() {}
  virtual bool Satisfy(
      const std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>
          &match_result) const = 0;
  virtual bool Satisfy1(
      const std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>
          &match_result) const = 0;
  virtual double makex_cs_score(
      const DataGraphVertexConstPtr &graph_node, const std::map<DataGraphVertexAttributeKeyType, double> &attributes_frequency) const = 0;
  virtual std::string info() const = 0;
  virtual Predicate<Pattern, DataGraph> *Copy(Pattern &pattern) const = 0;
  virtual bool PredicateCheck(
      const std::map<PatternVertexConstPtr,
                     std::vector<DataGraphVertexConstPtr>> &candidate_set)
      const = 0;
  virtual enum Operation GetOpType() const = 0;
  virtual const double GetPredicateScore() const = 0;
  virtual const DataGraphVertexAttributeKeyType GetAttribute() const = 0;
  virtual const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> GetX() const = 0;
  virtual const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> GetY() const = 0;
  virtual void PublicBuildAttrSet(const std::map<PatternVertexConstPtr, std::vector<DataGraphVertexConstPtr>> &candidate_set,
      std::set<std::variant<int64_t, std::string, double>> &attr_set, bool need_x) const = 0;
};
template <class Pattern, class DataGraph>
class VariablePredicate : public Predicate<Pattern, DataGraph> {
  using BaseLiteralType = Predicate<Pattern, DataGraph>;
  using PatternVertexConstPtr = typename BaseLiteralType::PatternVertexConstPtr;
  using PatternVertexIDType = typename BaseLiteralType::PatternVertexIDType;
  using DataGraphVertexConstPtr =
      typename BaseLiteralType::DataGraphVertexConstPtr;
  using DataGraphVertexPtr = typename BaseLiteralType::DataGraphVertexPtr;
  using DataGraphVertexAttributeKeyType =
      typename BaseLiteralType::DataGraphVertexAttributeKeyType;
  using DataGraphAttributeConstPtr =
      typename BaseLiteralType::DataGraphAttributeConstPtr;
  using DataGraphAttributePtr = typename BaseLiteralType::DataGraphAttributePtr;
  using DataGraphEdgeIDType = typename BaseLiteralType::DataGraphEdgeIDType;
  using DataGraphEdgePtr = typename GUNDAM::EdgeHandle<const DataGraph>::type;
  using DataGraphVertexIDType = typename BaseLiteralType::DataGraphIDType;

 private:
  std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> x_, y_;
  double predicate_score_;
  enum Operation op_;

 public:
  enum Operation GetOpType() const override {return op_;}
  const double GetPredicateScore() const override {return predicate_score_;}
  const DataGraphVertexAttributeKeyType GetAttribute() const override {return this->x_.second;}
  const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> GetX() const override {return x_;}
  const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> GetY() const override {return y_;}
  void PublicBuildAttrSet(const std::map<PatternVertexConstPtr, std::vector<DataGraphVertexConstPtr>> &candidate_set,
      std::set<std::variant<int64_t, std::string, double>> &attr_set, bool need_x) const override {
    if (need_x) {
      BuildAttrSet(this->x_, candidate_set, attr_set);
    } else {
      BuildAttrSet(this->y_, candidate_set, attr_set);
    }
  }

  VariablePredicate(const Pattern &pattern, PatternVertexIDType x_id,
                    DataGraphVertexAttributeKeyType x_attr_key,
                    PatternVertexIDType y_id,
                    DataGraphVertexAttributeKeyType y_attr_key,
                    enum Operation op,
                    double predicate_score) {
    this->x_.first = pattern.FindVertex(x_id);
    this->x_.second = x_attr_key;
    this->y_.first = pattern.FindVertex(y_id);
    this->y_.second = y_attr_key;
    this->op_ = op;
    this->predicate_score_ = predicate_score;
  }
  ~VariablePredicate() {}
  virtual bool Satisfy(
      const std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>
          &match_result) const override {
    if (match_result.find(this->x_.first) == match_result.end() ||
        match_result.find(this->y_.first) == match_result.end()) {
      return true;
    }
    DataGraphVertexConstPtr match_x_ptr =
        match_result.find(this->x_.first)->second;
    DataGraphVertexConstPtr match_y_ptr =
        match_result.find(this->y_.first)->second;
    DataGraphAttributeConstPtr x_attr_ptr =
        match_x_ptr->FindAttribute(this->x_.second);
    DataGraphAttributeConstPtr y_attr_ptr =
        match_y_ptr->FindAttribute(this->y_.second);
    if (x_attr_ptr.IsNull() || y_attr_ptr.IsNull()) return true;
    GUNDAM::BasicDataType x_value_type = x_attr_ptr->value_type();
    GUNDAM::BasicDataType y_value_type = y_attr_ptr->value_type();
    if (x_value_type != y_value_type) return false;
    switch (x_value_type) {
      case GUNDAM::BasicDataType::kTypeInt:
        if (!CheckOp(this->op_, x_attr_ptr->template const_value<int>(),
                     y_attr_ptr->template const_value<int>())) {
          return false;
        }
        break;
      case GUNDAM::BasicDataType::kTypeDouble:
        if (!CheckOp(this->op_, x_attr_ptr->template const_value<double>(),
                     y_attr_ptr->template const_value<double>())) {
          return false;
        }
        break;
      case GUNDAM::BasicDataType::kTypeString:
        if (!CheckOp(this->op_, x_attr_ptr->template const_value<std::string>(),
                     y_attr_ptr->template const_value<std::string>())) {
          return false;
        }
        break;
      default:
        break;
    }
    return true;
  }
  virtual bool Satisfy1(
      const std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>
          &match_result) const override {
    return true;
  }
  virtual double makex_cs_score(
      const DataGraphVertexConstPtr &graph_node, const std::map<DataGraphVertexAttributeKeyType, double> &attributes_frequency) const override {
        return 1.0;
  }

  virtual std::string info() const override { return "Variable"; }
  virtual Predicate<Pattern, DataGraph> *Copy(Pattern &pattern) const override {
    PatternVertexIDType x_id = this->x_.first->id();
    PatternVertexIDType y_id = this->y_.first->id();
    Predicate<Pattern, DataGraph> *literal_ptr =
        new VariablePredicate<Pattern, DataGraph>(
            pattern, x_id, this->x_.second, y_id, this->y_.second, this->op_, this->predicate_score_);
    return literal_ptr;
  }
  virtual bool PredicateCheck(
      const std::map<PatternVertexConstPtr,
                     std::vector<DataGraphVertexConstPtr>> &candidate_set)
      const override {
    PatternVertexIDType x_id = this->x_.first->id();
    PatternVertexIDType y_id = this->y_.first->id();

    

    std::set<std::variant<int64_t, std::string, double>> xa_set, yb_set;

    BuildAttrSet(this->x_, candidate_set, xa_set);
    BuildAttrSet(this->y_, candidate_set, yb_set);



    if (xa_set.empty() || yb_set.empty()) {
      return false;
    }
    if (xa_set.begin()->index() != yb_set.begin()->index()) {
      return false;
    }

    if (!AttributeCheck(xa_set, yb_set)) {
      return false;
    }
    return true;
  }

 private:
  void BuildAttrSet(
      const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType>
          &attr_pair,
      const std::map<PatternVertexConstPtr,
                     std::vector<DataGraphVertexConstPtr>> &candidate_set,
      std::set<std::variant<int64_t, std::string, double>> &attr_set) const {
    for (auto &attr_candidate : candidate_set.find(attr_pair.first)->second) {
      DataGraphAttributeConstPtr attr_ptr =
          attr_candidate->FindAttribute(attr_pair.second);
      if (attr_ptr.IsNull()) {
        continue;
      }

      GUNDAM::BasicDataType value_type = attr_ptr->value_type();
      switch (value_type) {
        case GUNDAM::BasicDataType::kTypeInt:
          attr_set.insert(std::variant<int64_t, std::string, double>(
              attr_ptr->template const_value<int64_t>()));
          break;
        case GUNDAM::BasicDataType::kTypeDouble:
          attr_set.insert(std::variant<int64_t, std::string, double>(
              attr_ptr->template const_value<double>()));
          break;
        case GUNDAM::BasicDataType::kTypeString:
          attr_set.insert(std::variant<int64_t, std::string, double>(
              attr_ptr->template const_value<std::string>()));
          break;
        default:
          break;
      }
    }
  }
  bool AttributeCheck(
      std::set<std::variant<int64_t, std::string, double>> &xa_set,
      std::set<std::variant<int64_t, std::string, double>> &yb_set) const {
    if (this->op_ == Operation::kLess) {
      std::set<std::variant<int64_t, std::string, double>>::iterator iter_min_xa = std::min_element(xa_set.begin(), xa_set.end());
      std::set<std::variant<int64_t, std::string, double>>::iterator iter_max_yb = std::max_element(yb_set.begin(), yb_set.end());
      if (*iter_min_xa < *iter_max_yb) {
        return true;
      }
    }
    else {
      for (auto &it : xa_set) {
        switch (this->op_) {
          case Operation::kEqual:
            if (yb_set.count(it)) {
              return true;
            }
            break;
          case Operation::kNotEqual:
            if (yb_set.size() > 1) {
              return true;
            }
            if (yb_set.size() == 1 && yb_set.count(it) == 0) {
              return true;
            }
            break;

          case Operation::kLess: {
            auto find_it = yb_set.lower_bound(it);
            if (find_it != yb_set.begin()) {
              return true;
            }
            break;
          }

          case Operation::kLessEqual: {
            auto find_it = yb_set.upper_bound(it);
            if (find_it != yb_set.begin()) {
              return true;
            }
            break;
          }

          case Operation::kGreat: {
            auto find_it = yb_set.upper_bound(it);
            if (find_it != yb_set.end()) {
              return true;
            }
            break;
          }

          case Operation::kGreatEqual: {
            auto find_it = yb_set.lower_bound(it);
            if (find_it != yb_set.end()) {
              return true;
            }
            break;
          }

          default:
            break;
        }
      }
    }

    return false;
  }
};



template <class Pattern, class DataGraph>
class WLPredicate : public Predicate<Pattern, DataGraph> {
  using BaseLiteralType = Predicate<Pattern, DataGraph>;
  using PatternVertexConstPtr = typename BaseLiteralType::PatternVertexConstPtr;
  using PatternVertexIDType = typename BaseLiteralType::PatternVertexIDType;
  using DataGraphVertexConstPtr =
      typename BaseLiteralType::DataGraphVertexConstPtr;
  using DataGraphVertexPtr = typename BaseLiteralType::DataGraphVertexPtr;
  using DataGraphVertexAttributeKeyType =
      typename BaseLiteralType::DataGraphVertexAttributeKeyType;
  using DataGraphAttributeConstPtr =
      typename BaseLiteralType::DataGraphAttributeConstPtr;
  using DataGraphAttributePtr = typename BaseLiteralType::DataGraphAttributePtr;
  using DataGraphEdgeIDType = typename BaseLiteralType::DataGraphEdgeIDType;
  using DataGraphEdgePtr = typename GUNDAM::EdgeHandle<const DataGraph>::type;
  using DataGraphVertexIDType = typename BaseLiteralType::DataGraphIDType;

 private:
  std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> x_, y_;
  double predicate_score_;
  enum Operation op_;

 public:
  enum Operation GetOpType() const override {return op_;}
  const double GetPredicateScore() const override {return predicate_score_;}
  const DataGraphVertexAttributeKeyType GetAttribute() const override {return this->x_.second;}
  const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> GetX() const override {return x_;}
  const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> GetY() const override {return y_;}
  void PublicBuildAttrSet(const std::map<PatternVertexConstPtr, std::vector<DataGraphVertexConstPtr>> &candidate_set,
      std::set<std::variant<int64_t, std::string, double>> &attr_set, bool need_x) const override {
    if (need_x) {
      BuildAttrSet(this->x_, candidate_set, attr_set);
    } else {
      BuildAttrSet(this->y_, candidate_set, attr_set);
    }
  }

  WLPredicate(const Pattern &pattern, PatternVertexIDType x_id,
                    DataGraphVertexAttributeKeyType x_attr_key,
                    PatternVertexIDType y_id,
                    DataGraphVertexAttributeKeyType y_attr_key,
                    enum Operation op,
                    double predicate_score) {
    this->x_.first = pattern.FindVertex(x_id);
    this->x_.second = x_attr_key;
    this->y_.first = pattern.FindVertex(y_id);
    this->y_.second = y_attr_key;
    this->op_ = op;
    this->predicate_score_ = predicate_score;
  }
  ~WLPredicate() {}
  virtual bool Satisfy(
      const std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>
          &match_result) const override {
    if (match_result.find(this->x_.first) == match_result.end() ||
        match_result.find(this->y_.first) == match_result.end()) {
      return true;
    }
    DataGraphVertexConstPtr match_x_ptr =
        match_result.find(this->x_.first)->second;
    DataGraphVertexConstPtr match_y_ptr =
        match_result.find(this->y_.first)->second;
    DataGraphAttributeConstPtr x_attr_ptr =
        match_x_ptr->FindAttribute(this->x_.second);
    DataGraphAttributeConstPtr y_attr_ptr =
        match_y_ptr->FindAttribute(this->y_.second);
    if (x_attr_ptr.IsNull() || y_attr_ptr.IsNull()) return true;
    GUNDAM::BasicDataType x_value_type = x_attr_ptr->value_type();
    GUNDAM::BasicDataType y_value_type = y_attr_ptr->value_type();
    if (x_value_type != y_value_type) return false;
    switch (x_value_type) {
      case GUNDAM::BasicDataType::kTypeInt:
        if (!CheckOp(this->op_, x_attr_ptr->template const_value<int>(),
                     y_attr_ptr->template const_value<int>())) {
          return false;
        }
        break;
      case GUNDAM::BasicDataType::kTypeDouble:
        if (!CheckOp(this->op_, x_attr_ptr->template const_value<double>(),
                     y_attr_ptr->template const_value<double>())) {
          return false;
        }
        break;
      case GUNDAM::BasicDataType::kTypeString:
        if (!CheckOp(this->op_, x_attr_ptr->template const_value<std::string>(),
                     y_attr_ptr->template const_value<std::string>())) {
          return false;
        }
        break;
      default:
        break;
    }
    return true;
  }
  virtual bool Satisfy1(
      const std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>
          &match_result) const override {
    return true;
  }
  virtual double makex_cs_score(
      const DataGraphVertexConstPtr &graph_node, const std::map<DataGraphVertexAttributeKeyType, double> &attributes_frequency) const override {
        return 1.0;
  }

  virtual std::string info() const override { return "WL"; }
  virtual Predicate<Pattern, DataGraph> *Copy(Pattern &pattern) const override {
    PatternVertexIDType x_id = this->x_.first->id();
    PatternVertexIDType y_id = this->y_.first->id();
    Predicate<Pattern, DataGraph> *literal_ptr =
        new WLPredicate<Pattern, DataGraph>(
            pattern, x_id, this->x_.second, y_id, this->y_.second, this->op_, this->predicate_score_);
    return literal_ptr;
  }
  virtual bool PredicateCheck(
      const std::map<PatternVertexConstPtr,
                     std::vector<DataGraphVertexConstPtr>> &candidate_set)
      const override {
    std::set<std::variant<int64_t, std::string, double>> xa_set, yb_set;
    BuildAttrSet(this->x_, candidate_set, xa_set);
    BuildAttrSet(this->y_, candidate_set, yb_set);
    if (xa_set.empty() || yb_set.empty()) {
      return false;
    }
    if (xa_set.begin()->index() != yb_set.begin()->index()) {
      return false;
    }

    if (!AttributeCheck(xa_set, yb_set)) {
      return false;
    }
    return true;
  }

 private:
  void BuildAttrSet(
      const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType>
          &attr_pair,
      const std::map<PatternVertexConstPtr,
                     std::vector<DataGraphVertexConstPtr>> &candidate_set,
      std::set<std::variant<int64_t, std::string, double>> &attr_set) const {
    for (auto &attr_candidate : candidate_set.find(attr_pair.first)->second) {
      DataGraphAttributeConstPtr attr_ptr =
          attr_candidate->FindAttribute(attr_pair.second);
      if (attr_ptr.IsNull()) {
        continue;
      }

      GUNDAM::BasicDataType value_type = attr_ptr->value_type();
      switch (value_type) {
        case GUNDAM::BasicDataType::kTypeInt:
          attr_set.insert(std::variant<int64_t, std::string, double>(
              attr_ptr->template const_value<int64_t>()));
          break;
        case GUNDAM::BasicDataType::kTypeDouble:
          attr_set.insert(std::variant<int64_t, std::string, double>(
              attr_ptr->template const_value<double>()));
          break;
        case GUNDAM::BasicDataType::kTypeString:
          attr_set.insert(std::variant<int64_t, std::string, double>(
              attr_ptr->template const_value<std::string>()));
          break;
        default:
          break;
      }
    }
  }
  bool AttributeCheck(
      std::set<std::variant<int64_t, std::string, double>> &xa_set,
      std::set<std::variant<int64_t, std::string, double>> &yb_set) const {
    if (this->op_ == Operation::kLess) {
      std::set<std::variant<int64_t, std::string, double>>::iterator iter_min_xa = std::min_element(xa_set.begin(), xa_set.end());
      std::set<std::variant<int64_t, std::string, double>>::iterator iter_max_yb = std::max_element(yb_set.begin(), yb_set.end());
      if (*iter_min_xa < *iter_max_yb) {
        return true;
      }
    }
    else {
      for (auto &it : xa_set) {
        switch (this->op_) {
          case Operation::kEqual:
            if (yb_set.count(it)) {
              return true;
            }
            break;
          case Operation::kNotEqual:
            if (yb_set.size() > 1) {
              return true;
            }
            if (yb_set.size() == 1 && yb_set.count(it) == 0) {
              return true;
            }
            break;

          case Operation::kLess: {
            auto find_it = yb_set.lower_bound(it);
            if (find_it != yb_set.begin()) {
              return true;
            }
            break;
          }

          case Operation::kLessEqual: {
            auto find_it = yb_set.upper_bound(it);
            if (find_it != yb_set.begin()) {
              return true;
            }
            break;
          }

          case Operation::kGreat: {
            auto find_it = yb_set.upper_bound(it);
            if (find_it != yb_set.end()) {
              return true;
            }
            break;
          }

          case Operation::kGreatEqual: {
            auto find_it = yb_set.lower_bound(it);
            if (find_it != yb_set.end()) {
              return true;
            }
            break;
          }

          default:
            break;
        }
      }
    }

    return false;
  }
};







template <class Pattern, class DataGraph, typename ConstantType>
class ConstantPredicate : public Predicate<Pattern, DataGraph> {
  using BaseLiteralType = Predicate<Pattern, DataGraph>;
  using PatternVertexConstPtr = typename BaseLiteralType::PatternVertexConstPtr;
  using PatternVertexIDType = typename BaseLiteralType::PatternVertexIDType;
  using DataGraphVertexConstPtr =
      typename BaseLiteralType::DataGraphVertexConstPtr;
  using DataGraphVertexPtr = typename BaseLiteralType::DataGraphVertexPtr;
  using DataGraphVertexAttributeKeyType =
      typename BaseLiteralType::DataGraphVertexAttributeKeyType;
  using DataGraphAttributeConstPtr =
      typename BaseLiteralType::DataGraphAttributeConstPtr;
  using DataGraphAttributePtr = typename BaseLiteralType::DataGraphAttributePtr;
  using DataGraphEdgeIDType = typename BaseLiteralType::DataGraphEdgeIDType;
  using DataGraphEdgePtr = typename GUNDAM::EdgeHandle<const DataGraph>::type;
  using DataGraphVertexIDType = typename BaseLiteralType::DataGraphIDType;
  
 private:
  std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> x_;
  ConstantType c_;
  enum Operation op_;
  double predicate_score_ = 1.0;
  

 public:
  enum Operation GetOpType() const override {return op_;}
  const double GetPredicateScore() const override {return predicate_score_;}
  const DataGraphVertexAttributeKeyType GetAttribute() const override {return this->x_.second;}
  const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> GetX() const override {return x_;}
  const std::pair<PatternVertexConstPtr, DataGraphVertexAttributeKeyType> GetY() const override {return x_;}
  void PublicBuildAttrSet(const std::map<PatternVertexConstPtr, std::vector<DataGraphVertexConstPtr>> &candidate_set,
      std::set<std::variant<int64_t, std::string, double>> &attr_set, bool need_x) const override {
    ; 
  }
  ConstantPredicate(const Pattern &pattern, PatternVertexIDType x_id,
                    DataGraphVertexAttributeKeyType attr_key, ConstantType c,
                    enum Operation op,
                    double predicate_score) {
    this->x_.first = pattern.FindVertex(x_id);
    this->x_.second = attr_key;
    this->c_ = c;
    this->op_ = op;
    this->predicate_score_ = predicate_score;
  }
  ~ConstantPredicate() {}
  virtual bool Satisfy(
      const std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>
          &match_result) const override {
    if (match_result.find(this->x_.first) == match_result.end()) {
      return true;
    }
    DataGraphVertexConstPtr match_x_ptr =
        match_result.find(this->x_.first)->second;
    DataGraphAttributeConstPtr x_attr_ptr =
        match_x_ptr->FindAttribute(this->x_.second);
    if (x_attr_ptr.IsNull()) {
      return true;
    }
    if (CheckOp(this->op_, x_attr_ptr->template const_value<ConstantType>(),
                this->c_))
      return false;
    return true;
  }


  virtual bool Satisfy1(
      const std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>
          &match_result) const override {
    if (match_result.find(this->x_.first) == match_result.end()) {
      return true;
    }
    DataGraphVertexConstPtr match_x_ptr =
        match_result.find(this->x_.first)->second;
    DataGraphAttributeConstPtr x_attr_ptr =
        match_x_ptr->FindAttribute(this->x_.second);
    if (x_attr_ptr.IsNull()) {
      return false;
    }
    if (CheckOp(this->op_, x_attr_ptr->template const_value<ConstantType>(),
                this->c_))
      return true;
    return false;
  }



  virtual double makex_cs_score(
    const DataGraphVertexConstPtr &graph_node, const std::map<DataGraphVertexAttributeKeyType, double> &attributes_frequency) const override {
    DataGraphAttributeConstPtr x_attr_ptr;

    if (attributes_frequency.find(static_cast<DataGraphVertexAttributeKeyType>(this->x_.second)) != attributes_frequency.end()) {
      x_attr_ptr = graph_node->FindAttribute(this->x_.second);
    }

    if (x_attr_ptr.IsNull()) {
      return 1.0;
    }
    double attributes_value = 0.0;

    GUNDAM::BasicDataType value_type = x_attr_ptr->value_type();
    switch (value_type) {
      case GUNDAM::BasicDataType::kTypeInt:
        attributes_value = (double)x_attr_ptr->template const_value<int64_t>();
        break;
      case GUNDAM::BasicDataType::kTypeDouble:
        attributes_value = (double)x_attr_ptr->template const_value<double>();
        break;
      default:
        attributes_value = 1.0;
        break;
    }

    auto graph_node_score = 1.0;
    
    if (attributes_frequency.find(static_cast<DataGraphVertexAttributeKeyType>(this->x_.second)) != attributes_frequency.end()) {
        double numeric_value = attributes_frequency.at(static_cast<DataGraphVertexAttributeKeyType>(this->x_.second));
        graph_node_score = 1.0 + 1.0 - score_transform_predicate(abs(numeric_value - attributes_value) / (0.5 * numeric_value));
    } 
    return graph_node_score;
  }


  virtual std::string info() const override { return "Constant"; }
  virtual Predicate<Pattern, DataGraph> *Copy(Pattern &pattern) const override {
    PatternVertexIDType x_id = this->x_.first->id();
    Predicate<Pattern, DataGraph> *literal_ptr =
        new ConstantPredicate<Pattern, DataGraph, ConstantType>(
            pattern, x_id, this->x_.second, this->c_, this->op_, this->predicate_score_);
    return literal_ptr;
  }
  virtual bool PredicateCheck(
      const std::map<PatternVertexConstPtr,
                     std::vector<DataGraphVertexConstPtr>> &candidate_set)
      const override {
    return true;
  }
};
}#endif