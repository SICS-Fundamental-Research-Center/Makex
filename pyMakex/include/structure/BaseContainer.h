#ifndef _BASE_CONTAINER_H
#define _BASE_CONTAINER_H
#include <vector>
namespace CFLogic {
template <class BaseClass>
class BaseContainer {
 private:
  using BaseContainerType = std::vector<BaseClass *>;
  using BaseContainerIterator = typename BaseContainerType::iterator;
  using BaseContainerConstIterator = typename BaseContainerType::const_iterator;
  BaseContainerType base_container_;

 public:
  BaseContainer() { this->base_container_.clear(); }
  ~BaseContainer() {
    for (auto &ptr : this->base_container_) {
      delete ptr;
    }
    this->base_container_.clear();
  }
  BaseContainer(const BaseContainer &b) {
    this->base_container_ = b.base_container_;
  }
  BaseContainerIterator begin() { return this->base_container_.begin(); }
  BaseContainerIterator end() { return this->base_container_.end(); }
  BaseContainerConstIterator begin() const {
    return this->base_container_.cbegin();
  }
  BaseContainerConstIterator end() const {
    return this->base_container_.cend();
  }
  template <class DerivedClass, typename... Args>
  void Add(Args &&...args) {
    BaseClass *base_ptr = new DerivedClass(args...);
    this->base_container_.push_back(base_ptr);
  }
};
}  // namespace CFLogic
#endif