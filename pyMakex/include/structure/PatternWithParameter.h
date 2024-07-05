#ifndef _PATTERN_WITH_PARA_H
#define _PATTERN_WITH_PARA_H
#include <cmath>
namespace CFLogic {
class Parameter {
 public:
  double data, m, v, t;
  Parameter() {
    data = 0;
    m = 0;
    v = 0;
    t = 0;
  }
  Parameter(const Parameter &b) : data(b.data), m(b.m), v(b.v), t(b.t) {}
  Parameter(Parameter &&b) = default;
  void clear() { data = m = v = t = 0; }
  void update(double grad, double learning_rate, double weight_decay = 0) {
    double g = grad - weight_decay * data;

    t += 1;
    m = 0.9 * m + 0.1 * g;
    v = 0.999 * v + 0.001 * g * g;

    double bias1 = 1 - exp(log(0.9) * t);
    double bias2 = 1 - exp(log(0.999) * t);

    double mt = m / bias1;
    double vt = sqrt(v) / sqrt(bias2) + 0.00000001;

    data += learning_rate * mt / vt;
  }
};

template <class Pattern>
class PatternWithParameter {
 private:
  Pattern pattern;
  Parameter weight;
  double H_score;
  double prior_probability;
};
}  // namespace CFLogic
#endif