#include "math_utils.h"

#include <cmath>
#include <stdexcept>

using std::invalid_argument;
using std::sqrt;
using std::vector;

namespace math_utils {

double
euclideanDistance(const vector<double>& a, const vector<double>& b)
{
  if (a.size() != b.size()) {
    throw invalid_argument("Vectors must have same dimension");
  }

  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sqrt(sum);
}

}  // namespace math_utils