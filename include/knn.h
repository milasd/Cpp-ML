#pragma once

#include <string>
#include <vector>

class KNearestNeighbors {
 private:
  int k;
  std::vector<std::vector<double>> X_train;
  std::vector<int> y_train;

 public:
  explicit KNearestNeighbors(int k = 3);

  void fit(
      const std::vector<std::vector<double>>& X, const std::vector<int>& y);

  int predict(const std::vector<double>& sample) const;
  std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

  double score(
      const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) const;

  static std::pair<std::vector<std::vector<double>>, std::vector<int>> loadData(
      const std::string& filepath);
};