#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "knn.h"
#include "math_utils.h"

using std::exception;
using std::getline;
using std::ifstream;
using std::invalid_argument;
using std::pair;
using std::partial_sort;
using std::runtime_error;
using std::sqrt;
using std::stod;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::vector;

KNearestNeighbors::KNearestNeighbors(int k) : k(k)
{
  if (k <= 0) {
    throw invalid_argument("k must be positive");
  }
}

void
KNearestNeighbors::fit(const vector<vector<double>>& X, const vector<int>& y)
{
  if (X.size() != y.size()) {
    throw invalid_argument("X and y must have same number of samples");
  }
  if (X.empty()) {
    throw invalid_argument("Training data cannot be empty");
  }

  X_train = X;
  y_train = y;
}


int
KNearestNeighbors::predict(const vector<double>& sample) const
{
  if (X_train.empty()) {
    throw runtime_error("Model must be fitted before prediction");
  }

  vector<pair<double, int>> distances;
  distances.reserve(X_train.size());

  for (size_t i = 0; i < X_train.size(); ++i) {
    double dist = math_utils::euclideanDistance(sample, X_train[i]);
    distances.emplace_back(dist, y_train[i]);
  }

  partial_sort(distances.begin(), distances.begin() + k, distances.end());

  unordered_map<int, int> votes;
  for (int i = 0; i < k; ++i) {
    votes[distances[i].second]++;
  }

  int predicted_class = votes.begin()->first;
  int max_votes = votes.begin()->second;

  for (const auto& vote : votes) {
    if (vote.second > max_votes) {
      max_votes = vote.second;
      predicted_class = vote.first;
    }
  }

  return predicted_class;
}

vector<int>
KNearestNeighbors::predict(const vector<vector<double>>& X) const
{
  vector<int> predictions;
  predictions.reserve(X.size());

  for (const auto& sample : X) {
    predictions.push_back(predict(sample));
  }

  return predictions;
}

double
KNearestNeighbors::score(
    const vector<vector<double>>& X, const vector<int>& y) const
{
  if (X.size() != y.size()) {
    throw invalid_argument("X and y must have same number of samples");
  }

  auto predictions = predict(X);
  int correct = 0;

  for (size_t i = 0; i < y.size(); ++i) {
    if (predictions[i] == y[i]) {
      correct++;
    }
  }

  return static_cast<double>(correct) / y.size();
}

pair<vector<vector<double>>, vector<int>>
KNearestNeighbors::loadData(const string& filepath)
{
  ifstream file(filepath);
  if (!file.is_open()) {
    throw runtime_error("Cannot open file: " + filepath);
  }

  vector<vector<double>> X;
  vector<int> y;
  string line;

  while (getline(file, line)) {
    if (line.empty())
      continue;

    stringstream ss(line);
    string cell;
    vector<double> row;

    while (getline(ss, cell, ',')) {
      try {
        row.push_back(stod(cell));
      }
      catch (const exception&) {
        throw runtime_error("Invalid data format in file");
      }
    }

    if (row.empty())
      continue;

    y.push_back(static_cast<int>(row.back()));
    row.pop_back();
    X.push_back(row);
  }

  return {X, y};
}