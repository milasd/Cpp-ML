#include <iostream>

#include "knn.h"

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::min;
using std::vector;

int
main()
{
  try {
    const std::string data_path = "data/sample/iris.csv";
    auto [X, y] = KNearestNeighbors::loadData(data_path);

    size_t split_idx = X.size() * 0.8;
    vector<vector<double>> X_train(X.begin(), X.begin() + split_idx);
    vector<vector<double>> X_test(X.begin() + split_idx, X.end());
    vector<int> y_train(y.begin(), y.begin() + split_idx);
    vector<int> y_test(y.begin() + split_idx, y.end());

    KNearestNeighbors knn(3);
    knn.fit(X_train, y_train);

    double accuracy = knn.score(X_test, y_test);
    cout << "Sample accuracy: " << accuracy << endl;

    auto predictions = knn.predict(X_test);
    cout << "First 5 predictions: ";
    for (size_t i = 0; i < min(size_t(5), predictions.size()); ++i) {
      cout << predictions[i] << " ";
    }
    cout << endl;
  }
  catch (const exception& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}