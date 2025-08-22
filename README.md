# Cpp-ML

Machine Learning algorithms from scratch in C++. Does not include Deep Learning.

```
data/               # Sample datasets
demos/              # Usage examples
include/            # Header files
src/
├── supervised/     # Supervised learning algorithms
├── unsupervised/   # Unsupervised learning algorithms  
└── utils/          # Common methods, such as distance calculation etc.
tests/              # Unit tests
```

## Run examples

Although the ML implementations in this repo are for personal practice, you can build and run the examples inside the `demos` folder. There should be a task command assigned for each demo.

Build and run the K-NN example:
```bash
task demo:knn
```

## Development

Clean build artifacts:
```bash
task clean
```

Format code:
```bash
task format
```
