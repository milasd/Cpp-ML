# Cpp-ML

Machine Learning algorithms from scratch in C++. Does not include Deep Learning.

```bash
cpp_ml/                 # ML algorithms codebase
    ├── src/
    │   ├── supervised/     # Supervised learning algorithms
    │   ├── unsupervised/   # Unsupervised learning algorithms  
    │   └── utils/          # Common functions (eg.: distance calculation etc.)
    ├── include/            # Header files
    └── tests/              # Unit tests
data/                   # Sample datasets
demos/                  # Usage examples
```

## Run examples

Although the ML implementations in this repo are for personal practice, you can build and run the examples inside the `demos` folder. There should be a task command assigned for each demo.

Build and run the K-NN example:
```bash
task demo:knn
```

## Development

There are multiple task commands for development; check full list with `task --list`. Some of the most used ones are listed in this section.

Clean build artifacts:
```bash
task clean
```

Format code:
```bash
task format
```
