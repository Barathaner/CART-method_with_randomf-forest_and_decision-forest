# CART Method with Random Forest and Decision Forest

Welcome to the GitHub repository for me implementation of the CART method with Random Forest and Decision Forest classifiers. This project aims to explore the functionality of ensemble classifiers in handling complex classification tasks using basic Python libraries like pandas and math, and without the use of advanced machine learning frameworks.

## Introduction

Random Forests and Decision Forests are integral in fields such as finance and healthcare due to their robustness against overfitting and capability to handle high-dimensional data. These ensemble classifiers improve prediction accuracy by combining multiple decision trees, making them ideal for applications like stock market analysis and fraud detection. This project implements these classifiers from scratch, focusing on their fundamental mechanisms and the use of parallelization for efficiency.

### Features

- Implementation of CART (Classification and Regression Trees) using Python.
- Use of Gini Impurity as the splitting criterion.
- Parallel building of trees to leverage computational resources.
- Comprehensive comparison between Random Forest and Decision Forest approaches.
- Visualization of Decision Trees using the Graphviz library.

## Repository Structure

- **src/**: Contains all the Python scripts for the project.
- **data/**: Sample datasets used for testing and demonstration.
- **docs/**: Documentation and results of the ensemble classifiers.
- **output/**: Results from the classifier executions and performance metrics.

## Environment Setup

### Prerequisites

- Python 3.11
- Graphviz (for tree visualization)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Barathaner/CART-method_with_randomf-forest_and_decision-forest.git
   cd CART-method_with_randomf-forest_and_decision-forest
   ```

2. Set up a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Navigate to the `src` directory and run the `main.py` script:
```bash
cd src
python main.py
```

## Expected Outcomes

- The application will process the datasets using both Random Forest and Decision Forest classifiers.
- Outputs including classification accuracy, tree structures, and performance metrics are saved in the `output` directory.
- Visualization files (.png) for each decision tree generated during the execution.

## Contributing

Contributions to this project are welcome! Please refer to the CONTRIBUTING.md file for guidelines on how to make contributions.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

- Thanks to the creators of the datasets used in this project, available through the UCI Machine Learning Repository.
- This project was inspired by the foundational papers on Random Forests and Decision Forests cited throughout the research.

Enjoy exploring ensemble classifiers with my CART method implementation! For any issues or further inquiries, please open an issue on this repository.
