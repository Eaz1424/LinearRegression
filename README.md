---

# Linear Regression From Scratch

## Overview

This repository contains an implementation of Linear Regression from scratch using **gradient descent**, a fundamental machine learning algorithm used for predictive and statistical analysis. The project is designed to help users understand how Linear Regression works and provides a hands-on approach to building and using this model with Python.

## 🔍 What It Does

- Reads training and test data from CSV files.
- Applies **gradient descent** to learn the parameters (slope and intercept).
- Uses a stopping condition based on the **change in cost function**.
- Evaluates the model on the test set.
- Prints key metrics: **MSE**, **RMSE**, and **R² Score**.

## Technologies Used

- **Python**: The programming language used for implementation.
- **NumPy**: For numerical operations and matrix computations.
- **pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting and visualizing data and results.
- **Git**: Version Control

## Project Structure

```
LinearRegression/
├── dataset/               # Contains example datasets
├── code/               # Implementation of Linear Regression
├── graph/              # Plots generated
└── README.md           # Project documentation
```
## 📚 Requirements

This project uses:

- Python 3.x
- numpy
- pandas
- matplotlib (for visualization)

## Getting Started

1. **Clone the Repository**  
   Clone the repository to your local machine using the following command:
   ```bash
   git clone https://github.com/Eaz1424/LinearRegression.git
   cd LinearRegression
   ```

2. **Install Dependencies**  
   Install the required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Code**  
   Execute the Linear Regression implementation:
   ```bash
   python code/linear_regression.py
   ```

4. **Use Your Own Data**  
   Place your dataset in the `dataset` folder and modify the script to load your file for analysis.



## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
