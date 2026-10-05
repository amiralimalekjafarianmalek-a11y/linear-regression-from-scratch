
# Linear Regression From Scratch

A simple implementation of Linear Regression from scratch using NumPy.

This project uses the California Housing dataset to predict median house values.

## Project Goal

The goal of this project is to understand how Linear Regression works internally without using machine learning libraries such as Scikit-Learn.

The model is trained using Gradient Descent.

## Dataset

The project uses the California Housing dataset.

The dataset contains information about California districts, including:

- Median income
- Housing median age
- Total rooms
- Total bedrooms
- Population
- Households
- Latitude
- Longitude

The target variable is:

- Median house value

Dataset size:

- 20,640 samples
- 8 input features
- 1 target variable

## Project Structure

```text
linear-regression-from-scratch/
│
├── data/
│   └── housing.csv
│
├── linear_regression.py
├── README.md
├── requirements.txt
└── .gitignore
