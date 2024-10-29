# LinearRegressionInC

Simple code in C to perform Linear Regression using Gradient Descent. The code allows the user to specify which variables to consider in the regression as well as defining the learning rate and number of iterations. This is a work in progress.

## Future Work

- Generate indicator variables for ordinal and categorical variables
- Add support for reading regression instructions from .txt file
- Add possibility of running multiple regression specifications at once
- Handle cases with missing values
- Add support for reading other file types

## Files

- NaiveMatrices.c : The C file with all the code for the necessary linear algebra operations to perform LinearRegression
- NaiveMatrices.h : Header file to make the functions and Matrix struct usable in the execution code
- Execution.c : The C file that runs the code to execute the linear regression and interact with the user
- data.csv : Weather forecasting dataset from [this Kaggle competition](https://www.kaggle.com/datasets/hanaksoy/customer-purchasing-behaviors)

## Usage

First compile the code by opening up your terminal and using:

```bash
gcc -o LinearReg NaiveMatrices.c Execution.c
```

Then you can run it by executing:

```bash
./LinearReg
```

The table below is an example output of the code:

| Independent Variable | Parameter Value | Confidence Interval    | p-value |
|----------------------|-----------------|------------------------|---------|
| purchase_amount      | 2.9228          | (2.9002, 2.9454)      | 0.0000  |
| age                  | -0.5634         | (-0.5763, -0.5505)    | 0.0000  |
| loyalty_score        | -1.3759         | (-2.5559, -0.1960)    | 0.0223  |


## Limitations

So far, the code only works with numerical variables. It also only does a simple regression and does not consider the possibility of interaction or squared terms.

