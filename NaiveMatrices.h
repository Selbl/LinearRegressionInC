// NaiveMatrices.h
#ifndef NAIVEMATRICES_H
#define NAIVEMATRICES_H

#include <stdbool.h>

/*
 * Struct:  Matrix 
 * --------------------
 * Defines a matrix by its dimensions and data
 * All of the elements in the matrices in my implementation are doubles
 *
 *  rows (int): number of rows
 * 
 *  cols (int): number of columns
 *
 *  data (double): pointer to the first element of the matrix.
 *               the data is flattened from a 2D array
 */

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

// Function declarations
bool checkDimensions(const Matrix *matrix_a, const Matrix *matrix_b);
bool isSquare(const Matrix *mat);
bool transposeMatrix(const Matrix *matrix, Matrix *result);
void multiplyScalar(const Matrix *matrix, double scalar);
double norm(const Matrix *vector);
void standardizeMatrix(Matrix *matrix);
bool sumMatrices(const Matrix *matrix_a, const Matrix *matrix_b, bool subtraction, Matrix *result);
bool multiplyMatrices(const Matrix *matrix_a, const Matrix *matrix_b, Matrix *result);
Matrix identityMatrix(int n);
bool inverseMatrix(Matrix *matrix, Matrix *inverse);
void printMatrix(const Matrix *matrix);
Matrix initMatrixZeros(int rows, int cols);
double boxMullerDraw();
Matrix initMatrixRandomNorm(int rows, int cols);
double oneIterGradientDescent(const Matrix *X, Matrix *Theta,const Matrix *Y, Matrix *residuals, double lr);
double linearRegression(const Matrix *X, Matrix *Theta,const Matrix *Y, Matrix *residuals, double lr,int epochs, double loss_tolerance, bool verbose);
void regressionMetrics(const Matrix *X, const Matrix *Theta, const Matrix *residuals, double loss, Matrix *lowerBounds, Matrix *upperBounds, Matrix *pValues);
#endif
