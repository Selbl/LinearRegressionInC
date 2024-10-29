#include "NaiveMatrices.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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


//typedef struct{
//    int rows;
//    int cols;
//    double *data;
//} Matrix;


/*
 * Function:  (bool) checkDimensions
 * --------------------
 * Verifies that two matrices have the same dimensions
 * This is crucial when doing certain operations such as sums
 *
 *  *matrix_a (pointer): a pointer to a matrix struct
 * 
 *  *matrix_b (pointer): a pointer to a matrix struct
 */

bool checkDimensions(const Matrix *matrix_a, const Matrix *matrix_b) {
    return (matrix_a->rows == matrix_b->rows && matrix_a->cols == matrix_b->cols);
}

/*
 * Function:  (bool) isSquare
 * --------------------
 * Verifies that a matrix is square
 * This is crucial when doing certain operations such as inversions
 *
 *  *mat (pointer): a pointer to a matrix struct
*/
bool isSquare(const Matrix *mat){
    return (mat->rows == mat->cols);
}

/*
 * Function: (bool) transposeMatrix
 * --------------------
 * Transposes a matrix. If the matrix is square, it transposes in place.
 * If the matrix is rectangular, it creates a new matrix for the result.
 *
 *  matrix (pointer): the original matrix
 *  result (pointer): the transposed matrix, allocated if not square
 *
 *  Returns true if successful, false on failure (e.g., memory allocation issues)
 */
bool transposeMatrix(const Matrix *matrix, Matrix *result) {
    // Define indices of for loops
    int r;
    int c;
    // Check if the matrix is square
    if (isSquare(matrix)) {
        // In-place transpose for square matrix
        for (r = 0; r < matrix->rows; r++) {
            for (c = r + 1; c < matrix->cols; c++) {
                // Swap elements at (r, c) and (c, r)
                double temp = *(matrix->data + r * matrix->cols + c);
                *(matrix->data + r * matrix->cols + c) = *(matrix->data + c * matrix->cols + r);
                *(matrix->data + c * matrix->cols + r) = temp;
            }
        }
        // If the matrix is square, the dimensions of the result are the same as the original
        result->rows = matrix->rows;
        result->cols = matrix->cols;
        result->data = matrix->data;
    } else {
        // Allocate memory for the result if the matrix is not square
        result->rows = matrix->cols;
        result->cols = matrix->rows;
        result->data = (double *)malloc(result->rows * result->cols * sizeof(double));
        if (result->data == NULL) {
            printf("Memory allocation failed for transposed matrix.\n");
            return false;
        }

        // Transpose the matrix into the result matrix
        for (r = 0; r < matrix->rows; r++) {
            for (c = 0; c < matrix->cols; c++) {
                *(result->data + c * result->cols + r) = *(matrix->data + r * matrix->cols + c);
            }
        }
    }
    return true;
}

/*
 * Function:  (void) multiplyScalar
 * --------------------
 * Multiplies a matrix by a scalar
 *
 *  *mat (pointer): a pointer to a matrix struct
 *  scalar (double): a scalar value
*/
void multiplyScalar(const Matrix *matrix, double scalar){
    int i;
    for(i = 0; i < matrix->rows + matrix->cols; i++){
        *(matrix->data + i) *= scalar;
    }
}

/*
 * Function: (double) norm
 * --------------------
 * Returns L2 norm of a vector
 *  vector (pointer): a pointer to vector (Matrix struct) to calculate the norm
*/
double norm(const Matrix *vector){
    // Check that it is in fact a vector
    if (vector->cols != 1){
        printf("Trying to get the norm of a non-vector");
        return 1;
    }
    int i;
    double ret = 0.0;
    for(i = 0; i < vector->rows; i++){
        ret += pow(vector->data[i],2);
    }
    ret = sqrt(ret);
    ret /= vector->rows;
    return ret;
}

/*
 * Function: (void) standardizeMatrix
 * --------------------
 * Standardizes a matrix by making sure all of its values are between -1 and 1
 *  matrix (pointer): a pointer to the matrix to normalize
*/
void standardizeMatrix(Matrix *matrix){
    // Do three passes, one to compute means, one to get standard devs and one to correct numbers
    // The other is to scale every element using those metrics
    double *means = (double *)malloc( matrix->cols * sizeof(double));
    double *stdDevs = (double *)malloc( matrix->cols * sizeof(double));
    // Declare variables of for loops
    int r;
    int c;
    // Initialize means and stdDevs to zero
    for (c = 0; c < matrix->cols; c++) {
        means[c] = 0.0;
        stdDevs[c] = 0.0;
    }
    // First pass: add all elements to the means indices
    for(r = 0;r < matrix->rows;r++){
        for(c=0;c < matrix->cols;c++){
            /*Print corresponding element*/
            means[c] += *(matrix->data + r * matrix->cols + c)/(matrix->rows);
        }
    }
    // Second pass: standard distribution
    for(r = 0;r < matrix->rows;r++){
        for(c=0;c < matrix->cols;c++){
            /*Print corresponding element*/
            stdDevs[c] += pow((*(matrix->data + r * matrix->cols + c)-means[c]),2)/(matrix->rows);
        }
    }
    // Take square root
    for(c = 0; c < matrix->cols; c++){
        stdDevs[c] = sqrt(stdDevs[c]);
    }
    // Standarize elements
    for(r = 0;r < matrix->rows;r++){
        for(c=0;c < matrix->cols;c++){
            /*Print corresponding element*/
            *(matrix->data + r * matrix->cols + c) = (*(matrix->data + r * matrix->cols + c)-means[c])/(stdDevs[c]);
        }
    }
    // free memory
    free(means);
    free(stdDevs);
}

/*
 * Function: (bool) sumMatrices
 * --------------------
 * Sums two matrices and returns whether the sum was successful
 * Fails if both matrices have different dimensions
 * Can handle subtraction as well and it operates as matrix_a - matrix_b
 *
 *  matrix_a (pointer): a pointer to the first matrix struct that you want to sum
 *  matrix_b (pointer): a pointer to the second matrix struct that you want to sum
 *  subtraction (bool): a boolean that indicates whether it is a subtraction instead
 * *result_matrix (pointer): a pointer to the result (a Matrix struct)
*/

bool sumMatrices(const Matrix *matrix_a, const Matrix *matrix_b, bool subtraction, Matrix *result){
    /* Check same dimensions*/
    if(!checkDimensions(matrix_a,matrix_b)){
        printf("Mismatch in the dimensions when summing");
        return false;
    }
    /* Enforce dimensions for result matrix */
    result->rows = matrix_a->rows;
    result->cols = matrix_a->cols;
    /* If it is a subtraction then change every element in matrix_b to its negative */
    if(subtraction){
        multiplyScalar(matrix_b, -1.0);
    }
    /*If it works, then sum the matrices in a flattened fashion*/
    int i;
    for(i = 0;i < matrix_a->rows*matrix_a->cols;i++){
        *(result->data+i) = *(matrix_a->data + i) + *(matrix_b->data + i);
    }
    /*After the routine is over, return success*/
    return true;
}

/*
 * Function: (bool) multiplyMatrices
 * --------------------
 * Multiplies two matrices and returns whether the multiplication was successful
 * Fails if both matrices have dimensions that do not match
 * This implementation requires that the number of columns of the first matrix
 * Equals the number of rows of the second matrix
 * 
 * The multiplication process is as follows:
 * result_{r,c} = sum^{columns_A - 1}_{k=0} A_{r,k} B_{k,c}
 *
 *  matrix_a (pointer): a pointer to the first matrix struct that you want to sum
 *  matrix_b (pointer): a pointer to the second matrix struct that you want to sum
 * *result_matrix (pointer): a pointer to the result (a Matrix struct)
*/

bool multiplyMatrices(const Matrix *matrix_a, const Matrix *matrix_b, Matrix *result){
    /* Check compatible dimensions*/
    if (matrix_a->cols!= matrix_b->rows){
        printf("Incompatible dimensions in matrix multiplication");
        return false;
    }
    /* Enforce dimensions for result matrix */
    result->rows = matrix_a->rows;
    result->cols = matrix_b->cols;
    /* Multiply */
    int r;
    int c;
    int k;
    for(r = 0; r < matrix_a->rows;r++){
        for(c=0;c<matrix_b->cols;c++){
            /* Initialize element at (r,c)*/
            /* Summing flattened array so have to adjust for dimensions*/
            *(result->data + r * result->cols + c) = 0;
            for(k = 0; k < matrix_a->cols; k++){
                *(result->data + r * result->cols + c) += 
                    (*(matrix_a->data + r * matrix_a->cols + k)) * 
                    (*(matrix_b->data + k * matrix_b->cols + c));
            }
        }
    }
    /*After the routine is over, return success*/
    return true;
}

/*
 * Function: (Matrix) identityMatrix
 * --------------------
 * Creates an identity matrix
 *
 * n (int): the dimension of the identity matrix 
 * 
*/
Matrix identityMatrix(int n) {
    Matrix identity = initMatrixZeros(n, n);
    for (int i = 0; i < n; i++) {
        identity.data[i * n + i] = 1.0;
    }
    return identity;
}

/*
 * Function: (bool) inverseMatrix
 * --------------------
 * Inverts a matrix using the Gaussian Elimination Method
 * Returns true if it can be inverted
 *
 * matrix (pointer to Matrix): a pointer to the matrix to invert
 * inverse (pointer to Matrix): a pointer to store the inverse matrix
 * 
*/
bool inverseMatrix(Matrix *matrix, Matrix *inverse) {
    // Check if the matrix is square
    if (!isSquare(matrix)) {
        printf("Matrix is not square.\n");
        return false;
    }
    // Store rows
    int n = matrix->rows;

    // Initialize inverse as an identity matrix
    *inverse = identityMatrix(n);

    // Temporary matrix that stores copy of original data
    Matrix temp = initMatrixZeros(n, n);
    memcpy(temp.data, matrix->data, n * n * sizeof(double));

    // Gaussian elimination
    for (int i = 0; i < n; i++) {
        // Find the pivot element
        double pivot = temp.data[i * n + i];
        if (fabs(pivot) < 1e-10) {  // Check for near-zero pivot
            printf("Matrix is singular or nearly singular.\n");
            free(temp.data);
            return false;
        }

        // Scale pivot row to make pivot element equal to 1
        for (int j = 0; j < n; j++) {
            temp.data[i * n + j] /= pivot;
            inverse->data[i * n + j] /= pivot;
        }

        // Make other rows zero in current column
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = temp.data[k * n + i];
                for (int j = 0; j < n; j++) {
                    temp.data[k * n + j] -= factor * temp.data[i * n + j];
                    inverse->data[k * n + j] -= factor * inverse->data[i * n + j];
                }
            }
        }
    }

    free(temp.data);
    return true;
}


/*
 * Function: (void) printMatrix
 * --------------------
 *  Prints a matrix in a 2D fashion
 *  Useful for debugging
 *
 *  matrix (pointer): a pointer to a Matrix struct
*/
void printMatrix(const Matrix *matrix){
    int r;
    int c;
    for(r = 0; r < matrix->rows;r++){
        for(c=0;c < matrix->cols;c++){
            /*Print corresponding element*/
            printf("%f ", *(matrix->data + r * matrix->cols + c));
        }
        /*Skip a line*/
        printf("\n");
    }
}

/*
 * Function: Matrix initMatrixZeros
 * --------------------
 *  Initializes a matrix to a matrix of zeros
 *
 *  rows (int): the number of rows of the matrix
 *  cols (int): the number of columns of the matrix
*/
Matrix initMatrixZeros(int rows, int cols) {
    // Allocate memory for the Matrix struct
    Matrix ret;
    ret.rows = rows;
    ret.cols = cols;
    
    // Allocate memory for the 2D matrix data (stored as a 1D array)
    ret.data = (double *)malloc(rows * cols * sizeof(double));
    
    // Check if memory allocation was successful
    if (ret.data == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);  // Exit the program if malloc fails
    }
    
    // Initialize the allocated memory to zero using memset
    memset(ret.data, 0, rows * cols * sizeof(double));
    
    return ret;
}

/*
 * Function: Box-Muller Draw
 * --------------------
 *  Generates a standard normal random variable draw
 *  Uses the Box-Muller transform to transform two uniform draws into
 *  standard normal
 *
*/
double boxMullerDraw(){
    // Generate two random uniform variables using the random function
    double u1 = (double)rand()/RAND_MAX;
    double u2 = (double)rand()/RAND_MAX;
    // Prevent log of zero
    if (u1 == 0.0) {
        u1 = 1e-10;
    }
    // Generate draws
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
    // Randomly return one or the other
    double det = (double)rand()/RAND_MAX;
    if(det >= 0.5){
        return z0;
    }
    return z1;
}

/*
 * Function: Matrix initMatrixRandomNorm
 * --------------------
 *  Initializes a matrix to random values according to a standard normal
 *
 *  rows (int): the number of rows of the matrix
 *  cols (int): the number of columns of the matrix
*/
Matrix initMatrixRandomNorm(int rows, int cols) {
    // Allocate memory for the Matrix struct
    Matrix ret;
    ret.rows = rows;
    ret.cols = cols;
    
    // Allocate memory for the 2D matrix data (stored as a 1D array)
    ret.data = (double *)malloc(rows * cols * sizeof(double));
    
    // Check if memory allocation was successful
    if (ret.data == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);  // Exit the program if malloc fails
    }
    
    // Call the BoxMuller draw for every element
    int i;
    for(i = 0; i < rows*cols; i++){
        ret.data[i] = boxMullerDraw();
    }
    return ret;
}

/*
 * Function: double OneIterGradientDescent
 * --------------------
 *  Performs one iteration of gradient descent for given matrices and learning rate
 *  Returns the cost function after that iteration
 * 
 *  X (pointer to Matrix): the Matrix with the data
 *  Theta (pointer to Matrix): the parameters Matrix
 *  Y (pointer to Matrix): the dependent variable
 *  residuals (pointer to Matrix): vector to store residuals of last iteration
 *  lr (double): the learning rate
*/
double oneIterGradientDescent(const Matrix *X, Matrix *Theta,const Matrix *Y, Matrix *residuals,double lr){
    // Multiply X and Theta, then subtract Y and then multiply that from the left with X transposed
    // Initialize intermediate results
    Matrix mult;
    mult.rows = X->rows;
    mult.cols = Theta->cols;
    mult.data = (double *)malloc(X->rows*Theta->cols*sizeof(double));
    multiplyMatrices(X, Theta, &mult);
    // Subtract to Y
    Matrix yPred;
    yPred.rows = Y->rows;
    yPred.cols = Y->cols;
    yPred.data = (double *)malloc(Y->rows*Y->cols*sizeof(double));
    sumMatrices(&mult, Y, true, &yPred);
    // Get loss
    double loss;
    // Initialize the difference of the values
    Matrix lossVec;
    lossVec.rows = Y->rows;
    lossVec.cols = Y->cols;
    lossVec.data = (double *)malloc(Y->rows*Y->cols*sizeof(double));
    sumMatrices(&yPred, Y, true, &lossVec);
    loss = norm(&lossVec);
    // Store residuals
    memcpy(residuals->data, lossVec.data, lossVec.rows * lossVec.cols * sizeof(double));
    // Free lossVec
    free(lossVec.data);
    // Free mult
    free(mult.data);
    // Multiply with X transposed
    Matrix xT;
    xT.rows = X->cols;
    xT.cols = X->rows;
    xT.data = (double *)malloc(xT.rows*xT.cols*sizeof(double));
    transposeMatrix(X, &xT);
    // Multiply x transposed and y predicted
    Matrix step;
    step.rows = Theta->rows;
    step.cols = Theta->cols;
    step.data = (double *)malloc(Theta->rows*Theta->cols*sizeof(double));
    multiplyMatrices(&xT, &yPred, &step);
    // Free memory from intermediate steps
    free(xT.data);
    free(yPred.data);
    // Multiply the step by the learning rate
    multiplyScalar(&step, lr);
    // Update theta
    sumMatrices(Theta, &step, true, Theta);
    // Free memory
    free(step.data);
    return loss;
}

/*
 * Function: double linearRegression
 * --------------------
 *  Performs linear Regression using gradient descent
 *  Returns the loss
 * 
 *  X (pointer to Matrix): the Matrix with the data
 *  Theta (pointer to Matrix): the parameters Matrix
 *  Y (pointer to Matrix): the dependent variable
 *  residuals (pointer to Matrix): vector to store residuals
 *  lr (double): the learning rate
 *  epochs (int): the maximum number of iterations to iterate for
 *  loss_tolerance (double): upper threshold on the loss that causes early stopping
 *  verbose (bool): prints the loss at each epoch
*/
double linearRegression(const Matrix *X, Matrix *Theta,const Matrix *Y, Matrix *residuals, double lr,int epochs, double loss_tolerance, bool verbose){
    double loss;
    int epoch;
    for(epoch = 0; epoch < epochs; epoch++){
        // Do an iteration
        loss = oneIterGradientDescent(X, Theta,Y,residuals,lr);
        // Print if needed
        if(verbose){
            printf("In iteration %d\n",epoch);
            printf("Current loss %f\n",loss);
        }
        // Check if early stop
        if (loss < loss_tolerance){
            break;
        }
    }
    return loss;

}

/*
 * Function: void regressionMetrics
 * --------------------
 *  Retrieves the confidence intervals and p values for the results of
 *  the linear regression
 * 
 *  X (pointer to Matrix): the independent variables matrix
 *  Theta (pointer to Matrix): the parameters Matrix
 *  residuals (pointer to Matrix): vector to store residuals
 *  loss (double): the loss of the linear regression
 *  lowerBounds (pointer to Matrix): lower bound of the confidence interval
 *  upperBounds (pointer to Matrix): upper bound of the confidence interval
 *  pValues (pointer to Matrix): p-values
*/
void regressionMetrics(const Matrix *X, const Matrix *Theta, const Matrix *residuals, double loss, Matrix *lowerBounds, Matrix *upperBounds, Matrix *pValues){
    // Get the residual variance
    double resVar = loss/(X->rows - X->cols);
    // Initialize variables to do the inverse
    Matrix inv;
    Matrix multX;
    multX.rows = X->cols;
    multX.cols = X->cols;
    multX.data = (double *)malloc(X->rows*X->cols*sizeof(double));
    Matrix xT;
    xT.rows = X->cols;
    xT.cols = X->rows;
    xT.data = (double *)malloc(xT.rows*xT.cols*sizeof(double));
    transposeMatrix(X, &xT);
    multiplyMatrices(&xT, X, &multX);
    inverseMatrix(&multX,&inv);
    // Get covariance matrix
    multiplyScalar(&inv, resVar);
    // Get lower and upper bounds of confidence intervals
    int c;
    // Get critical value
    double critVal = 1.96;
    for(c = 0; c < inv.cols; c++){
        double stdDev = sqrt(*(inv.data + c * inv.cols + c));
        double tStat = *(Theta->data + c)/stdDev;
        *(lowerBounds->data + c) = *(Theta->data + c)-critVal*stdDev;
        *(upperBounds->data + c) = *(Theta->data + c)+critVal*stdDev;
        *(pValues->data + c) = erfc(fabs(tStat)*sqrt(0.5));
    }
    // 
    // Free memory
    free(inv.data);
    free(multX.data);
    free(xT.data);
}


