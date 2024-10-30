#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NaiveMatrices.h"

/*
 * Function: (int) countColumns
 * ------------------------------------
 *  Counts the number of columns in a loaded .csv file
 *  Does this by counting the number of tokens in the first line
 *
 *  line (char pointer): a pointer to a line in the .csv file
 *                       it should be the first line
 * 
 *  delim (char): a string delimiter for separating the values in a line
 */
int countColumns(char *line, char *delim) {
    // Initialize counter
    int count = 0;
    // Split the line by the delimiter
    char *token = strtok(line, delim);
    // Keep splitting until you hit the end of the line
    while (token != NULL) {
        count++;
        token = strtok(NULL, delim);
    }
    return count;
}

/*
 * Function: (int) loadCSV
 * ------------------------------------
 *  Loads a CSV file into a matrix
 *
 *  filename (char): the filename of the file that is loaded
 *  headers (char): an array of strings with the names of the columns
 *  matrix (Matrix): a matrix struct that holds the loaded data
 */
int loadCSV(const char *filename, char ***headers, Matrix *matrix, char *delim) {
    // Open file
    FILE *file = fopen(filename, "r");
    // Check opening worked, if not return an error code
    if (file == NULL) {
        printf("Error opening file!\n");
        return 0;
    }
    // Initialize a char for the first memory of the file
    char line[1024];
    // Initialize matrix dimensions
    int row = 0;
    int col = 0;

    // Read header
    // fgets gets next character from a string and advances position indicator
    // for that string
    // Order of argumes: string that is reading, how much more to move the position indicator, which stream of text are we reading
    if (fgets(line, sizeof(line), file)) {
        // Count number of columns
        char tempLine[1024];
        // Copy line to count columns
        strcpy(tempLine, line);  
        // Get number of columns
        int columns = countColumns(tempLine,delim);

        // Allocate memory for headers
        *headers = (char **)malloc(columns * sizeof(char *));
        if (*headers == NULL) {
            printf("Memory allocation failed for headers!\n");
            fclose(file);
            return 0;
        }

        // Parse and store column headers
        char *token = strtok(line, delim);
        for (col = 0; col < columns; col++) {
            (*headers)[col] = strdup(token);  // Duplicate the string for each header
            token = strtok(NULL, delim);
        }

        // Initialize matrix dimensions
        matrix->cols = columns;
        // Rows and data are allocated dynamically later
        matrix->rows = 0; 
        matrix->data = NULL; 
    }

    // Read numerical data
    while (fgets(line, sizeof(line), file)) {
        // Copy line for checking missing value
        char *tempLine = strdup(line);
        char *token = strtok(tempLine, delim);
        // Raise a flag for whether the row is valid or not
        int validRow = 1;
        
        // Check for missing values
        for (int i = 0; i < matrix->cols; i++) {
            if (token == NULL || strcmp(token, "") == 0) {
                validRow = 0;
                break;
            }
            token = strtok(NULL, delim);
        }

        free(tempLine); // Free the copy of the line after checking

        // If row has no missing values, load it into matrix
        if (validRow) {
            matrix->rows++;
            matrix->data = (double *)realloc(matrix->data, matrix->rows * matrix->cols * sizeof(double));
            if (matrix->data == NULL) {
                printf("Memory allocation failed for matrix data!\n");
                fclose(file);
                return 0;
            }

            token = strtok(line, delim);
            for (col = 0; col < matrix->cols; col++) {
                matrix->data[(matrix->rows - 1) * matrix->cols + col] = atof(token);
                token = strtok(NULL, delim);
            }
        }
    }

    fclose(file);
    return 1;  // Success
}

/*
 * Function: (void) printHeaders
 * ------------------------------------
 *  Prints the headers that are contained in a list of headers
 *  Use mostly for debugging and reference
 *
 *  **headers (char): the list of headers
 *  cols (int): the number of columns that the headers represent
 */
void printHeaders(char **headers, int cols) {
    for (int i = 0; i < cols; i++) {
        printf("%s ", headers[i]);
    }
    printf("\n");
}

/*
 * Function: (int) getColumnIndex
 * ------------------------------------
 *  Retrieves the column index from the column name
 *
 *  headers (char): list with the headers
 *  numCols (int): number of Columns
 *  colName (char): a matrix struct that holds the loaded data
 */
int getColumnIndex(char **headers, int numCols, const char *colName) {
    for (int i = 0; i < numCols; i++) {
        if (strcmp(headers[i], colName) == 0) {
            return i;
        }
    }
    return -1; // Return -1 if column name is not found
}

/*
 * Function: (int) createDependentAndIndependentMatrices
 * ------------------------------------
 *  Formats the input data into matrices for use in linear regression
 *  Receives user input to check which is the dependent variable and which
 *  are the independent variables
 * 
 *  headers (char): names of the variables
 *  matrix (Matrix): a pointer to the matrix with the loaded raw data from the file
 *  X (Matrix): a pointer to matrix to store independent variables
 *  Y (Matrix): a pointer to matrix to store dependent variable
 *  selectedIndices (int): a list of integers that stores which indices are selected as independent variables
 */
int createDependentAndIndependentMatrices(char **headers, Matrix *matrix, Matrix *Y, Matrix *X, int **selectedIndices) {
    int depVarIndex;
    int numIndepVars = 0;
    int *indepVarIndices = NULL;

    // Allocate memory for selectedIndices
    *selectedIndices = NULL;
    // Get dependent variable from user
    char depVar[100];
    printf("Enter the name of the dependent variable: ");
    scanf("%99s", depVar);
    depVarIndex = getColumnIndex(headers, matrix->cols, depVar);
    if (depVarIndex == -1) {
        printf("Dependent variable not found in headers.\n");
        return 1;
    }

    // Get independent variables from user
    printf("Enter the names of independent variables, one at a time.\n");
    printf("Type 'endIndep' to stop, or 'all' to include all variables except the dependent variable.\n");

    while (1) {
        char indepVar[100];
        printf("Independent variable %d: ", numIndepVars + 1);
        scanf("%99s", indepVar);

        // Check if the user wants to include all variables
        if (strcmp(indepVar, "all") == 0) {
            indepVarIndices = (int *)malloc((matrix->cols - 1) * sizeof(int));
            if (indepVarIndices == NULL) {
                printf("Memory allocation failed!\n");
                return 1;
            }

            for (int i = 0; i < matrix->cols; i++) {
                if (i != depVarIndex) {
                    indepVarIndices[numIndepVars++] = i;
                }
            }
            break;
        }

        // Check for end condition
        if (strcmp(indepVar, "endIndep") == 0) {
            break;
        }

        // Keep iterating if variable not found
        int indepVarIndex = getColumnIndex(headers, matrix->cols, indepVar);
        if (indepVarIndex == -1) {
            printf("Independent variable '%s' not found in headers.\n", indepVar);
            continue;
        }

        // Stop iterating if ran out of memory
        indepVarIndices = (int *)realloc(indepVarIndices, (numIndepVars + 1) * sizeof(int));
        if (indepVarIndices == NULL) {
            printf("Memory allocation failed!\n");
            return 1;
        }
        indepVarIndices[numIndepVars++] = indepVarIndex;
    }
    // Store the selected indices into selectedIndices
    *selectedIndices = (int *)malloc(numIndepVars * sizeof(int));
    memcpy(*selectedIndices, indepVarIndices, numIndepVars * sizeof(int)); // Copy selected indices
    // Continue with setting Y, X as before

    // Allocate and populate Y and X matrices
    Y->rows = matrix->rows;
    Y->cols = 1;
    Y->data = malloc(matrix->rows * sizeof(double));

    X->rows = matrix->rows;
    X->cols = numIndepVars;
    X->data = malloc(matrix->rows * numIndepVars * sizeof(double));
    // Consider case of failure
    if (Y->data == NULL || X->data == NULL) {
        printf("Memory allocation failed for Y or X matrix.\n");
        free(Y->data);
        free(X->data);
        free(indepVarIndices);
        return 1;
    }

    // Populate Y
    for (int r = 0; r < matrix->rows; r++) {
        Y->data[r] = matrix->data[r * matrix->cols + depVarIndex];
    }

    // Populate X
    for (int r = 0; r < matrix->rows; r++) {
        for (int c = 0; c < numIndepVars; c++) {
            X->data[r * numIndepVars + c] = matrix->data[r * matrix->cols + indepVarIndices[c]];
        }
    }
    // Free memory and end
    free(indepVarIndices);
    return 0;
}

// Main function
int main() {
    // Initialize 
    const char *filename = "dataMissing.csv";
    char **headers = NULL;
    Matrix matrix = {0, 0, NULL};
    // Consider failed load case
    if (!loadCSV(filename, &headers, &matrix, ",")) {
        printf("Failed to load CSV.\n");
        return 1;
    }

    // Show the headers for easy reference for the user
    printf("Variables in file:\n");
    printf("\n");
    printHeaders(headers, matrix.cols);

    int iter;
    double lr;
    // Receive number of iterations and learning rates
    printf("Please input the number of iterations\n");
    scanf("%d", &iter);
    printf("Please input the learning rate\n");
    scanf("%lf", &lr);

    Matrix Y = {0, 0, NULL};
    Matrix X = {0, 0, NULL};

    // Initialize array to store the indices of the independent variables
    int *selectedIndices = NULL;
    int independent_var_count = createDependentAndIndependentMatrices(headers, &matrix, &Y, &X, &selectedIndices);
    // Generate output that the data was formatted properly
    if (independent_var_count > 0) {
        printf("Data formatting successful!\n");
    }
    // Initialize parameters and residuals
    Matrix Theta = initMatrixRandomNorm(X.cols, 1);
    Matrix residuals = {Y.rows, Y.cols, (double *)malloc(Y.rows * Y.cols * sizeof(double))};
    standardizeMatrix(&X);
    standardizeMatrix(&Y);
    // Execute linear regression
    double loss = linearRegression(&X, &Theta, &Y, &residuals, lr, iter, 0.0001, true);
    printf("Linear Regression was successful. Computing relevant metrics\n");

    Matrix ICLow = {X.cols, 1, (double *)malloc(X.cols * sizeof(double))};
    Matrix ICUp = {X.cols, 1, (double *)malloc(X.cols * sizeof(double))};
    Matrix PVals = {X.cols, 1, (double *)malloc(X.cols * sizeof(double))};
    // Retrieve the regression metrics
    regressionMetrics(&X, &Theta, &residuals, loss, &ICLow, &ICUp, &PVals);

    printf("Independent Variable\tParameter Value\tConfidence Interval\tp-value\n");

    for (int i = 0; i < X.cols; i++) {
        int varIndex = selectedIndices[i]; 
        printf("%-22s | %-16.4f | (%-6.4f, %-6.4f) | %-8.4f\n", headers[varIndex], Theta.data[i],
               ICLow.data[i], ICUp.data[i], PVals.data[i]);
    }

    // Free used memory
    for (int i = 0; i < matrix.cols; i++) {
        free(headers[i]);
    }
    free(headers);
    free(matrix.data);
    free(Y.data);
    free(X.data);
    free(Theta.data);
    free(residuals.data);
    free(ICLow.data);
    free(ICUp.data);
    free(PVals.data);
    free(selectedIndices);
    return 0;
}
