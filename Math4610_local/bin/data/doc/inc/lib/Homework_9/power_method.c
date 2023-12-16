#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 4  // Define the size of the matrix

void matrix_vector_multiply(double matrix[N][N], double vector[N], double result[N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        result[i] = 0.0;
        for (int j = 0; j < N; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

double dot_product(double vec1[N], double vec2[N]) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

double norm(double vector[N]) {
    return sqrt(dot_product(vector, vector));
}

void normalize(double vector[N]) {
    double vector_norm = norm(vector);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        vector[i] /= vector_norm;
    }
}

int main() {
    double A[N][N] = {
        {4, 1, 2, 3},
        {0, 3, 4, 5},
        {0, 0, 1, 2},
        {0, 0, 0, 1}
    };
    double x[N] = {1, 1, 1, 1};  // Initial guess
    double y[N];
    double eigenvalue = 0.0;
    double tolerance = 1e-6;
    double error = 1.0;
    int max_iterations = 1000;
    int iteration = 0;

    while (error > tolerance && iteration < max_iterations) {
        matrix_vector_multiply(A, x, y);
        double new_eigenvalue = norm(y);
        error = fabs(new_eigenvalue - eigenvalue);
        eigenvalue = new_eigenvalue;
        normalize(y);
        for (int i = 0; i < N; i++) {
            x[i] = y[i];
        }
        iteration++;
    }

    printf("Dominant Eigenvalue: %lf\n", eigenvalue);
    return 0;
}
