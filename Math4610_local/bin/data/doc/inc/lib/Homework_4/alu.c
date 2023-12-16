#include <stdio.h>

#define N 10

void upper_triangular(double a[N][N + 1]) {
    for (int k = 0; k < N - 1; k++) {
        for (int i = k + 1; i < N; i++) {
            double factor = a[i][k] / a[k][k];
            for (int j = k; j < N + 1; j++) {
                a[i][j] -= factor * a[k][j];
            }
        }
    }
}

void back_substitution(double a[N][N+1], double x[N]) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = a[i][N];
        for (int j = i + 1; j < N; j++) {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
}
