#include <stdio.h>
#include "homework_4.h"
#define N 10 //using this for a constant size

int main(void) {
    machineEps32();
    machineEps64();
    printf("%f\n", backward_difference(3, .000001));
    printf("%f\n", forward_difference(3, .000001));

    double u[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double v[] = {2.1, 4.0, 6.2, 8.0, 10.3};

    l_1_distance(u, v, sizeof(u)/sizeof(double));
    printf("\n");
    l_1_norm(v, sizeof(v)/sizeof(double));
    printf("\n");
    l_2_distance(u, v, sizeof(u)/sizeof(double));
    printf("\n");
    l_2_norm(u, sizeof(u)/sizeof(double));
    printf("\n");
    printf("%f\n", infinity_norm_distance(u, v, sizeof(v)/sizeof(double)));
    printf("%f\n", infinity_norm(u, sizeof(u)/sizeof(double)));
    
    double m, b;
    least_sqaures(sizeof(u)/sizeof(double), u, v, &m, &b);
    printf("Best fit line: y = %lf x + %lf\n", m, b);

    double mat[N][N + 1] = {
        {3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 2},
        {0, 1, 5, 1, 0, 0, 0, 0, 0, 0, 3},
        {0, 0, 1, 6, 1, 0, 0, 0, 0, 0, 4},
        {0, 0, 0, 1, 7, 1, 0, 0, 0, 0, 5},
        {0, 0, 0, 0, 1, 8, 1, 0, 0, 0, 6},
        {0, 0, 0, 0, 0, 1, 9, 1, 0, 0, 7},
        {0, 0, 0, 0, 0, 0, 1, 10, 1, 0, 8},
        {0, 0, 0, 0, 0, 0, 0, 1, 11, 1, 9},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 12, 10}
    };
    double x[N];

    upper_triangular(mat, sizeof(mat)/sizeof(double));
    back_substitution(mat, x, sizeof(mat)/sizeof(double));
    printf("Solution:\n");
    for (int i = 0; i < N; i++)
        printf("x%d = %f\n", i + 1, x[i]);

    return 0;
}