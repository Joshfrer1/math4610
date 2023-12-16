#define N 10
void machineEps32();
void machineEps64();
double backward_difference(double x, double h);
double forward_difference(double x, double h);
double l_1_distance(double *u, double *v, int size);
double l_1_norm(double *vector, int size);
double l_2_distance(double *u, double *v, int size);
double l_2_norm(double *vector, int size);
double infinity_norm_distance(double *u, double *v, int size);
double infinity_norm(double *vector, int size);
void least_sqaures(int n, double *u, double *v, double* m, double* b);
void upper_triangular(double a[N][N + 1], int size);
void back_substitution(double a[N][N + 1], double x[N], int size);



