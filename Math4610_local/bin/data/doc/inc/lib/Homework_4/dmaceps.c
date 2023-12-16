#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void machineEps64() {
    double previous_eps = 0.0;
    double eps = 1.0;
    while ((1.0 + eps) != 1.0){
        previous_eps = eps;
        eps /= 2.0;
    }
    printf("%e\n", previous_eps);
}