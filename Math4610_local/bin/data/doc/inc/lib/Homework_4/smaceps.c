#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void machineEps32() {
    float previous_eps = 0.0f;
    float eps = 1.0f;
    while ((1.0f + eps) != 1.0f){
        previous_eps = eps;
        eps /= 2.0f;
    }
    printf("%e\n", previous_eps);
}