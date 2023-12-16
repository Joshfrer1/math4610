#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>  // Include OpenMP header

#define matrix_dimension 25

int n = matrix_dimension;
float sum;

int main()
{
    float A[n][n];
    float ones[n];
    float x0[n];
    float b[n];
    //
    // create a matrix
    //
    srand((unsigned int)(time(NULL)));
    float a = 5.0;
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            A[i][j] = ((float)rand()/(float)(RAND_MAX) * a);
        }
        x0[i] = ((float)rand()/(float)(RAND_MAX) * a);
    }
    //
    // modify the diagonal entries for diagonal dominance
    // --------------------------------------------------
    //
    for(int i=0; i<n; i++)
    {
        sum = 0.0;
        for(int j=0; j<n; j++)
        {
            sum = sum + fabs(A[i][j]);
        }
        A[i][i] = A[i][i] + sum;
    }
    //
    // generate a vector of ones
    // -------------------------
    //
    for(int j=0; j<n; j++)
    {
        ones[j] = 1.0;
    }
    //
    // use the vector of ones to generate a right hand side for the testing
    // operation in the code
    // ---------------------
    // 
    for(int i=0; i<n; i++)
    {
        sum = 0.0;
        for(int j=0; j<n; j++)
        {
            sum = sum + A[i][j];
        }
        b[i] = sum;
    }

    //
    // Jacobi iteration test
    // ---------------------
    //
    float tol = 0.0001;
    float error = 10.0 * tol;
    float x1[n];
    float res[n];
    int maxiter = 100;
    int iter = 0;

    // Parallelize this loop
    #pragma omp parallel for
    for(int i=0; i<n; i++)
    {
        sum = b[i];
        for(int j=0; j<n; j++)
        {
            sum = sum - A[i][j] * x0[i];
        }
        res[i] = sum;
    } 
    //
    // loop starts here for Jacobi
    // ---------------------------
    //
    while (error > tol && iter < maxiter) 
    {
        // Parallelize this loop
        #pragma omp parallel for
        for(int i=0; i<n; i++)
        {
            x1[i] = x0[i] + res[i] / A[i][i];
        }
        //
        // compute the error
        // -----------------
        //
        sum = 0.0;
        for(int i=0; i<n; i++)
        {
            float val = x1[i] - x0[i];
            sum = sum + val * val;
        }
        error = sqrt(sum);
        //
        // reset the input for the next loop through
        // -----------------------------------------
        //
        for(int i=0; i<n; i++)
        {
            x0[i] = x1[i];
        } 
        //
        // compute the next residual
        // -------------------------
        //
        // Parallelize this loop
        #pragma omp parallel for
        for(int i=0; i<n; i++)
        {
            sum = b[i];
            for(int j=0; j<n; j++)
            {
                sum = sum - A[i][j] * x0[j];
            }
            res[i] = sum;
        }
        //
        // update the iteration counter
        // ----------------------------
        //
        iter++;
        //
        // end of loop
        // -----------
    }

    for(int i=0; i<n; i++)
        printf("x[%d] = %6f \t res[%d] = %6f\n", i, x1[i], i, res[i]);

    return 0;
}
