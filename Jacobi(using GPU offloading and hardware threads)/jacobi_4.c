#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i) * (n_cells + 2) + (j)])
#define T_new(i, j) (T_new[(i) * (n_cells + 2) + (j)])

int n_cells = 1000; // Grid size (N x N)
double MAX_RESIDUAL = 1.e-8;

// OpenMP GPU implementation (with vectorization)
void kernel_omp_gpu_vector(double *T, int max_iterations) {
    int iteration = 0;
    double residual = 1.e6;
    double *T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

    while (residual > MAX_RESIDUAL && iteration < max_iterations) {
        // Update grid using the average of neighbors
        #pragma omp target teams distribute parallel for simd collapse(2) map(tofrom: T[:(n_cells + 2) * (n_cells + 2)]) map(to: T_new[:(n_cells + 2) * (n_cells + 2)])
        for (unsigned i = 1; i <= n_cells; i++) {
            for (unsigned j = 1; j <= n_cells; j++) {
                T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
            }
        }

        residual = 0.0;

        // Update the residual and transfer the updated values to the original array
        #pragma omp target teams distribute parallel for simd reduction(max:residual) collapse(2) map(tofrom: T[:(n_cells + 2) * (n_cells + 2)]) map(to: T_new[:(n_cells + 2) * (n_cells + 2)])
        for (unsigned i = 1; i <= n_cells; i++) {
            for (unsigned j = 1; j <= n_cells; j++) {
                double diff = fabs(T_new(i, j) - T(i, j));
                residual = MAX(diff, residual);
                T(i, j) = T_new(i, j);

                // Debugging: print some values
                if (i == 1 && j == 1) { // Print for the first element
                    printf("T[%d][%d]: %.9f, T_new[%d][%d]: %.9f, Residual: %.9f\n", i, j, T(i, j), i, j, T_new(i, j), residual);
                }
            }
        }

        iteration++;
        // Print residual and iteration count for debugging
        printf("Iteration %d, Residual: %.9e\n", iteration, residual);
    }

    printf("Residual after %d iterations: %.9e\n", iteration, residual);
    free(T_new);
}

// Initialize grid and boundary conditions
void initialize_grid(double *T) {
    for (unsigned i = 0; i <= n_cells + 1; i++) {
        for (unsigned j = 0; j <= n_cells + 1; j++) {
            if ((j == 0) || (j == (n_cells + 1))) {
                T(i, j) = 1.0; // Boundary conditions
            } else {
                T(i, j) = 0.0; // Interior grid
            }
        }
    }
}

int main() {
    double *T = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

    // Loop over iterations from 10^0 to 10^6
    for (int exp = 0; exp <= 6; exp++) {
        int max_iterations = (int)pow(10, exp); // 10^exp iterations

        // Initialize grid
        initialize_grid(T);

        // OpenMP GPU (with vectorization) implementation
        double start_omp_gpu_vector = omp_get_wtime();
        kernel_omp_gpu_vector(T, max_iterations);
        double end_omp_gpu_vector = omp_get_wtime();
        printf("OpenMP GPU (With Vectorization) Execution Time for %d iterations: %.6f seconds\n", max_iterations, end_omp_gpu_vector - start_omp_gpu_vector);
    }

    free(T);
    return 0;
}
