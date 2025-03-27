#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_RESIDUAL 1e-15
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define T(i, j) T[(i) * (n_cells + 2) + (j)]
#define T_new(i, j) T_new[(i) * (n_cells + 2) + (j)]

void initialize_grid(double *T, int n_cells) {
    for (unsigned i = 0; i <= n_cells + 1; i++) {
        for (unsigned j = 0; j <= n_cells + 1; j++) {
            if ((j == 0) || (j == (n_cells + 1)))
                T(i, j) = 1.0;  // Boundary conditions
            else
                T(i, j) = 0.0;  // Interior grid
        }
    }
}

void jacobi_omp(double *T, int n_cells, int max_iterations) {
    double *T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));
    double residual = 1e6;
    int iteration = 0;

    while (residual > MAX_RESIDUAL && iteration < max_iterations) {
        residual = 0.0;

        // Parallelized Jacobi iteration
        #pragma omp parallel for collapse(2)
        for (unsigned i = 1; i <= n_cells; i++) {
            for (unsigned j = 1; j <= n_cells; j++) {
                T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
            }
        }

        // Compute residual and update T in parallel
        #pragma omp parallel for reduction(max: residual) collapse(2)
        for (unsigned i = 1; i <= n_cells; i++) {
            for (unsigned j = 1; j <= n_cells; j++) {
                residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
                T(i, j) = T_new(i, j);
            }
        }
        iteration++;
    }
    printf("%% Using number of cells = %d\n", n_cells);
    printf("%% Using maximum iteration count = %d\n", max_iterations);
    printf("%% Parallel Residual = %.9e\n", residual);
    free(T_new);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <grid_size> <max_iterations>\n", argv[0]);
        return 1;
    }

    int n_cells = atoi(argv[1]);
    int max_iterations = atoi(argv[2]);

    double *T = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));
    initialize_grid(T, n_cells);
    
    double start = omp_get_wtime();
    jacobi_omp(T, n_cells, max_iterations);
    double end = omp_get_wtime();
    
    printf("%% OpenMP CPU execution time: %f seconds\n", end - start);
    
    free(T);
    return 0;
}
