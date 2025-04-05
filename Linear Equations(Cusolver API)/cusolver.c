#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <time.h>


void generateHilbertMatrix(int N, double *H) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            H[i * N + j] = 1.0 / (i + j + 1);
        }
    }
}


void generateBVector(int N, double *b) {
    for (int i = 0; i < N; i++) {
        b[i] = 1.0;
    }
}


void perturbBVector(int N, double *b) {
    for (int i = 0; i < N; i++) {
        b[i] += ((double)rand() / RAND_MAX);
    }
}
int main() {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    for (int N = 2; N <= 8192; N *= 2) {
        int lda = N, ldb = N;
        double *H = (double*)malloc(N * N * sizeof(double));
        double *b = (double*)malloc(N * sizeof(double));
        double *x = (double*)malloc(N * sizeof(double));
        generateHilbertMatrix(N, H);
        generateBVector(N, b);

        double *d_H, *d_b;
        int *d_Ipiv, *d_info;
        cudaMalloc((void**)&d_H, sizeof(double) * N * N);
        cudaMalloc((void**)&d_b, sizeof(double) * N);
        cudaMalloc((void**)&d_Ipiv, sizeof(int) * N);
        cudaMalloc((void**)&d_info, sizeof(int));
        cudaMemcpy(d_H, H, sizeof(double) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(double) * N, cudaMemcpyHostToDevice);

        int lwork;
        double *d_work;
        cusolverDnDgetrf_bufferSize(cusolverH, N, N, d_H, lda, &lwork);
        cudaMalloc((void**)&d_work, sizeof(double) * lwork);

        // Start timing
        cudaEventRecord(start, stream);

        cusolverDnDgetrf(cusolverH, N, N, d_H, lda, d_work, d_Ipiv, d_info);
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_H, lda, d_Ipiv, d_b, ldb, d_info);

        // Stop timing after synchronizing
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        int info;
        cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (info < 0) {
            printf("N = %d: Parameter %d is invalid\n", N, -info);
            continue;
        }

        perturbBVector(N, b);
        cudaMemcpy(d_b, b, sizeof(double) * N, cudaMemcpyHostToDevice);

        // Start timing for perturbed solve
        cudaEventRecord(start, stream);

        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_H, lda, d_Ipiv, d_b, ldb, d_info);

        // Stop timing after synchronizing
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float milliseconds2 = 0;
        cudaEventElapsedTime(&milliseconds2, start, stop);

        printf("N = %d, LU factorization + solve time = %f ms, Solve with existing LU time = %f ms\n", N, milliseconds, milliseconds2);

        free(H); free(b); free(x);
        cudaFree(d_H); cudaFree(d_b); cudaFree(d_Ipiv); cudaFree(d_info); cudaFree(d_work);
    }

    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);
    cudaDeviceReset();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
