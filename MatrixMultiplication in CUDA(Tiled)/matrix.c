#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_WIDTH 16
#define NUM_RUNS 10  // Run multiple times for averaging

__global__ void matmul(float *A, float *B, float *C, int m, int p, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float value = 0.0;
        for (int k = 0; k < p; ++k) {
            value += A[row * p + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

__global__ void matTileMul(float *A, float *B, float *C, int m, int p, int n) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0;

    for (int t = 0; t < (p - 1) / TILE_WIDTH + 1; t++) {
        if (row < m && t * TILE_WIDTH + threadIdx.x < p)
            tileA[threadIdx.y][threadIdx.x] = A[row * p + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < n && t * TILE_WIDTH + threadIdx.y < p)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = value;
}

void MatInit(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 10);
    }
}

void matrixMultiplication(int m, int p, int n) {
    float *hA, *hB, *hC, *dA, *dB, *dC;
    size_t sizeA = m * p * sizeof(float);
    size_t sizeB = p * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    hA = (float *)malloc(sizeA);
    hB = (float *)malloc(sizeB);
    hC = (float *)malloc(sizeC);

    MatInit(hA, m * p);
    MatInit(hB, p * n);

    cudaMalloc((void **)&dA, sizeA);
    cudaMalloc((void **)&dB, sizeB);
    cudaMalloc((void **)&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    dim3 dimGrid((n - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Warm-up run
    matmul<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
    cudaDeviceSynchronize();

    // Measure Basic Kernel
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++)
        matmul<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Basic Kernel Execution Time: %.3f ms\n", milliseconds / NUM_RUNS);

    // Warm-up run
    matTileMul<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
    cudaDeviceSynchronize();

    // Measure Tiled Kernel
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++)
        matTileMul<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiled Kernel Execution Time: %.3f ms\n", milliseconds / NUM_RUNS);

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    free(hA);
    free(hB);
    free(hC);
}

int main() {
    int sizes[] = {32,64, 128, 256, 512, 1024, 2048, 4096,8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        printf("For N = %d\n", sizes[i]);
        matrixMultiplication(sizes[i], sizes[i], sizes[i]);
    }

    return 0;
}
