#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define BLOCK_SIZE 512

__global__ void reduction(float *input, float *output, int len){
  __shared__ float partialSum[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;

    if (start + t < len)
    partialSum[t] = input[start + t];
  else
    partialSum[t] = 0;

  if (start + BLOCK_SIZE + t < len)
    partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
  else
    partialSum[BLOCK_SIZE + t] = 0;

   for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1){
    __syncthreads();
    if (t < stride)
      partialSum[t] += partialSum[t + stride];
  }

    if (t == 0)
    output[blockIdx.x] = partialSum[0];
}

double sumation(float *arr, int N){
  double sum = 0.0;
  for (int i = 0; i < N; i++)  
    sum += arr[i];
  return sum;
}

int main(){
  for (int power = 1; power <= 30; power++) {
    int N = 1 << power;

    int numOutput = (N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);  

    float *hostInput, *hostOutput;
    float *deviceInput, *deviceOutput;

    hostInput = (float *)malloc(N * sizeof(float));
    hostOutput = (float *)malloc(numOutput * sizeof(float));


    for (int i = 0; i < N; i++) {
      hostInput[i] = i + 1;
    }


    cudaMalloc(&deviceInput, sizeof(float) * N);
    cudaMalloc(&deviceOutput, sizeof(float) * numOutput);
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * N, cudaMemcpyHostToDevice);


    clock_t t = clock();
    float cpu_result = sumation(hostInput, N);
    t = clock() - t;
    double cpu_time = ((double)t) / CLOCKS_PER_SEC * 1000;
printf("For N =%d\n",N);
    printf("CPU Result: %f\n", cpu_result);
    printf("CPU elapsed time: %.2e ms\n", cpu_time);

    cudaEvent_t start, stop;
    float gpu_time;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    dim3 dimGrid(numOutput, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    reduction<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, N);
    cudaDeviceSynchronize();

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU elapsed time: %.2e ms\n", gpu_time);

    
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * numOutput, cudaMemcpyDeviceToHost);

    
    float gpu_result = sumation(hostOutput, numOutput);
    printf("GPU Result: %f\n", gpu_result);

   
    double speedup = cpu_time / gpu_time;
    printf("Speedup (CPU Time / GPU Time): %f\n", speedup);

    // Free memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    free(hostInput);
    free(hostOutput);
  }
  return 0;
}
