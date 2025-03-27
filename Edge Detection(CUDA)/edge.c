#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define TILE_WIDTH 16

int deviceMask[5][5]={
    {2, 2, 4, 2, 2},
    {1, 1, 2, 1, 1},
    {0, 0, 0, 0, 0},
    {-1, -1, -2, -1, -1},
    {-2, -2, -4, -2, -2}
};

__global__ void convolution(unsigned int *input, int *output,int channels,int width,int height){

  int col = blockDim.x*blockIdx.x+threadIdx.x;
  int row = blockDim.y*blockIdx.y+threadIdx.y;
  
  if(col<width && row<height){
  
  for(int c=0;c<channels;c++){
  int sum =0;
  for(int i=-2;i<=2;i++){
    for(int j=-2;j<=2;j++){
      int curRow = row+i;
      int curCol = col+j;
      
      if(curRow>=0 && curRow<height && curCol>=0 && curCol<width){
        int index = (curRow *width +curCol)*channels +c;
        sum+=input[index]*deviceMask[i+2][j+2];
        }
        }
        }
        int outputIndex = (row*width+col) *channels +c;
        output[outputIndex] = sum;
        }
        }
        }
        
int main(int argc, char *argv[]) {
    unsigned int *hostInputImage;
    int *hostOutputImage;
    unsigned int inputLength = 384 * 512 * 3; // 384 * 512 * 3 = 589824 for RGB
    printf("Importing image data and allocating memory on host...\n");

 
    hostInputImage = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
    hostOutputImage = (int *)malloc(inputLength * sizeof(int));

  
    FILE *f = fopen("peppers.dat", "r");
    unsigned int pixelValue, i = 0;
    while (!feof(f) && i < inputLength) {
        fscanf(f, "%d", &pixelValue);
        hostInputImage[i++] = pixelValue;
    }
    fclose(f);

    int maskRows = 5, maskColumns = 5;
    int imageChannels = 3, imageWidth = 512, imageHeight = 384;

    unsigned int *deviceInputImage;
    int *deviceOutputImage;

    cudaMalloc((void **)&deviceInputImage, inputLength * sizeof(unsigned int));
    cudaMalloc((void **)&deviceOutputImage, inputLength * sizeof(int));

  
    cudaMemcpy(deviceInputImage, hostInputImage, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 dimGrid((imageWidth + TILE_WIDTH - 1) / TILE_WIDTH, (imageHeight + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    convolution<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, imageChannels, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    cudaMemcpy(hostOutputImage, deviceOutputImage, inputLength * sizeof(int), cudaMemcpyDeviceToHost);

    f = fopen("peppers.out", "w");
    for (int i = 0; i < inputLength; ++i) {
        fprintf(f, "%d\n", hostOutputImage[i]);
    }
    fclose(f);

    // Free memory
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
    free(hostInputImage);
    free(hostOutputImage);

    printf("Convolution completed successfully!\n");
    return 0;
}