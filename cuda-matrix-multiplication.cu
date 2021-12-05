#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 2

typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

__device__ Matrix getSubMatrix(Matrix X, int row, int col);
__device__ float getElement(const Matrix X, int row, int col);
__device__ void setElement(Matrix X, int row, int col, float value);

 __global__ void multipleMatrixesKernel(Matrix X, Matrix Y, Matrix Z) {

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix sZ = getSubMatrix(Z, blockRow, blockCol);

    float Zvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (X.width / BLOCK_SIZE); ++m) {

        Matrix sX = getSubMatrix(X, blockRow, m);

        Matrix sY = getSubMatrix(Y, m, blockCol);

        __shared__ float Xs[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Ys[BLOCK_SIZE][BLOCK_SIZE];

        Xs[row][col] = getElement(sX, row, col);
        Ys[row][col] = getElement(sY, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Zvalue += Xs[row][e] * Ys[e][col];

        __syncthreads();
    }
    setElement(sZ, row, col, Zvalue);
}

void multipleMatrixes(const Matrix X, const Matrix Y, Matrix Z) { 
    Matrix d_X;
    d_X.width = d_X.stride = X.width; 
    d_X.height = X.height;
    size_t size = X.width * X.height * sizeof(float);
    cudaMalloc(&d_X.elements, size);
    cudaMemcpy(d_X.elements, X.elements, size, cudaMemcpyHostToDevice);

    Matrix d_Y;
    d_Y.width = d_Y.stride = Y.width; 
    d_Y.height = Y.height;
    size = Y.width * Y.height * sizeof(float);
    cudaMalloc(&d_Y.elements, size);
    cudaMemcpy(d_Y.elements, Y.elements, size,
    cudaMemcpyHostToDevice);

    Matrix d_Z;
    d_Z.width = d_Z.stride = Z.width; 
    d_Z.height = Z.height;
    size = Z.width * Z.height * sizeof(float);
    cudaMalloc(&d_Z.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(Y.width / dimBlock.x, X.height / dimBlock.y);
    multipleMatrixesKernel<<<dimGrid, dimBlock>>>(d_X, d_Y, d_Z);

    cudaMemcpy(Z.elements, d_Z.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_X.elements);
    cudaFree(d_Y.elements);
    cudaFree(d_Z.elements);
}

__device__ Matrix getSubMatrix(Matrix X, int row, int col) {
    Matrix sX;
    sX.width = BLOCK_SIZE;
    sX.height = BLOCK_SIZE;
    sX.stride = X.stride;
    sX.elements = &X.elements[X.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return sX;
}

__device__ float getElement(const Matrix X, int row, int col) {
    return X.elements[row * X.stride + col];
}

__device__ void setElement(Matrix X, int row, int col, float value) {
    X.elements[row * X.stride + col] = value;
}

void fillMatrixRandomValues(float* matrix, int rows, int columns){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            matrix[i * columns + j] = rand() % 9 + 1;
        }
    }
}

void printMatrix(float* matrix, int rows, int columns){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            printf("%.2f ", matrix[i * columns + j]);
        }
        printf("\n");
    }
}

int main()
{
    int width = 4;
    int height = 4;
    int stride = 1;
    float *elementsX = (float *)malloc(width * height * sizeof(float));
    float *elementsY = (float *)malloc(width * height * sizeof(float));
    float *elementsZ = (float *)malloc(width * height * sizeof(float));

    fillMatrixRandomValues(elementsX, width, height);
    fillMatrixRandomValues(elementsY, width, height);

    Matrix X;
    X.width = width;
    X.height = height;
    X.stride = stride;
    X.elements = elementsX;

    Matrix Y;
    Y.width = width;
    Y.height = height;
    Y.stride = stride;
    Y.elements = elementsY;

    Matrix Z;
    Z.width = width;
    Z.height = height;
    Z.stride = stride;
    Z.elements = elementsZ;

    printMatrix(X.elements, width, height);
    printf("\n");
    printMatrix(Y.elements, width, height);
    printf("\n");

    multipleMatrixes(X, Y, Z);

    printMatrix(Z.elements, width, height);

    free(elementsX);
    free(elementsY);
    free(elementsZ);

    return 0;

}
