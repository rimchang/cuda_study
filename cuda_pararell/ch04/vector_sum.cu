#include <stdlib.h>
#include <iostream>

__global__ void VectorAdd(int* a, int* b, int* c, int size){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}


int main()
{
    
    const int size = 512 * 65535; // max thread size per block 512 * max block size per grid 65535
    const int BufferSize = size * sizeof(int);

    int* h_inputA;
    int* h_inputB;
    int* h_result;

    h_inputA = (int*)malloc(BufferSize);
    h_inputB = (int*)malloc(BufferSize);
    h_result = (int*)malloc(BufferSize);
    

    for(int i=0; i < size; i++){
        h_inputA[i] = i;
        h_inputB[i] = i;
        h_result[i] = 0;
       
    }
    
    int* d_A;
    int* d_B;
    int* d_result;

    //GPU 디바이스에 메모리 할당
    cudaMalloc((void**)&d_A, size*sizeof(int));
    cudaMalloc((void**)&d_B, size*sizeof(int));
    cudaMalloc((void**)&d_result, size*sizeof(int));
    
    //호스트에서 디바이스로 입력 데이터 카피

    cudaMemcpy(d_A, h_inputA, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_inputB, size*sizeof(int), cudaMemcpyHostToDevice);

    //33,553,920개의 스레드를 생성하여 덧셈 계산
    VectorAdd<<<65535, 512>>>(d_A, d_B, d_result, size);

    //디바이스 에서 호스트로 결과 데이터 카피
    cudaMemcpy(h_result, d_result, size*sizeof(int), cudaMemcpyDeviceToHost);

    //결과 출력

    for(int i=0; i<5; i++){
        std::cout << "result "<< i << ": " << h_result[i] << std::endl;
    }

    for(int i=size-5; i<size; i++){
        std::cout << "result "<< i << ": " << h_result[i] << std::endl;
    }



   // 디바이스 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);

    free(h_inputA);
    free(h_inputB);
    free(h_result);
    
    return 0;
}
