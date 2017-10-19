#include <stdlib.h>
#include <iostream>

void matrixMulC(int* m, int* n, int* p, int width){
    int dst = 0;

    for(int i=0; i<width; i++){
        for(int j=0; j<width; j++){
            dst = i*width + j;
            for(int idx=0; idx<width; idx++){
                p[dst] += m[i*width+idx]*n[idx*width+j];
            }
        }
    }
}

__global__ void MatrixMul(int* m , int* n, int* r, int width){
    int tid, tx, ty;

    tx = blockDim.x * blockIdx.x + threadIdx.x; // 현재 스레드의 x축 인덱스
    ty = blockDim.y * blockIdx.y + threadIdx.y; // 현재 스레드의 y축 인덱스
    tid = width*ty + tx; // 결과매트릭스의 인덱스

    int value = 0;
    int mval = 0;
    int nval= 0;

    for(int i=0; i<width; i++){

        mval = m[ty * width + i];
        nval = n[i*width + tx];

        value += mval * nval;
    }

    r[tid] = value;

}

main(){
    
    const int matWidth = 12;
    const int matHeight = 12;
    const int matSize = matWidth * matHeight;
    const int memSize = matSize * sizeof(int);

    int* h_m;
    int* h_n;
    int* p_cuda; //매트릭스 곱 cuda 커널의 결과를 가르킬 포인터
    int* p_c; // 매트릭스 곱 c 함수의 결과를 가르킬 포인터

    //호스트 메모리 할당
    
    h_m = (int*)malloc(memSize);
    h_n = (int*)malloc(memSize);
    p_cuda = (int*)malloc(memSize);
    p_c = (int*)malloc(memSize);

    //호스트 배열 초기화
    for(int i=0; i<matSize; i++){
        h_m[i] = i;
        h_n[i] = i;
        p_cuda[i] = 0;
        p_c[i] = 0;
    }
    
    //cuda 커널 결과랑 비교할 c함수 호출
    matrixMulC(h_m, h_n, p_c, matWidth); 
    
    int* d_m;
    int* d_n;
    int* d_r;

    //디바이스 메모리 할당
    cudaMalloc((void**)&d_m, memSize);
    cudaMalloc((void**)&d_n, memSize);
    cudaMalloc((void**)&d_r, memSize);

    //호스트메모리 디바이스로 카피
    cudaMemcpy(d_m, h_m, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, memSize, cudaMemcpyHostToDevice);

    dim3 Dg(3, 4, 1);
    dim3 Db(4, 3, 1);

    //cuda 커널 호출
    MatrixMul<<<Dg, Db>>>(d_m, d_n, d_r, matWidth); 

    cudaMemcpy(p_cuda, d_r, memSize, cudaMemcpyDeviceToHost);
    
    bool flag = true;
    for(int i=0; i<matSize; i++){
        if(p_cuda[i] != p_c[i])
            flag = false;
    }

    std::cout << "matrix mul success " << flag << std::endl;

    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_r);

    free(h_n);
    free(h_m);
    free(p_c);
    free(p_cuda);

    return 0;
}
