#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 16
using namespace std;

void matrixMulC(int* m, int* n, int* r, int mheight, int mwidth, int nheight, int nwidth){

    int rheight = mheight;
    int rwidth = nwidth;
    int dest = 0;
    for(int rrow=0; rrow<rheight; rrow++){
        for(int rcol=0; rcol<rwidth; rcol++){
            dest = rrow * rwidth + rcol;
            
            for(int idx=0; idx<mwidth; idx++){
                r[dest] += m[rrow*mwidth + idx] * n[idx * nwidth + rcol];
            }
        }
    }
}


__global__ void MatrixMul(int* m, int* n, int* r, int mheight, int mwidth, int nheight, int nwidth){
    
    int CValue = 0;
	
	// 실제.. thread의 인덱스는 아니다.
    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    
    // matrix m의 ceil(cols/BLOCK_SIZE) 만큼 증가시킴
    for (int k=0; k < (BLOCK_SIZE + mwidth -1 )/BLOCK_SIZE; k++){
        
        for (int i=0; i<BLOCK_SIZE; i++){

            // 현재 thread의 ROw가 매트릭스 m의 height를 넘지 않고 넉넉히 잡은 그리드의 col이 매트릭스 m의 width를 넘지 않고
            // 동시에 현재의 thread의 Col이 매트릭스 n의 width를 넘지 않고 넉넉히 잡은 그리드의 row가 매트릭스 n의 height를 넘지 않으면
            // 값을 할당!! 즉 k*BLOCK_SIZE+n 이 넉넉히 잡아준 ROW, COL이다. 
            if((k*BLOCK_SIZE + i < mwidth && Row < mheight) && (k*BLOCK_SIZE + i < nheight && Col < nwidth)){
                CValue += m[Row*mwidth + k*BLOCK_SIZE + i] * n[(k*BLOCK_SIZE + i)*nwidth + Col];
            }
        }
		
		// 현재 thread의 ROW, COL이 모두 
        if (Row < mheight && Col < nwidth){
			
			r[((blockIdx.y * blockDim.y + threadIdx.y)*nwidth)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
		} 
	}
}

main(int argc, char* argv[]){
    
    int mwidth, mheight, nwidth, nheight;
    
    // parsing args

    if (argc == 1) {
        cout << "no args , two matrix width, height assign 12" << endl;
        mwidth = 12;
        mheight = 12;
        nwidth = 12;
        nheight = 12;

    } else {
        
        mheight = atoi(argv[1]);
        mwidth = atoi(argv[2]);
        nheight = atoi(argv[3]);
        nwidth = atoi(argv[4]);

        if (mwidth != nheight)
            cout << "not matched matrix dim" << endl;


    }
    
    const int mSize = mheight * mwidth;
    const int nSize = nheight * nwidth;
    const int rSize = mheight * nwidth;
    const int mMemSize = mSize * sizeof(int);
    const int nMemSize = nSize * sizeof(int);
    const int rMemSize = rSize * sizeof(int); 
    // 호스트 메모리 할당

    int* h_m;
    int* h_n;
    int* h_cuda_r;
    int* h_c_r;

    h_m = (int*)malloc(mMemSize);
    h_n = (int*)malloc(nMemSize);
    h_cuda_r = (int*)malloc(rMemSize);
    h_c_r = (int*)malloc(rMemSize);



    // 호스트 매트릭스 초기화
    for(int i=0; i<mSize; i++){
        h_m[i] = i;
    }

    for(int i=0; i<nSize; i++){
        h_n[i] = i;
    }

    for(int i=0; i<rSize; i++){
        h_cuda_r[i] = 0;
        h_c_r[i] = 0;
    }

    // 디바이스 메모리 할당
    int* d_m;
    int* d_n;
    int* d_r;

    cudaMalloc((void**)&d_m,mMemSize);
    cudaMalloc((void**)&d_n,nMemSize);
    cudaMalloc((void**)&d_r,rMemSize);
    
    // 호스트에서 디바이스로 메모리 카피
    cudaMemcpy(d_m, h_m, mMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, nMemSize, cudaMemcpyHostToDevice);
    
    
    // gt 1050-ti 에서는 block이 6의배수여야 좋다.
    // gt 1050-ti 에서는 최대 3072 스레드 생성가능
   
    //dim3 dg(3, 4, 1); //grid가 안맞으면.. 쓰레기값이 나온다
    //dim3 db(4, 3, 1); //block도 안맞으면.. 쓰레기값이 나온다 


    dim3 db(BLOCK_SIZE, BLOCK_SIZE);
   
    // tile이 전체 매트릭스를 덮는것을 보장하기 위해 좀더 크게 잡아
    // int/int 하면 floor되는데.. 이를 ceiling 해주기 위한것이네 그냥.
    int dg_x = (nwidth + db.x -1) / db.x;
    int dg_y = (mheight +db.y - 1) / db.y;
    dim3 dg(dg_x, dg_y);


    MatrixMul<<<dg, db>>>(d_m, d_n, d_r, mheight, mwidth, nheight, nwidth);

    cudaMemcpy(h_cuda_r, d_r, rMemSize, cudaMemcpyDeviceToHost);
    
   
    // C 함수 호출
    matrixMulC(h_m, h_n, h_c_r, mheight, mwidth, nheight, nwidth);

    bool flag = true;
    for(int i=0; i<rSize; i++){
        if(h_cuda_r[i] != h_c_r[i]){
            flag = false;
        }
    }
    cout << "success matmul : " << flag << endl;
    
    for(int row=0; row<mheight; row++){
        for(int col=0; col<nwidth; col++){
            cout << h_c_r[row*nwidth + col] << " ";
        }

        cout << endl;
    }
    
    for(int row=0; row<mheight; row++){
        for(int col=0; col<nwidth; col++){
            cout << h_cuda_r[row*nwidth + col] << " ";
        }

        cout << endl;
    }
    
    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_r);
   
    free(h_m);
    free(h_n);
    free(h_cuda_r);
    free(h_c_r);
    
    
    return 0;
}
