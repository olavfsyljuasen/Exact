#include<iostream>
#include<vector>
using namespace std;

#include "sparsematrix_new.h"


__global__ void cuda_mat_vec_multiply_double(int M, int Nc ,
					     int* indices , double* data , double* x, double* y) 
{
  int row = blockDim.x*blockIdx.x + threadIdx.x ;
  
  if ( row < M ){
    double dot = 0;
    for ( int n = 0; n < Nc ; n ++)
      {
	int col = indices [n*M + row];
	double val = data [n*M + row];
	if ( val != 0)
	  dot += val * x[col];
      }
    y[row] += dot ;
  }
}



int matrixtimesvector(vector<MyField>& v2,ELLmatrix& m,vector<MyField>& v1)
{    
  cudaError_t err = cudaSuccess;

  // the matrix:
  int M =m.NRows(); // # rows in the matrix
  int N =m.NCols(); // # cols in the matrix, the total not the actual stored #cols
  int Nc=m.Nc();    // # of stored columns in the matrix

  int Ntot=m.GetNelements(); // = M*Nc;


  MyField* m_val=m.GetValStartPtr();
  int*     m_col=m.GetColStartPtr();

  // allocate space on GPU
  MyField* d_m_val=NULL;
  if (cudaMalloc((void **)&d_m_val, Ntot* sizeof(m_val[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocatioNerror (allocate d_m_val)\n");
        return EXIT_FAILURE;
    }

  int* d_m_col=NULL;
  if (cudaMalloc((void **)&d_m_col, Ntot* sizeof(m_col[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocatioNerror (allocate d_m_col)\n");
        return EXIT_FAILURE;
    }

  // upload the matrix to the GPU
  err = cudaMemcpy(d_m_val, m_val, Ntot, cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy vector m_val from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err = cudaMemcpy(d_m_col, m_col, Ntot, cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy vector m_col from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }


  // also allocate the vectors

  
  if (cudaMalloc((void **)&d_v1, N* sizeof(v1[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocatioNerror (allocate d_v1)\n");
        return EXIT_FAILURE;
    }

  if (cudaMalloc((void **)&d_v2, N* sizeof(v2[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocatioNerror (allocate d_v2)\n");
        return EXIT_FAILURE;
    }

  // upload the input vector to the GPU
  err = cudaMemcpy(d_v1, &v1[0], N, cudaMemcpyHostToDevice);
  
  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy vector v1 from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  // Launch the CUDA Kernel
  // maxThreadsPerBlock is a variable
  // 

  int threadsPerBlock = 256; // Should experiment with this
    int blocksPerGrid =(M + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cuda_mat_vec_multiply_double<<<blocksPerGrid, threadsPerBlock>>>(M,Nc,d_m_col,d_m_val,d_v1,d_v2);
    err = cudaGetLastError();
    
    if (err != cudaSuccess)
      {
	fprintf(stderr, "Failed to launch cuda_mat_vec_multiply_double kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    
    
    // get the result from the GPU
    err = cudaMemcpy(&v2[0], d_v2, N, cudaMemcpyDeviceToHost);

    if(err != cudaSuccess )
      {
	fprintf(stderr, "Failed to copy vector v2 from device to host (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
      }

    //free up the memory on the GPU
    err = cudaFree(d_m_col);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_m_col (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_m_val);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_m_val (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_v1);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_v1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_v2);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_v2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    return 0;
}




int main()
{
  const int Rows=10;
  const int Cols=10;

  COOmatrix A(Rows,Cols);
  
  for(int r=0; r<Rows; r++)
    {
      A(r,0)=1.;
      A(r,1)=1.;
    }

  vector<MyField> vec(Cols);
  for(int i=0; i<Cols; i++) vec[i]=1.;

  ELLmatrix B(A);

  vector<MyField> vec2(Cols);

  matrixtimesvector(vec2,B,vec);

  cout << "vec2: " << endl;
  for(int i=0; i<Cols; i++) cout << vec[2] << " ";
  cout << endl;
}
