#include<stdio.h>
#include<cuda.h>

#define N 32 //8192
#define dx 16
#define dy 16

void myCudaCheck(cudaError_t err) {
        if(err != cudaSuccess) {
                printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
                exit(EXIT_FAILURE);
        }
}

void fill_matrix(double *mat, unsigned numRows, unsigned numCols)
{
	for(unsigned i=0; i < numRows; i++)
	for(unsigned j=0; j < numCols; j++)
	{
		mat[i*numCols + j] = i*2.1f + j*3.2f;
	}	
}

void print_matrix_to_file(double *mat, unsigned numRows, unsigned numCols)
{
	const char *fname = "assignment2_out";
	FILE *f = fopen(fname, "a");
	for(unsigned i=0; i < numRows; i++)	
	{
		for(unsigned j=0; j < numCols; j++)
			fprintf(f,"%4.4f ", mat[i*numCols + j]);
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void matrixMultiplySharedMem(double *A, double *B, double *C ){

	int row, col;
	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	__shared__ double s_A[ dy ][ N ];
	__shared__ double s_B[ N ][ dx ];

	for( int i = 0; i < N/dx; i++ ){
		s_A[ty][tx + i*dx] = A[ row*N + tx +  i*dx ];
		s_B[ty + i*dy][tx] = B[ (ty + i*dy)*N + col ];
	}
	__syncthreads();
	
	C[ row*N + col ] = 0;
	for( int i = 0; i < N; i++ ){
		C[ row*N + col ] += s_A[ ty ][ i ]*s_B[ i ][ tx ];
	}

}

int main(){

	size_t mat_size = sizeof(double)*N*N;
	
	double *d_A, *d_B, *d_C;

	double *h_A = (double *) malloc( mat_size );
	double *h_B = (double *) malloc( mat_size );
	double *h_C = (double *) malloc( mat_size );
	
	dim3 threads( 16, 16 );
	dim3 blocks( N/16, N/16 );

	cudaEvent_t start, stop;
	
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	
	fill_matrix( h_A, N, N );
	fill_matrix( h_B, N, N );
	fill_matrix( h_C, N, N );

	myCudaCheck( cudaMalloc( &d_A, mat_size ) );
	myCudaCheck( cudaMalloc( &d_B, mat_size ) );
	myCudaCheck( cudaMalloc( &d_C, mat_size ) );

	myCudaCheck( cudaMemcpy( d_A, h_A, mat_size, cudaMemcpyHostToDevice ) );
	myCudaCheck( cudaMemcpy( d_B, h_B, mat_size, cudaMemcpyHostToDevice ) );

	cudaEventRecord( start );
	matrixMultiplySharedMem<<< blocks, threads >>>( d_A, d_B, d_C );
	cudaEventRecord( stop );
	
	myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
	
	cudaEventSynchronize( stop );

	float milliseconds = 0;
	cudaEventElapsedTime( &milliseconds, start, stop );
	printf( "%f\n", milliseconds );

	print_matrix_to_file( h_C, N, N );
	return 0;
}
