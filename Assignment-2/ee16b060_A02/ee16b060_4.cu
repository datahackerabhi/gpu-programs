#include<stdio.h>
#include<cuda.h>

#define N 32 
#define dx 16
#define dy 16
#define TILE_WIDTH 16

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

__global__ void matrixMultiplyTiling(double *A, double *B, double *C ){

	int row, col;
	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	double Pvalue = 0;

	__shared__ double ts_A[ TILE_WIDTH ][ TILE_WIDTH ];
	__shared__ double ts_B[ TILE_WIDTH ][ TILE_WIDTH ];

	C[ row*N + col ] = 0;

	for( int p = 0; p < N/TILE_WIDTH; p++ ){

		ts_A[ty][tx] = A[ row*N + tx +  p*TILE_WIDTH ];
		ts_B[ty][tx] = B[ (ty + p*TILE_WIDTH)*N + col ];
		__syncthreads();
	
		for( int i = 0; i < TILE_WIDTH; i++ ){
			Pvalue += ts_A[ ty ][ i ]*ts_B[ i ][ tx ];
		}
		__syncthreads();

	}

	C[ row*N + col ] = Pvalue;
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
	matrixMultiplyTiling<<< blocks, threads >>>( d_A, d_B, d_C );
	cudaEventRecord( stop );
	
	myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
	
	cudaEventSynchronize( stop );
	
	float milliseconds = 0;
	cudaEventElapsedTime( &milliseconds, start, stop );
	printf("%f\n", milliseconds );

	print_matrix_to_file( h_C, N, N );
	return 0;
}
