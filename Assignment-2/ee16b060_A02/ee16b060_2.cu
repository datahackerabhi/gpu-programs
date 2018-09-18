#include<stdio.h>
#include<cuda.h>

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

__global__ void matrixMultiply(double *A, double *B, double *C, unsigned N ){

	int row, col;
	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
	
	C[ row*N + col ] = 0;
	for( int i = 0; i < N; i++ ){
		C[ row*N + col ] += A[ row*N + i ]*B[ i*N + col ];
	}

}

int main(){

	unsigned N = 8192;  //size of matrix
	size_t mat_size = sizeof(double)*N*N;
	
	double *d_A, *d_B, *d_C;
	
	int num_threadsx = 1;
	int num_threadsy = 1;

	double *h_A = (double *) malloc( mat_size );
	double *h_B = (double *) malloc( mat_size );
	double *h_C = (double *) malloc( mat_size );
	
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
	
	for( num_threadsy = 1; num_threadsy < 64 ; num_threadsy = num_threadsy*2 )
	for( num_threadsx = 1; num_threadsx < 64 ; num_threadsx = num_threadsx*2 ){
		
		dim3 threads( num_threadsx, num_threadsy );
		dim3 blocks( N/num_threadsx, N/num_threadsy );

		cudaEventRecord( start );
		matrixMultiply<<< blocks, threads >>>( d_A, d_B, d_C, N );
		cudaEventRecord( stop );

		cudaEventSynchronize( stop );
	
		float milliseconds = 0;
		cudaEventElapsedTime( &milliseconds, start, stop );
		printf( "%d %d %f\n", num_threadsx, num_threadsy,  milliseconds );
		
		myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
		//print_matrix_to_file( h_C, N, N );

	}

	print_matrix_to_file( h_C, N, N );
	return 0;
}
