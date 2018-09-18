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


__global__ void matrixMultiply1(double *A, double *B, double *C, unsigned N ){

	int row, col;
	row = blockIdx.x*blockDim.x + threadIdx.x;
	col = blockIdx.y*blockDim.y + threadIdx.y;
	
	C[ row*N + col ] = 0; 
	for( int i = 0; i < N; i++ ){
		C[ row*N + col ] += A[ row*N + i ]*B[ i*N + col ];
	}

}
__global__ void matrixMultiply2(double *A, double *B, double *C, unsigned N ){

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

	double *h_A = (double *) malloc( mat_size );
	double *h_B = (double *) malloc( mat_size );
	double *h_C = (double *) malloc( mat_size );
	
	dim3 threads( 16, 16 );
	dim3 blocks( N/16, N/16 );

	cudaEvent_t start1, stop1, start2, stop2;
	
	cudaEventCreate( &start1 );
	cudaEventCreate( &stop1 );
	cudaEventCreate( &start2 );
	cudaEventCreate( &stop2 );
	
	fill_matrix( h_A, N, N );
	fill_matrix( h_B, N, N );
	fill_matrix( h_C, N, N );

	myCudaCheck( cudaMalloc( &d_A, mat_size ) );
	myCudaCheck( cudaMalloc( &d_B, mat_size ) );
	myCudaCheck( cudaMalloc( &d_C, mat_size ) );

	myCudaCheck( cudaMemcpy( d_A, h_A, mat_size, cudaMemcpyHostToDevice ) );
	myCudaCheck( cudaMemcpy( d_B, h_B, mat_size, cudaMemcpyHostToDevice ) );

	cudaEventRecord( start1 );
	matrixMultiply1<<< blocks, threads >>>( d_A, d_B, d_C, N );
	cudaEventRecord( stop1 );
	
	myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
	print_matrix_to_file( h_C, N, N );
	
	cudaEventRecord( start2 );
	matrixMultiply2<<< blocks, threads >>>( d_A, d_B, d_C, N );
	cudaEventRecord( stop2 );

	myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
	
	cudaEventSynchronize( stop1 );
	cudaEventSynchronize( stop2 );
	
	print_matrix_to_file( h_C, N, N );
	
	float t1,t2;
	cudaEventElapsedTime( &t1, start1, stop1 );
	cudaEventElapsedTime( &t2, start2, stop2 );
	
	printf("%f\n", t1);
	printf("%f\n", t2);
	
	return 0;
}
