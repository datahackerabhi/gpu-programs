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

__global__ void matrixMultiply(double *A, double *B, double *C, unsigned numColsA ){

	int row, col;
	int bx = blockIdx.x; 
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	col = bx*blockDim.x + tx;
	row = by*blockDim.y + ty;
	
	C[ row*gridDim.x + col ] = 0;
	for( int i = 0; i < numColsA; i++ ){
		C[ row*gridDim.x*blockDim.x + col ] += A[ row*numColsA + i ]*B[ i*gridDim.x*blockDim.x + col ];
	}

}

int main(){

	unsigned numRowsA = 4096;  
	unsigned numColsA = 8192;
	unsigned numRowsB  = numColsA;
	unsigned numColsB = 16384;

	cudaEvent_t start, stop;

	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	size_t mat_sizeA = sizeof(double)*numRowsA*numColsA;
	size_t mat_sizeB = sizeof(double)*numRowsB*numColsB;
	size_t mat_sizeC = sizeof(double)*numRowsA*numColsB;

	double *d_A, *d_B, *d_C;
	
	int num_threads = 16;

	double *h_A = (double *) malloc( mat_sizeA );
	double *h_B = (double *) malloc( mat_sizeB );
	double *h_C = (double *) malloc( mat_sizeC );
	
	fill_matrix( h_A, numRowsA, numColsA );
	fill_matrix( h_B, numRowsB, numColsB );

	myCudaCheck( cudaMalloc( &d_A, mat_sizeA ) );
	myCudaCheck( cudaMalloc( &d_B, mat_sizeB ) );
	myCudaCheck( cudaMalloc( &d_C, mat_sizeC ) );

	myCudaCheck( cudaMemcpy( d_A, h_A, mat_sizeA, cudaMemcpyHostToDevice ) );
	myCudaCheck( cudaMemcpy( d_B, h_B, mat_sizeB, cudaMemcpyHostToDevice ) );
	
		
	dim3 threads( num_threads, num_threads );
	dim3 blocks( numColsB/num_threads, numRowsA/num_threads );
	
	cudaEventRecord( start );
	matrixMultiply<<< blocks, threads >>>( d_A, d_B, d_C, numColsA );
	cudaEventRecord( stop );

	myCudaCheck( cudaMemcpy( h_C, d_C, mat_sizeC, cudaMemcpyDeviceToHost ) );
	
	cudaEventSynchronize( stop );
	float milliseconds = 0;
	cudaEventElapsedTime( &milliseconds, start, stop );
	
	printf( "%f\n", milliseconds );

	print_matrix_to_file( h_C, numRowsA, numColsB );
	return 0;
}
