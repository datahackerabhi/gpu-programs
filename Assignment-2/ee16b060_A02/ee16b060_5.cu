#include<stdio.h>
#include<cuda.h>

#define N 8192 
#define TILE_WIDTH1 4
#define TILE_WIDTH2 8
#define TILE_WIDTH3 16
#define TILE_WIDTH4 32

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

__global__ void matrixMultiplyTiling1(double *A, double *B, double *C ){

	int row, col;
	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	double Pvalue = 0;

	__shared__ double ts_A[ TILE_WIDTH1 ][ TILE_WIDTH1 ];
	__shared__ double ts_B[ TILE_WIDTH1 ][ TILE_WIDTH1 ];

	C[ row*N + col ] = 0;

	for( int p = 0; p < N/TILE_WIDTH1; p++ ){

		ts_A[ty][tx] = A[ row*N + tx +  p*TILE_WIDTH1 ];
		ts_B[ty][tx] = B[ (ty + p*TILE_WIDTH1)*N + col ];
		__syncthreads();
	
		for( int i = 0; i < TILE_WIDTH1; i++ ){
			Pvalue += ts_A[ ty ][ i ]*ts_B[ i ][ tx ];
		}
		__syncthreads();

	}

	C[ row*N + col ] = Pvalue;
}

__global__ void matrixMultiplyTiling2(double *A, double *B, double *C ){

	int row, col;
	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	double Pvalue = 0;

	__shared__ double ts_A[ TILE_WIDTH2 ][ TILE_WIDTH2 ];
	__shared__ double ts_B[ TILE_WIDTH2 ][ TILE_WIDTH2 ];

	C[ row*N + col ] = 0;

	for( int p = 0; p < N/TILE_WIDTH2; p++ ){

		ts_A[ty][tx] = A[ row*N + tx +  p*TILE_WIDTH2 ];
		ts_B[ty][tx] = B[ (ty + p*TILE_WIDTH2)*N + col ];
		__syncthreads();
	
		for( int i = 0; i < TILE_WIDTH2; i++ ){
			Pvalue += ts_A[ ty ][ i ]*ts_B[ i ][ tx ];
		}
		__syncthreads();

	}

	C[ row*N + col ] = Pvalue;
}

__global__ void matrixMultiplyTiling3(double *A, double *B, double *C ){

	int row, col;
	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	double Pvalue = 0;

	__shared__ double ts_A[ TILE_WIDTH3 ][ TILE_WIDTH3 ];
	__shared__ double ts_B[ TILE_WIDTH3 ][ TILE_WIDTH3 ];

	C[ row*N + col ] = 0;

	for( int p = 0; p < N/TILE_WIDTH3; p++ ){

		ts_A[ty][tx] = A[ row*N + tx +  p*TILE_WIDTH3 ];
		ts_B[ty][tx] = B[ (ty + p*TILE_WIDTH3)*N + col ];
		__syncthreads();
	
		for( int i = 0; i < TILE_WIDTH3; i++ ){
			Pvalue += ts_A[ ty ][ i ]*ts_B[ i ][ tx ];
		}
		__syncthreads();

	}

	C[ row*N + col ] = Pvalue;
}

__global__ void matrixMultiplyTiling4(double *A, double *B, double *C ){

	int row, col;
	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	double Pvalue = 0;

	__shared__ double ts_A[ TILE_WIDTH4 ][ TILE_WIDTH4 ];
	__shared__ double ts_B[ TILE_WIDTH4 ][ TILE_WIDTH4 ];

	C[ row*N + col ] = 0;

	for( int p = 0; p < N/TILE_WIDTH4; p++ ){

		ts_A[ty][tx] = A[ row*N + tx +  p*TILE_WIDTH4 ];
		ts_B[ty][tx] = B[ (ty + p*TILE_WIDTH4)*N + col ];
		__syncthreads();
	
		for( int i = 0; i < TILE_WIDTH4; i++ ){
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
	
	dim3 threads1( TILE_WIDTH1, TILE_WIDTH1 );
	dim3 blocks1( N/TILE_WIDTH1, N/TILE_WIDTH1 );
	
	dim3 threads2( TILE_WIDTH2, TILE_WIDTH2 );
	dim3 blocks2( N/TILE_WIDTH2, N/TILE_WIDTH2 );
	
	dim3 threads3( TILE_WIDTH3, TILE_WIDTH3 );
	dim3 blocks3( N/TILE_WIDTH3, N/TILE_WIDTH3 );
	
	dim3 threads4( TILE_WIDTH4, TILE_WIDTH4 );
	dim3 blocks4( N/TILE_WIDTH4, N/TILE_WIDTH4 );

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
	matrixMultiplyTiling1<<< blocks1, threads1 >>>( d_A, d_B, d_C );
	cudaEventRecord( stop );
	
	myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
	print_matrix_to_file( h_C, N, N );
	cudaEventSynchronize( stop );
	
	float milliseconds = 0;
	cudaEventElapsedTime( &milliseconds, start, stop );
	printf( "%f\n", milliseconds );
	
	cudaEventRecord( start );
	matrixMultiplyTiling2<<< blocks2, threads2 >>>( d_A, d_B, d_C );
	cudaEventRecord( stop );
	
	myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
	print_matrix_to_file( h_C, N, N );
	cudaEventSynchronize( stop );
	
	milliseconds = 0;
	cudaEventElapsedTime( &milliseconds, start, stop );
	printf( "%f\n", milliseconds );
	
	cudaEventRecord( start );
	matrixMultiplyTiling3<<< blocks3, threads3 >>>( d_A, d_B, d_C );
	cudaEventRecord( stop );
	
	myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
	print_matrix_to_file( h_C, N, N );
	cudaEventSynchronize( stop );
	
	milliseconds = 0;
	cudaEventElapsedTime( &milliseconds, start, stop );
	printf( "%f\n", milliseconds );
	
	cudaEventRecord( start );
	matrixMultiplyTiling4<<< blocks4, threads4 >>>( d_A, d_B, d_C );
	cudaEventRecord( stop );
	
	myCudaCheck( cudaMemcpy( h_C, d_C, mat_size, cudaMemcpyDeviceToHost ) );
	print_matrix_to_file( h_C, N, N );
	cudaEventSynchronize( stop );
	
	milliseconds = 0;
	cudaEventElapsedTime( &milliseconds, start, stop );
	printf( "%f\n", milliseconds );

	
	return 0;
}
