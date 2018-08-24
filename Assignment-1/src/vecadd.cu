#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

void myCudaCheck() {
	cudaError_t err  = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
		exit(EXIT_FAILURE);
	}
}

__global__ void addVec( int* A, int* B, double* C, int N ){
	
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	
	if( i < N ) 
		C[i] = A[i] + B[i];

}

int main(int argc, char **argv) {
	
	if( argc != 3) {
		printf("Usage: <executable> file1.dat file2.dat\n");
		exit(1);
	}
	
	int h_num = 32768;
	size_t size_int = 32768*sizeof(int);
	size_t size_double = 32768*sizeof(double);
	int h_A[h_num], h_B[h_num];
	double h_C[h_num];
	int *d_A, *d_B;
	double *d_C;
	FILE *fp, *in1, *in2;

	in1 = fopen( argv[1], "r" );
	in2 = fopen( argv[2], "r" );

	fp = fopen( "ee16b060_3_out.txt", "w" );

	for(int i = 0; i < h_num; i++ ){
		fscanf(in1, "%d", &h_A[i]);
		fscanf(in2, "%d", &h_B[i]);
	}

	cudaMalloc( &d_A, size_int );
	cudaMalloc( &d_B, size_int );
	cudaMalloc( &d_C, size_double );

	cudaMemcpy( d_A, h_A, size_int, cudaMemcpyHostToDevice );
	cudaMemcpy( d_B, h_B, size_int, cudaMemcpyHostToDevice );
	
	addVec<<<128,256>>>(d_A, d_B, d_C, h_num );

	cudaMemcpy( h_C, d_C, size_double, cudaMemcpyDeviceToHost );
	
	cudaDeviceSynchronize();

	for(int i = 0; i < h_num; i++){
		fprintf( fp, "%d ", h_A[i] );
		fprintf( fp, "%d ", h_B[i] );
		fprintf( fp, "%lf\n", h_C[i] );
	}

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );
	
	fclose(fp);
	fclose(in1);
	fclose(in2);
	
	myCudaCheck();

	return 0;
}

