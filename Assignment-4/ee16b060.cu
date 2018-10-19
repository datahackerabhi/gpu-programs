/*
Template code for convolution. CS6023, IITM */
#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define W 1024 // Input DIM
#define OW (W-4) // Output DIM
#define D 8   // Input and Kernel Depth
#define T 5  // Kernel DIM
#define N 128 // Number of kernels

void fillMatrix(unsigned char *matrix){

unsigned char (*m)[W][D]=(unsigned char (*)[W][D])matrix;

for(int i=0;i<W;i++){
	for(int j=0;j<W;j++){
		for(int k=0;k<D;k++){
			m[i][j][k]=(i*j+j*k+i*k+i*2+j*3+k*4)%255;
				}
			}
		}
}



void fillKernel(float *kernel){

float (*t)[T][T][D]=(float (*)[T][T][D])kernel;

for(int i=0;i<N;i++){
	for(int j=0;j<T;j++){
		for(int k=0;k<T;k++){
			for(int l=0;l<D;l++){
			t[i][j][k][l]=fmod(-(i+1)*2.1+(j+1)*3.2-(k+1)*4.8+(l+1)*7.1,1.0);
				}
			}
		}
	}
}

void print_matrix_to_file(float *m){

	const char *fname = "assignment4_out";
	FILE *f = fopen(fname, "w");

	float (*mat)[OW][OW]=(float (*)[OW][OW])m;		

	for(unsigned i=0; i < N; i++) {
		for(unsigned j=0; j < OW; j++)
			for(unsigned k=0;k<OW;k++)
				fprintf(f,"%4.4f ", mat[i][j][k]);
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void mykernel( unsigned char *Dmatrix,const float* __restrict__ Dkernel, float *Doutput ){

	float (*kernel)[T][T][D]=(float (*)[T][T][D])Dkernel;	
	unsigned char (*matrix)[W][D]=(unsigned char (*)[W][D])Dmatrix;
	float (*output)[OW][OW]=(float (*)[OW][OW])Doutput;

	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	
	int width = (T-1)/2;
	int centerx = tidx + width;
	int centery = tidy + width;
	
	float val = 0;

	if( (tidx<OW) && (tidy<OW) ){
		
		for( int l = 0; l < N; l++ ){
		for( int i = -width; i <= width; i++ ){
			for( int j = -width; j <= width; j++){
				for( int k = 0; k < D; k++ ){
					val += ((float) matrix[centery+i][centerx+j][k])*kernel[l][width+i][width+j][k];
				}
			}
		}
		output[l][tidy][tidx] = val;
		val = 0;
		}

	}
}

int main()
{

	unsigned char *matrix=(unsigned char*)malloc(sizeof(unsigned char)*W*W*D);
	float *kernel=(float*)malloc(sizeof(float)*T*T*D*N);
	float *output=(float *)malloc(sizeof(float)*N*OW*OW);
	
	dim3 threads( 32, 32);
	dim3 blocks(32, 32);

	fillMatrix(matrix);
	fillKernel(kernel);


	unsigned char *Dmatrix;cudaMalloc(&Dmatrix,sizeof(unsigned char)*W*W*D);
	float *Dkernel;cudaMalloc(&Dkernel,sizeof(float)*N*T*T*D);
	float *Doutput;cudaMalloc(&Doutput,sizeof(float)*N*OW*OW);

	cudaMemcpy(Dmatrix, matrix, sizeof(unsigned char)*W*W*D,cudaMemcpyHostToDevice);
	cudaMemcpy(Dkernel, kernel, sizeof(float)*T*T*D*N,cudaMemcpyHostToDevice);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);

	//Make your cuda kernel call
	mykernel<<< blocks, threads >>>( Dmatrix, Dkernel, Doutput );	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
	    printf("Error: %s\n", cudaGetErrorString(err));

	cudaDeviceSynchronize();


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f\n",milliseconds);


	cudaMemcpy(output, Doutput, sizeof(float)*N*OW*OW,cudaMemcpyDeviceToHost);

	//Use print_matrix_to_file function only 
	print_matrix_to_file(output);

}
