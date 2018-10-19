#include<stdio.h>
#include<string.h>
#include<time.h>
#include<cuda.h>

#define MAXWORDS 20000
void myCudaCheck(cudaError_t err) {
        if(err != cudaSuccess) {
                printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
                exit(EXIT_FAILURE);
        }
}
__host__ __device__ int power( int a, int b ){
	
	int val=1;
	for( int i = 0; i<b; i++){
		val *= a;
	}
	return val;
}


__global__ void count_grams( int *d_word_size, int *d_counts, int N, int shared_mem_size, int word_count){
	extern __shared__ int array[];

	int *count_index = array;
	int *counts = &count_index[shared_mem_size/2];
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int threadid = threadIdx.x;
	int i =threadid;
	int index = 0;
	while( i < shared_mem_size/2 ){
		count_index[i] = i;
		counts[i] = 0;
		i += blockDim.x;
	}
	__syncthreads();
	id = blockIdx.x*blockDim.x + threadIdx.x;

	while( id < word_count - N + 1 ){
		for( i = 0; i < N; i++ ){
			index += (d_word_size[id + i]-1)*power(20,i);
		}
		int index_shared = 0;
		for( i = 0; i< shared_mem_size/2; i++){
			if( index == count_index[i] ){
				atomicAdd(&counts[i], 1 );
				index_shared = 1;
			}
			
	
		}
		if( index_shared == 0){
			
			atomicAdd( &d_counts[index] , 1);
		}
		id = id + gridDim.x*blockDim.x;
		index = 0;
	}
	__syncthreads();	
	threadid = threadIdx.x;
	while( threadid < shared_mem_size/2 ){
		d_counts[count_index[threadid]] = counts[threadid];
		threadid += blockDim.x;
	}
}

int main(int argc, char **argv){
	
	unsigned int N = atoi(argv[1]);
	int i=0;
	
	int word_size[MAXWORDS];
	int *counts;
	
	int *d_word_size;
	int *d_counts;
	int shared_mem_size = 1<<12;
	
	char curWord[40];
	int totalWordCount = 0;
	FILE *ip;
	ip = fopen(argv[2], "r" );
	
	while(fscanf(ip,"%s",curWord) != EOF && totalWordCount < MAXWORDS){
		int size = strlen(curWord);
		int j = 0;
		int reduce = 0;
		for( i = 0;i<size;i++){
			if( curWord[i] == '-' ){
				word_size[totalWordCount] = i - j;
				word_size[totalWordCount] -= reduce;
				reduce = 0;
				j = i+1;
				totalWordCount += 1;
			}
			else {
				if(i == size-1){
					if(!( (('a'<=curWord[i])&& (curWord[i]<='z')) || (('A'<=curWord[i])&&(curWord[i]<='Z'))  )){
						int k = i;
						while(!( (('a'<=curWord[k])&& (curWord[k]<='z')) || (('A'<=curWord[k])&&(curWord[k]<='Z'))  )){
							k = k - 1;
						}
						word_size[totalWordCount] = k - j + 1;
						word_size[totalWordCount] -= reduce;
						reduce = 0;
						totalWordCount += 1;
					}
					else{
						word_size[totalWordCount] = i - j+1;
						word_size[totalWordCount] -= reduce;
						reduce = 0;
						totalWordCount += 1;
					}
				}
				if(i == 0){
					int k = i;
					while(!( (('a'<=curWord[k])&& (curWord[k]<='z')) || (('A'<=curWord[k])&&(curWord[k]<='Z'))  )){
						k = k + 1;
					}
					reduce = k;

				}
			}

		}
		
			
	}
	size_t size_bytes = totalWordCount*sizeof(int);
	size_t count_bytes = pow(20,N)*sizeof(int);
	counts = (int*) malloc(count_bytes);
	for(int j = 0; j<pow(20,N);j++){
		counts[j] = 0;
	}

	myCudaCheck( cudaMalloc( &d_word_size, size_bytes ) );
	myCudaCheck( cudaMalloc( &d_counts, count_bytes ) );
	myCudaCheck( cudaMemcpy( d_word_size, word_size, size_bytes, cudaMemcpyHostToDevice ));
	myCudaCheck( cudaMemcpy( d_counts, counts, count_bytes, cudaMemcpyHostToDevice ));
	//kernel call
	count_grams<<<1,128, shared_mem_size*sizeof(int)>>>( d_word_size, d_counts, N, shared_mem_size, totalWordCount);
	myCudaCheck( cudaMemcpy( counts, d_counts, count_bytes, cudaMemcpyDeviceToHost ) );
	FILE *fp = fopen("ee16b060_out.txt", "w");
	for( int j = 0; j< pow(20,N); j++){
		if( counts[j] !=0){
			int index = j;
			for( int k = 0; k < N; k++ ){
				fprintf(fp, "%d ",  (index)%20 + 1); 
				index = (int) (index/20);
			}
			fprintf(fp, "%d\n", counts[j]);
		}
	}
	return 0;
}
