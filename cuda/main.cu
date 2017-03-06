
#include "header.cuh"

// 
__global__ void CopyArrayA(bool ** handler, bool** d_A)
{
	*handler = &d_A[0][0];
}

bool Populate(int sum, bool A ){
	
	return (sum == 3) || (A == 1 && sum == 2);
}

// Caculations
__global__ void Update(bool **A, bool** B, struct Parms* d_parms)
{
	// Take Indexes
	int i, j,temp,sum,down,up,left,right;
	temp = blockIdx.x * blockDim.x + threadIdx.x;
	i = temp / NPROB;
	j = temp % NPROB;


	// Check for errors and values of edges
	if (i < 0 || i > NPROB - 1 || j < 0 || j > NPROB - 1)
	{
		return;
	}

	down = (i != size-1) ? i+1 : 0;
	up = (i != 0) ? i-1 : size-1;
	left = (j != 0) ? j-1 : size-1;
	right = (j != size-1) ? j+1 : 0;

	sum = A[down][j] + A[up][j] + A[i][left] + A[i][right] + A[down][left] + A[up][left] + A[down][right] + A[up][right] ;
	// Calculate formula
	B[i][j] = Populate(sum,A[i][j]);
}

int main()
{

	// Create CPU array A
	bool** A = SeqAllocate(NPROB);

	// Initialize and save data
	inidat(NPROB, A);
	prtdat(NPROB, A, "initial.dat");

	// Create GPU array A
	bool** d_A = cudaSeqAllocate(NPROB);

	// Create GPU array B
	bool** d_B = cudaSeqAllocate(NPROB);

	struct timeval  tv1, tv2;
	gettimeofday(&tv1, NULL);

	// Copy memory from CPU to GPU
	bool** d_temp;
	cudaMalloc((void ***)&d_temp,sizeof(bool*));
	CopyArrayA << <1,1>> >(d_temp, d_A);
	bool* temp;
	cudaMemcpy(&temp, d_temp, sizeof(bool*), cudaMemcpyDeviceToHost);
	cudaMemcpy( temp, &A[0][0], NPROB*NPROB*sizeof(bool), cudaMemcpyHostToDevice);

	// Pass parms to GPU space
	struct Parms* d_parms;
	cudaMalloc((void **)&d_parms, sizeof(struct Parms));
	cudaMemcpy(d_parms, &parms, sizeof(struct Parms), cudaMemcpyHostToDevice);

	bool** handler;
	int i;
	for (i = 0; i<GENERATION; i++)
	{
		// Do calculations
		Update << < (NPROB*NPROB + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> >(d_A, d_B, d_parms);
		cudaDeviceSynchronize();

		// set curent result to the new one
		handler = d_A;
		d_A = d_B;
		d_B = handler; 
	}

	//Copy data from GPU to CPU
	CopyArrayA << <1, 1 >> >(d_temp, d_A);
	cudaMemcpy(&temp, d_temp, sizeof(bool*), cudaMemcpyDeviceToHost);
	cudaMemcpy(*A, temp, NPROB*NPROB*sizeof(bool), cudaMemcpyDeviceToHost);

	// Wait for all threads
	cudaDeviceSynchronize();
	gettimeofday(&tv2, NULL);
	printf("Finished after %f seconds\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
	printf("Writing final.dat file \n");

	// Print Results
	prtdat(NPROB, A, "final.dat");

	// Free GPU array A
	cudaSeqFree(d_A);

	// Free GPU array B
	cudaSeqFree(d_B);

	// Free CPU array A
	SeqFree(A);

    return 0;
}

