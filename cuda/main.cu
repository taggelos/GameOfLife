
#include "header.cuh"

// 
__global__ void CopyArrayA(float ** handler, float** d_A)
{
	*handler = &d_A[0][0];
}

// Caculations
__global__ void Update(float **A, float** B, struct Parms* d_parms)
{
	// Take Indexes
	int i, j,temp;
	temp = blockIdx.x * blockDim.x + threadIdx.x;
	i = temp / NPROB;
	j = temp % NPROB;

	// Check for errors and values of edges
	if (i < 0 || i > NPROB - 1 || j < 0 || j > NPROB - 1)
	{
		return;
	}
	else if (i == 0 || i == NPROB - 1 || j == 0 || j == NPROB - 1)
	{
		B[i][j] = 0; 
		return;

	}

	// Calculate formula
	B[i][j] = A[i][j] +
		d_parms->cx * (A[i + 1][j] + A[i - 1][j] - 2 * A[i][j]) +
		d_parms->cy * (A[i][j + 1] + A[i][j - 1] - 2 * A[i][j]);
}

int main()
{

	// Create CPU array A
	float** A = SeqAllocate(NPROB);

	// Initialize and save data
	inidat(NPROB, A);
	prtdat(NPROB, A, "initial.dat");

	// Create GPU array A
	float** d_A = cudaSeqAllocate(NPROB);

	// Create GPU array B
	float** d_B = cudaSeqAllocate(NPROB);

	struct timeval  tv1, tv2;
	gettimeofday(&tv1, NULL);

	// Copy memory from CPU to GPU
	float** d_temp;
	cudaMalloc((void ***)&d_temp,sizeof(float*));
	CopyArrayA << <1,1>> >(d_temp, d_A);
	float* temp;
	cudaMemcpy(&temp, d_temp, sizeof(float*), cudaMemcpyDeviceToHost);
	cudaMemcpy( temp, &A[0][0], NPROB*NPROB*sizeof(float), cudaMemcpyHostToDevice);

	// Pass parms to GPU space
	struct Parms* d_parms;
	cudaMalloc((void **)&d_parms, sizeof(struct Parms));
	cudaMemcpy(d_parms, &parms, sizeof(struct Parms), cudaMemcpyHostToDevice);

	float** handler;
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
	cudaMemcpy(&temp, d_temp, sizeof(float*), cudaMemcpyDeviceToHost);
	cudaMemcpy(*A, temp, NPROB*NPROB*sizeof(float), cudaMemcpyDeviceToHost);

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

