
#include "header.cuh"

struct Parms parms = { 0.1, 0.1 };

/*****************************************************************************
* subroutine inidat - Initialize Array
*****************************************************************************/
void inidat(int nx, float **u)
{
	int ix, iy;
	for (ix = 0; ix <= nx - 1; ix++)
	{
		for (iy = 0; iy <= nx - 1; iy++)
		{
			u[ix][iy] = (float)(ix * (nx - ix - 1) * iy * (nx - iy - 1));
		}
	}
}

/**************************************************************************
* subroutine prtdat - Print the results
**************************************************************************/
void prtdat(int nx, float** u, char *fnam)
{
	int ix, iy;
	FILE *fp;
	fp = fopen(fnam, "w");
	for (ix = 0; ix < nx; ix++)
	{
		for (iy = 0; iy < nx; iy++)
		{
			fprintf(fp, "%6.1f", u[ix][iy]);
			if (iy != nx - 1)
			{
				fprintf(fp, " ");
			}
			else
			{
				fprintf(fp, "\n");
			}
		}
	}
	fclose(fp);
}

// Create 2D array with sequential memory positions
float** SeqAllocate(int size_of_matrix) {
	float* sequence = (float*) malloc(size_of_matrix*size_of_matrix*sizeof(float));
	float** matrix = (float**) malloc(size_of_matrix*sizeof(float *));
	int i;
	for (i = 0; i<size_of_matrix; i++)
		matrix[i] = &(sequence[i*size_of_matrix]);

	return matrix;
}

// Free 2D array with sequential memory positions
void SeqFree(float** memory_ptr)
{
	free(memory_ptr[0]);
	free(memory_ptr);
}

// Assigh value to sequential 2D Array which is in GPU
__global__ void Assign(float *d_sequence, float** d_matrix )
{
	int i = threadIdx.x;
	d_matrix[i] = &(d_sequence[i* blockDim.x]);
}

// Create 2D array with sequential memory positions in GPU
float** cudaSeqAllocate(int size_of_matrix) {
	
	float* d_sequence;
	cudaMalloc((void **)&d_sequence, size_of_matrix*size_of_matrix*sizeof(float));

	float** d_matrix;
	cudaMalloc((void ***)&d_matrix, size_of_matrix*sizeof(float *));

	Assign << <1 , size_of_matrix >> >(d_sequence, d_matrix);
	cudaDeviceSynchronize();

	return d_matrix;
}

// Free Kernel
__global__ void cudaFreeKernel(float** memory_ptr, float** d_memorystart)
{
	*d_memorystart = memory_ptr[0];
}

// Free 2D array with sequential memory positions in GPU
void cudaSeqFree(float** memory_ptr)
{
	float** d_memorystart;
	cudaMalloc((void ***)&d_memorystart, sizeof(float*) );
	cudaFreeKernel << <1, 1 >> >(memory_ptr, d_memorystart);
	float* memorystart;
	cudaMemcpy(&memorystart, d_memorystart, sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(memorystart);
	cudaFree(memory_ptr);
}