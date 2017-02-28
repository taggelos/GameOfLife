#include "header.h"
#include <omp.h>

struct Parms parms = {0.1, 0.1};

/**************************************************************************
* subroutine update
****************************************************************************/
void Independent_Update(int** A, int** B, int size)
{
	int i, j;
#ifdef __OMP__
	int thread_count = (size < MAXTHREADS) ? size : MAXTHREADS;
	// Create size threads with openmp or max
	#pragma omp parallel for num_threads( thread_count ) collapse(2)
#endif
	// Calcalate inside data
	for (i = 1; i < size-1; i++)
	{
		for (j = 1; j < size-1; j++)
		{
			int sum = A[i+1][j] + A[i][j+1] + A[i-1][j] + A[i][j-1] + A[i-1][j-1] + A[i-1][j+1] + A[i+1][j-1] +A[i+1][j+1];
			if(A[i][j]==1){
				if(sum <=1)
					B[i][j]=0;
				else if (sum<=3)
					B[i][j]=1;
				else
					B[i][j]=0;
				
			}
			else{
				if(sum == 3)
					B[i][j]=1;
				else
					B[i][j]=0;
			}
		}
	}
}

void Dependent_Update(int** A, int** B, int* size, int** Row)
{
	int i, j;
	int T1, T2, Τ3 , Τ4;
#ifdef __OMP__
	int thread_count = (size < MAXTHREADS) ? size : MAXTHREADS;
	// Create size threads with openmp or max
	#pragma omp parallel for num_threads( thread_count ) private(T1) private(T2) collapse(2)
#endif
	// Calculate the edges of array
	// 2 for instructions for better split to threads
	for (i = 1; i < size-1; i++)
	{
		for (j = 0; j < 4; j++)
		{
			if (j == 0)	// South Neighbor. Same format for the below if statement
			{
				int sum = A[size-1][i+1] + A[size-1][i-1] + A[size-2][i] + Row[DOWN][i] + A[size-2][i-1] + A[size-2][i+1] + Row[DOWN][i-1] +Row[DOWN][i+1];  
				B[size-1][i] = A[size-1][i] ; 								  // Calculate formula
					
			}
			else if(j == 1) // North Neighbor
			{
				T1 = (i != size-1) ? A[0][i + 1] : Row[RIGHT][0];
				T2 = (i != 0) ? A[0][i-1] : Row[LEFT][0];
				B[0][i] = A[0][i] + 
					parms.cx * ( A[1][i] + Row[UP][i] - 2.0 * A[0][i] ) + 
					parms.cy * ( T1 + T2 - 2.0 * A[0][i] );
			}
			else if(j == 2) // West Neighbor
			{
				T1 = (i != size-1) ? A[i + 1][0] : Row[DOWN][0];
				T2 = (i != 0) ? A[i-1][0] : Row[UP][0];
				B[i][0] = A[i][0] + 
					parms.cx * ( T1 + T2 - 2.0 * A[i][0] ) + 
					parms.cy * ( A[i][1] + Row[LEFT][i] - 2.0 * A[i][0] );
			}
			else	// East Neighbor
			{
				T1 = (i != size-1) ? A[i + 1][size-1] : Row[DOWN][size-1];
				T2 = (i != 0) ? A[i-1][size-1] : Row[UP][size-1];
				B[i][size-1] = A[i][size-1] + 
					parms.cx * ( T1 + T2 - 2.0 * A[i][size-1] ) + 
					parms.cy * ( Row[RIGHT][i] + A[i][size-2] - 2.0 * A[i][size-1] );
			}
		}
	}
}


inline int diffa(int** A, int** B, int size)
{
	int diff, sum = 0;
	int i, j;
#ifdef __OMP__
	int thread_count = (size < MAXTHREADS) ? size : MAXTHREADS;
	// Create size threads with openmp or max
	#pragma omp parallel for num_threads( thread_count ) reduction(+:sum) private(diff) collapse(2)
#endif
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			diff = B[i][j]-A[i][j];
			sum += (diff*diff);
		}
	}
	return sum;
}

/*****************************************************************************
* subroutine inidat   - Initialize Array
*****************************************************************************/
inline void inidat(int size, int** u)
{
	int ix, iy;

	for (ix = 0; ix < size; ix++)
	{
		for (iy = 0; iy < size; iy++)
		{
			u[ix][iy] = (ix + 1) * ((size + 2) - ix - 2) * (iy + 1) * ((size + 2) - iy - 2);
		}
	}
}

/**************************************************************************
* subroutine prtdat - Print the results
**************************************************************************/
inline void prtdat(int size, int** u, char *fnam)
{
	int ix, iy;
	FILE *fp;
	fp = fopen(fnam, "w");
	for (ix = 0; ix < size; ix++)
	{
		for (iy = 0; iy < size-1; iy++)
		{
			fprintf(fp, "%6.1f ", u[ix][iy]);
		}
		fprintf(fp, "%6.1f\n", u[ix][iy]);
	}
	fclose(fp);
}

// Create 2D array with sequential memory positions
inline int** SeqAllocate(int size) {
	int* sequence = malloc(size*size*sizeof(int));
	int** matrix = malloc(size*sizeof(int *));
	int i;
	for (i = 0; i < size; i++)
		matrix[i] = &(sequence[i*size]);

	return matrix;
}

// Free 2D array with sequential memory positions
inline void SeqFree(int** memory_ptr)
{
	memory_ptr[0][2] = -3;
	free(memory_ptr[0]);
	free(memory_ptr);
}

// Free worker structures
inline void finalize(Finalize* fin)
{
	MPI_Wait(fin->request, MPI_STATUS_IGNORE);
	SeqFree(fin->A);
	free(fin->request);
	free(fin);
}
