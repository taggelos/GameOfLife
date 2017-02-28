#include "header.h"
#include <omp.h>

struct Parms parms = {0.1, 0.1};

/**************************************************************************
* subroutine update
****************************************************************************/

inline int Populate(int i, int j, int sum, int** A ){
	int res;
	if(A[i][j]==1){
		if(sum <=1)
			res=0;
		else if (sum<=3)
			res=1;
		else
			res=0;
		
	}
	else{
		if(sum == 3)
			res=1;
		else
			res=0;
	}	
	return res;
}


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
			
			B[i][j] = Populate(i,j,sum,A);
		}
	}
}

void Dependent_Update(int** A, int** B, int size, int** Row)
{
	int i, j, sum;
	int T1, T2;
#ifdef __OMP__
	int thread_count = (size < MAXTHREADS) ? size : MAXTHREADS;
	// Create size threads with openmp or max
	#pragma omp parallel for num_threads( thread_count ) collapse(2)
#endif
	// Calculate the edges of array
	// 2 for instructions for better split to threads
	for (i = 1; i < size-1; i++)
	{
		for (j = 0; j < 4; j++)
		{
			if (j == 0)	// South Neighbor. Same format for the below if statement
			{
				sum = A[size-1][i+1] + A[size-1][i-1] + A[size-2][i] + Row[DOWN][i] + A[size-2][i-1] + A[size-2][i+1] + Row[DOWN][i-1] +Row[DOWN][i+1];  
				B[size-1][i] = Populate(size-1,i,sum,A);
									
			}
			else if(j == 1) // North Neighbor
			{
				
				sum = A[0][i+1] + A[0][i-1] + A[1][i] + Row[UP][i] + A[1][i-1] + A[1][i+1] + Row[UP][i+1] +Row[UP][i-1];

				B[0][i] = Populate(0,i,sum,A);
			}
			else if(j == 2) // West Neighbor
			{

				sum = A[i][1] + A[i-1][0] + A[i+1][0] + Row[LEFT][i] + A[i+1][1] + A[i-1][1] + Row[LEFT][i+1] + Row[LEFT][i-1];
				
				B[i][0] = Populate(i,0,sum,A);
			}
			else	// East Neighbor
			{
				sum = A[i][size-2] + A[i-1][size-1] + A[i+1][size-1] + Row[RIGHT][i] + A[i-1][size-2] + A[i+1][size-2] + Row[RIGHT][i+1] + Row[RIGHT][i-1];

				B[i][size-1] = Populate(i,size-1,sum,A);
			}
		}
	}
}

void UpdateDiag(int** A, int** B, int size, int** DiagRecvTable , int** Row)
{
	int i, j, sum;

#ifdef __OMP__
	int thread_count = (size < MAXTHREADS) ? size : MAXTHREADS;
	// Create size threads with openmp or max
	#pragma omp parallel for num_threads( thread_count ) 
#endif

	for (j = 0; j < 4; j++)
	{
		if (j == 0) {
			//Upper Left
			sum = A[0][1] + A[1][0] + A[1][1] + Row[LEFT][0] + Row[LEFT][1] + Row[UP][0] + Row[UP][1] + DiagRecvTable[UP][DIAGLEFT];
			B[0][0]=Populate(0,0,sum,A);
		}
		else if (j == 1){
			//Upper Right
			sum = A[1][size-1] + A[0][size-2] + A[1][size-2] + Row[RIGHT][0] + Row[RIGHT][1] + Row[UP][size-1] + Row[UP][size-2] + DiagRecvTable[UP][DIAGRIGHT];
			B[0][size-1]=Populate(0,size-1,sum,A);
		}
		else if (j == 2){
			//Bottom Left
			sum = A[size-1][1] + A[size-2][0] + A[size-2][1] + Row[LEFT][size-1] + Row[LEFT][size-2] + Row[DOWN][0] + Row[DOWN][1] + DiagRecvTable[DOWN][DIAGLEFT];
			B[size-1][0]=Populate(size-1,0,sum,A);
		}
		else {
			//Bottom Right
			sum = A[size-2][size-1] + A[size-1][size-2] + A[size-2][size-2] + Row[RIGHT][size-1] + Row[RIGHT][size-2] + Row[DOWN][size-1] + Row[DOWN][size-2] + DiagRecvTable[DOWN][DIAGRIGHT];
			B[size-1][size-1]=Populate(size-1,size-1,sum,A);
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

#ifdef __OMP__
	int thread_count = (size < MAXTHREADS) ? size : MAXTHREADS;
	// Create size threads with openmp or max
	#pragma omp parallel for num_threads( thread_count ) collapse(2)
#endif

	for (ix = 0; ix < size; ix++)
	{
		for (iy = 0; iy < size; iy++)
		{
			u[ix][iy] = rand() % 2;
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
			fprintf(fp, "%d ", u[ix][iy]);
		}
		fprintf(fp, "%d\n", u[ix][iy]);
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
	//memory_ptr[0][2] = -3;
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
